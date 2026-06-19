"""Model-selection harness — optimise the no-flat **test accuracy** safely.

The headline metric is accuracy on the no-flat test slice (flat ``Open == Close``
bars excluded). To tune toward it without contaminating the test set, this module
selects hyperparameters on a **no-flat validation fold carved from the training
half** (time-ordered: the last ``val_frac`` of the train half), then performs the
**single** final evaluation on the real test set.

Workflow per algorithm:
  1. Grid-search hyperparameters; score each combo by no-flat **validation**
     accuracy (model fit on the inner-train, flat rows dropped).
  2. (optional W3) tune the decision threshold on the validation fold.
  3. Retrain the best config on the **full** no-flat training half (optionally
     with |move| sample weighting), predict the test set, and report no-flat
     **test** accuracy / MCC.

Guardrails honoured: seed 42; flat rows dropped only after features are built;
scalers fit on train only; the test half is touched once, at the end; every
prediction set is persisted to ``data/processed/`` (Rule 7).

Run with::

    python -m src.tuning --algos baseline rf gbm svm --featset v2
    python -m src.tuning --algos rf --featset v3 --move-weight --tune-threshold
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import src.statistics as statistics
from src.config import load_config
from src.features import build_features
from src.labels import build_labels, flat_mask
from src.models import baseline, rf
from src.models.gbm import predict as gbm_predict
from src.models.gbm import train as gbm_train
from src.models.svm import predict as svm_predict
from src.models.svm import train as svm_train
from src.split import split

_PROC = Path("data/processed")
_TMP = _PROC / "_tuning_tmp"
_DOCS = Path("docs/notes")

_ALGOS: tuple[str, ...] = ("baseline", "rf", "gbm", "svm")
_DISPLAY = {
    "baseline": "Logistic Regression",
    "rf": "Random Forest",
    "gbm": "Gradient Boosting (XGBoost)",
    "svm": "SVM (RBF kernel)",
}

# SVM RBF training is O(n²); cap the inner-train sample during the sweep.
_SVM_SWEEP_N = 30_000

# Curated, seed-42 grids. Lists of explicit param dicts (not full cross-products)
# to bound runtime. RF/GBM grids attack the overfit diagnosed on the no-flat test
# (RF's unbounded depth; GBM depth/shrinkage/regularisation); LogReg sweeps the
# regularisation strength/penalty that made it the current champion.
_GRIDS: dict[str, list[dict]] = {
    # sklearn 1.8 unified API: l1_ratio (0=L2, 1=L1, between=elastic-net) + C.
    # L2 uses lbfgs (fast, robust on raw-scale features — matches the champion);
    # L1/elastic-net need saga (higher max_iter, converges well on v2/v3 scale).
    "baseline": [
        {"l1_ratio": 0.0, "C": 0.1, "solver": "lbfgs", "max_iter": 2000},
        {"l1_ratio": 0.0, "C": 1.0, "solver": "lbfgs", "max_iter": 2000},
        {"l1_ratio": 0.0, "C": 10.0, "solver": "lbfgs", "max_iter": 2000},
        {"l1_ratio": 1.0, "C": 1.0, "solver": "saga", "max_iter": 5000},
        {"l1_ratio": 0.5, "C": 1.0, "solver": "saga", "max_iter": 5000},
    ],
    "rf": [
        {"n_estimators": 300, "max_depth": d, "min_samples_leaf": m,
         "max_features": f}
        for d in (6, 12)
        for m in (50, 200)
        for f in ("sqrt", 0.3)
    ],
    "gbm": [
        {"n_estimators": 400, "max_depth": 2, "learning_rate": 0.05,
         "reg_lambda": 5.0, "reg_alpha": 1.0, "min_child_weight": 20},
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03,
         "reg_lambda": 5.0, "reg_alpha": 0.0, "min_child_weight": 5},
        {"n_estimators": 600, "max_depth": 3, "learning_rate": 0.02,
         "reg_lambda": 10.0, "reg_alpha": 1.0, "min_child_weight": 20},
        {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
         "reg_lambda": 1.0, "reg_alpha": 0.0, "min_child_weight": 1},
    ],
    "svm": [
        {"C": 0.5, "gamma": "scale"},
        {"C": 2.0, "gamma": "scale"},
        {"C": 1.0, "gamma": 0.05},
        {"C": 2.0, "gamma": 0.01},
    ],
}

def _feature_fn(featset: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return the feature builder for a feature-set key ('v1'/'v2'/'v3')."""
    if featset == "v1":
        return build_features
    if featset == "v2":
        from src.features_v2 import load_or_build_features_v2
        return load_or_build_features_v2
    if featset == "v3":
        from src.features_v3 import load_or_build_features_v3
        return load_or_build_features_v3
    raise ValueError(f"Unknown featset: {featset!r}")


# ---------------------------------------------------------------------------
# Train / predict / score dispatch (forwards sample_weight where given)
# ---------------------------------------------------------------------------

def _fit(algo: str, X: pd.DataFrame, y: pd.Series, params: dict,
         save_path: Path, sample_weight: np.ndarray | None = None) -> Any:
    """Fit one algorithm, forwarding sample_weight, persisting to save_path."""
    if algo == "baseline":
        return baseline.train(X, y, params=params, save_path=save_path,
                              sample_weight=sample_weight)
    if algo == "rf":
        return rf.train(X, y, params=params, save_path=save_path,
                        sample_weight=sample_weight)
    if algo == "gbm":
        return gbm_train(X, y, params=params, save_path=save_path,
                         sample_weight=sample_weight)
    if algo == "svm":
        return svm_train(X, y, params=params, save_path=save_path,
                         sample_weight=sample_weight)
    raise ValueError(f"Unknown algo: {algo!r}")


def _predict(algo: str, model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict binary labels with a fitted model (default threshold)."""
    if algo == "baseline":
        return baseline.predict(model, X)
    if algo == "rf":
        return rf.predict(model, X)
    if algo == "gbm":
        return gbm_predict(model, X)
    if algo == "svm":
        return svm_predict(model, X)
    raise ValueError(f"Unknown algo: {algo!r}")


def _scores(algo: str, model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return continuous decision scores for threshold tuning.

    Probabilistic models (baseline/rf/gbm) return P(class=1); the RBF SVM returns
    the signed distance to the hyperplane (``decision_function``) since it is fit
    with probability=False.
    """
    if algo == "svm":
        return model["clf"].decision_function(model["scaler"].transform(X))
    return model.predict_proba(X)[:, 1]


def _default_threshold(algo: str) -> float:
    """Default decision threshold: 0.0 for the SVM margin, else 0.5 probability."""
    return 0.0 if algo == "svm" else 0.5


def predict_with_threshold(
    algo: str, model: Any, X: pd.DataFrame, threshold: float | None = None
) -> np.ndarray:
    """Predict binary labels, optionally at a tuned decision threshold.

    With ``threshold=None`` this is the plain label prediction (``_predict``).
    Otherwise labels are ``_scores(...) >= threshold`` — probability ≥ threshold
    for baseline/rf/gbm, or signed SVM margin ≥ threshold for the RBF SVM. Used by
    the GUI to apply the threshold stored in ``tuned_params_{featset}.json``.

    Args:
        algo: One of baseline/rf/gbm/svm.
        model: A fitted model (or SVMModel dict for svm).
        X: Feature matrix to score.
        threshold: Tuned decision threshold; None → default label prediction.

    Returns:
        Integer ndarray of predicted labels (0 or 1).
    """
    if threshold is None:
        return _predict(algo, model, X)
    return (_scores(algo, model, X) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------

def _load_splits(
    cfg: dict, feat_fn: Callable[[pd.DataFrame], pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load + feature-build once → (X_train, raw_train, X_test, raw_test).

    All four are aligned to ``df.iloc[4:]`` and split 50/50 by time, matching the
    binary-suite / production prediction ordering.
    """
    from src.load import load_raw

    train_size = cfg["data"].get("train_size", 0.5)
    data_path = Path(cfg["data"]["path"])
    df = load_raw(data_path)
    features = feat_fn(df)
    raw_align = df.iloc[4:].reset_index(drop=True)
    X_train, X_test = split(features, train_size=train_size)
    raw_train, raw_test = split(raw_align, train_size=train_size)
    return X_train, raw_train, X_test, raw_test


class SelectionSplit:
    """Inner-train / no-flat-validation carved from the TRAIN half only.

    The test half is never represented here — ``val`` is the last ``val_frac`` of
    the training half, so all validation timestamps precede the test-half start
    (asserted by the leakage test).
    """

    def __init__(self, X_inner: pd.DataFrame, y_inner: pd.Series,
                 move_inner: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
                 val_start: pd.Timestamp):
        self.X_inner = X_inner
        self.y_inner = y_inner
        self.move_inner = move_inner
        self.X_val = X_val          # no-flat validation features
        self.y_val = y_val          # no-flat validation labels
        self.val_start = val_start  # first validation timestamp


def build_selection_split(
    cfg: dict,
    feat_fn: Callable[[pd.DataFrame], pd.DataFrame],
    val_frac: float = 0.2,
) -> SelectionSplit:
    """Carve a no-flat validation fold from the training half (no test leakage).

    Args:
        cfg: Parsed config dict (data.path, data.train_size).
        feat_fn: Feature builder applied to the raw DataFrame.
        val_frac: Fraction of the train half (most recent rows) used as validation.

    Returns:
        SelectionSplit with the flat-dropped inner-train (X/y/move) and the
        flat-dropped validation slice (X/y) plus the first validation timestamp.
    """
    X_train, raw_train, _, _ = _load_splits(cfg, feat_fn)
    n = len(X_train)
    cut = int(n * (1.0 - val_frac))

    X_inner_all, raw_inner = X_train.iloc[:cut], raw_train.iloc[:cut]
    X_val_all, raw_val = X_train.iloc[cut:], raw_train.iloc[cut:]

    # Inner-train: drop flat rows (focus on decisive up/down bars).
    keep_inner = ~flat_mask(raw_inner)
    X_inner = X_inner_all[keep_inner].reset_index(drop=True)
    y_inner = build_labels(raw_inner)[keep_inner].reset_index(drop=True)
    move_inner = (raw_inner["Close"].values - raw_inner["Open"].values)[keep_inner]

    # Validation: flat-dropped to match the no-flat metric.
    keep_val = ~flat_mask(raw_val)
    X_val = X_val_all[keep_val].reset_index(drop=True)
    y_val = build_labels(raw_val)[keep_val].reset_index(drop=True).to_numpy()
    val_start = pd.Timestamp(raw_val["Date and Time"].iloc[0])

    return SelectionSplit(X_inner, y_inner, move_inner, X_val, y_val, val_start)


def _full_train_test(
    cfg: dict, feat_fn: Callable[[pd.DataFrame], pd.DataFrame]
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame, np.ndarray,
           np.ndarray, np.ndarray]:
    """Build the final fit/eval arrays.

    Returns (X_train_nf, y_train_nf, move_train_nf, X_test, y_test, move_test,
    keep_test): the FULL no-flat training half, plus the whole test set, its
    per-bar move (Close - Open), and the no-flat keep mask (True where Close != Open).
    """
    X_train, raw_train, X_test, raw_test = _load_splits(cfg, feat_fn)

    keep_train = ~flat_mask(raw_train)
    X_train_nf = X_train[keep_train].reset_index(drop=True)
    y_train_nf = build_labels(raw_train)[keep_train].reset_index(drop=True)
    move_train_nf = (raw_train["Close"].values - raw_train["Open"].values)[keep_train]

    y_test = build_labels(raw_test).to_numpy()
    move_test = (raw_test["Close"].values - raw_test["Open"].values)
    keep_test = ~flat_mask(raw_test)
    return X_train_nf, y_train_nf, move_train_nf, X_test, y_test, move_test, keep_test


# ---------------------------------------------------------------------------
# Search + threshold tuning
# ---------------------------------------------------------------------------

def _move_weight(move: np.ndarray) -> np.ndarray:
    """Per-row weight from |move|, normalised by its median (≈1 on a typical bar)."""
    a = np.abs(np.asarray(move, dtype=float))
    med = np.median(a[a > 0]) if np.any(a > 0) else 1.0
    return a / med if med > 0 else np.ones_like(a)


def grid_search(
    algo: str,
    sel: SelectionSplit,
    grid: list[dict] | None = None,
) -> tuple[dict, float, list[tuple[dict, float]]]:
    """Grid-search one algorithm; score combos by no-flat validation accuracy.

    Args:
        algo: One of baseline/rf/gbm/svm.
        sel: SelectionSplit (inner-train + no-flat validation).
        grid: Param dicts to try; defaults to the module's curated grid.

    Returns:
        (best_params, best_val_accuracy, all_results) where all_results is a list
        of (params, val_accuracy) for every combo tried.
    """
    grid = grid or _GRIDS[algo]
    _TMP.mkdir(parents=True, exist_ok=True)

    X_inner, y_inner = sel.X_inner, sel.y_inner
    if algo == "svm" and len(X_inner) > _SVM_SWEEP_N:
        rng = np.random.RandomState(42)
        idx = np.sort(rng.choice(len(X_inner), _SVM_SWEEP_N, replace=False))
        X_inner = X_inner.iloc[idx].reset_index(drop=True)
        y_inner = y_inner.iloc[idx].reset_index(drop=True)

    results: list[tuple[dict, float]] = []
    best_params, best_acc = grid[0], -1.0
    for params in grid:
        model = _fit(algo, X_inner, y_inner, params, _TMP / f"{algo}_sweep.joblib")
        y_hat = _predict(algo, model, sel.X_val)
        acc = float((y_hat == sel.y_val).mean())
        results.append((params, acc))
        print(f"    [{algo}] val_acc={acc:.4f}  {params}")
        if acc > best_acc:
            best_acc, best_params = acc, params
    return best_params, best_acc, results


def tune_threshold(scores_val: np.ndarray, y_val: np.ndarray) -> float:
    """Pick the decision threshold maximising no-flat validation accuracy.

    Args:
        scores_val: Continuous decision scores on the validation fold.
        y_val: Binary validation labels (no-flat).

    Returns:
        The score threshold (>=) that maximises accuracy on the validation fold.
    """
    qs = np.quantile(scores_val, np.linspace(0.02, 0.98, 49))
    candidates = np.unique(np.concatenate([qs, [np.median(scores_val)]]))
    best_thr, best_acc = float(candidates[0]), -1.0
    for thr in candidates:
        acc = float(((scores_val >= thr).astype(int) == y_val).mean())
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr


def select_features(
    sel: SelectionSplit,
    score_algo: str = "baseline",
    k_grid: list[int] | None = None,
) -> tuple[list[str], list[tuple[int, float]]]:
    """Pick a feature subset that maximises no-flat validation accuracy.

    Ranks features by the magnitude of an L1 logistic regression's coefficients
    (fit on the standardised inner-train — train-only scaling, no leakage), then
    sweeps the top-k for several k and scores each subset on the no-flat
    validation fold with ``score_algo``.

    Args:
        sel: SelectionSplit (inner-train + no-flat validation).
        score_algo: Algorithm used to score each candidate subset (default
            logistic regression — fast and the current champion family).
        k_grid: Candidate subset sizes; defaults to {10,15,20,30,all}.

    Returns:
        (best_columns, results) where results is [(k, val_accuracy), ...].
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cols = list(sel.X_inner.columns)
    scaler = StandardScaler().fit(sel.X_inner)
    ranker = LogisticRegression(l1_ratio=1.0, C=0.1, solver="saga",
                                max_iter=2000, random_state=42)
    ranker.fit(scaler.transform(sel.X_inner), sel.y_inner)
    order = np.argsort(-np.abs(ranker.coef_).ravel())
    ranked = [cols[i] for i in order]

    n = len(cols)
    k_grid = k_grid or [k for k in (10, 15, 20, 30, n) if k <= n]
    k_grid = sorted(set(k_grid))
    _TMP.mkdir(parents=True, exist_ok=True)

    results: list[tuple[int, float]] = []
    best_cols, best_acc = ranked, -1.0
    for k in k_grid:
        subset = ranked[:k]
        model = _fit(score_algo, sel.X_inner[subset], sel.y_inner,
                     _GRIDS[score_algo][0], _TMP / "select.joblib")
        acc = float((_predict(score_algo, model, sel.X_val[subset]) == sel.y_val).mean())
        results.append((k, acc))
        print(f"    [select] k={k:<3} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_cols = acc, subset
    print(f"    [select] best k={len(best_cols)}  val_acc={best_acc:.4f}")
    return best_cols, results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_tuning(
    cfg: dict,
    algos: tuple[str, ...] = _ALGOS,
    featset: str = "v2",
    use_move_weight: bool = False,
    tune_thr: bool = False,
    val_frac: float = 0.2,
    select: bool = False,
) -> list[dict[str, Any]]:
    """Tune, retrain, and evaluate each algorithm on the no-flat test slice.

    Persists ``tuned_{featset}_{algo}_predictions.npz`` (y_true, y_pred, move),
    ``tuned_params.json``, and the markdown report ``docs/notes/tuning_stats.md``.
    When ``select`` is set, a feature subset chosen on the validation fold (via
    ``select_features``) is applied to all models and saved to
    ``selected_{featset}.json``.

    Returns:
        A list of per-algo result dicts (best params, val/test accuracy, MCC, …).
    """
    feat_fn = _feature_fn(featset)
    print(f"[tuning] feature set={featset}  move_weight={use_move_weight}  "
          f"tune_threshold={tune_thr}  select={select}  val_frac={val_frac}")

    sel = build_selection_split(cfg, feat_fn, val_frac=val_frac)
    X_tr, y_tr, move_tr, X_te, y_te, move_te, keep_te = _full_train_test(cfg, feat_fn)
    sw_full = _move_weight(move_tr) if use_move_weight else None
    sw_inner = _move_weight(sel.move_inner) if use_move_weight else None
    _TMP.mkdir(parents=True, exist_ok=True)
    _PROC.mkdir(parents=True, exist_ok=True)

    if select:
        cols, _ = select_features(sel)
        print(f"[tuning] selected {len(cols)} of {sel.X_inner.shape[1]} features")
        (_PROC / f"selected_{featset}.json").write_text(json.dumps(cols, indent=2))
        sel.X_inner = sel.X_inner[cols]
        sel.X_val = sel.X_val[cols]
        X_tr, X_te = X_tr[cols], X_te[cols]

    n_keep = int(keep_te.sum())
    print(f"[tuning] inner-train={len(sel.X_inner):,}  no-flat val={len(sel.X_val):,}  "
          f"full no-flat train={len(X_tr):,}  no-flat test={n_keep:,}")

    results: list[dict[str, Any]] = []
    params_out: dict[str, Any] = {}
    for algo in algos:
        print(f"\n  === {_DISPLAY[algo]} ===")
        best_params, best_val_acc, _ = grid_search(algo, sel)

        # Threshold tuned on the validation fold (fit best params on inner-train).
        thr = _default_threshold(algo)
        if tune_thr:
            m_inner = _fit(algo, sel.X_inner, sel.y_inner, best_params,
                           _TMP / f"{algo}_inner.joblib", sample_weight=sw_inner)
            thr = tune_threshold(_scores(algo, m_inner, sel.X_val), sel.y_val)
            print(f"    [{algo}] tuned threshold={thr:.4f} "
                  f"(default {_default_threshold(algo)})")

        # Final fit on the FULL no-flat training half; evaluate on the test set.
        model = _fit(algo, X_tr, y_tr, best_params,
                     _PROC / f"tuned_{featset}_{algo}_model.joblib",
                     sample_weight=sw_full)
        if tune_thr:
            y_pred = (_scores(algo, model, X_te) >= thr).astype(int)
        else:
            y_pred = _predict(algo, model, X_te)

        npz = _PROC / f"tuned_{featset}_{algo}_predictions.npz"
        np.savez(npz, y_true=y_te, y_pred=y_pred, move=move_te, keep=keep_te)

        res_nf = statistics.compute(y_te[keep_te], y_pred[keep_te],
                                    name=f"{_DISPLAY[algo]} (tuned, {featset})")
        print(f"    [{algo}] no-flat TEST acc={res_nf['accuracy']:.4f}  "
              f"mcc={res_nf['mcc']:.4f}  (val acc={best_val_acc:.4f})")

        params_out[algo] = {"params": best_params, "threshold": thr,
                            "val_accuracy": best_val_acc,
                            "test_noflat_accuracy": res_nf["accuracy"],
                            "test_noflat_mcc": res_nf["mcc"]}
        results.append({"algo": algo, "stats": res_nf,
                        "val_acc": best_val_acc, "params": best_params,
                        "threshold": thr})

    # Per-featset paths so v1/v2/v3 runs don't clobber each other.
    params_path = _PROC / f"tuned_params_{featset}.json"
    report_path = _DOCS / f"tuning_stats_{featset}.md"
    params_path.write_text(json.dumps(
        {"featset": featset, "move_weight": use_move_weight,
         "tune_threshold": tune_thr, "models": params_out}, indent=2))
    _write_report(results, featset, use_move_weight, tune_thr, n_keep, report_path)
    print(f"\n[tuning] params → {params_path}")
    print(f"[tuning] report → {report_path}")
    return results


def _write_report(results: list[dict[str, Any]], featset: str,
                  move_weight: bool, tune_thr: bool, n_keep: int,
                  report_path: Path) -> None:
    """Write the tuning report (no-flat TEST metrics + chosen config)."""
    _DOCS.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Tuned Models — No-Flat Test Slice ({featset})\n\n"
        f"Hyperparameters selected on a **no-flat validation fold from the "
        f"training half** (the test set is touched once). Feature set: {featset}; "
        f"|move| weighting: {move_weight}; threshold tuned: {tune_thr}. "
        f"No-flat test rows: {n_keep:,}.\n\n"
    )
    rows = ["| Model | Val acc | Test acc (no-flat) | Test MCC | Threshold | Params |",
            "|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda d: d["stats"]["accuracy"], reverse=True):
        rows.append(
            f"| {_DISPLAY[r['algo']]} | {r['val_acc']:.4f} | "
            f"{r['stats']['accuracy']:.4f} | {r['stats']['mcc']:.4f} | "
            f"{r['threshold']:.3f} | `{r['params']}` |"
        )
    body = "\n".join(rows) + "\n\n---\n\n"
    body += "\n---\n\n".join(statistics.format_markdown(r["stats"]) for r in results)
    report_path.write_text(header + body + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune models toward no-flat test accuracy (validation from train)"
    )
    parser.add_argument("--algos", nargs="+", choices=_ALGOS, default=list(_ALGOS))
    parser.add_argument("--featset", choices=["v1", "v2", "v3"], default="v2")
    parser.add_argument("--move-weight", action="store_true",
                        help="Weight training rows by |Close - Open|.")
    parser.add_argument("--tune-threshold", action="store_true",
                        help="Tune the decision threshold on the validation fold.")
    parser.add_argument("--select", action="store_true",
                        help="Select a feature subset on the validation fold.")
    parser.add_argument("--val-frac", type=float, default=0.2)
    args = parser.parse_args()

    cfg = load_config()
    run_tuning(cfg, algos=tuple(args.algos), featset=args.featset,
               use_move_weight=args.move_weight, tune_thr=args.tune_threshold,
               val_frac=args.val_frac, select=args.select)


if __name__ == "__main__":
    main()
