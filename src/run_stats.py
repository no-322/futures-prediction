"""Compute standardised statistics for all binary models.

Section A — Production models (logistic, rf, gbm):
  Load existing joblibs, infer on the 50% test set (no retraining), save
  predictions, write docs/notes/all_stats.md.

Section C — No-flat binary suite (20-feat) + HMM-regime binary:
  Train logistic/rf/gbm with flat (Close==Open) rows removed from training,
  plus the HMM-regime binary model; write docs/notes/binary_noflat_stats.md.

Section D — 49-feature binary suites (flat-included + no-flat):
  Train both variants on the cached features_v2 matrix; write
  docs/notes/binary_v2_stats.md.

Section LB — Model leaderboard:
  Rank every saved binary prediction set by no-flat test accuracy then full-test
  MCC; write docs/notes/model_leaderboard.md.

The walk-forward leaderboard and per-model report live in leaderboard-walk-forward.md
and results.md (see src.evaluate / the two-leaderboard generators).

Run with:
    python -m src.run_stats                       # sections a c d
    python -m src.run_stats --sections a          # production only
    python -m src.run_stats --sections c d --skip-existing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import src.evaluate as evaluate
import src.statistics as statistics
from src.config import load_config, model_params
from src.features_v2 import load_or_build_features_v2
from src.labels import build_labels
from src.load import load_raw
from src.pipeline import _build_dataset
from src.split import split
from src.statistics import StatsResult

_PROD_MODELS = {
    "logistic": ("Logistic Regression",
                 Path("data/processed/logistic_model.joblib")),
    "rf":       ("Random Forest",
                 Path("data/processed/rf_model.joblib")),
    "gbm":      ("Gradient Boosting (XGBoost)",
                 Path("data/processed/gbm_model.joblib")),
}

_REPORT_PATH = Path("docs/notes/all_stats.md")
_NOFLAT_REPORT_PATH = Path("docs/notes/binary_noflat_stats.md")
_V2_REPORT_PATH = Path("docs/notes/binary_v2_stats.md")


_LEADERBOARD_PATH = Path("docs/notes/leaderboard.md")
_PROC = Path("data/processed")

_REGIME_LABELS = {0: "low-vol / calm", 1: "high-vol / active"}

# 49-feature binary variant: (prefix, display_suffix, section title). Flat is dropped
# globally, so there is a single (no-flat) variant.
_V2_VARIANTS = [
    ("exp_noflat_v2", "no-flat, v2", "No-flat (49 features)"),
]


def section_a(cfg: dict) -> list[StatsResult]:
    """Load production joblibs, predict on the test set, compute stats."""
    from src.models import logistic, rf
    from src.models.gbm import load as gbm_load, predict as gbm_predict

    train_size = cfg["data"].get("train_size", 0.5)
    data_path  = Path(cfg["data"]["path"])

    print("\n[Section A] Production models — loading joblibs + inference")
    _, X_test, _, y_test = _build_dataset(data_path, train_size)
    y_true = y_test.to_numpy()

    loaders = {
        "logistic": (logistic.load,   logistic.predict),
        "rf":       (rf.load,         rf.predict),
        "gbm":      (gbm_load,        gbm_predict),
    }

    results: list[StatsResult] = []
    for algo, (display, jpath) in _PROD_MODELS.items():
        if not jpath.exists():
            print(f"  [{algo}] Joblib not found, skipping")
            continue
        print(f"  [{algo}] Loading from {jpath}...")
        load_fn, predict_fn = loaders[algo]
        model  = load_fn(jpath)
        y_pred = predict_fn(model, X_test)
        np.savez(
            Path(f"data/processed/{algo}_predictions.npz"),
            y_true=y_true, y_pred=y_pred,
        )
        result = statistics.compute(y_true, y_pred, name=display)
        results.append(result)
        print(f"    acc={result['accuracy']:.4f}  "
              f"macro_f1={result['macro_f1']:.4f}  mcc={result['mcc']:.4f}")

    return results


def section_c(cfg: dict, skip_existing: bool = False) -> None:
    """No-flat binary suite (20-feat) + HMM-regime binary → binary_noflat_stats.md."""
    from src import binary_suite
    from src.models import regime_binary

    print("\n[Section C] No-flat binary suite — flat rows removed from training")
    results: list[StatsResult] = []

    algos = tuple(
        a for a in binary_suite._ALGOS
        if not (skip_existing
                and Path(f"data/processed/exp_noflat_{a}_predictions.npz").exists())
    )
    if algos:
        binary_suite.run(config=cfg, algos=algos)
    for algo in binary_suite._ALGOS:
        npz = Path(f"data/processed/exp_noflat_{algo}_predictions.npz")
        if not npz.exists():
            print(f"    Warning: {npz} not found, skipping stats")
            continue
        d = np.load(npz)
        res = statistics.compute(d["y_true"], d["y_pred"],
                                 name=binary_suite._DISPLAY[algo])
        results.append(res)
        print(f"    {binary_suite._DISPLAY[algo]:<38} acc={res['accuracy']:.4f}  "
              f"macro_f1={res['macro_f1']:.4f}  mcc={res['mcc']:.4f}")

    # -- HMM-regime binary model ---------------------------------------------
    hmm_npz = Path("data/processed/exp_regime_binary_predictions.npz")
    if not (skip_existing and hmm_npz.exists()):
        regime_binary.run(config=cfg)

    per_regime_md = ""
    if hmm_npz.exists():
        d   = np.load(hmm_npz)
        res = statistics.compute(d["y_true"], d["y_pred"], name="HMM-regime binary")
        results.append(res)
        print(f"    {'HMM-regime binary':<38} acc={res['accuracy']:.4f}  "
              f"macro_f1={res['macro_f1']:.4f}  mcc={res['mcc']:.4f}")

        blocks = ["\n---\n\n## HMM-regime binary — per-regime breakdown\n\n"]
        for r in [0, 1]:
            mask = d["regime"] == r
            if mask.sum() == 0:
                continue
            sub = statistics.compute(
                d["y_true"][mask], d["y_pred"][mask],
                name=f"Regime {r} ({_REGIME_LABELS[r]})",
            )
            blocks.append(statistics.format_markdown(sub))
            blocks.append("\n")
        per_regime_md = "".join(blocks)
    else:
        print(f"    Warning: {hmm_npz} not found, skipping HMM stats")

    _NOFLAT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# No-Flat Binary Suite — Model Evaluation Statistics\n\n"
        "Binary up/down classification on the 50/50 time-ordered test set, with "
        "flat bars (Close == Open) removed from the **training** set only "
        "(test set whole). Per-label metrics: class 0 = down, class 1 = up.\n\n"
    )
    body = "\n---\n\n".join(statistics.format_markdown(r) for r in results)
    _NOFLAT_REPORT_PATH.write_text(header + body + "\n" + per_regime_md)
    print(f"\nNo-flat binary statistics written to {_NOFLAT_REPORT_PATH}")


def section_d(cfg: dict, skip_existing: bool = False) -> None:
    """49-feature binary suites (flat-incl + no-flat) → binary_v2_stats.md."""
    from src import binary_suite

    print("\n[Section D] 49-feature binary suites (flat-included + no-flat)")
    report_blocks: list[str] = []

    for prefix, suffix, title in _V2_VARIANTS:
        algos = tuple(
            a for a in binary_suite._ALGOS
            if not (skip_existing
                    and Path(f"data/processed/{prefix}_{a}_predictions.npz").exists())
        )
        if algos:
            binary_suite.run(
                config=cfg, algos=algos,
                build_features_fn=load_or_build_features_v2,
                prefix=prefix, display_suffix=suffix,
            )

        results: list[StatsResult] = []
        for algo in binary_suite._ALGOS:
            npz = Path(f"data/processed/{prefix}_{algo}_predictions.npz")
            if not npz.exists():
                print(f"    Warning: {npz} not found, skipping stats")
                continue
            d   = np.load(npz)
            res = statistics.compute(d["y_true"], d["y_pred"],
                                     name=binary_suite._display(algo, suffix))
            results.append(res)
            print(f"    {binary_suite._display(algo, suffix):<42} "
                  f"acc={res['accuracy']:.4f}  macro_f1={res['macro_f1']:.4f}  "
                  f"mcc={res['mcc']:.4f}")

        if results:
            block = [f"## {title}\n\n"]
            block.append("\n---\n\n".join(statistics.format_markdown(r) for r in results))
            report_blocks.append("".join(block))

    _V2_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# 49-Feature Binary Suites — Model Evaluation Statistics\n\n"
        "Binary up/down classification on the 50/50 time-ordered test set using the "
        "**49-feature v2 matrix**. Two variants: flat-included (all training rows) and "
        "no-flat (flat `Close == Open` rows removed from **training** only; test whole). "
        "Per-label metrics: class 0 = down, class 1 = up.\n\n"
    )
    _V2_REPORT_PATH.write_text(header + "\n\n---\n\n".join(report_blocks) + "\n")
    print(f"\n49-feature binary statistics written to {_V2_REPORT_PATH}")
def _leaderboard_name(stem: str, registry: dict[str, str]) -> str:
    """Human-readable model name for a prediction-set stem."""
    if stem in registry:
        return registry[stem]
    if stem.startswith("exp_noflat_v1rel_"):
        from src.binary_suite import BASE_DISPLAY
        algo = stem[len("exp_noflat_v1rel_"):]
        return f"{BASE_DISPLAY.get(algo, algo)} (v1-rel)"
    if stem.startswith("tuned_"):
        from src.binary_suite import BASE_DISPLAY
        _, feat, algo = stem.split("_", 2)
        return f"{BASE_DISPLAY.get(algo, algo)} (tuned, {feat})"
    return stem


def _aum_pct(y_pred: np.ndarray, returns: np.ndarray) -> float:
    """% increase in AUM from the compounding long/short backtest (= total_return·100)."""
    bt = statistics.backtest(np.asarray(y_pred), np.asarray(returns))
    return float(bt["total_return"]) * 100.0


def _test_reference(cfg: dict) -> tuple[int, np.ndarray, np.ndarray]:
    """(test length, y_true, per-bar returns) for the flat-free test split."""
    from src.labels import build_labels, drop_flat
    from src.features import build_features
    train_size = cfg["data"].get("train_size", 0.5)
    df = load_raw(Path(cfg["data"]["path"]))
    feats = build_features(df)
    raw = df.iloc[4:].reset_index(drop=True)
    feats, raw = drop_flat(feats, raw)
    _, raw_test = split(raw, train_size=train_size)
    y = build_labels(raw_test).to_numpy()
    returns = ((raw_test["Close"] - raw_test["Open"]) / raw_test["Open"]).to_numpy()
    return int(len(y)), y, returns


def rank_models(cfg: dict) -> list[tuple[str, str, float, float, float]]:
    """Rank every saved binary prediction set on the single test split, best first.

    Reads each ``data/processed/{stem}_predictions.npz`` (no retraining) and computes
    test accuracy, recall (class 1), and MCC — all on the flat-free test set, since flat
    bars are dropped from the modelling set. Rows are sorted by (accuracy, MCC)
    descending. Sets whose length differs from the current test split are skipped.

    Returns:
        List of (stem, display_name, accuracy, recall, mcc, aum_pct), best first.
    """
    from src import backtest

    n_total, _, returns = _test_reference(cfg)
    registry = backtest._build_registry()
    proc = Path("data/processed")

    rows: list[tuple[str, str, float, float, float]] = []
    for npz in sorted(proc.glob("*_predictions.npz")):
        stem = npz.name[: -len("_predictions.npz")]
        if stem.startswith("backtest_"):
            continue
        d = np.load(npz)
        if "y_pred" not in d or "y_true" not in d or len(d["y_pred"]) != n_total:
            continue
        res = statistics.compute(d["y_true"], d["y_pred"])
        recall = res["per_class"].get(1, {}).get("recall", float("nan"))
        aum = _aum_pct(d["y_pred"], returns)
        rows.append((stem, _leaderboard_name(stem, registry),
                     res["accuracy"], float(recall), res["mcc"], aum))

    rows.sort(key=lambda r: (r[2], r[4]), reverse=True)
    return rows


def leaderboard(cfg: dict) -> None:
    """Write ``docs/notes/leaderboard.md`` — single test-set model comparison.

    A table from rank_models(): accuracy, recall, MCC on the flat-free 50/50 test set,
    sorted by accuracy then MCC. Length-mismatched sets are skipped and listed.
    """
    n_total, _, _ = _test_reference(cfg)
    rows = rank_models(cfg)
    ranked_stems = {r[0] for r in rows}
    skipped = [
        npz.name[: -len("_predictions.npz")]
        for npz in sorted(Path("data/processed").glob("*_predictions.npz"))
        if not npz.name.startswith("backtest_")
        and npz.name[: -len("_predictions.npz")] not in ranked_stems
    ]

    lines = [
        "# Model Leaderboard — single test set\n\n",
        f"Every binary up/down model on the flat-free 50/50 time-ordered test set "
        f"({n_total:,} decisive bars). **Sorted by accuracy, then MCC.**\n\n",
    ]
    if skipped:
        lines.append("- Excluded (length mismatch / non-binary): "
                     + ", ".join(f"`{s}`" for s in sorted(skipped)) + ".\n")
    lines += [
        "*AUM %* = total return of the compounding long/short backtest on the test bars.\n",
        "\n| Model | Accuracy | Recall | MCC | AUM % |\n",
        "|-------|----------|--------|-----|-------|\n",
    ]
    for _stem, name, acc, recall, mcc, aum in rows:
        lines.append(f"| {name} | {acc:.4f} | {recall:.4f} | {mcc:.4f} | {aum:+.1f}% |\n")

    _LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LEADERBOARD_PATH.write_text("".join(lines))
    print(f"Leaderboard ({len(rows)} models) → {_LEADERBOARD_PATH}")


def _featset_builder(featset: str) -> "Callable":
    """Return the feature-matrix builder for a feature-set key ('v1'/'v2'/'v3')."""
    if featset == "v1":
        from src.features import build_features
        return build_features
    if featset == "v2":
        from src.features_v2 import load_or_build_features_v2
        return load_or_build_features_v2
    if featset == "v3":
        from src.features_v3 import load_or_build_features_v3
        return load_or_build_features_v3
    if featset == "orderflow":
        from src.features_orderflow import load_or_build_features_orderflow
        return load_or_build_features_orderflow
    if featset == "v1rel":
        from src.features_v1_rel import build_features_v1_rel
        return build_features_v1_rel
    raise ValueError(f"Unknown featset: {featset!r}")


# Curated core evaluated by walk-forward: 3 algos × 4 feature pipelines.
_WF_ALGOS = ("logistic", "rf", "gbm")
_WF_FEATSETS = ("v1", "v1rel", "v2", "v3")
_WF_LEADERBOARD_PATH = Path("docs/notes/leaderboard-walk-forward.md")


class _AlwaysUp:
    """Naive walk-forward baseline estimator: predict up (1) for every bar."""

    def fit(self, X, y):  # noqa: D401 - sklearn-style stub
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=int)


def _wf_xy(cfg: dict, featset: str):
    """Flat-free (X, y, timestamps, returns) for a feature set — walk-forward input."""
    from src.labels import build_labels, drop_flat
    df = load_raw(Path(cfg["data"]["path"]))
    X = _featset_builder(featset)(df)
    raw_align = df.iloc[4:].reset_index(drop=True)
    X, raw_align = drop_flat(X, raw_align)
    y = build_labels(raw_align)
    ts = raw_align["Date and Time"].reset_index(drop=True)
    returns = ((raw_align["Close"] - raw_align["Open"]) / raw_align["Open"]).to_numpy()
    return X, y, ts, returns


def walkforward_curated(cfg: dict) -> None:
    """Run rolling walk-forward for the curated core + the always-up baseline.

    For each (feature set × algorithm) and the naive always-up baseline, fit fresh per
    fold and persist per-fold predictions as ``wf_{featset}_{algo}`` (Rule 7). These feed
    ``leaderboard_walkforward``.
    """
    import src.walkforward as walkforward
    from src.models import gbm, logistic, rf

    modules = {"logistic": logistic, "rf": rf, "gbm": gbm}
    tmp = _PROC / "_wf_curated_tmp" / "model.joblib"

    for featset in _WF_FEATSETS:
        X, y, ts, rets = _wf_xy(cfg, featset)
        for algo in _WF_ALGOS:
            factory = walkforward.module_factory(
                modules[algo], model_params(cfg, algo), tmp)
            print(f"  walk-forward {algo} on {featset}")
            walkforward.walk_forward(X, y, ts, factory, config=cfg, returns=rets,
                                     name=f"wf_{featset}_{algo}")
    # Always-up baseline (feature-set independent — use v1's rows).
    X, y, ts, rets = _wf_xy(cfg, "v1")
    walkforward.walk_forward(X, y, ts, (lambda: _AlwaysUp()), config=cfg, returns=rets,
                             name="wf_baseline_alwaysup")


def _threshold_predict_fn(thr: float | None):
    """(model, X) -> labels applying a stored decision threshold; None → default predict."""
    if thr is None:
        return None
    return lambda model, Xte: (model.predict_proba(Xte)[:, 1] >= thr).astype(int)


def walkforward_curated_tuned(cfg: dict) -> None:
    """Walk-forward the **tuned (regularized)** models, mirroring `walkforward_curated`.

    For each feature set with a saved ``tuned_params_{featset}.json`` (v1/v2/v3), each fold
    fits the model with the validation-selected regularized hyperparameters and applies the
    stored decision threshold. Persists per-fold predictions as ``wf_tuned_{featset}_{algo}``,
    which `leaderboard_walkforward` / `walkforward_results` pick up as `tuned_{featset}_{algo}`
    rows alongside the default-hyperparameter ones.
    """
    import src.walkforward as walkforward
    from src.models import gbm, logistic, rf

    modules = {"logistic": logistic, "rf": rf, "gbm": gbm}
    tmp = _PROC / "_wf_tuned_tmp" / "model.joblib"

    for featset in _WF_FEATSETS:
        spec_path = _PROC / f"tuned_params_{featset}.json"
        if not spec_path.exists():
            print(f"  (skip {featset}: no {spec_path.name})")
            continue
        spec = json.loads(spec_path.read_text())
        X, y, ts, rets = _wf_xy(cfg, featset)
        for algo in _WF_ALGOS:
            entry = spec["models"].get(algo)
            if entry is None:
                continue
            params = dict(entry["params"])
            thr = entry.get("threshold") if spec.get("tune_threshold") else None
            factory = walkforward.module_factory(modules[algo], params, tmp)
            print(f"  walk-forward tuned {algo} on {featset} (thr={thr})")
            walkforward.walk_forward(
                X, y, ts, factory, config=cfg, returns=rets,
                predict_fn=_threshold_predict_fn(thr),
                name=f"wf_tuned_{featset}_{algo}",
            )


def _hmm_fold_transform(X_base: "pd.DataFrame", X_regime_full: np.ndarray,
                        vol15_idx: int, mode: str = "feature"):
    """Build a per-fold ``walk_forward`` transform that fits a fresh HMM on the train
    block and assigns the causal (filtered) regime.

    The HMM + scaler are fit on the fold's train regime descriptors only; the canonical
    high-vol state is identified from the TRAIN filtered states; then the filtered
    posterior is computed causally over ``concat(train, test)`` (time-ordered), so no
    row peeks at the future.

    mode == "feature": append ``regime_hi_prob`` = P(high-vol | data<=t) to X.
    mode == "gate":    return a test gate selecting high-vol bars (filtered argmax == hi).
    """
    from src.models.regime_hmm import (
        canonical_regime_labels,
        filter_regime,
        filter_regime_posterior,
        fit_regime,
    )

    def _transform(full_train_idx: np.ndarray, test_idx: np.ndarray):
        Xtr_reg = X_regime_full[full_train_idx]
        hmm, scaler = fit_regime(Xtr_reg)                       # fit on train block only
        train_states = filter_regime(hmm, scaler, Xtr_reg)      # causal train states
        remap = canonical_regime_labels(train_states, Xtr_reg, vol15_idx)
        hi_raw = next(raw for raw, canon in remap.items() if canon == 1)

        order = np.concatenate([full_train_idx, test_idx])      # time-ordered
        post = filter_regime_posterior(hmm, scaler, X_regime_full[order])
        hi_prob = post[:, hi_raw]
        n_tr = len(full_train_idx)
        hi_tr, hi_te = hi_prob[:n_tr], hi_prob[n_tr:]

        if mode == "feature":
            X_tr = X_base.iloc[full_train_idx].copy()
            X_tr["regime_hi_prob"] = hi_tr
            X_te = X_base.iloc[test_idx].copy()
            X_te["regime_hi_prob"] = hi_te
            return X_tr, X_te, None
        # gate: trade only high-vol bars (filtered argmax == high-vol state).
        gate = hi_te >= 0.5
        return X_base.iloc[full_train_idx], X_base.iloc[test_idx], gate

    return _transform


def _combo_xy(cfg: dict, base_featset: str, of_variant: str):
    """Flat-free ``(X, X_regime, y, timestamps, returns)`` for a base+order-flow combo.

    ``X`` = the base feature set concatenated with the 20 order-flow columns (``of_variant``
    ``"raw"``/``"linear"``). ``X_regime`` = v2's ``REGIME_COLS`` descriptor array, sourced
    independently so a v1 base still carries the HMM regime feature. All matrices are built
    on ``df.iloc[4:]`` and flat-dropped with one shared mask so rows stay aligned.
    """
    from src.features_orderflow import load_or_build_features_orderflow
    from src.labels import build_labels, flat_mask
    from src.models.regime_hmm import REGIME_COLS

    df = load_raw(Path(cfg["data"]["path"]))
    X_base = _featset_builder(base_featset)(df).reset_index(drop=True)
    X_of = load_or_build_features_orderflow(df, of_variant).reset_index(drop=True)
    X_reg = load_or_build_features_v2(df)[REGIME_COLS].reset_index(drop=True)
    raw_align = df.iloc[4:].reset_index(drop=True)

    X = pd.concat([X_base, X_of], axis=1)
    keep = ~flat_mask(raw_align)
    X = X[keep].reset_index(drop=True)
    X_reg = X_reg[keep].reset_index(drop=True).to_numpy()
    raw_align = raw_align[keep].reset_index(drop=True)
    y = build_labels(raw_align)
    ts = raw_align["Date and Time"].reset_index(drop=True)
    returns = ((raw_align["Close"] - raw_align["Open"]) / raw_align["Open"]).to_numpy()
    return X, X_reg, y, ts, returns


# base + order-flow + HMM-regime-feature combos: best base set per algo (retired leaderboard).
_COMBO_RECIPES = [
    ("v1", "linear", "logistic"),   # logistic: v1 + scale-stable order-flow
    ("v3", "raw",    "rf"),         # trees: v3 + raw order-flow
    ("v3", "raw",    "gbm"),
]


def walkforward_curated_regime_orderflow(cfg: dict) -> None:
    """Walk-forward the base + order-flow + HMM-regime-feature combos.

    Three recipes (best base set per algo from the retired single-split leaderboard):
    logistic on v1 + order-flow[linear], rf/gbm on v3 + order-flow[raw], each with a
    per-fold causal ``regime_hi_prob`` appended (HMM fit on the fold's train block only —
    no look-ahead). Persists ``wf_ofhmm_{base}_{algo}`` sets that ``leaderboard_walkforward``
    / ``walkforward_results`` pick up as ``ofhmm_{base}_{algo}`` rows.
    """
    import src.walkforward as walkforward
    from src.models import gbm, logistic, rf
    from src.models.regime_hmm import REGIME_COLS

    modules = {"logistic": logistic, "rf": rf, "gbm": gbm}
    vol15_idx = REGIME_COLS.index("lag1_vol15")
    tmp = _PROC / "_wf_combo_tmp" / "model.joblib"

    for base, variant, algo in _COMBO_RECIPES:
        X, X_reg, y, ts, rets = _combo_xy(cfg, base, variant)
        ft = _hmm_fold_transform(X, X_reg, vol15_idx, mode="feature")
        factory = walkforward.module_factory(modules[algo], model_params(cfg, algo), tmp)
        print(f"  walk-forward {algo} on {base}+orderflow[{variant}]+regime "
              f"({X.shape[1]} feats)")
        walkforward.walk_forward(
            X, y, ts, factory, config=cfg, returns=rets, fold_transform=ft,
            name=f"wf_ofhmm_{base}_{algo}",
        )


def leaderboard_walkforward(cfg: dict, proc: Path = _PROC,
                            out: Path = _WF_LEADERBOARD_PATH) -> None:
    """Write ``docs/notes/leaderboard-walk-forward.md`` ranked by mean fold accuracy.

    Reads every ``wf_*_predictions.npz`` (per-fold ``accuracies`` + concatenated
    ``y_true``/``y_pred``), reports mean ± std accuracy across folds, recall (class 1),
    and **folds won** = number of folds the model beats the always-up baseline. Ranked by
    mean accuracy; the folds-won column is the regime-stability signal.
    """
    base_npz = proc / "walkforward_wf_baseline_alwaysup_predictions.npz"
    base_acc = (np.load(base_npz)["accuracies"] if base_npz.exists() else None)

    rows = []
    for npz in sorted(proc.glob("walkforward_wf_*_predictions.npz")):
        stem = npz.name[: -len("_predictions.npz")]
        if stem == "walkforward_wf_baseline_alwaysup":
            continue
        d = np.load(npz)
        accs = d["accuracies"]
        res = statistics.compute(d["y_true"], d["y_pred"])
        recall = res["per_class"].get(1, {}).get("recall", float("nan"))
        won = (int(np.sum(accs > base_acc)) if base_acc is not None
               and len(base_acc) == len(accs) else None)
        aum = _aum_pct(d["y_pred"], d["returns"]) if "returns" in d else float("nan")
        name = stem[len("walkforward_wf_"):]
        rows.append((name, float(np.nanmean(accs)), float(np.nanstd(accs)),
                     float(recall), won, len(accs), aum))

    rows.sort(key=lambda r: r[1], reverse=True)
    lines = [
        "# Walk-Forward Leaderboard\n\n",
        "Rolling walk-forward (3-month train / 1-month test, from `config.yaml`) over the "
        "flat-free modelling set. **Ranked by mean fold accuracy.** *Folds won* = folds "
        "beating the always-up baseline; *AUM %* = total return of the compounding "
        "long/short backtest over the concatenated walk-forward test bars.\n\n",
        "| Model | Mean acc ± std | Folds won | Recall | AUM % |\n",
        "|-------|----------------|-----------|--------|-------|\n",
    ]
    for name, mean, std, recall, won, nfolds, aum in rows:
        won_s = f"{won}/{nfolds}" if won is not None else "—"
        lines.append(f"| {name} | {mean * 100:.1f}% ± {std * 100:.1f}% "
                     f"| {won_s} | {recall:.4f} | {aum:+.1f}% |\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines))
    print(f"Walk-forward leaderboard ({len(rows)} models) → {out}")


_RESULTS_PATH = Path("docs/results.md")


def walkforward_results(cfg: dict, proc: Path = _PROC,
                        out: Path = _RESULTS_PATH) -> None:
    """Write ``docs/results.md`` — per-model walk-forward report (the eval format).

    For each ``wf_{featset}_{algo}`` set: mean ± std accuracy across folds, recall,
    confusion matrix, Δ vs the always-up baseline, and backtest AUM %. Flat is dropped
    globally, so all metrics are on decisive up/down bars.
    """
    base_npz = proc / "walkforward_wf_baseline_alwaysup_predictions.npz"
    base_mean = (float(np.nanmean(np.load(base_npz)["accuracies"]))
                 if base_npz.exists() else None)

    blocks = [
        "# Model Evaluation Results — walk-forward\n\n",
        "Rolling walk-forward (3-month train / 1-month test). Each model reports the "
        "mean ± std across folds; Δ is vs the always-up baseline; AUM % is the "
        "compounding backtest total return.\n",
    ]
    for npz in sorted(proc.glob("walkforward_wf_*_predictions.npz")):
        stem = npz.name[: -len("_predictions.npz")]
        if stem == "walkforward_wf_baseline_alwaysup":
            continue
        d = np.load(npz)
        accs = d["accuracies"]
        res = statistics.compute(d["y_true"], d["y_pred"], name=stem[len("wf_"):])
        recall = res["per_class"].get(1, {}).get("recall", float("nan"))
        cm = res["confusion_matrix"]
        mean = float(np.nanmean(accs))
        delta = (mean - base_mean) * 100 if base_mean is not None else float("nan")
        aum = _aum_pct(d["y_pred"], d["returns"]) if "returns" in d else float("nan")
        blocks.append(
            f"\n---\n\n## {stem[len('walkforward_wf_'):]}\n\n"
            f"- **Accuracy:** {mean * 100:.1f}% ± {float(np.nanstd(accs)) * 100:.1f}% "
            f"across {len(accs)} folds (range "
            f"{float(np.nanmin(accs)) * 100:.1f}–{float(np.nanmax(accs)) * 100:.1f}%)\n"
            f"- **Recall (up):** {recall:.4f}\n"
            f"- **Δ vs always-up baseline:** {delta:+.1f} pp\n"
            f"- **Backtest AUM %:** {aum:+.1f}%\n\n"
            f"Confusion matrix (rows=actual, cols=predicted):\n\n"
            f"| | Pred 0 | Pred 1 |\n|---|---|---|\n"
            f"| **Actual 0** | {cm[0][0]:,} | {cm[0][1]:,} |\n"
            f"| **Actual 1** | {cm[1][0]:,} | {cm[1][1]:,} |\n"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(blocks))
    print(f"Walk-forward results → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute stats for all binary models")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip suites whose .npz predictions already exist",
    )
    parser.add_argument(
        "--sections", nargs="+",
        choices=["a", "c", "d", "lb", "wf", "wftuned", "wfcombo", "lbwf"],
        default=["a", "c", "d"],
        help="Which sections to run: a=production stats, c=no-flat 20-feat suite + HMM, "
             "d=49-feature binary suites, lb=leaderboard.md (single test set), "
             "wf=walk-forward the curated core (default hyperparameters), "
             "wftuned=walk-forward the tuned (regularized) models from tuned_params_*.json, "
             "wfcombo=walk-forward the base+order-flow+HMM-regime-feature combos, "
             "lbwf=leaderboard-walk-forward.md + results.md from the wf_* sets. "
             "lb/lbwf read existing .npz (no retraining). Default: a c d.",
    )
    args = parser.parse_args()

    cfg = load_config()

    if "a" in args.sections:
        results = section_a(cfg)
        if results:
            statistics.write_results(results, _REPORT_PATH)
            print(f"\nProduction statistics written to {_REPORT_PATH}")
    if "c" in args.sections:
        section_c(cfg, skip_existing=args.skip_existing)
    if "d" in args.sections:
        section_d(cfg, skip_existing=args.skip_existing)
    if "lb" in args.sections:
        leaderboard(cfg)
    if "wf" in args.sections:
        walkforward_curated(cfg)
    if "wftuned" in args.sections:
        walkforward_curated_tuned(cfg)
    if "wfcombo" in args.sections:
        walkforward_curated_regime_orderflow(cfg)
    if "lbwf" in args.sections:
        leaderboard_walkforward(cfg)
        walkforward_results(cfg)


if __name__ == "__main__":
    main()
