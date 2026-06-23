"""Rolling walk-forward validation — the project's iteration metric.

The ``evaluation`` skill mandates a **rolling fixed-width** walk-forward as the
day-to-day out-of-sample metric: a train block then a test block, stepped forward
in calendar time (default **3 months train / 1 month test**, sizes from config).
Rolling — not expanding — because edge here is regime-dependent and a fixed window
tracks regime shifts and contract rolls.

Design:
  * Window sizes come from ``config.yaml`` (``walk_forward`` section), never
    hardcoded. Explicit kwargs override config; config overrides the skill default.
  * Calendar-time windows via ``pd.DateOffset`` — timestamp arithmetic, not row
    offsets, so the data's gaps (overnight, weekends, contract rolls) are honoured.
  * A **fresh model instance per fold** (``model_factory()``); never reuse a fitted
    estimator across folds.
  * Reports **per-fold accuracy AND mean ± std** — per-fold is mandatory: the
    between-period spread is the regime signal.
  * Leakage guards: monotonic-time assertion before splitting; per-fold
    ``train_idx.max() < test_idx.min()``; ``purge``/``embargo`` params (default 0,
    inert under the current single-bar label — kept for forward-horizon labels).

The harness accepts **any sklearn-style model** through a factory callable returning
a fresh object exposing ``fit``/``predict``. ``project_factories`` ships ready
factories for this repo's four models; the SVM factory wraps a ``StandardScaler`` in
an sklearn ``Pipeline`` so scaling is fit **inside the fold, on the train block only**.

Run with::

    python -m src.walkforward --algo baseline --featset v1
    python -m src.walkforward --algo rf --featset v2 --train-months 6 --test-months 1
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.config import load_config, model_params

_PROC = Path("data/processed")

# Skill defaults — used only when neither explicit kwargs nor config supply sizes.
_DEFAULT_TRAIN_MONTHS = 3
_DEFAULT_TEST_MONTHS = 1


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    """One rolling walk-forward fold: contiguous train block then test block.

    Indices are positional into the row-aligned ``(X, y, timestamps)`` arrays.
    Timestamps bound the calendar window each block was drawn from.
    """

    train_idx: np.ndarray   # int positions of the train block
    test_idx:  np.ndarray   # int positions of the test block
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp


class WalkForwardResult(TypedDict):
    """Aggregated walk-forward outcome across all folds."""

    name:          str
    n_folds:       int
    mean_accuracy: float
    std_accuracy:  float          # population std (ddof=0) across folds
    min_accuracy:  float
    max_accuracy:  float
    per_fold:      list[dict[str, Any]]
    npz_path:      str | None


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------

def make_folds(
    timestamps: pd.Series,
    train_months: int,
    test_months: int,
    step_months: int,
    purge: int = 0,
    embargo: int = 0,
) -> list[Fold]:
    """Build rolling calendar-time walk-forward folds.

    The window slides over the time axis: train spans ``[t0, t0 + train_months)``
    and test spans ``[t0 + train_months, t0 + train_months + test_months)``; ``t0``
    advances by ``step_months`` each fold. Windows are defined by timestamp
    arithmetic (``pd.DateOffset``), so gaps in the minute grid never distort widths.
    A fold is emitted only when both its train and test blocks are non-empty, which
    drops any partial window at the tail of the series.

    Args:
        timestamps: Per-row timestamps aligned to the feature/label rows; must be
            monotonically non-decreasing (asserted).
        train_months: Width of the train block in calendar months.
        test_months: Width of the test block in calendar months.
        step_months: Calendar months to advance the window between folds. Set equal
            to ``test_months`` for non-overlapping test blocks.
        purge: Number of train rows nearest the train/test boundary to drop. Inert
            (0) for the single-bar label; load-bearing under forward-horizon labels
            whose window crosses the boundary.
        embargo: Number of rows immediately after the train block to skip before the
            test block begins; cheap insurance against adjacent-boundary effects.

    Returns:
        List of ``Fold`` in time order. Empty if the series is shorter than one
        train+test window.

    Raises:
        ValueError: If ``timestamps`` is not monotonically non-decreasing, or any of
            the window sizes is not a positive integer.
    """
    if train_months <= 0 or test_months <= 0 or step_months <= 0:
        raise ValueError(
            "train_months, test_months, step_months must all be positive; "
            f"got {train_months}, {test_months}, {step_months}"
        )
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    if not ts.is_monotonic_increasing:
        raise ValueError(
            "timestamps must be monotonically non-decreasing before folding; "
            "contract-roll concatenation can silently break forward-only order."
        )
    values = ts.to_numpy()
    origin = ts.iloc[0]
    last = ts.iloc[-1]

    folds: list[Fold] = []
    step = pd.DateOffset(months=step_months)
    train_off = pd.DateOffset(months=train_months)
    test_off = pd.DateOffset(months=test_months)

    t0 = origin
    while t0 + train_off <= last:
        train_lo = t0
        train_hi = t0 + train_off            # exclusive
        test_lo = train_hi
        test_hi = train_hi + test_off        # exclusive

        train_pos = np.flatnonzero((values >= train_lo) & (values < train_hi))
        test_pos = np.flatnonzero((values >= test_lo) & (values < test_hi))

        if embargo > 0 and train_pos.size > 0:
            # Skip the first `embargo` test rows adjacent to the train boundary.
            test_pos = test_pos[embargo:]
        if purge > 0 and train_pos.size > 0:
            # Drop the `purge` train rows nearest the boundary.
            train_pos = train_pos[:-purge] if purge < train_pos.size else train_pos[:0]

        if train_pos.size > 0 and test_pos.size > 0:
            assert train_pos.max() < test_pos.min(), (
                "leakage: a train row is positioned at/after a test row"
            )
            folds.append(
                Fold(
                    train_idx=train_pos,
                    test_idx=test_pos,
                    train_start=ts.iloc[train_pos[0]],
                    train_end=ts.iloc[train_pos[-1]],
                    test_start=ts.iloc[test_pos[0]],
                    test_end=ts.iloc[test_pos[-1]],
                )
            )
        t0 = t0 + step

    return folds


# ---------------------------------------------------------------------------
# Core harness
# ---------------------------------------------------------------------------

def _resolve_sizes(
    config: dict | None,
    train_months: int | None,
    test_months: int | None,
    step_months: int | None,
    purge: int | None,
    embargo: int | None,
) -> tuple[int, int, int, int, int]:
    """Resolve window params: explicit kwargs → config → skill defaults."""
    wf: dict = (config or {}).get("walk_forward", {}) if config else {}
    tr = train_months if train_months is not None else wf.get("train_months", _DEFAULT_TRAIN_MONTHS)
    te = test_months if test_months is not None else wf.get("test_months", _DEFAULT_TEST_MONTHS)
    st = step_months if step_months is not None else wf.get("step_months", te)
    pu = purge if purge is not None else wf.get("purge", 0)
    em = embargo if embargo is not None else wf.get("embargo", 0)
    return int(tr), int(te), int(st), int(pu), int(em)


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    model_factory: Callable[[], Any],
    *,
    config: dict | None = None,
    train_months: int | None = None,
    test_months: int | None = None,
    step_months: int | None = None,
    purge: int | None = None,
    embargo: int | None = None,
    keep: np.ndarray | None = None,
    include_flat: bool = False,
    drop_flat_train: bool = False,
    predict_fn: Callable[[Any, pd.DataFrame], np.ndarray] | None = None,
    name: str = "wf",
    save: bool = True,
) -> WalkForwardResult:
    """Run rolling walk-forward validation for one sklearn-style model.

    For each fold a **fresh** estimator is built via ``model_factory()``, fit on the
    fold's train block, and scored on its test block. Per-fold accuracy plus the
    mean ± std across folds are returned; predictions are persisted to disk before
    returning (CLAUDE.md Rule 7).

    Window sizes resolve from explicit kwargs first, then ``config['walk_forward']``,
    then the skill defaults (3 months train / 1 month test, step = test width).

    No-flat handling (this project's default): flat bars (``Close == Open``) are
    ambiguous up/down targets. When a ``keep`` mask is supplied, per-fold accuracy is
    computed on the **non-flat test rows by default** (``include_flat=False``) — the
    no-flat test slice. With ``keep=None`` there is no flatness signal, so the full
    test block is scored (back-compatible).

    Args:
        X: Feature matrix, shape (n, d). Row-aligned with ``y`` and ``timestamps``.
        y: Integer class labels, shape (n,).
        timestamps: Per-row timestamps (the aligned ``"Date and Time"`` column),
            shape (n,). Must be monotonically non-decreasing.
        model_factory: Zero-arg callable returning a fresh, unfitted estimator that
            exposes sklearn-style ``fit(X, y)`` and ``predict(X) -> labels``. Called
            once per fold so no fitted state is shared across folds.
        config: Optional config dict (from ``load_config()``) supplying window sizes.
        train_months, test_months, step_months, purge, embargo: Optional explicit
            overrides; each takes precedence over ``config`` when not None.
        keep: Optional boolean non-flat mask (True = ``Close != Open``), aligned to the
            full X/y/timestamps. Enables the no-flat scoring / no-flat training options.
        include_flat: If False (default) and ``keep`` is given, per-fold accuracy uses
            only the non-flat test rows; True scores the full test block.
        drop_flat_train: If True and ``keep`` is given, flat rows are dropped from each
            fold's train block before fitting (default: train on the full block).
        predict_fn: Optional ``(model, X) -> labels`` override (e.g. to apply a decision
            threshold via ``predict_proba``); defaults to ``model.predict``.
        name: Short tag used in the report and the persisted npz filename.
        save: If True, write predictions to
            ``data/processed/walkforward_{name}_predictions.npz``.

    Returns:
        WalkForwardResult with per-fold detail and the mean/std/min/max accuracy.

    Raises:
        ValueError: If X/y/timestamps lengths differ, or no fold fits in the series.
    """
    if not (len(X) == len(y) == len(timestamps)):
        raise ValueError(
            f"X, y, timestamps must be row-aligned; got {len(X)}, {len(y)}, "
            f"{len(timestamps)}"
        )
    tr, te, st, pu, em = _resolve_sizes(
        config, train_months, test_months, step_months, purge, embargo
    )
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    if keep is not None:
        keep = np.asarray(keep, dtype=bool)
        if keep.shape[0] != len(X):
            raise ValueError(
                f"keep must align to X; got {keep.shape[0]} vs {len(X)}"
            )

    folds = make_folds(ts, tr, te, st, purge=pu, embargo=em)
    if not folds:
        raise ValueError(
            f"No folds fit in the series span with train={tr}mo/test={te}mo; "
            "the series may be shorter than one train+test window."
        )

    per_fold: list[dict[str, Any]] = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    fold_id_all: list[np.ndarray] = []
    kept_all: list[np.ndarray] = []
    accuracies: list[float] = []

    for i, fold in enumerate(folds):
        train_idx = fold.train_idx
        if drop_flat_train and keep is not None:
            train_idx = train_idx[keep[train_idx]]
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[fold.test_idx], y.iloc[fold.test_idx]

        model = model_factory()                       # fresh per fold
        model.fit(X_tr, y_tr)
        if predict_fn is not None:
            y_hat = np.asarray(predict_fn(model, X_te))
        else:
            y_hat = np.asarray(model.predict(X_te))
        y_te_arr = y_te.to_numpy()

        # No-flat test scoring (default when keep given and include_flat is False).
        test_keep = (
            keep[fold.test_idx] if keep is not None
            else np.ones(y_te_arr.shape, dtype=bool)
        )
        score_mask = (
            np.ones(y_te_arr.shape, dtype=bool) if include_flat else test_keep
        )
        acc = float(accuracy_score(y_te_arr[score_mask], y_hat[score_mask]))

        accuracies.append(acc)
        y_true_all.append(y_te_arr)
        y_pred_all.append(y_hat)
        fold_id_all.append(np.full(y_te_arr.shape, i, dtype=int))
        kept_all.append(test_keep)
        per_fold.append(
            {
                "fold": i,
                "accuracy": acc,
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "n_train": int(train_idx.size),
                "n_test": int(score_mask.sum()),
            }
        )

    accs = np.asarray(accuracies, dtype=float)
    npz_path: str | None = None
    if save:
        _PROC.mkdir(parents=True, exist_ok=True)
        npz_path = str(_PROC / f"walkforward_{name}_predictions.npz")
        np.savez(
            npz_path,
            y_true=np.concatenate(y_true_all),
            y_pred=np.concatenate(y_pred_all),
            fold_id=np.concatenate(fold_id_all),
            kept=np.concatenate(kept_all),
            accuracies=accs,
            test_starts=np.array([f["test_start"] for f in per_fold]),
            test_ends=np.array([f["test_end"] for f in per_fold]),
        )

    return WalkForwardResult(
        name=name,
        n_folds=len(folds),
        mean_accuracy=float(accs.mean()),
        std_accuracy=float(accs.std()),
        min_accuracy=float(accs.min()),
        max_accuracy=float(accs.max()),
        per_fold=per_fold,
        npz_path=npz_path,
    )


def summarize(result: WalkForwardResult) -> str:
    """Render a walk-forward result in the skill's reporting convention.

    Produces the headline ``"X% ± y% across N folds (range a–b)"`` line followed by
    a per-fold table. Per-fold detail is included deliberately: a stable improvement
    (wins in every fold) reads differently from a single-period fluke.

    Args:
        result: A ``WalkForwardResult`` from ``walk_forward``.

    Returns:
        Multi-line markdown-ish string suitable for stdout or a report file.
    """
    lines = [
        f"## Walk-forward — {result['name']}",
        "",
        f"Accuracy: {result['mean_accuracy'] * 100:.1f}% "
        f"± {result['std_accuracy'] * 100:.1f}% "
        f"across {result['n_folds']} folds "
        f"(range {result['min_accuracy'] * 100:.1f}–{result['max_accuracy'] * 100:.1f}%)",
        "",
        "| Fold | Test window | n_train | n_test | Accuracy |",
        "|------|-------------|---------|--------|----------|",
    ]
    for f in result["per_fold"]:
        window = f"{f['test_start'][:10]} → {f['test_end'][:10]}"
        lines.append(
            f"| {f['fold']} | {window} | {f['n_train']:,} | "
            f"{f['n_test']:,} | {f['accuracy'] * 100:.1f}% |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model factories ("any sklearn-style model")
# ---------------------------------------------------------------------------

def sklearn_factory(estimator_cls: type, params: dict) -> Callable[[], Any]:
    """Wrap an sklearn estimator class + params into a fresh-instance factory.

    The returned callable builds a brand-new estimator each call with seed 42
    injected (project rule), so ``walk_forward`` gets an unfitted model per fold.

    Args:
        estimator_cls: An sklearn-style class implementing ``fit``/``predict``.
        params: Constructor kwargs; ``random_state`` is forced to 42.

    Returns:
        Zero-arg callable returning a fresh ``estimator_cls`` instance.
    """
    p = dict(params)
    p["random_state"] = 42

    def _make() -> Any:
        return estimator_cls(**p)

    return _make


class _ModuleEstimator:
    """Fresh fit/predict/predict_proba adapter backed by a project model module.

    Delegates fitting to ``module.train(X, y, params, save_path=tmp_path)`` so the
    estimator is constructed exactly as the production pipeline builds it — the
    module's full default hyperparameters (e.g. RF ``class_weight="balanced"``,
    ``oob_score``, ``bootstrap``) with ``params`` overlaid and seed 42 — with no
    duplicated default dicts. Prediction delegates to ``module.predict``;
    ``predict_proba`` exposes the underlying estimator's probabilities for thresholding.
    """

    def __init__(self, module: Any, params: dict, tmp_path: Path) -> None:
        self._module = module
        self._params = dict(params)
        self._tmp = Path(tmp_path)
        self._model: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_ModuleEstimator":
        self._model = self._module.train(
            X, y, params=self._params, save_path=self._tmp
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._module.predict(self._model, X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)


def module_factory(module: Any, params: dict, tmp_path: Path) -> Callable[[], Any]:
    """Factory yielding a fresh ``_ModuleEstimator`` per fold for a project model module.

    Reproduces the exact training recipe of ``src.models.{baseline,rf,gbm}`` — full
    default hyperparameters with ``params`` overlaid, seed 42 — so a walk-forward fold
    fits the same estimator the production/tuning pipeline would.

    Args:
        module: A project model module exposing ``train(X, y, params, save_path)`` and
            ``predict(model, X)`` (``src.models.baseline`` / ``rf`` / ``gbm``).
        params: Hyperparameter overrides overlaid on the module's defaults.
        tmp_path: Scratch path the module's ``train`` writes its joblib to each fold
            (overwritten; required because the module persists on every fit).

    Returns:
        Zero-arg callable returning a fresh fit/predict/predict_proba adapter.
    """
    def _make() -> _ModuleEstimator:
        return _ModuleEstimator(module, params, tmp_path)

    return _make


def project_factories(config: dict) -> dict[str, Callable[[], Any]]:
    """Build ready walk-forward factories for the four project models.

    Each factory yields a fresh, seed-42 estimator configured from
    ``model_params(config, algo)``. The SVM factory returns a ``Pipeline`` of
    ``StandardScaler`` → ``SVC`` so the scaler is fit **inside each fold on the train
    block only** — never on the full series (leakage rule).

    Args:
        config: Config dict from ``load_config()`` supplying per-algo hyperparameters.

    Returns:
        Dict mapping ``{'baseline','rf','gbm','svm'}`` to zero-arg factories.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    baseline_p = model_params(config, "baseline") or {"max_iter": 1000}
    rf_p = model_params(config, "rf")
    gbm_p = model_params(config, "gbm")
    svm_p = model_params(config, "svm")

    def _make_svm() -> Any:
        p = dict(svm_p)
        p["random_state"] = 42
        return make_pipeline(StandardScaler(), SVC(**p))

    return {
        "baseline": sklearn_factory(LogisticRegression, baseline_p),
        "rf": sklearn_factory(RandomForestClassifier, rf_p),
        "gbm": sklearn_factory(XGBClassifier, gbm_p),
        "svm": _make_svm,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_xy(featset: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load raw data and build (X, y, timestamps) aligned to df.iloc[4:]."""
    from src.labels import build_labels
    from src.load import load_raw

    cfg = load_config()
    df = load_raw(Path(cfg["data"]["path"]))
    if featset == "v1":
        from src.features import build_features
        X = build_features(df)
    elif featset == "v2":
        from src.features_v2 import load_or_build_features_v2
        X = load_or_build_features_v2(df)
    elif featset == "v3":
        from src.features_v3 import load_or_build_features_v3
        X = load_or_build_features_v3(df)
    else:
        raise ValueError(f"Unknown featset: {featset!r}")

    raw_align = df.iloc[4:].reset_index(drop=True)
    y = build_labels(raw_align)
    timestamps = raw_align["Date and Time"].reset_index(drop=True)
    return X, y, timestamps


def main() -> None:
    """CLI entry: run walk-forward for one algo on a feature set and print the report."""
    parser = argparse.ArgumentParser(description="Rolling walk-forward validation.")
    parser.add_argument("--algo", default="baseline",
                        choices=("baseline", "rf", "gbm", "svm"))
    parser.add_argument("--featset", default="v1", choices=("v1", "v2", "v3"))
    parser.add_argument("--train-months", type=int, default=None)
    parser.add_argument("--test-months", type=int, default=None)
    parser.add_argument("--step-months", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    X, y, timestamps = _build_xy(args.featset)
    factory = project_factories(cfg)[args.algo]

    result = walk_forward(
        X, y, timestamps, factory,
        config=cfg,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        name=f"{args.algo}_{args.featset}",
    )
    print(summarize(result))
    if result["npz_path"]:
        print(f"\nPredictions saved → {result['npz_path']}")


if __name__ == "__main__":
    main()
