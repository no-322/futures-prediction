"""Experiment 4 — Binary classification suite (configurable feature set + flat toggle).

Trains the three classifiers (baseline, rf, gbm) on the canonical
50/50 time-ordered split as binary up/down models. Two knobs select the variant:
  - feature set: v1 `build_features` (20-dim) or v2 `build_features_v2` (49-dim);
  - `drop_flat`: remove flat bars (Close == Open) from the *training* set only,
    to focus on pure binary up/down — or keep them (flat-included, like production).

Artifacts use a per-variant ``{prefix}_*`` name (e.g. exp_noflat, exp_noflat_v2,
exp_v2) so variants never collide and the originals are never overwritten.

Design guarantees (see plan):
  - Flat removal happens AFTER feature vectors are built, so it never changes any
    other row's features — no look-ahead / data-mining leakage.
  - The test set is sacred and whole; test flat bars keep binary label 0 (down)
    per the project label spec. The per-bar move (Close - Open) is saved into
    each .npz so flat rows can optionally be masked at analysis time.
  - Artifacts use a distinct ``exp_noflat_*`` prefix; the original production
    models, predictions, and metadata are never overwritten (purely additive).

Run with::

    python -m src.binary_suite
    python -m src.binary_suite --algos rf gbm
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, model_params
from src.features import build_features
from src.labels import build_labels, drop_flat
from src.load import load_raw
from src.models import baseline, rf
from src.models.gbm import predict as gbm_predict
from src.models.gbm import train as gbm_train
from src.split import split

_PROC = Path("data/processed")
_ALGOS: tuple[str, ...] = ("baseline", "rf", "gbm")
BASE_DISPLAY: dict[str, str] = {
    "baseline": "Logistic Regression",
    "rf":       "Random Forest",
    "gbm":      "Gradient Boosting (XGBoost)",
}


def _display(algo: str, suffix: str) -> str:
    """Compose a model display name, e.g. 'Random Forest (no-flat, v2)'."""
    return f"{BASE_DISPLAY[algo]} ({suffix})"


# Back-compat: v1 no-flat display names (referenced by run_all_stats.section_c).
_DISPLAY: dict[str, str] = {a: _display(a, "no-flat") for a in BASE_DISPLAY}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _build_dataset(
    cfg: dict,
    build_features_fn: "callable | None" = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray]:
    """Build the binary 50/50 split with flat (Close==Open) rows dropped globally.

    Features are built on the full series, then flat rows are removed from the whole
    modelling set (train and test) before splitting, so labels are strictly binary.

    Args:
        cfg: Parsed config dict (provides data.path and data.train_size).
        build_features_fn: Feature builder `(df) -> DataFrame`; defaults to the
            v1 `build_features`. Pass e.g. `features_v2.load_or_build_features_v2`.

    Returns:
        (X_train, X_test, y_train, y_test, move_test) — feature matrices and binary
        labels, plus the per-bar signed move (Close - Open) for the test set.
    """
    train_size = cfg["data"].get("train_size", 0.5)
    data_path  = Path(cfg["data"]["path"])

    df        = load_raw(data_path)
    feat_fn   = build_features_fn or build_features
    features  = feat_fn(df)
    raw_align = df.iloc[4:].reset_index(drop=True)
    features, raw_align = drop_flat(features, raw_align)   # binary 0/1 modelling set

    X_train, X_test     = split(features, train_size=train_size)
    raw_train, raw_test = split(raw_align, train_size=train_size)
    y_train = build_labels(raw_train)
    y_test  = build_labels(raw_test)

    move_test = (raw_test["Close"] - raw_test["Open"]).to_numpy()
    return X_train, X_test, y_train, y_test, move_test


# ---------------------------------------------------------------------------
# Train / predict dispatch (reuses src/models, distinct save paths)
# ---------------------------------------------------------------------------

def _train(algo: str, X: pd.DataFrame, y: pd.Series, params: dict, save_path: Path) -> Any:
    """Train one algorithm, persisting to the no-flat save_path."""
    if algo == "baseline":
        return baseline.train(X, y, params=params, save_path=save_path)
    if algo == "rf":
        return rf.train(X, y, params=params, save_path=save_path)
    if algo == "gbm":
        return gbm_train(X, y, params=params, save_path=save_path)
    raise ValueError(f"Unknown algo: {algo!r}")


def _predict(algo: str, model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict binary labels with a fitted model."""
    if algo == "baseline":
        return baseline.predict(model, X)
    if algo == "rf":
        return rf.predict(model, X)
    if algo == "gbm":
        return gbm_predict(model, X)
    raise ValueError(f"Unknown algo: {algo!r}")


def _feature_importance(
    algo: str,
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame | None:
    """Return a ranked feature-importance table, or None if unavailable.

    rf/gbm expose ``feature_importances_``; baseline (logistic regression) uses
    the absolute coefficient magnitude. Any other model returns None.

    Args:
        algo: One of 'baseline', 'rf', 'gbm'.
        model: Fitted model artifact.
        feature_names: Column names of the training feature matrix.

    Returns:
        DataFrame with columns ['feature', 'importance'] sorted descending, or
        None when the model exposes no importance measure.
    """
    if algo in ("rf", "gbm"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif algo == "baseline":
        imp = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        return None
    return (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    config: dict | None = None,
    algos: tuple[str, ...] = _ALGOS,
    build_features_fn: "callable | None" = None,
    prefix: str = "exp_noflat",
    display_suffix: str = "no-flat",
) -> list[dict[str, Any]]:
    """Train a binary model suite and persist all artifacts.

    Flat (`Close==Open`) rows are dropped from the whole modelling set (rule 7), so
    every model trains and is scored on decisive up/down bars. For each algorithm:
    fit, predict the test set, and save the model joblib, the predictions .npz
    (y_true, y_pred, move), and a feature-importance CSV where available. Artifacts
    are written as ``{prefix}_{algo}_*`` so different feature-set variants never collide.

    Args:
        config: Parsed config dict; defaults to load_config().
        algos: Which algorithms to run (subset of baseline/rf/gbm).
        build_features_fn: Feature builder; defaults to v1 `build_features`.
        prefix: Artifact filename prefix (e.g. 'exp_noflat', 'exp_v2').
        display_suffix: Suffix used in display names (e.g. 'no-flat, v2').

    Returns:
        List of {'algo', 'display', 'npz'} dicts for the runs that completed.
    """
    cfg = config or load_config()
    X_train, X_test, y_train, y_test, move_test = _build_dataset(
        cfg, build_features_fn
    )
    y_true        = y_test.to_numpy()
    feature_names = list(X_train.columns)
    _PROC.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for algo in algos:
        params     = model_params(cfg, algo)
        model_path = _PROC / f"{prefix}_{algo}_model.joblib"
        npz_path   = _PROC / f"{prefix}_{algo}_predictions.npz"

        print(f"  [{algo}] training on {len(X_train):,} rows × "
              f"{X_train.shape[1]} features [{display_suffix}]...")
        model  = _train(algo, X_train, y_train, params, model_path)
        y_pred = _predict(algo, model, X_test)

        np.savez(npz_path, y_true=y_true, y_pred=y_pred, move=move_test)
        print(f"    Model      → {model_path}")
        print(f"    Predictions → {npz_path}")

        imp_df = _feature_importance(algo, model, feature_names)
        if imp_df is not None:
            imp_path = _PROC / f"{prefix}_{algo}_feature_importance.csv"
            imp_df.to_csv(imp_path, index=False)
            print(f"    Importances → {imp_path}")
        else:
            print("    Importances → skipped (no native measure)")

        up_rate = float((y_pred == 1).mean())
        acc     = float((y_pred == y_true).mean())
        print(f"    pred_up_rate={up_rate:.4f}  raw_accuracy={acc:.4f}")

        runs.append({"algo": algo,
                     "display": _display(algo, display_suffix),
                     "npz": str(npz_path)})

    return runs


if __name__ == "__main__":
    import argparse

    import src.statistics as statistics

    parser = argparse.ArgumentParser(
        description="Experiment 4 — binary classification suite (v1/v2, flat toggle)"
    )
    parser.add_argument(
        "--algos", nargs="+", choices=_ALGOS, default=list(_ALGOS),
        help="Subset of algorithms to train (default: all four)",
    )
    parser.add_argument(
        "--features", choices=["v1", "v2"], default="v1",
        help="Feature set: v1 (20-dim) or v2 (49-dim). Default: v1.",
    )
    args = parser.parse_args()

    if args.features == "v2":
        from src.features_v2 import load_or_build_features_v2 as feat_fn
    else:
        feat_fn = None
    # Flat rows are always dropped now; the prefix just encodes the feature set.
    prefix = "exp_noflat" if args.features == "v1" else "exp_noflat_v2"
    suffix = "no-flat" if args.features == "v1" else "no-flat, v2"

    print(f"Running Experiment 4 — binary suite [{suffix}] "
          f"[{', '.join(args.algos)}]")
    completed = run(algos=tuple(args.algos), build_features_fn=feat_fn,
                    prefix=prefix, display_suffix=suffix)

    print("\nStatistics (binary up/down, full test set):")
    for r in completed:
        d   = np.load(r["npz"])
        res = statistics.compute(d["y_true"], d["y_pred"], name=r["display"])
        print(f"  {r['display']:<38} acc={res['accuracy']:.4f}  "
              f"macro_f1={res['macro_f1']:.4f}  mcc={res['mcc']:.4f}")
