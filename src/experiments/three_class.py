"""Experiment 1 — Three-class direction classifier with TimeSeriesSplit CV.

Labels: 0 = down (Close < Open), 1 = up (Close > Open), 2 = flat (Close == Open).

Run with:
    python -m src.experiments.three_class                    # v1 features, fixed HP
    python -m src.experiments.three_class --features v2     # v2 features, fixed HP
    python -m src.experiments.three_class --features v2 --tune-hp  # v2 + walk-forward HP
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.config import load_config, model_params
from src.features import build_features
from src.load import load_raw
from src.experiments.labels import three_class_labels
from src.experiments.metrics import mcc, macro_f1, per_class_recall

_CLASS_NAMES = {0: "down", 1: "up", 2: "flat"}
_RESULTS_PATH = Path("docs/exp_three_class_results.md")
_RESULTS_PATH_V2 = Path("docs/exp_three_class_v2_results.md")

_HP_GRID = {
    "n_estimators":    [50, 100],      # lightweight inner models
    "max_depth":       [None, 5, 10],
    "min_samples_leaf": [1, 5, 10],
    "max_features":    ["sqrt", "log2"],
}
_HP_N_ITER    = 4    # halved from 8 — combined with n_jobs=4 gives ~4× speedup
_HP_INNER_CV  = 3


def _build_rf(params: dict) -> RandomForestClassifier:
    p = dict(params)
    p["random_state"] = 42
    p["class_weight"] = "balanced"
    return RandomForestClassifier(**p)


def _tune_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """Walk-forward inner CV to select RF hyperparameters.

    Uses RandomizedSearchCV with TimeSeriesSplit(n_splits=3) on the outer
    fold's training window. Inner estimators use n_estimators=100, n_jobs=1
    to bound memory; the final model upgrades to n_estimators=500, n_jobs=-1.

    Args:
        X_train: Training features for the current outer fold.
        y_train: Training labels.

    Returns:
        Dict of best hyperparameters ready to pass to _build_rf().
    """
    search = RandomizedSearchCV(
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        ),
        param_distributions=_HP_GRID,
        n_iter=_HP_N_ITER,
        cv=TimeSeriesSplit(n_splits=_HP_INNER_CV),
        scoring="f1_macro",
        n_jobs=4,
        random_state=42,
        refit=False,
        error_score=0.0,
    )
    search.fit(X_train, y_train)
    best = dict(search.best_params_)
    best["n_estimators"] = 500
    best["n_jobs"] = -1
    return best


def run(
    n_splits: int = 5,
    config: dict | None = None,
    build_features_fn: Callable | None = None,
    tune_hp: bool = False,
) -> list[dict[str, Any]]:
    """Run TimeSeriesSplit CV for the three-class labelling scheme.

    Args:
        n_splits: Number of outer TimeSeriesSplit folds.
        config: Parsed config dict; RF params read from config['models']['rf'].
                Defaults applied if None.
        build_features_fn: Feature-building callable with signature
            (df: DataFrame) -> DataFrame.  Defaults to build_features (v1).
        tune_hp: If True, use nested walk-forward CV (inner TSS(3),
            RandomizedSearchCV n_iter=8) to select RF hyperparameters per fold.

    Returns:
        List of per-fold result dicts.
    """
    cfg = config or load_config()
    rf_params = model_params(cfg, "rf")
    feat_fn = build_features_fn or build_features

    data_path = Path(cfg["data"]["path"])
    df = load_raw(data_path)
    features = feat_fn(df)
    raw_align = df.iloc[4:].reset_index(drop=True)

    X = features.values
    y = three_class_labels(raw_align).values

    tag = "v2" if build_features_fn is not None else "v1"

    tss = TimeSeriesSplit(n_splits=n_splits)
    results: list[dict[str, Any]] = []
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    last_model = None

    for fold_i, (train_idx, test_idx) in enumerate(tss.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if tune_hp:
            print(f"  Fold {fold_i}: tuning hyperparameters...")
            best_params = _tune_hyperparams(X_train, y_train)
        else:
            best_params = rf_params

        model = _build_rf(best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        last_model = model

        all_y_true.append(y_test)
        all_y_pred.append(y_pred)

        results.append({
            "fold":               fold_i,
            "n_train":            int(len(train_idx)),
            "n_test":             int(len(test_idx)),
            "accuracy":           float(accuracy_score(y_test, y_pred)),
            "mcc":                mcc(y_test, y_pred),
            "macro_f1":           macro_f1(y_test, y_pred),
            "per_class_recall":   per_class_recall(y_test, y_pred),
            "best_hp":            best_params,
            "class_counts_train": {int(k): int(v) for k, v in
                                   zip(*np.unique(y_train, return_counts=True))},
            "class_counts_test":  {int(k): int(v) for k, v in
                                   zip(*np.unique(y_test,  return_counts=True))},
        })
        print(f"  Fold {fold_i}: acc={results[-1]['accuracy']:.4f}  "
              f"mcc={results[-1]['mcc']:.4f}  "
              f"macro_f1={results[-1]['macro_f1']:.4f}")

    # Persist predictions and last-fold model
    _save_predictions(all_y_true, all_y_pred, f"three_class_{tag}")
    if last_model is not None:
        import joblib as _jl
        model_path = Path(f"data/processed/exp_three_class_{tag}_model.joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _jl.dump(last_model, model_path)
        print(f"  Model saved to {model_path}")
    return results


def _save_predictions(
    all_y_true: list[np.ndarray],
    all_y_pred: list[np.ndarray],
    name: str,
) -> None:
    """Save concatenated cross-fold predictions to data/processed/.

    Args:
        all_y_true: List of per-fold ground-truth arrays.
        all_y_pred: List of per-fold prediction arrays.
        name: Experiment name used to form the filename.
    """
    out = Path("data/processed") / f"exp_{name}_predictions.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             y_true=np.concatenate(all_y_true),
             y_pred=np.concatenate(all_y_pred))
    print(f"  Predictions saved to {out}")


def summarise(results: list[dict[str, Any]], label: str = "") -> str:
    """Format per-fold results as a markdown report."""
    scalar_keys = ["accuracy", "mcc", "macro_f1"]
    all_classes = sorted({k for r in results for k in r["per_class_recall"]})
    tag = f" [{label}]" if label else ""

    lines = [
        f"# Experiment 1 — Three-Class Direction Classifier{tag}\n\n",
        f"Folds: {len(results)}  |  Model: Random Forest  |  "
        "Labels: 0=down, 1=up, 2=flat\n\n",
        "## Scalar metrics (mean ± std across folds)\n\n",
        "| Metric | Mean | Std |\n",
        "|--------|------|-----|\n",
    ]
    for key in scalar_keys:
        vals = np.array([r[key] for r in results])
        lines.append(f"| {key} | {vals.mean():.4f} | {vals.std():.4f} |\n")

    lines += [
        "\n## Per-class recall (mean ± std across folds)\n\n",
        "| Class | Mean recall | Std |\n",
        "|-------|-------------|-----|\n",
    ]
    for cls in all_classes:
        vals = np.array([r["per_class_recall"].get(cls, 0.0) for r in results])
        lines.append(f"| {cls} ({_CLASS_NAMES.get(cls, '?')}) | "
                     f"{vals.mean():.4f} | {vals.std():.4f} |\n")

    lines += [
        "\n## Per-fold detail\n\n",
        "| Fold | n_train | n_test | accuracy | mcc | macro_f1 | "
        "recall_down | recall_up | recall_flat |\n",
        "|------|---------|--------|----------|-----|----------|"
        "------------|-----------|-------------|\n",
    ]
    for r in results:
        pcr = r["per_class_recall"]
        lines.append(
            f"| {r['fold']} | {r['n_train']:,} | {r['n_test']:,} | "
            f"{r['accuracy']:.4f} | {r['mcc']:.4f} | {r['macro_f1']:.4f} | "
            f"{pcr.get(0, 0):.4f} | {pcr.get(1, 0):.4f} | {pcr.get(2, 0):.4f} |\n"
        )

    return "".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 1 — Three-class RF classifier"
    )
    parser.add_argument(
        "--features", choices=["v1", "v2"], default="v1",
        help="Feature set: v1 = 20 lagged OHLCV, v2 = 49 engineered features",
    )
    parser.add_argument(
        "--tune-hp", action="store_true",
        help="Enable nested walk-forward hyperparameter selection per fold",
    )
    args = parser.parse_args()

    if args.features == "v2":
        from src.experiments.features_v2 import build_features_v2
        feat_fn    = build_features_v2
        out_path   = _RESULTS_PATH_V2
        run_label  = "v2 features"
    else:
        feat_fn    = None
        out_path   = _RESULTS_PATH
        run_label  = "v1 features"

    if args.tune_hp:
        run_label += " + walk-forward HP tuning"

    print(f"Running Experiment 1 — Three-class RF  [{run_label}]")
    results = run(build_features_fn=feat_fn, tune_hp=args.tune_hp)
    report  = summarise(results, label=run_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\nResults written to {out_path}")
    print(report)
