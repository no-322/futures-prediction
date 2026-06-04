"""Experiment 2 — Two-stage cascade classifier with TimeSeriesSplit CV.

Stage 1 (Gate): predicts whether the next bar will move beyond a threshold
                (tradeable=1 vs skip=0).
Stage 2 (Direction): predicts direction (1=up, 0=down) for bars the gate
                     selects as tradeable.

Combined output labels: no-trade (gate=0), long (gate=1, dir=1), short (gate=1, dir=0).

Threshold is tuned on the inner training window for each fold.

Run with:
    python -m src.experiments.two_stage
    python -m src.experiments.two_stage --features v2
    python -m src.experiments.two_stage --features v2 --tune-hp
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.config import load_config, model_params
from src.features import build_features
from src.load import load_raw
from src.experiments.labels import direction_labels, gate_labels, move_series
from src.experiments.metrics import (
    conditional_hit_rate,
    coverage,
    direction_mcc_debug,
    gate_recall_debug,
)

_RESULTS_PATH = Path("docs/exp_two_stage_results.md")
_RESULTS_PATH_V2 = Path("docs/exp_two_stage_v2_results.md")

_HP_GRID = {
    "n_estimators":    [50, 100],      # lightweight inner models
    "max_depth":       [None, 5, 10],
    "min_samples_leaf": [1, 5, 10],
    "max_features":    ["sqrt", "log2"],
}
_HP_N_ITER   = 4    # halved from 8 — combined with n_jobs=4 gives ~4× speedup
_HP_INNER_CV = 3


def _build_rf(params: dict) -> RandomForestClassifier:
    p = dict(params)
    p["random_state"] = 42
    p["class_weight"] = "balanced"
    return RandomForestClassifier(**p)


def _tune_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scoring: str = "f1",
) -> dict:
    """Walk-forward inner CV to select RF hyperparameters.

    Args:
        X_train: Training features for the current outer fold.
        y_train: Training labels (binary).
        scoring: sklearn scoring string passed to RandomizedSearchCV.

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
        scoring=scoring,
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


def _tune_threshold(
    X_inner_train: np.ndarray,
    move_inner_train: np.ndarray,
    X_inner_val: np.ndarray,
    move_inner_val: np.ndarray,
    percentiles: list[float],
    rf_params: dict,
) -> float:
    """Select the threshold percentile that maximises gate F1 on the inner validation set."""
    best_threshold = float(np.percentile(np.abs(move_inner_train), 50))
    best_f1 = -1.0

    for pct in percentiles:
        threshold = float(np.percentile(np.abs(move_inner_train), pct))
        gate_train = gate_labels(pd.Series(move_inner_train), threshold).values
        gate_val   = gate_labels(pd.Series(move_inner_val),   threshold).values

        if gate_train.sum() < 10 or gate_val.sum() == 0:
            continue

        _inner_params = dict(rf_params)
        _inner_params.update({"n_estimators": 50, "n_jobs": 1})
        gate_model = _build_rf(_inner_params)
        gate_model.fit(X_inner_train, gate_train)
        gate_pred_val = gate_model.predict(X_inner_val)

        score = f1_score(gate_val, gate_pred_val, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold


def run(
    n_splits: int = 5,
    config: dict | None = None,
    build_features_fn: Callable | None = None,
    tune_hp: bool = False,
) -> list[dict[str, Any]]:
    """Run the full two-stage cascade with TimeSeriesSplit CV.

    For each fold:
      1. Hold out inner_val_fraction of the training window for threshold tuning.
      2. Tune threshold on (inner_train, inner_val).
      3. Optionally tune RF hyperparameters via inner walk-forward CV.
      4. Retrain gate RF on the full training fold with the tuned threshold.
      5. Train direction RF on training bars that genuinely moved beyond threshold.
      6. Evaluate coverage, conditional_hit_rate, and debug diagnostics on test fold.

    Args:
        n_splits: Number of TimeSeriesSplit folds.
        config: Parsed config dict. RF and experiment params read from config.
        build_features_fn: Feature-building callable. Defaults to build_features (v1).
        tune_hp: If True, use nested walk-forward CV to select RF hyperparameters.

    Returns:
        List of per-fold result dicts.
    """
    cfg = config or load_config()
    rf_params  = model_params(cfg, "rf")
    exp_cfg    = cfg.get("experiments", {})
    pcts       = exp_cfg.get("two_stage", {}).get("threshold_percentiles",
                                                   [30, 40, 50, 60, 70, 80])
    inner_frac = exp_cfg.get("two_stage", {}).get("inner_val_fraction", 0.2)
    feat_fn    = build_features_fn or build_features

    data_path = Path(cfg["data"]["path"])
    df        = load_raw(data_path)
    features  = feat_fn(df)
    raw_align = df.iloc[4:].reset_index(drop=True)

    X     = features.values
    move  = move_series(raw_align).values
    y_dir = direction_labels(raw_align).values

    tss = TimeSeriesSplit(n_splits=n_splits)
    results: list[dict[str, Any]] = []

    for fold_i, (train_idx, test_idx) in enumerate(tss.split(X)):
        X_train, X_test   = X[train_idx],   X[test_idx]
        mv_train, mv_test = move[train_idx], move[test_idx]
        dir_train         = y_dir[train_idx]
        dir_test          = y_dir[test_idx]

        # --- threshold tuning on inner split ---------------------------------
        inner_cut = int(len(train_idx) * (1 - inner_frac))
        X_inner_train, X_inner_val    = X_train[:inner_cut],  X_train[inner_cut:]
        mv_inner_train, mv_inner_val  = mv_train[:inner_cut], mv_train[inner_cut:]

        threshold = _tune_threshold(
            X_inner_train, mv_inner_train,
            X_inner_val,   mv_inner_val,
            pcts, rf_params,
        )

        # --- optional HP tuning for both stages ------------------------------
        if tune_hp:
            print(f"  Fold {fold_i}: tuning gate hyperparameters...")
            gate_labels_train = gate_labels(pd.Series(mv_train), threshold).values
            gate_hp = _tune_hyperparams(X_train, gate_labels_train, scoring="f1")

            tradeable_mask_train = np.abs(mv_train) > threshold
            if tradeable_mask_train.sum() >= 1000:
                print(f"  Fold {fold_i}: tuning direction hyperparameters...")
                dir_hp = _tune_hyperparams(
                    X_train[tradeable_mask_train],
                    dir_train[tradeable_mask_train],
                    scoring="f1",
                )
            else:
                dir_hp = gate_hp  # not enough data — reuse gate params
        else:
            gate_hp = rf_params
            dir_hp  = rf_params

        # --- Stage 1: gate ---------------------------------------------------
        gate_labels_full = gate_labels(pd.Series(mv_train), threshold).values
        gate_model = _build_rf(gate_hp)
        gate_model.fit(X_train, gate_labels_full)
        gate_pred_test = gate_model.predict(X_test)

        # --- Stage 2: direction ----------------------------------------------
        tradeable_mask_train = np.abs(mv_train) > threshold
        if tradeable_mask_train.sum() < 10:
            dir_pred_gated = np.array([], dtype=int)
        else:
            dir_model = _build_rf(dir_hp)
            dir_model.fit(
                X_train[tradeable_mask_train],
                dir_train[tradeable_mask_train],
            )
            gated_test_mask = gate_pred_test == 1
            dir_pred_gated = (
                dir_model.predict(X_test[gated_test_mask])
                if gated_test_mask.sum() > 0
                else np.array([], dtype=int)
            )

        # --- metrics ---------------------------------------------------------
        gated_test_mask      = gate_pred_test == 1
        genuinely_moved_test = np.abs(mv_test) > threshold

        fold_result: dict[str, Any] = {
            "fold":                   fold_i,
            "n_train":                int(len(train_idx)),
            "n_test":                 int(len(test_idx)),
            "threshold":              float(threshold),
            "coverage":               coverage(gate_pred_test),
            "conditional_hit_rate":   conditional_hit_rate(
                dir_test[gated_test_mask], dir_pred_gated
            ) if gated_test_mask.sum() > 0 else 0.0,
            "gate_debug":             gate_recall_debug(mv_test, threshold,
                                                         gate_pred_test),
            "direction_mcc_debug":    direction_mcc_debug(
                dir_test, dir_pred_gated,
                genuinely_moved_test, gated_test_mask,
            ),
            "best_gate_hp":           gate_hp,
            "best_dir_hp":            dir_hp,
            "n_gated_test":           int(gated_test_mask.sum()),
            "n_truly_tradeable_test": int(genuinely_moved_test.sum()),
        }
        results.append(fold_result)

        print(
            f"  Fold {fold_i}: threshold={threshold:.4f}  "
            f"coverage={fold_result['coverage']:.3f}  "
            f"hit_rate={fold_result['conditional_hit_rate']:.4f}  "
            f"gate_recall={fold_result['gate_debug']['recall']:.4f}  "
            f"dir_mcc={fold_result['direction_mcc_debug']:.4f}"
        )

    return results


def summarise(results: list[dict[str, Any]], label: str = "") -> str:
    """Format per-fold results as a markdown report."""
    tag = f" [{label}]" if label else ""
    lines = [
        f"# Experiment 2 — Two-Stage Cascade Classifier{tag}\n\n",
        "Stage 1: Gate (tradeable vs skip) | Stage 2: Direction (long vs short)\n",
        f"Folds: {len(results)}  |  Model: Random Forest (both stages)\n\n",
        "## Primary metrics (mean ± std across folds)\n\n",
        "| Metric | Mean | Std | Description |\n",
        "|--------|------|-----|-------------|\n",
    ]

    def _row(label_str, vals, desc):
        return f"| {label_str} | {vals.mean():.4f} | {vals.std():.4f} | {desc} |\n"

    cov     = np.array([r["coverage"]             for r in results])
    chr_    = np.array([r["conditional_hit_rate"]  for r in results])
    g_rec   = np.array([r["gate_debug"]["recall"]  for r in results])
    g_prec  = np.array([r["gate_debug"]["precision"] for r in results])
    g_f1    = np.array([r["gate_debug"]["f1"]      for r in results])
    dir_mcc = np.array([r["direction_mcc_debug"]   for r in results])
    thr     = np.array([r["threshold"]             for r in results])

    lines += [
        _row("coverage",             cov,    "fraction of bars the gate trades"),
        _row("conditional_hit_rate", chr_,   "directional accuracy on traded bars"),
        "\n## Debug diagnostics (mean ± std across folds)\n\n",
        "| Metric | Mean | Std | Description |\n",
        "|--------|------|-----|-------------|\n",
        _row("gate_recall",     g_rec,   "recall on bars that genuinely moved > threshold"),
        _row("gate_precision",  g_prec,  "precision of gate predictions"),
        _row("gate_f1",         g_f1,    "F1 of gate model on test fold"),
        _row("direction_mcc",   dir_mcc, "MCC on bars that genuinely moved (regardless of gate)"),
        _row("threshold (pts)", thr,     "tuned threshold per fold"),
        "\n## Per-fold detail\n\n",
        "| Fold | n_train | n_test | threshold | coverage | hit_rate | "
        "gate_recall | dir_mcc |\n",
        "|------|---------|--------|-----------|----------|----------|"
        "------------|----------|\n",
    ]
    for r in results:
        lines.append(
            f"| {r['fold']} | {r['n_train']:,} | {r['n_test']:,} | "
            f"{r['threshold']:.4f} | {r['coverage']:.4f} | "
            f"{r['conditional_hit_rate']:.4f} | "
            f"{r['gate_debug']['recall']:.4f} | "
            f"{r['direction_mcc_debug']:.4f} |\n"
        )
    return "".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 2 — Two-stage cascade RF classifier"
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
        feat_fn   = build_features_v2
        out_path  = _RESULTS_PATH_V2
        run_label = "v2 features"
    else:
        feat_fn   = None
        out_path  = _RESULTS_PATH
        run_label = "v1 features"

    if args.tune_hp:
        run_label += " + walk-forward HP tuning"

    print(f"Running Experiment 2 — Two-stage cascade  [{run_label}]")
    results = run(build_features_fn=feat_fn, tune_hp=args.tune_hp)
    report  = summarise(results, label=run_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\nResults written to {out_path}")
    print(report)
