"""Experiment 2 — Two-stage cascade classifier with TimeSeriesSplit CV.

Stage 1 (Gate): predicts whether the next bar will move beyond a threshold
                (tradeable=1 vs skip=0).
Stage 2 (Direction): predicts direction (1=up, 0=down) for bars the gate
                     selects as tradeable.

Combined output labels: no-trade (gate=0), long (gate=1, dir=1), short (gate=1, dir=0).

Threshold is tuned on the inner training window for each fold.

Run with:
    python -m src.experiments.two_stage
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

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


def _build_rf(params: dict) -> RandomForestClassifier:
    p = dict(params)
    p["random_state"] = 42
    p["class_weight"] = "balanced"
    return RandomForestClassifier(**p)


def _tune_threshold(
    X_inner_train: np.ndarray,
    move_inner_train: np.ndarray,
    X_inner_val: np.ndarray,
    move_inner_val: np.ndarray,
    percentiles: list[float],
    rf_params: dict,
) -> float:
    """Select the threshold percentile that maximises gate F1 on the inner validation set.

    Args:
        X_inner_train: Features for the inner training slice.
        move_inner_train: Signed moves for the inner training slice.
        X_inner_val: Features for the inner validation slice.
        move_inner_val: Signed moves for the inner validation slice.
        percentiles: Candidate percentiles of |move_inner_train| to try.
        rf_params: RF hyperparameters.

    Returns:
        Best threshold (in points) found on the inner validation set.
    """
    best_threshold = float(np.percentile(np.abs(move_inner_train), 50))
    best_f1 = -1.0

    for pct in percentiles:
        threshold = float(np.percentile(np.abs(move_inner_train), pct))
        gate_train = gate_labels(pd.Series(move_inner_train), threshold).values
        gate_val   = gate_labels(pd.Series(move_inner_val),   threshold).values

        if gate_train.sum() < 10 or gate_val.sum() == 0:
            continue  # too few positive examples — skip

        # Use a lightweight model for inner tuning — fewer trees, single job.
        # Only used for relative threshold comparison, not final predictions.
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
) -> list[dict[str, Any]]:
    """Run the full two-stage cascade with TimeSeriesSplit CV.

    For each fold:
      1. Hold out inner_val_fraction of the training window for threshold tuning.
      2. Tune threshold on (inner_train, inner_val).
      3. Retrain gate RF on the full training fold with the tuned threshold.
      4. Train direction RF on training bars that genuinely moved beyond threshold.
      5. Evaluate coverage, conditional_hit_rate, and debug diagnostics on test fold.

    Args:
        n_splits: Number of TimeSeriesSplit folds.
        config: Parsed config dict. RF and experiment params read from config.

    Returns:
        List of per-fold result dicts.
    """
    cfg = config or load_config()
    rf_params   = model_params(cfg, "rf")
    exp_cfg     = cfg.get("experiments", {})
    pcts        = exp_cfg.get("two_stage", {}).get("threshold_percentiles",
                                                    [30, 40, 50, 60, 70, 80])
    inner_frac  = exp_cfg.get("two_stage", {}).get("inner_val_fraction", 0.2)

    data_path = Path(cfg["data"]["path"])
    df = load_raw(data_path)
    features  = build_features(df)
    raw_align = df.iloc[4:].reset_index(drop=True)

    X     = features.values
    move  = move_series(raw_align).values          # shape (N,)
    y_dir = direction_labels(raw_align).values     # shape (N,)

    tss = TimeSeriesSplit(n_splits=n_splits)
    results: list[dict[str, Any]] = []

    for fold_i, (train_idx, test_idx) in enumerate(tss.split(X)):
        X_train, X_test   = X[train_idx],    X[test_idx]
        mv_train, mv_test = move[train_idx],  move[test_idx]
        dir_train         = y_dir[train_idx]
        dir_test          = y_dir[test_idx]

        # --- threshold tuning on inner split --------------------------------
        inner_cut = int(len(train_idx) * (1 - inner_frac))
        X_inner_train, X_inner_val    = X_train[:inner_cut],  X_train[inner_cut:]
        mv_inner_train, mv_inner_val  = mv_train[:inner_cut], mv_train[inner_cut:]

        threshold = _tune_threshold(
            X_inner_train, mv_inner_train,
            X_inner_val,   mv_inner_val,
            pcts, rf_params,
        )

        # --- Stage 1: gate — trained on full training fold ------------------
        gate_train = gate_labels(pd.Series(mv_train), threshold).values
        gate_model = _build_rf(rf_params)
        gate_model.fit(X_train, gate_train)
        gate_pred_test = gate_model.predict(X_test)

        # --- Stage 2: direction — trained on bars that genuinely moved ------
        tradeable_mask_train = np.abs(mv_train) > threshold
        if tradeable_mask_train.sum() < 10:
            # degenerate fold — skip direction model
            dir_pred_gated = np.array([], dtype=int)
        else:
            dir_model = _build_rf(rf_params)
            dir_model.fit(X_train[tradeable_mask_train],
                          dir_train[tradeable_mask_train])
            gated_test_mask = gate_pred_test == 1
            dir_pred_gated = (
                dir_model.predict(X_test[gated_test_mask])
                if gated_test_mask.sum() > 0
                else np.array([], dtype=int)
            )

        # --- metrics --------------------------------------------------------
        gated_test_mask = gate_pred_test == 1
        genuinely_moved_test = np.abs(mv_test) > threshold

        fold_result: dict[str, Any] = {
            "fold":                 fold_i,
            "n_train":              int(len(train_idx)),
            "n_test":               int(len(test_idx)),
            "threshold":            float(threshold),
            "coverage":             coverage(gate_pred_test),
            "conditional_hit_rate": conditional_hit_rate(
                dir_test[gated_test_mask], dir_pred_gated
            ) if gated_test_mask.sum() > 0 else 0.0,
            "gate_debug":           gate_recall_debug(mv_test, threshold,
                                                       gate_pred_test),
            "direction_mcc_debug":  direction_mcc_debug(
                dir_test, dir_pred_gated,
                genuinely_moved_test, gated_test_mask,
            ),
            "n_gated_test":         int(gated_test_mask.sum()),
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


def summarise(results: list[dict[str, Any]]) -> str:
    """Format per-fold results as a markdown report.

    Args:
        results: List returned by run().

    Returns:
        Markdown string ready to write to docs/exp_two_stage_results.md.
    """
    lines = [
        "# Experiment 2 — Two-Stage Cascade Classifier\n\n",
        "Stage 1: Gate (tradeable vs skip) | Stage 2: Direction (long vs short)\n",
        f"Folds: {len(results)}  |  Model: Random Forest (both stages)\n\n",
        "## Primary metrics (mean ± std across folds)\n\n",
        "| Metric | Mean | Std | Description |\n",
        "|--------|------|-----|-------------|\n",
    ]

    def _row(label, vals, desc):
        return f"| {label} | {vals.mean():.4f} | {vals.std():.4f} | {desc} |\n"

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
    print("Running Experiment 2 — Two-stage cascade with TimeSeriesSplit CV")
    results = run()
    report = summarise(results)
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(report)
    print(f"\nResults written to {_RESULTS_PATH}")
    print(report)
