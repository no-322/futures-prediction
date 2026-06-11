"""Experiment 5 — Binary HMM-regime direction model.

A binary-classification counterpart to the regime cascade (Experiment 3), built
for the no-flat binary study. There is NO gate and NO HOLD class: a Gaussian HMM
detects 2 latent market regimes (low-vol / high-vol) from 5 regime-descriptor
features, and a separate binary direction RandomForest is trained per regime.
Every test bar is assigned a regime and predicted up (1) / down (0).

Design guarantees (see plan):
  - 50/50 time-ordered split (same test set as the production models), so y_true
    is directly comparable across models.
  - Flat bars (Close == Open) are removed from the *training* split only, after
    features are built — no look-ahead. The test set is whole; test flat bars
    keep binary label 0 (down). Per-bar regime and move (Close - Open) are saved
    so per-regime and flat-aware confusion matrices are recomputable offline
    without retraining.
  - Artifacts use the ``exp_regime_binary_*`` prefix; nothing existing is
    overwritten.

Reuses the HMM helpers from src.models.regime_hmm (``fit_regime``, ``assign_regime``,
``canonical_regime_labels``, ``build_rf``, ``REGIME_COLS``).

Run with::

    python -m src.models.regime_binary
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, model_params
from src.features_v2 import build_features_v2
from src.labels import direction_labels, flat_mask
from src.models.regime_hmm import (
    MIN_REGIME_ROWS,
    REGIME_COLS,
    assign_regime,
    build_rf,
    canonical_regime_labels,
    fit_regime,
)
from src.load import load_raw
from src.split import split

_PROC = Path("data/processed")
_NPZ = _PROC / "exp_regime_binary_predictions.npz"


def _save_importance(model: Any, feature_names: list[str], path: Path) -> None:
    """Write a per-regime RF feature-importance CSV (MDI)."""
    df = (
        pd.DataFrame({
            "feature":    feature_names,
            "importance": np.asarray(model.feature_importances_, dtype=float),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    df.to_csv(path, index=False)


def run(config: dict | None = None) -> dict[str, Any]:
    """Train the binary HMM-regime model and persist all artifacts.

    Args:
        config: Parsed config dict; RF params read from config['models']['rf'].
            Defaults applied if None.

    Returns:
        Summary dict with regime counts and the path of the saved predictions.
    """
    cfg        = config or load_config()
    rf_params  = model_params(cfg, "rf")
    train_size = cfg["data"].get("train_size", 0.5)
    data_path  = Path(cfg["data"]["path"])

    df        = load_raw(data_path)
    features  = build_features_v2(df)
    raw_align = df.iloc[4:].reset_index(drop=True)

    # -- 50/50 split ---------------------------------------------------------
    X_train_df, X_test_df = split(features, train_size=train_size)
    raw_train, raw_test   = split(raw_align, train_size=train_size)

    # -- drop flat rows from training only -----------------------------------
    flat   = flat_mask(raw_train)
    keep   = ~flat
    n_drop = int(flat.sum())
    print(f"  Dropping {n_drop:,} flat (Close==Open) training rows of "
          f"{len(flat):,} ({100 * n_drop / len(flat):.2f}%); "
          f"test set untouched ({len(X_test_df):,} rows)")
    X_train_df = X_train_df.loc[keep].reset_index(drop=True)
    raw_train  = raw_train.loc[keep].reset_index(drop=True)

    X_train = X_train_df.values
    X_test  = X_test_df.values
    y_train = direction_labels(raw_train).values
    y_test  = direction_labels(raw_test).values
    move_test = (raw_test["Close"] - raw_test["Open"]).to_numpy()

    # -- regime detection (HMM) ----------------------------------------------
    regime_col_idx  = [list(features.columns).index(c) for c in REGIME_COLS]
    vol15_local_idx = REGIME_COLS.index("lag1_vol15")
    X_reg_train = X_train[:, regime_col_idx]
    X_reg_test  = X_test[:,  regime_col_idx]

    hmm, scaler      = fit_regime(X_reg_train)
    regime_raw_train = assign_regime(hmm, scaler, X_reg_train)
    regime_raw_test  = assign_regime(hmm, scaler, X_reg_test)

    remap        = canonical_regime_labels(regime_raw_train, X_reg_train, vol15_local_idx)
    regime_train = np.array([remap[r] for r in regime_raw_train])
    regime_test  = np.array([remap[r] for r in regime_raw_test])

    # -- per-regime binary direction models ----------------------------------
    dir_models: dict[int, Any] = {0: None, 1: None}
    for r in [0, 1]:
        mask_r = regime_train == r
        if mask_r.sum() >= MIN_REGIME_ROWS:
            dir_models[r] = build_rf(rf_params)
            dir_models[r].fit(X_train[mask_r], y_train[mask_r])
        else:
            print(f"  Regime {r} has only {int(mask_r.sum())} training rows — "
                  "will use pooled fallback")

    if any(m is None for m in dir_models.values()):
        pooled = build_rf(rf_params)
        pooled.fit(X_train, y_train)
        for r in [0, 1]:
            if dir_models[r] is None:
                dir_models[r] = pooled

    # -- predict every test bar via its regime's model -----------------------
    y_pred = np.full(len(X_test), -1, dtype=int)
    for r in [0, 1]:
        mask = regime_test == r
        if mask.sum() > 0:
            y_pred[mask] = dir_models[r].predict(X_test[mask])
    assert (y_pred >= 0).all(), "every test bar must receive a prediction"

    # -- persist predictions (overall + per-regime + flat-aware) -------------
    _PROC.mkdir(parents=True, exist_ok=True)
    np.savez(_NPZ,
             y_true=y_test, y_pred=y_pred,
             regime=regime_test, move=move_test)
    print(f"  Predictions saved to {_NPZ}")

    # -- persist models + per-regime importances -----------------------------
    joblib.dump(hmm,    _PROC / "exp_regime_binary_hmm.joblib")
    joblib.dump(scaler, _PROC / "exp_regime_binary_scaler.joblib")
    feature_names = list(features.columns)
    for r in [0, 1]:
        joblib.dump(dir_models[r], _PROC / f"exp_regime_binary_dir_r{r}.joblib")
        _save_importance(
            dir_models[r], feature_names,
            _PROC / f"exp_regime_binary_dir_r{r}_feature_importance.csv",
        )
    print("  Models + importances saved to data/processed/exp_regime_binary_*")

    summary = {
        "n_train":            int(len(X_train)),
        "n_test":             int(len(X_test)),
        "n_flat_dropped":     n_drop,
        "regime_counts_train": {r: int((regime_train == r).sum()) for r in [0, 1]},
        "regime_counts_test":  {r: int((regime_test == r).sum()) for r in [0, 1]},
        "regime_vol_centres":  {
            r: float(X_reg_train[regime_train == r, vol15_local_idx].mean())
            if (regime_train == r).any() else float("nan")
            for r in [0, 1]
        },
        "npz": str(_NPZ),
    }
    return summary


if __name__ == "__main__":
    import src.statistics as statistics

    print("Running Experiment 5 — binary HMM-regime direction model")
    summary = run()

    d   = np.load(_NPZ)
    res = statistics.compute(d["y_true"], d["y_pred"],
                             name="HMM-regime binary")
    print(f"\nRegime counts (test): {summary['regime_counts_test']}")
    print(f"Regime vol centres:   {summary['regime_vol_centres']}")
    print(f"\nOverall: acc={res['accuracy']:.4f}  "
          f"macro_f1={res['macro_f1']:.4f}  mcc={res['mcc']:.4f}")
    for r in [0, 1]:
        mask = d["regime"] == r
        if mask.sum() == 0:
            continue
        rr = statistics.compute(d["y_true"][mask], d["y_pred"][mask],
                                name=f"regime {r}")
        print(f"  regime {r}: n={int(mask.sum()):,}  acc={rr['accuracy']:.4f}  "
              f"mcc={rr['mcc']:.4f}")
