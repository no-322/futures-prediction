"""Experiment 3 — Regime-conditional two-stage cascade with Gaussian HMM.

Extends Experiment 2 (two-stage v2) by inserting a Gaussian HMM between the
gate and the direction classifier.  The HMM detects 2 latent market regimes
(typically low-vol/calm vs high-vol/active) from 5 regime-descriptor features.
A separate direction RF is trained per regime, allowing the model to specialise
its directional view for each market condition.

Full cascade per test bar:
    Gate (regime-blind) → tradeable? yes/no
      └─ yes → HMM regime label {0,1}
                  └─ direction_model[regime].predict(X) → long/short

Run with:
    python -m src.experiments.regime_two_stage
    python -m src.experiments.regime_two_stage --tune-hp
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.config import load_config, model_params
from src.experiments.features_v2 import build_features_v2
from src.experiments.labels import direction_labels, gate_labels, move_series
from src.experiments.metrics import conditional_hit_rate, coverage, direction_mcc_debug, gate_recall_debug
from src.load import load_raw

_RESULTS_PATH = Path("docs/exp_regime_two_stage_results.md")

# Features used to characterise the market regime
_REGIME_COLS = [
    "lag1_vol15",
    "lag1_macd_hist",
    "lag1_rsi15",
    "lag1_tick_delta",
    "lag1_return",
]

_HP_GRID = {
    "n_estimators":    [50, 100],
    "max_depth":       [None, 5, 10],
    "min_samples_leaf": [1, 5, 10],
    "max_features":    ["sqrt", "log2"],
}
_HP_N_ITER   = 4
_HP_INNER_CV = 3
_MIN_REGIME_ROWS = 200   # minimum bars per regime to train a direction model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Walk-forward inner CV hyperparameter search (n_iter=4, inner TSS(3))."""
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=1),
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
    X_inner: np.ndarray,
    mv_inner: np.ndarray,
    X_val: np.ndarray,
    mv_val: np.ndarray,
    percentiles: list[float],
    rf_params: dict,
) -> float:
    """Select gate threshold maximising F1 on inner validation slice."""
    best_thr = float(np.percentile(np.abs(mv_inner), 50))
    best_f1  = -1.0
    for pct in percentiles:
        thr = float(np.percentile(np.abs(mv_inner), pct))
        g_train = gate_labels(pd.Series(mv_inner), thr).values
        g_val   = gate_labels(pd.Series(mv_val),   thr).values
        if g_train.sum() < 10 or g_val.sum() == 0:
            continue
        p = dict(rf_params)
        p.update({"n_estimators": 50, "n_jobs": 1})
        gm = _build_rf(p)
        gm.fit(X_inner, g_train)
        score = f1_score(g_val, gm.predict(X_val), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = thr
    return best_thr


def _fit_regime(
    X_regime: np.ndarray,
) -> tuple[GaussianHMM, StandardScaler]:
    """Fit Gaussian HMM(2 states) on the regime descriptor features.

    Args:
        X_regime: Array of shape (n_train, 5) containing regime descriptor cols.

    Returns:
        (hmm, scaler) — fitted HMM and the scaler used to normalise inputs.
    """
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_regime)
    hmm    = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=100,
        random_state=42,
    )
    hmm.fit(X_sc)
    return hmm, scaler


def _assign_regime(
    hmm: GaussianHMM,
    scaler: StandardScaler,
    X_regime: np.ndarray,
) -> np.ndarray:
    """Decode hidden state sequence via Viterbi algorithm."""
    return hmm.predict(scaler.transform(X_regime))


def _canonical_regime_labels(
    regime_train: np.ndarray,
    X_regime_train: np.ndarray,
    vol15_col_idx: int,
) -> dict[int, int]:
    """Map raw HMM state labels to canonical labels.

    Regime 0 = low vol (calm), Regime 1 = high vol (active).
    Returns a remapping dict {raw_label → canonical_label}.
    """
    mean_vol = {
        s: float(X_regime_train[regime_train == s, vol15_col_idx].mean())
        for s in [0, 1]
    }
    # State with higher mean vol15 → canonical 1 (high-vol)
    if mean_vol[0] > mean_vol[1]:
        return {0: 1, 1: 0}
    return {0: 0, 1: 1}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    n_splits: int = 5,
    config: dict | None = None,
    tune_hp: bool = False,
) -> list[dict[str, Any]]:
    """Run the regime-conditional two-stage cascade with TimeSeriesSplit CV.

    Args:
        n_splits: Number of outer TimeSeriesSplit folds.
        config: Parsed config dict.
        tune_hp: If True, use nested walk-forward HP selection per fold.

    Returns:
        List of per-fold result dicts.
    """
    cfg        = config or load_config()
    rf_params  = model_params(cfg, "rf")
    exp_cfg    = cfg.get("experiments", {})
    pcts       = exp_cfg.get("two_stage", {}).get("threshold_percentiles",
                                                   [30, 40, 50, 60, 70, 80])
    inner_frac = exp_cfg.get("two_stage", {}).get("inner_val_fraction", 0.2)

    data_path = Path(cfg["data"]["path"])
    df        = load_raw(data_path)
    features  = build_features_v2(df)
    raw_align = df.iloc[4:].reset_index(drop=True)

    X       = features.values
    X_df    = features                               # keep column names for regime cols
    move    = move_series(raw_align).values
    y_dir   = direction_labels(raw_align).values

    # Indices of regime descriptor columns within features_v2
    regime_col_idx  = [list(X_df.columns).index(c) for c in _REGIME_COLS]
    vol15_local_idx = _REGIME_COLS.index("lag1_vol15")  # within regime cols

    tss     = TimeSeriesSplit(n_splits=n_splits)
    results: list[dict[str, Any]] = []
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    last_gate_model  = None
    last_dir_models: dict[int, RandomForestClassifier] = {}

    for fold_i, (train_idx, test_idx) in enumerate(tss.split(X)):
        X_train,    X_test    = X[train_idx],    X[test_idx]
        mv_train,   mv_test   = move[train_idx],  move[test_idx]
        dir_train,  dir_test  = y_dir[train_idx], y_dir[test_idx]

        X_reg_train = X_train[:, regime_col_idx]
        X_reg_test  = X_test[:,  regime_col_idx]

        # -- threshold tuning --------------------------------------------------
        inner_cut = int(len(train_idx) * (1 - inner_frac))
        threshold = _tune_threshold(
            X_train[:inner_cut], mv_train[:inner_cut],
            X_train[inner_cut:], mv_train[inner_cut:],
            pcts, rf_params,
        )

        # -- gate model --------------------------------------------------------
        gate_lbl_train = gate_labels(pd.Series(mv_train), threshold).values
        if tune_hp:
            print(f"  Fold {fold_i}: tuning gate HP...")
            gate_hp = _tune_hyperparams(X_train, gate_lbl_train, scoring="f1")
        else:
            gate_hp = rf_params
        gate_model = _build_rf(gate_hp)
        gate_model.fit(X_train, gate_lbl_train)
        gate_pred  = gate_model.predict(X_test)

        # -- regime detection (HMM) -------------------------------------------
        hmm, scaler = _fit_regime(X_reg_train)
        regime_raw_train = _assign_regime(hmm, scaler, X_reg_train)
        regime_raw_test  = _assign_regime(hmm, scaler, X_reg_test)

        # Canonicalise labels: 0=low-vol, 1=high-vol
        remap = _canonical_regime_labels(
            regime_raw_train, X_reg_train, vol15_local_idx
        )
        regime_train = np.array([remap[r] for r in regime_raw_train])
        regime_test  = np.array([remap[r] for r in regime_raw_test])

        # -- direction models per regime ---------------------------------------
        dir_models: dict[int, RandomForestClassifier | None] = {0: None, 1: None}
        pooled_dir_model = None   # fallback when regime has too few bars

        for r in [0, 1]:
            mask_r = (regime_train == r) & (np.abs(mv_train) > threshold)
            if mask_r.sum() >= _MIN_REGIME_ROWS:
                if tune_hp:
                    print(f"  Fold {fold_i}: tuning direction HP (regime {r})...")
                    dir_hp = _tune_hyperparams(X_train[mask_r], dir_train[mask_r])
                else:
                    dir_hp = rf_params
                dir_models[r] = _build_rf(dir_hp)
                dir_models[r].fit(X_train[mask_r], dir_train[mask_r])
            else:
                print(f"  Fold {fold_i}: regime {r} has only {mask_r.sum()} "
                      f"tradeable bars — will use pooled fallback")

        # Pooled fallback direction model (trained on all tradeable bars)
        tradeable_train = np.abs(mv_train) > threshold
        if tradeable_train.sum() >= _MIN_REGIME_ROWS:
            if pooled_dir_model is None:
                pooled_dir_model = _build_rf(rf_params)
                pooled_dir_model.fit(X_train[tradeable_train],
                                     dir_train[tradeable_train])
            for r in [0, 1]:
                if dir_models[r] is None:
                    dir_models[r] = pooled_dir_model

        # Track last-fold models for joblib saving
        last_gate_model = gate_model
        last_dir_models = {r: m for r, m in dir_models.items() if m is not None}

        # -- predict on test set ----------------------------------------------
        gated_mask           = gate_pred == 1
        genuinely_moved_test = np.abs(mv_test) > threshold

        dir_pred_full = np.full(len(X_test), -1, dtype=int)  # −1 = no prediction
        for r in [0, 1]:
            if dir_models[r] is None:
                continue
            regime_gated_mask = gated_mask & (regime_test == r)
            if regime_gated_mask.sum() > 0:
                dir_pred_full[regime_gated_mask] = dir_models[r].predict(
                    X_test[regime_gated_mask]
                )

        # Only gated bars with a valid prediction
        valid_pred_mask = (gated_mask) & (dir_pred_full >= 0)
        dir_pred_gated  = dir_pred_full[gated_mask]

        # -- per-regime metrics -----------------------------------------------
        per_regime: dict[int, dict] = {}
        for r in [0, 1]:
            reg_gated = gated_mask & (regime_test == r)
            if reg_gated.sum() > 0:
                dir_pred_r = dir_pred_full[reg_gated]
                valid      = dir_pred_r >= 0
                per_regime[r] = {
                    "n_test":          int((regime_test == r).sum()),
                    "n_gated":         int(reg_gated.sum()),
                    "hit_rate":        conditional_hit_rate(
                        dir_test[reg_gated][valid], dir_pred_r[valid]
                    ) if valid.sum() > 0 else float("nan"),
                    "direction_mcc":   float(
                        direction_mcc_debug(
                            dir_test,
                            dir_pred_full[gated_mask],
                            genuinely_moved_test,
                            gated_mask,
                        )
                    ) if gated_mask.sum() > 0 else float("nan"),
                }
            else:
                per_regime[r] = {"n_test": 0, "n_gated": 0,
                                  "hit_rate": float("nan"),
                                  "direction_mcc": float("nan")}

        # Overall metrics (consistent with v2 two_stage for comparison)
        overall_dir_pred = dir_pred_full[gated_mask]
        valid_overall    = overall_dir_pred >= 0

        fold_result: dict[str, Any] = {
            "fold":       fold_i,
            "n_train":    int(len(train_idx)),
            "n_test":     int(len(test_idx)),
            "threshold":  float(threshold),
            "regime_counts_train": {
                r: int((regime_train == r).sum()) for r in [0, 1]
            },
            "regime_counts_test": {
                r: int((regime_test == r).sum()) for r in [0, 1]
            },
            "regime_vol_centres": {
                r: float(X_reg_train[regime_train == r, vol15_local_idx].mean())
                for r in [0, 1]
            },
            "overall": {
                "coverage":             float(coverage(gate_pred)),
                "conditional_hit_rate": conditional_hit_rate(
                    dir_test[gated_mask][valid_overall],
                    overall_dir_pred[valid_overall],
                ) if valid_overall.sum() > 0 else float("nan"),
                "direction_mcc": direction_mcc_debug(
                    dir_test, overall_dir_pred,
                    genuinely_moved_test, gated_mask,
                ) if gated_mask.sum() > 0 else float("nan"),
            },
            "gate_debug": gate_recall_debug(mv_test, threshold, gate_pred),
            "per_regime": per_regime,
        }
        results.append(fold_result)

        # Accumulate gated predictions for post-hoc statistics
        valid_overall_pred = dir_pred_full[gated_mask]
        valid_mask = valid_overall_pred >= 0
        if valid_mask.sum() > 0:
            all_y_true.append(dir_test[gated_mask][valid_mask])
            all_y_pred.append(valid_overall_pred[valid_mask])

        print(
            f"  Fold {fold_i}: threshold={threshold:.4f}  "
            f"coverage={fold_result['overall']['coverage']:.3f}  "
            f"hit_rate={fold_result['overall']['conditional_hit_rate']:.4f}  "
            f"dir_mcc={fold_result['overall']['direction_mcc']:.4f}  "
            f"regime_counts={fold_result['regime_counts_test']}"
        )

    _save_predictions(all_y_true, all_y_pred, "regime_v2")
    import joblib as _jl
    proc = Path("data/processed")
    proc.mkdir(parents=True, exist_ok=True)
    if last_gate_model is not None:
        _jl.dump(last_gate_model, proc / "exp_regime_v2_gate.joblib")
    for r, m in last_dir_models.items():
        _jl.dump(m, proc / f"exp_regime_v2_dir_r{r}.joblib")
    if last_gate_model is not None:
        print(f"  Models saved to data/processed/exp_regime_v2_*.joblib")
    return results


def _save_predictions(
    all_y_true: list[np.ndarray],
    all_y_pred: list[np.ndarray],
    name: str,
) -> None:
    """Save concatenated gated-bar predictions to data/processed/."""
    if not all_y_true:
        return
    out = Path("data/processed") / f"exp_{name}_predictions.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             y_true=np.concatenate(all_y_true),
             y_pred=np.concatenate(all_y_pred))
    print(f"  Predictions saved to {out}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def summarise(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Experiment 3 — Regime-Conditional Two-Stage Cascade (Gaussian HMM)\n\n",
        f"Folds: {len(results)}  |  Regime model: GaussianHMM(2 states)  |  "
        "Direction: per-regime RF\n\n",
        "## Overall metrics (mean ± std — comparable to Exp 2 v2)\n\n",
        "| Metric | Mean | Std | Exp 2 v2 baseline |\n",
        "|--------|------|-----|-------------------|\n",
    ]

    def _m(key, path="overall"):
        vals = np.array([r[path][key] for r in results
                         if not np.isnan(r[path][key])])
        return vals

    cov     = _m("coverage")
    chr_    = _m("conditional_hit_rate")
    dir_mcc = _m("direction_mcc")
    g_rec   = np.array([r["gate_debug"]["recall"]    for r in results])
    g_prec  = np.array([r["gate_debug"]["precision"] for r in results])

    lines += [
        f"| coverage | {cov.mean():.4f} | {cov.std():.4f} | 0.3651 |\n",
        f"| conditional_hit_rate | {chr_.mean():.4f} | {chr_.std():.4f} | 0.4646 |\n",
        f"| direction_mcc | {dir_mcc.mean():.4f} | {dir_mcc.std():.4f} | 0.0415 |\n",
        f"| gate_recall | {g_rec.mean():.4f} | {g_rec.std():.4f} | 0.4335 |\n",
        f"| gate_precision | {g_prec.mean():.4f} | {g_prec.std():.4f} | 0.6110 |\n\n",
        "## Per-regime metrics (mean ± std across folds)\n\n",
        "| Regime | Interpretation | n_test (mean) | n_gated (mean) | "
        "hit_rate | direction_mcc |\n",
        "|--------|---------------|---------------|----------------|"
        "---------|---------------|\n",
    ]

    for r in [0, 1]:
        lbl = "low-vol / calm" if r == 0 else "high-vol / active"
        n_test   = np.array([res["regime_counts_test"][r] for res in results])
        n_gated  = np.array([res["per_regime"][r]["n_gated"] for res in results])
        hit_arr  = np.array([res["per_regime"][r]["hit_rate"] for res in results
                             if not np.isnan(res["per_regime"][r]["hit_rate"])])
        mcc_arr  = np.array([res["per_regime"][r]["direction_mcc"] for res in results
                             if not np.isnan(res["per_regime"][r]["direction_mcc"])])
        lines.append(
            f"| {r} ({lbl}) | {n_test.mean():.0f} | {n_gated.mean():.0f} | "
            f"{hit_arr.mean():.4f} ± {hit_arr.std():.4f} | "
            f"{mcc_arr.mean():.4f} ± {mcc_arr.std():.4f} |\n"
        )

    lines += [
        "\n## Regime vol centres (mean lag1_vol15 per regime, across folds)\n\n",
        "| Fold | Regime 0 vol15 | Regime 1 vol15 |\n",
        "|------|----------------|----------------|\n",
    ]
    for res in results:
        vc = res["regime_vol_centres"]
        lines.append(
            f"| {res['fold']} | {vc[0]:.6f} | {vc[1]:.6f} |\n"
        )

    lines += [
        "\n## Per-fold detail\n\n",
        "| Fold | threshold | coverage | hit_rate | dir_mcc | "
        "r0_hit | r1_hit | r0_mcc | r1_mcc |\n",
        "|------|-----------|----------|----------|---------|"
        "--------|--------|--------|--------|\n",
    ]
    for res in results:
        ov  = res["overall"]
        pr  = res["per_regime"]
        lines.append(
            f"| {res['fold']} | {res['threshold']:.4f} | "
            f"{ov['coverage']:.4f} | "
            f"{ov['conditional_hit_rate']:.4f} | "
            f"{ov['direction_mcc']:.4f} | "
            f"{pr[0]['hit_rate']:.4f} | "
            f"{pr[1]['hit_rate']:.4f} | "
            f"{pr[0]['direction_mcc']:.4f} | "
            f"{pr[1]['direction_mcc']:.4f} |\n"
        )

    return "".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 3 — Regime-conditional cascade with Gaussian HMM"
    )
    parser.add_argument(
        "--tune-hp", action="store_true",
        help="Enable nested walk-forward HP selection per fold",
    )
    args = parser.parse_args()

    label = "v2 features + HMM regimes"
    if args.tune_hp:
        label += " + walk-forward HP tuning"

    print(f"Running Experiment 3 — Regime-conditional cascade  [{label}]")
    results = run(tune_hp=args.tune_hp)
    report  = summarise(results)
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(report)
    print(f"\nResults written to {_RESULTS_PATH}")
    print(report)
