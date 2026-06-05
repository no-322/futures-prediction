"""Compute standardised statistics for all models and experiments.

Section A — Production models (baseline, rf, gbm, svm):
  Load existing joblibs, infer on the 50% test set. No retraining. ~20 min.

Section B — Experiment pipelines (5 variants):
  Re-run without HP tuning, using 200 trees (fast approximation).
  features_v2 is built ONCE and cached to data/processed/features_v2.parquet
  — subsequent v2 experiments load from cache, saving ~12 min per experiment.
  Saves .npz predictions and last-fold model joblibs.

  Pass --skip-existing to skip any experiment whose .npz already exists.

Run with:
    python -m src.experiments.run_all_stats
    python -m src.experiments.run_all_stats --skip-existing
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

import src.statistics as statistics
from src.config import load_config
from src.experiments.features_v2 import build_features_v2
from src.pipeline import _build_dataset, _get_predictions
from src.statistics import StatsResult

_FEATURES_V2_CACHE = Path("data/processed/features_v2.parquet")

_PROD_MODELS = {
    "baseline": ("Logistic Regression",
                 Path("data/processed/baseline_model.joblib")),
    "rf":       ("Random Forest",
                 Path("data/processed/rf_model.joblib")),
    "gbm":      ("Gradient Boosting (XGBoost)",
                 Path("data/processed/gbm_model.joblib")),
    "svm":      ("SVM (RBF kernel)",
                 Path("data/processed/svm_model.joblib")),
}

_REPORT_PATH = Path("docs/notes/all_stats.md")


def _fast_config(cfg: dict, n_estimators: int = 200) -> dict:
    """Return a config copy with n_estimators overridden for speed."""
    fast = copy.deepcopy(cfg)
    fast["models"]["rf"]["n_estimators"] = n_estimators
    return fast


def section_a(cfg: dict) -> list[StatsResult]:
    """Load production joblibs, predict on test set, compute stats."""
    from src.models import baseline, rf
    from src.models.gbm import load as gbm_load, predict as gbm_predict
    from src.models.svm import load as svm_load, predict as svm_predict
    import joblib

    train_size = cfg["data"].get("train_size", 0.5)
    data_path  = Path(cfg["data"]["path"])

    print("\n[Section A] Production models — loading joblibs + inference")
    _, X_test, _, y_test = _build_dataset(data_path, train_size)
    y_true = y_test.to_numpy()

    loaders = {
        "baseline": (baseline.load,   baseline.predict),
        "rf":       (rf.load,         rf.predict),
        "gbm":      (gbm_load,        gbm_predict),
        "svm":      (svm_load,        svm_predict),
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
              f"macro_f1={result['macro_f1']:.4f}  "
              f"mcc={result['mcc']:.4f}")

    return results


def _load_or_build_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Load features_v2 from parquet cache if it exists; build and cache otherwise."""
    if _FEATURES_V2_CACHE.exists():
        print(f"  Loading cached features_v2 from {_FEATURES_V2_CACHE}...")
        return pd.read_parquet(_FEATURES_V2_CACHE)
    print("  Building features_v2 (will cache for future runs)...")
    features = build_features_v2(df)
    _FEATURES_V2_CACHE.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(_FEATURES_V2_CACHE)
    print(f"  Cached to {_FEATURES_V2_CACHE}")
    return features


def section_b(cfg: dict, skip_existing: bool = False) -> list[StatsResult]:
    """Re-run all experiment pipelines (no HP tuning, 200 trees) and compute stats.

    Args:
        cfg: Parsed config dict.
        skip_existing: If True, skip experiments whose .npz predictions already exist.
    """
    from src.experiments import three_class, two_stage, regime_two_stage

    fast_cfg = _fast_config(cfg, n_estimators=200)

    runs = [
        ("Three-class v1 (20 features)",  "three_class_v1",
         lambda: three_class.run(config=fast_cfg)),
        ("Three-class v2 (49 features)",  "three_class_v2",
         lambda: three_class.run(config=fast_cfg,
                                  build_features_fn=_load_or_build_features_v2)),
        ("Two-stage v1 (20 features)",    "two_stage_v1",
         lambda: two_stage.run(config=fast_cfg)),
        ("Two-stage v2 (49 features)",    "two_stage_v2",
         lambda: two_stage.run(config=fast_cfg,
                               build_features_fn=_load_or_build_features_v2)),
        ("Regime cascade v2 (HMM)",       "regime_v2",
         lambda: regime_two_stage.run(config=fast_cfg)),
    ]

    print("\n[Section B] Experiments — re-running without HP tuning (200 trees)")
    results: list[StatsResult] = []

    for display_name, npz_tag, run_fn in runs:
        npz_path = Path(f"data/processed/exp_{npz_tag}_predictions.npz")

        if skip_existing and npz_path.exists():
            print(f"\n  [{display_name}] Skipping — predictions already exist")
        else:
            print(f"\n  [{display_name}]")
            run_fn()  # saves exp_{npz_tag}_predictions.npz

        if not npz_path.exists():
            print(f"    Warning: {npz_path} not found, skipping stats")
            continue

        d = np.load(npz_path)
        result = statistics.compute(d["y_true"], d["y_pred"], name=display_name)
        results.append(result)
        print(f"    acc={result['accuracy']:.4f}  "
              f"macro_f1={result['macro_f1']:.4f}  "
              f"mcc={result['mcc']:.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute stats for all models")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip experiments whose .npz predictions already exist",
    )
    args = parser.parse_args()

    cfg = load_config()
    all_results: list[StatsResult] = []

    all_results.extend(section_a(cfg))
    all_results.extend(section_b(cfg, skip_existing=args.skip_existing))

    statistics.write_results(all_results, _REPORT_PATH)
    print(f"\nAll statistics written to {_REPORT_PATH}")


if __name__ == "__main__":
    main()
