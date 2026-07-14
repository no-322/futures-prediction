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


def _test_reference(cfg: dict) -> tuple[int, np.ndarray]:
    """(test length, y_true) for the flat-free test split — the leaderboard reference."""
    train_size = cfg["data"].get("train_size", 0.5)
    _, _, _, y_test = _build_dataset(Path(cfg["data"]["path"]), train_size)
    y = y_test.to_numpy()
    return int(len(y)), y


def rank_models(cfg: dict) -> list[tuple[str, str, float, float, float]]:
    """Rank every saved binary prediction set on the single test split, best first.

    Reads each ``data/processed/{stem}_predictions.npz`` (no retraining) and computes
    test accuracy, recall (class 1), and MCC — all on the flat-free test set, since flat
    bars are dropped from the modelling set. Rows are sorted by (accuracy, MCC)
    descending. Sets whose length differs from the current test split are skipped.

    Returns:
        List of (stem, display_name, accuracy, recall, mcc), best first.
    """
    from src import backtest

    n_total, _ = _test_reference(cfg)
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
        rows.append((stem, _leaderboard_name(stem, registry),
                     res["accuracy"], float(recall), res["mcc"]))

    rows.sort(key=lambda r: (r[2], r[4]), reverse=True)
    return rows


def leaderboard(cfg: dict) -> None:
    """Write ``docs/notes/leaderboard.md`` — single test-set model comparison.

    A table from rank_models(): accuracy, recall, MCC on the flat-free 50/50 test set,
    sorted by accuracy then MCC. Length-mismatched sets are skipped and listed.
    """
    n_total, _ = _test_reference(cfg)
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
        "\n| Model | Accuracy | Recall | MCC |\n",
        "|-------|----------|--------|-----|\n",
    ]
    for _stem, name, acc, recall, mcc in rows:
        lines.append(f"| {name} | {acc:.4f} | {recall:.4f} | {mcc:.4f} |\n")

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
def main() -> None:
    parser = argparse.ArgumentParser(description="Compute stats for all binary models")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip suites whose .npz predictions already exist",
    )
    parser.add_argument(
        "--sections", nargs="+",
        choices=["a", "c", "d", "lb"],
        default=["a", "c", "d"],
        help="Which sections to run (a=production stats, c=no-flat 20-feat suite + HMM, "
             "d=49-feature binary suites, lb=leaderboard.md across all saved prediction "
             "sets). The lb section reads existing .npz (no retraining), so run a/c/d "
             "first. Default: a c d.",
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


if __name__ == "__main__":
    main()
