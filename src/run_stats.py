"""Compute standardised statistics for all binary models.

Section A — Production models (baseline, rf, gbm, svm):
  Load existing joblibs, infer on the 50% test set (no retraining), save
  predictions, write docs/notes/all_stats.md.

Section C — No-flat binary suite (20-feat) + HMM-regime binary:
  Train baseline/rf/gbm/svm with flat (Close==Open) rows removed from training,
  plus the HMM-regime binary model; write docs/notes/binary_noflat_stats.md.

Section D — 49-feature binary suites (flat-included + no-flat):
  Train both variants on the cached features_v2 matrix; write
  docs/notes/binary_v2_stats.md.

Section LB — Model leaderboard:
  Rank every saved binary prediction set by no-flat test accuracy then full-test
  MCC; write docs/notes/model_leaderboard.md.

Section TOP5 — Top-5 walk-forward evaluation:
  Reconstruct each top-5 leaderboard model's recipe (feature set, tuned params, no-flat
  training, stored threshold) and run the rolling walk-forward harness, retraining per
  fold; report per-fold accuracy + mean±std on the no-flat test slice; write
  docs/notes/top5_evaluation.md.

Section TOP5OF — Order-flow-augmented top-5 walk-forward:
  Re-run the same top-5 models with the 20 lagged features_orderflow columns
  concatenated onto each model's base feature set; append an "Order-Flow Augmented"
  section (with Δ vs baseline) to docs/notes/top5_evaluation.md.

Sections HMMFEAT / HMMGATE — causal-HMM-regime top-5 walk-forward:
  Per fold, fit a 2-state Gaussian HMM on the train block and assign the regime by
  causal *filtering* (no look-ahead). HMMFEAT appends the filtered posterior P(high-vol)
  as a feature; HMMGATE scores only high-vol ("high-risk") bars and reports coverage.
  Both append a section to docs/notes/top5_evaluation.md.

The 3-class / gated-cascade experiments are not part of this driver (they live on
the experimentation branch).

Run with:
    python -m src.run_stats                       # sections a c d
    python -m src.run_stats --sections a          # production only
    python -m src.run_stats --sections c d --skip-existing
    python -m src.run_stats --sections top5       # evaluate top-5 (reads existing .npz)
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
from src.labels import build_labels, flat_mask
from src.load import load_raw
from src.pipeline import _build_dataset
from src.split import split
from src.statistics import StatsResult

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
_NOFLAT_REPORT_PATH = Path("docs/notes/binary_noflat_stats.md")
_V2_REPORT_PATH = Path("docs/notes/binary_v2_stats.md")

# No-flat-test evaluation slice (test rows with Open == Close excluded). Sibling
# reports mirror the full-test ones above for direct side-by-side comparison.
_NFT_REPORT_PATH = Path("docs/notes/all_stats_noflat_test.md")
_NFT_NOFLAT_REPORT_PATH = Path("docs/notes/binary_noflat_stats_noflat_test.md")
_NFT_V2_REPORT_PATH = Path("docs/notes/binary_v2_stats_noflat_test.md")

_LEADERBOARD_PATH = Path("docs/notes/model_leaderboard.md")
_TOP5_EVAL_PATH = Path("docs/notes/top5_evaluation.md")
_PROC = Path("data/processed")

_REGIME_LABELS = {0: "low-vol / calm", 1: "high-vol / active"}

# 49-feature binary variants: (prefix, drop_flat, display_suffix, section title).
_V2_VARIANTS = [
    ("exp_v2",        False, "v2, flat-incl", "Flat-included (49 features)"),
    ("exp_noflat_v2", True,  "no-flat, v2",   "No-flat (49 features)"),
]


def section_a(cfg: dict) -> list[StatsResult]:
    """Load production joblibs, predict on the test set, compute stats."""
    from src.models import baseline, rf
    from src.models.gbm import load as gbm_load, predict as gbm_predict
    from src.models.svm import load as svm_load, predict as svm_predict

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

    for prefix, drop_flat, suffix, title in _V2_VARIANTS:
        algos = tuple(
            a for a in binary_suite._ALGOS
            if not (skip_existing
                    and Path(f"data/processed/{prefix}_{a}_predictions.npz").exists())
        )
        if algos:
            binary_suite.run(
                config=cfg, algos=algos,
                build_features_fn=load_or_build_features_v2,
                drop_flat=drop_flat, prefix=prefix, display_suffix=suffix,
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


def test_flat_mask(cfg: dict) -> np.ndarray:
    """Reconstruct the test-set "keep" mask that drops flat (Open == Close) bars.

    Rebuilds the contiguous second-50% test slice exactly as the prediction sets
    were generated (df.iloc[4:] → 50/50 split), then returns the boolean keep
    mask (True = keep) aligned 1-to-1 with every binary model's saved predictions.

    Args:
        cfg: Parsed config dict (uses cfg["data"]["path"] and train_size).

    Returns:
        Boolean ndarray, True where Close != Open, same length and order as the
        50/50 test predictions (and as `raw_test`).
    """
    train_size = cfg["data"].get("train_size", 0.5)
    data_path  = Path(cfg["data"]["path"])

    df          = load_raw(data_path)
    raw_align   = df.iloc[4:].reset_index(drop=True)
    _, raw_test = split(raw_align, train_size=train_size)
    return ~flat_mask(raw_test)


def _nft_stats(stem: str, display: str, keep: np.ndarray) -> StatsResult | None:
    """Load an existing prediction set and compute stats on the keep-masked slice.

    Returns None (with a warning) when the .npz is missing or its length does not
    match the test mask (e.g. 3-class / two-stage / TimeSeriesSplit sets).
    """
    npz = Path(f"data/processed/{stem}_predictions.npz")
    if not npz.exists():
        print(f"    Warning: {npz} not found, skipping")
        return None
    d = np.load(npz)
    if len(d["y_pred"]) != len(keep):
        print(f"    Skipping {stem}: length {len(d['y_pred']):,} != test mask "
              f"{len(keep):,} (not a contiguous 50/50 binary set)")
        return None
    res = statistics.compute(d["y_true"][keep], d["y_pred"][keep], name=display)
    print(f"    {display:<42} acc={res['accuracy']:.4f}  "
          f"macro_f1={res['macro_f1']:.4f}  mcc={res['mcc']:.4f}")
    return res


def section_noflat_test(cfg: dict) -> None:
    """No-flat-test slice: recompute stats for all saved prediction sets.

    Reads existing {stem}_predictions.npz only (no retraining), drops test rows
    where Open == Close, and writes three sibling reports mirroring sections
    A/C/D. Run sections a/c/d first so the prediction artifacts exist.
    """
    from src import binary_suite

    keep    = test_flat_mask(cfg)
    n_keep  = int(keep.sum())
    n_total = int(keep.size)
    n_drop  = n_total - n_keep
    slice_note = (
        f"**No-flat test slice:** flat (`Open == Close`) bars removed from "
        f"**evaluation only** — predictions are unchanged (they were generated on "
        f"the whole test set, blind to flatness). Kept {n_keep:,} of {n_total:,} "
        f"test rows ({n_drop:,} flat dropped, {100 * n_drop / n_total:.2f}%). "
        f"Per-label metrics: class 0 = down, class 1 = up.\n\n"
    )
    print(f"\n[Section NFT] No-flat-test slice — keeping {n_keep:,}/{n_total:,} "
          f"test rows ({n_drop:,} flat dropped)")

    # -- Group 1: production models (v1, flat-incl) --------------------------
    print("  Production models:")
    prod = [r for algo, (display, _) in _PROD_MODELS.items()
            if (r := _nft_stats(algo, display, keep)) is not None]
    if prod:
        _NFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        header = ("# Production Models — Stats on the No-Flat Test Slice\n\n"
                  + slice_note)
        body = "\n---\n\n".join(statistics.format_markdown(r) for r in prod)
        _NFT_REPORT_PATH.write_text(header + body + "\n")
        print(f"  → {_NFT_REPORT_PATH}")

    # -- Group 2: v1 no-flat suite + HMM-regime binary -----------------------
    print("  No-flat binary suite (v1) + HMM:")
    results: list[StatsResult] = []
    for algo in binary_suite._ALGOS:
        r = _nft_stats(f"exp_noflat_{algo}", binary_suite._DISPLAY[algo], keep)
        if r is not None:
            results.append(r)

    per_regime_md = ""
    hmm_npz = Path("data/processed/exp_regime_binary_predictions.npz")
    if hmm_npz.exists():
        d = np.load(hmm_npz)
        if len(d["y_pred"]) == len(keep):
            r = statistics.compute(d["y_true"][keep], d["y_pred"][keep],
                                   name="HMM-regime binary")
            results.append(r)
            print(f"    {'HMM-regime binary':<42} acc={r['accuracy']:.4f}  "
                  f"macro_f1={r['macro_f1']:.4f}  mcc={r['mcc']:.4f}")
            regime = d["regime"][keep]
            yt, yp = d["y_true"][keep], d["y_pred"][keep]
            blocks = ["\n---\n\n## HMM-regime binary — per-regime breakdown\n\n"]
            for rg in [0, 1]:
                m = regime == rg
                if m.sum() == 0:
                    continue
                sub = statistics.compute(yt[m], yp[m],
                                         name=f"Regime {rg} ({_REGIME_LABELS[rg]})")
                blocks.append(statistics.format_markdown(sub))
                blocks.append("\n")
            per_regime_md = "".join(blocks)

    if results:
        _NFT_NOFLAT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        header = ("# No-Flat Binary Suite — Stats on the No-Flat Test Slice\n\n"
                  + slice_note)
        body = "\n---\n\n".join(statistics.format_markdown(r) for r in results)
        _NFT_NOFLAT_REPORT_PATH.write_text(header + body + "\n" + per_regime_md)
        print(f"  → {_NFT_NOFLAT_REPORT_PATH}")

    # -- Group 3: 49-feature variants (flat-incl + no-flat) ------------------
    print("  49-feature variants:")
    report_blocks: list[str] = []
    for prefix, _drop, suffix, title in _V2_VARIANTS:
        grp = [r for algo in binary_suite._ALGOS
               if (r := _nft_stats(f"{prefix}_{algo}",
                                   binary_suite._display(algo, suffix), keep))
               is not None]
        if grp:
            block = [f"## {title}\n\n"]
            block.append("\n---\n\n".join(statistics.format_markdown(r) for r in grp))
            report_blocks.append("".join(block))
    if report_blocks:
        _NFT_V2_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        header = ("# 49-Feature Binary Suites — Stats on the No-Flat Test Slice\n\n"
                  + slice_note)
        _NFT_V2_REPORT_PATH.write_text(header + "\n\n---\n\n".join(report_blocks) + "\n")
        print(f"  → {_NFT_V2_REPORT_PATH}")


# Single-split leaderboard-variant prefixes → display suffix. The remainder of the
# stem after the prefix is a base top-5 stem, decoded recursively for its model name.
_SS_VARIANT_LABELS: dict[str, str] = {
    "ss_hmmfeat_": "+ HMM regime feature",
    "ss_hmmgate_": "+ HMM gate (high-vol)",
    "ss_offeat_":  "+ order-flow + regime feature",
    "ss_ofgate_":  "+ order-flow + HMM gate",
}


def _leaderboard_name(stem: str, registry: dict[str, str]) -> str:
    """Human-readable model name for a prediction-set stem."""
    if stem in registry:
        return registry[stem]
    for prefix, label in _SS_VARIANT_LABELS.items():
        if stem.startswith(prefix):
            base = stem[len(prefix):]
            return f"{_leaderboard_name(base, registry)} {label}"
    if stem.startswith("exp_noflat_v1rel_"):
        from src.binary_suite import BASE_DISPLAY
        algo = stem[len("exp_noflat_v1rel_"):]
        return f"{BASE_DISPLAY.get(algo, algo)} (no-flat, v1-rel)"
    if stem.startswith("tuned_"):
        from src.binary_suite import BASE_DISPLAY
        _, feat, algo = stem.split("_", 2)
        return f"{BASE_DISPLAY.get(algo, algo)} (tuned, {feat})"
    return stem


def _score_predset(
    d, keep: np.ndarray
) -> tuple[float, float, float, float | None]:
    """Score one loaded prediction set against the no-flat test mask.

    Standard sets are scored on all non-flat test bars. A set carrying a ``gate``
    boolean (the single-split HMM-gate variants) is scored only on the **traded**
    high-vol population: no-flat accuracy on ``keep & gate``, full accuracy / MCC on
    the gated bars, plus the coverage fraction (traded / non-flat).

    Args:
        d: A loaded ``np.load`` mapping with ``y_true``/``y_pred`` (and optional ``gate``).
        keep: Boolean no-flat test mask aligned to the predictions.

    Returns:
        (no_flat_acc, full_acc, full_mcc, coverage) — coverage is None unless gated.
    """
    yt, yp = d["y_true"], d["y_pred"]
    if "gate" in d:
        gate = np.asarray(d["gate"], dtype=bool)
        nf = keep & gate
        nf_acc = statistics.compute(yt[nf], yp[nf])["accuracy"]
        gfull = statistics.compute(yt[gate], yp[gate])
        coverage = float(nf.sum()) / float(keep.sum())
        return nf_acc, gfull["accuracy"], gfull["mcc"], coverage
    full = statistics.compute(yt, yp)
    nf_acc = statistics.compute(yt[keep], yp[keep])["accuracy"]
    return nf_acc, full["accuracy"], full["mcc"], None


def rank_models(cfg: dict) -> list[tuple[str, str, float, float, float]]:
    """Rank every saved binary prediction set, best first.

    Reads each data/processed/{stem}_predictions.npz (no retraining) and computes
    no-flat test accuracy (Open==Close bars dropped from evaluation only), full-test
    accuracy, and full-test MCC. Rows are sorted by (no-flat accuracy, full-test MCC)
    descending — the same key the leaderboard uses. Non-binary / different-length sets
    (3-class, two-stage, regime_v2, the concatenated walk-forward set) are skipped.

    Args:
        cfg: Parsed config dict (used to rebuild the no-flat test mask).

    Returns:
        List of (stem, display_name, no_flat_acc, full_acc, full_mcc), best first.
    """
    from src import backtest

    keep = test_flat_mask(cfg)
    n_total = int(keep.size)
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
        nf_acc, full_acc, full_mcc, coverage = _score_predset(d, keep)
        name = _leaderboard_name(stem, registry)
        if coverage is not None:
            name += f" (cov {coverage * 100:.0f}%)"
        rows.append((stem, name, nf_acc, full_acc, full_mcc))

    # Sort by no-flat accuracy, then full-test MCC (both descending).
    rows.sort(key=lambda r: (r[2], r[4]), reverse=True)
    return rows


def leaderboard(cfg: dict) -> None:
    """Write docs/notes/model_leaderboard.md comparing every binary model.

    Thin formatter over rank_models(): emits a 4-column table sorted by no-flat
    accuracy then full-test MCC. Non-binary sets (3-class / two-stage / regime_v2,
    which have a different length) are skipped by rank_models and listed here.
    """
    keep = test_flat_mask(cfg)
    n_keep, n_total = int(keep.sum()), int(keep.size)

    rows = rank_models(cfg)
    ranked_stems = {r[0] for r in rows}
    skipped = [
        npz.name[: -len("_predictions.npz")]
        for npz in sorted(Path("data/processed").glob("*_predictions.npz"))
        if not npz.name.startswith("backtest_")
        and npz.name[: -len("_predictions.npz")] not in ranked_stems
    ]

    lines = [
        "# Model Leaderboard\n\n",
        "Every binary up/down model on the 50/50 time-ordered test set, compared "
        "in one place. **Sorted by no-flat test accuracy, then full-test MCC.**\n\n",
        f"- *No-flat test accuracy*: accuracy on the {n_keep:,} of {n_total:,} test "
        "bars where `Close != Open` (flat bars dropped from evaluation only).\n",
        "- *Accuracy* and *MCC*: computed on the full test set.\n",
        "- `+ HMM gate` rows are scored **only on high-vol bars** (the rest are not "
        "traded); their `(cov NN%)` is the fraction of non-flat bars traded, and all "
        "three metrics are over that gated subset.\n",
    ]
    if skipped:
        lines.append("- Excluded (non-binary / different length): "
                     + ", ".join(f"`{s}`" for s in sorted(skipped)) + ".\n")
    lines += [
        "\n| Model | No-flat test acc | Accuracy | MCC |\n",
        "|-------|------------------|----------|-----|\n",
    ]
    for _stem, name, nf_acc, acc, mcc in rows:
        lines.append(f"| {name} | {nf_acc:.4f} | {acc:.4f} | {mcc:.4f} |\n")

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


def _top5_recipe(stem: str, cfg: dict) -> dict:
    """Decode a leaderboard stem into a walk-forward training recipe.

    Maps a saved prediction-set stem to everything needed to retrain it per fold:
    algorithm, feature set, hyperparameters, and (for tuned models) the stored
    decision threshold. ``exp_noflat_baseline`` uses config defaults and no threshold;
    ``tuned_{featset}_{algo}`` reads ``data/processed/tuned_params_{featset}.json``.

    Args:
        stem: Prediction-set stem from the leaderboard (e.g. ``tuned_v3_gbm``).
        cfg: Parsed config dict (for baseline default hyperparameters).

    Returns:
        Dict with keys: ``algo``, ``featset``, ``params`` (dict), ``threshold``
        (float or None).

    Raises:
        ValueError: If the stem is not a recognised top-model recipe.
    """
    if stem == "exp_noflat_baseline":
        return {"algo": "baseline", "featset": "v1",
                "params": model_params(cfg, "baseline"), "threshold": None}
    if stem.startswith("tuned_"):
        _, featset, algo = stem.split("_", 2)
        spec = json.loads(
            Path(f"data/processed/tuned_params_{featset}.json").read_text()
        )
        entry = spec["models"][algo]
        thr = entry.get("threshold") if spec.get("tune_threshold") else None
        return {"algo": algo, "featset": featset,
                "params": dict(entry["params"]), "threshold": thr}
    raise ValueError(f"No walk-forward recipe for stem {stem!r}")


def _is_top5_recipe_stem(stem: str) -> bool:
    """True if a leaderboard stem is a base walk-forward recipe (not a derived set).

    The top-5 walk-forward / variant generators operate on the base model recipes
    (`exp_noflat_baseline`, `tuned_{featset}_{algo}`). Derived leaderboard rows — the
    single-split order-flow/HMM variants (`ss_*`), v2/v3 binary suites, HMM-regime
    binary, walk-forward sets — are not walk-forward recipes and must be skipped so
    they cannot recurse into the selection (e.g. an `ss_hmmgate_*` ranking top-2).
    """
    return stem == "exp_noflat_baseline" or stem.startswith("tuned_")


def _top5_ranked(cfg: dict, k: int) -> list[tuple[str, str, float, float, float]]:
    """Top-k leaderboard rows restricted to base walk-forward recipes (best first)."""
    return [r for r in rank_models(cfg) if _is_top5_recipe_stem(r[0])][:k]


def walkforward_top5(cfg: dict, k: int = 5, path: Path = _TOP5_EVAL_PATH) -> None:
    """Rolling walk-forward evaluation of the top-k leaderboard models → markdown.

    Reconstructs each top model's recipe (feature set, tuned hyperparameters, no-flat
    training, stored decision threshold) and retrains it fresh per walk-forward fold
    (rolling 3mo-train / 1mo-test from config), reporting per-fold accuracy + mean ± std
    on the **no-flat test slice** (the default). Writes ``docs/notes/top5_evaluation.md``
    and persists each model's per-fold predictions (Rule 7).

    Args:
        cfg: Parsed config dict.
        k: Number of top models to evaluate (default 5).
        path: Destination markdown file.
    """
    import src.walkforward as walkforward
    from src.models import baseline as m_baseline
    from src.models import gbm as m_gbm
    from src.models import rf as m_rf

    modules = {"baseline": m_baseline, "rf": m_rf, "gbm": m_gbm}

    df = load_raw(Path(cfg["data"]["path"]))
    raw_align = df.iloc[4:].reset_index(drop=True)
    y = build_labels(raw_align)
    timestamps = raw_align["Date and Time"].reset_index(drop=True)
    keep = ~flat_mask(raw_align)                       # full-series non-flat mask
    n_keep, n_total = int(keep.sum()), int(keep.size)

    ranked = _top5_ranked(cfg, k)
    feat_cache: dict[str, "pd.DataFrame"] = {}
    tmp = _PROC / "_walkforward_tmp" / "model.joblib"

    sections: list[str] = []
    for rank, (stem, name, nf_acc, _acc, _mcc) in enumerate(ranked, start=1):
        recipe = _top5_recipe(stem, cfg)
        featset = recipe["featset"]
        X = feat_cache.get(featset)
        if X is None:
            X = _featset_builder(featset)(df)
            feat_cache[featset] = X

        thr = recipe["threshold"]
        predict_fn = None
        if thr is not None:
            predict_fn = (lambda thr: (
                lambda model, Xte: (model.predict_proba(Xte)[:, 1] >= thr).astype(int)
            ))(thr)

        factory = walkforward.module_factory(
            modules[recipe["algo"]], recipe["params"], tmp
        )
        print(f"  #{rank} {name} — walk-forward (featset={featset}, "
              f"thr={thr if thr is not None else 0.5})")
        res = walkforward.walk_forward(
            X, y, timestamps, factory,
            config=cfg, keep=keep, drop_flat_train=True, predict_fn=predict_fn,
            name=f"top5_{stem}",
        )
        headline = (
            f"## #{rank} {name}\n\n"
            f"Walk-forward no-flat accuracy: **{res['mean_accuracy'] * 100:.1f}% "
            f"± {res['std_accuracy'] * 100:.1f}%** across {res['n_folds']} folds "
            f"(range {res['min_accuracy'] * 100:.1f}–{res['max_accuracy'] * 100:.1f}%). "
            f"Recipe: {recipe['algo']} on {featset}, no-flat training, "
            f"threshold {thr if thr is not None else 0.5:.4f}. "
            f"Single-split leaderboard no-flat acc: {nf_acc:.4f}.\n\n"
        )
        # summarize() emits its own "## Walk-forward — name" heading + table; keep the
        # per-fold table only (drop its heading) under our richer headline.
        table = walkforward.summarize(res).split("\n", 2)[2].split("\n\n", 1)[1]
        sections.append(headline + table)

    intro = (
        "# Top-5 Models — Walk-Forward Evaluation\n\n"
        f"The top {len(sections)} models from the **Model Leaderboard** "
        "(`docs/notes/model_leaderboard.md`), each retrained fresh per **rolling "
        "walk-forward** fold (3-month train / 1-month test, stepped 1 month, from "
        "`config.yaml`). Each model keeps its recipe: feature set, tuned "
        "hyperparameters, no-flat training, and stored decision threshold. Per-fold "
        "accuracy is on the **no-flat test slice** (flat `Close == Open` bars excluded "
        f"from scoring; {n_keep:,} of {n_total:,} bars are non-flat overall). The "
        "between-fold spread is the regime signal; compare the walk-forward mean to "
        "each model's single-split leaderboard no-flat accuracy.\n\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(intro + "\n---\n\n".join(sections) + "\n")
    print(f"Walk-forward top-{len(sections)} → {path}")


def _baseline_top5_mean(stem: str) -> float | None:
    """Mean walk-forward accuracy of the un-augmented top-5 run for ``stem``, if saved.

    Reads ``data/processed/walkforward_top5_{stem}_predictions.npz`` (written by
    ``walkforward_top5``) and returns the mean of its per-fold ``accuracies``. Returns
    None when the baseline run has not been persisted, so the delta line is omitted.
    """
    npz = _PROC / f"walkforward_top5_{stem}_predictions.npz"
    if not npz.exists():
        return None
    d = np.load(npz)
    if "accuracies" not in d:
        return None
    return float(np.mean(d["accuracies"]))


def walkforward_top5_orderflow(cfg: dict, k: int = 5,
                               path: Path = _TOP5_EVAL_PATH) -> None:
    """Walk-forward the top-k models with order-flow features concatenated on.

    Mirrors ``walkforward_top5`` but augments each model's base feature set (v1/v2/v3)
    with the 20 lagged order-flow columns from ``features_orderflow`` before running the
    rolling walk-forward harness. Each model keeps its full recipe (tuned params, no-flat
    training, stored threshold). Per-fold predictions are persisted as
    ``walkforward_top5of_{stem}`` (Rule 7), and an ``# Order-Flow Augmented`` section is
    **appended** to ``docs/notes/top5_evaluation.md`` (the baseline tables are preserved;
    any prior order-flow section is replaced so re-runs do not duplicate).

    Args:
        cfg: Parsed config dict.
        k: Number of top models to evaluate (default 5).
        path: Destination markdown file (appended to in place).
    """
    import src.walkforward as walkforward
    from src.features_orderflow import load_or_build_features_orderflow
    from src.models import baseline as m_baseline
    from src.models import gbm as m_gbm
    from src.models import rf as m_rf

    modules = {"baseline": m_baseline, "rf": m_rf, "gbm": m_gbm}

    df = load_raw(Path(cfg["data"]["path"]))
    raw_align = df.iloc[4:].reset_index(drop=True)
    y = build_labels(raw_align)
    timestamps = raw_align["Date and Time"].reset_index(drop=True)
    keep = ~flat_mask(raw_align)                       # full-series non-flat mask
    n_keep, n_total = int(keep.sum()), int(keep.size)

    # Two order-flow variants: "raw" (signed_vol = Volume·sign) for tree models, which
    # benefited from it; "linear" (signed *relative* volume, O(1) scale) for logistic
    # models, whose unscaled fit was dominated by raw volume magnitudes.
    X_of_raw = load_or_build_features_orderflow(df, "raw").reset_index(drop=True)
    X_of_lin = load_or_build_features_orderflow(df, "linear").reset_index(drop=True)

    ranked = _top5_ranked(cfg, k)
    feat_cache: dict[str, "pd.DataFrame"] = {}
    tmp = _PROC / "_walkforward_tmp" / "model.joblib"

    sections: list[str] = []
    for rank, (stem, name, nf_acc, _acc, _mcc) in enumerate(ranked, start=1):
        recipe = _top5_recipe(stem, cfg)
        featset = recipe["featset"]
        X_base = feat_cache.get(featset)
        if X_base is None:
            X_base = _featset_builder(featset)(df).reset_index(drop=True)
            feat_cache[featset] = X_base
        # Logistic models (algo == "baseline") get the scale-stable linear variant.
        is_linear = recipe["algo"] == "baseline"
        of_variant = "linear" if is_linear else "raw"
        X_of = X_of_lin if is_linear else X_of_raw
        # Both matrices are aligned to df.iloc[4:]; concat positionally.
        X_aug = pd.concat([X_base, X_of], axis=1)

        thr = recipe["threshold"]
        predict_fn = None
        if thr is not None:
            predict_fn = (lambda thr: (
                lambda model, Xte: (model.predict_proba(Xte)[:, 1] >= thr).astype(int)
            ))(thr)

        factory = walkforward.module_factory(
            modules[recipe["algo"]], recipe["params"], tmp
        )
        print(f"  #{rank} {name} +orderflow[{of_variant}] — walk-forward "
              f"(featset={featset}+OF, {X_aug.shape[1]} cols, "
              f"thr={thr if thr is not None else 0.5})")
        res = walkforward.walk_forward(
            X_aug, y, timestamps, factory,
            config=cfg, keep=keep, drop_flat_train=True, predict_fn=predict_fn,
            name=f"top5of_{stem}",
        )

        base_mean = _baseline_top5_mean(stem)
        if base_mean is not None:
            delta = (res["mean_accuracy"] - base_mean) * 100.0
            delta_line = (
                f" Baseline (no order-flow) mean: {base_mean * 100:.1f}%; "
                f"**Δ {delta:+.1f} pp**."
            )
        else:
            delta_line = ""
        of_label = ("order-flow [linear-scaled: signed relative volume]"
                    if is_linear else "order-flow [raw]")
        headline = (
            f"## #{rank} {name} + {of_label}\n\n"
            f"Walk-forward no-flat accuracy: **{res['mean_accuracy'] * 100:.1f}% "
            f"± {res['std_accuracy'] * 100:.1f}%** across {res['n_folds']} folds "
            f"(range {res['min_accuracy'] * 100:.1f}–{res['max_accuracy'] * 100:.1f}%). "
            f"Recipe: {recipe['algo']} on {featset}+orderflow[{of_variant}] "
            f"({X_aug.shape[1]} cols), no-flat training, "
            f"threshold {thr if thr is not None else 0.5:.4f}.{delta_line}\n\n"
        )
        table = walkforward.summarize(res).split("\n", 2)[2].split("\n\n", 1)[1]
        sections.append(headline + table)

    section = (
        "# Order-Flow Augmented (base feature set + features_orderflow)\n\n"
        f"The same top-{len(sections)} models as above, each re-run with the 20 lagged "
        "**order-flow** columns from `src/features_orderflow.py` (`norm_vol`, "
        "`signed_vol`, `cum_td5/10/15`, lagged t-1…t-4) concatenated onto its base "
        "feature set. **Logistic-regression models use the scale-stable `linear` "
        "variant** (`signed_vol = sign(ΔO,C)·Volume / trailing-mean Volume`, an O(1) "
        "signed relative volume) because the raw volume magnitude dominates an unscaled "
        "linear fit; **tree models keep the `raw` variant** (`signed_vol = "
        "Volume·sign(ΔO,C)`), which helped them. Identical rolling walk-forward (3mo "
        "train / 1mo test) and recipe; per-fold accuracy on the no-flat test slice "
        f"({n_keep:,} of {n_total:,} bars non-flat). The Δ compares each model's "
        "order-flow-augmented walk-forward mean to its no-order-flow baseline mean from "
        "the section above.\n\n"
        + "\n---\n\n".join(sections) + "\n"
    )

    _ORDERFLOW_MARKER = "# Order-Flow Augmented"
    existing = path.read_text() if path.exists() else ""
    idx = existing.find(_ORDERFLOW_MARKER)
    if idx != -1:
        existing = existing[:idx].rstrip() + "\n"   # strip prior order-flow section
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(existing.rstrip() + "\n\n---\n\n" + section)
    print(f"Walk-forward top-{len(sections)} +orderflow appended → {path}")


def _append_top5_section(path: Path, marker: str, section: str) -> None:
    """Append a section to top5_evaluation.md, replacing any prior copy (idempotent)."""
    existing = path.read_text() if path.exists() else ""
    idx = existing.find(marker)
    if idx != -1:
        existing = existing[:idx].rstrip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(existing.rstrip() + "\n\n---\n\n" + section)


def _hmm_fold_transform(
    X_base: "pd.DataFrame",
    X_regime_full: np.ndarray,
    vol15_idx: int,
    mode: str,
):
    """Build a per-fold ``walk_forward`` transform that fits a fresh HMM on the train
    block and assigns the causal (filtered) regime.

    The HMM + scaler are fit on the fold's train regime descriptors only; the canonical
    high-vol state is identified from the TRAIN filtered states; then the filtered
    posterior is computed causally over ``concat(train, test)`` (time-ordered).

    mode == "feature": append ``regime_hi_prob`` = P(high-vol | data<=t-1) to X.
    mode == "gate":    return a test gate selecting high-vol bars (argmax regime == hi).
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


def walkforward_top5_hmm(cfg: dict, mode: str, k: int = 5,
                         path: Path = _TOP5_EVAL_PATH) -> None:
    """Walk-forward the top-k models with a causal HMM regime, as feature or gate.

    Per fold a fresh 2-state Gaussian HMM is fit on the train block's regime descriptors
    (``REGIME_COLS`` from features_v2) and the regime is assigned by **filtering** (no
    look-ahead). ``mode="feature"`` appends the filtered posterior P(high-vol) as a
    column to each model's base feature set; ``mode="gate"`` restricts scoring to
    high-vol ("high-risk") bars. Predictions persisted per Rule 7; a section is appended
    to ``top5_evaluation.md``.

    Args:
        cfg: Parsed config dict.
        mode: ``"feature"`` or ``"gate"``.
        k: Number of top models (default 5).
        path: Destination markdown file (appended in place).
    """
    if mode not in ("feature", "gate"):
        raise ValueError(f"mode must be 'feature' or 'gate'; got {mode!r}")
    import src.walkforward as walkforward
    from src.models import baseline as m_baseline
    from src.models import gbm as m_gbm
    from src.models import rf as m_rf
    from src.models.regime_hmm import REGIME_COLS

    modules = {"baseline": m_baseline, "rf": m_rf, "gbm": m_gbm}

    df = load_raw(Path(cfg["data"]["path"]))
    raw_align = df.iloc[4:].reset_index(drop=True)
    y = build_labels(raw_align)
    timestamps = raw_align["Date and Time"].reset_index(drop=True)
    keep = ~flat_mask(raw_align)
    n_keep, n_total = int(keep.sum()), int(keep.size)

    # Regime descriptors (from features_v2), aligned to df.iloc[4:].
    features_v2 = load_or_build_features_v2(df)
    X_regime_full = features_v2[REGIME_COLS].to_numpy()
    vol15_idx = REGIME_COLS.index("lag1_vol15")

    tag = "feat" if mode == "feature" else "gate"
    ranked = _top5_ranked(cfg, k)
    feat_cache: dict[str, "pd.DataFrame"] = {}
    tmp = _PROC / "_walkforward_tmp" / "model.joblib"

    sections: list[str] = []
    for rank, (stem, name, nf_acc, _acc, _mcc) in enumerate(ranked, start=1):
        recipe = _top5_recipe(stem, cfg)
        featset = recipe["featset"]
        X_base = feat_cache.get(featset)
        if X_base is None:
            X_base = _featset_builder(featset)(df).reset_index(drop=True)
            feat_cache[featset] = X_base

        thr = recipe["threshold"]
        predict_fn = None
        if thr is not None:
            predict_fn = (lambda thr: (
                lambda model, Xte: (model.predict_proba(Xte)[:, 1] >= thr).astype(int)
            ))(thr)

        factory = walkforward.module_factory(
            modules[recipe["algo"]], recipe["params"], tmp
        )
        ft = _hmm_fold_transform(X_base, X_regime_full, vol15_idx, mode)
        print(f"  #{rank} {name} +hmm[{mode}] — walk-forward "
              f"(featset={featset}, thr={thr if thr is not None else 0.5})")
        res = walkforward.walk_forward(
            X_base, y, timestamps, factory,
            config=cfg, keep=keep, drop_flat_train=True, predict_fn=predict_fn,
            fold_transform=ft, name=f"top5hmm{tag}_{stem}",
        )

        base_mean = _baseline_top5_mean(stem)
        if mode == "feature":
            if base_mean is not None:
                delta = (res["mean_accuracy"] - base_mean) * 100.0
                ctx = (f" Baseline (no regime) mean: {base_mean * 100:.1f}%; "
                       f"**Δ {delta:+.1f} pp**.")
            else:
                ctx = ""
            headline = (
                f"## #{rank} {name} + regime feature\n\n"
                f"Walk-forward no-flat accuracy: **{res['mean_accuracy'] * 100:.1f}% "
                f"± {res['std_accuracy'] * 100:.1f}%** across {res['n_folds']} folds "
                f"(range {res['min_accuracy'] * 100:.1f}–{res['max_accuracy'] * 100:.1f}%). "
                f"Recipe: {recipe['algo']} on {featset}+regime_hi_prob, no-flat training, "
                f"threshold {thr if thr is not None else 0.5:.4f}.{ctx}\n\n"
            )
        else:
            cov = float(np.nanmean([f["coverage"] for f in res["per_fold"]])) * 100.0
            ctx = (f" Mean coverage (high-vol bars traded): **{cov:.1f}%** of non-flat "
                   "bars.")
            if base_mean is not None:
                ctx += (f" Baseline mean on all non-flat bars: {base_mean * 100:.1f}% "
                        "(different population — context only).")
            headline = (
                f"## #{rank} {name} — high-vol gate\n\n"
                f"Walk-forward no-flat accuracy on high-vol bars: "
                f"**{res['mean_accuracy'] * 100:.1f}% ± {res['std_accuracy'] * 100:.1f}%** "
                f"across {res['n_folds']} folds "
                f"(range {res['min_accuracy'] * 100:.1f}–{res['max_accuracy'] * 100:.1f}%). "
                f"Recipe: {recipe['algo']} on {featset}, no-flat training, "
                f"threshold {thr if thr is not None else 0.5:.4f}.{ctx}\n\n"
            )
        table = walkforward.summarize(res).split("\n", 2)[2].split("\n\n", 1)[1]
        sections.append(headline + table)

    if mode == "feature":
        marker = "# HMM Regime Feature"
        intro = (
            f"{marker} (causal filtered posterior P(high-vol))\n\n"
            f"The same top-{len(sections)} models as above, each re-run with a single "
            "extra column **`regime_hi_prob`** = the causal forward-**filtered** "
            "posterior P(high-vol regime | data ≤ t-1) from a per-fold 2-state Gaussian "
            "HMM (fit on the train block's `REGIME_COLS` descriptors). The regime is "
            "assigned by **filtering, not Viterbi smoothing**, so no future bar informs "
            "the regime at t. Identical rolling walk-forward (3mo train / 1mo test) and "
            "recipe; per-fold accuracy on the no-flat test slice "
            f"({n_keep:,} of {n_total:,} bars non-flat). The Δ compares each model's "
            "regime-augmented walk-forward mean to its no-regime baseline mean.\n\n"
        )
    else:
        marker = "# HMM Regime Gate"
        intro = (
            f"{marker} (trade high-vol / high-risk bars only)\n\n"
            f"The same top-{len(sections)} models as above, but per fold a 2-state "
            "Gaussian HMM (fit on the train block, regime assigned by causal "
            "**filtering**) gates scoring to **high-vol ('high-risk') bars only** — the "
            "model still trains on all non-flat train bars, but accuracy is measured "
            "only where the filtered regime is high-vol. Reported with **coverage** "
            "(fraction of non-flat bars actually traded): a conditional, high-confidence "
            "metric. Accuracy here is on a different (smaller) population than the "
            "baseline, so the baseline mean is context only, not a clean Δ.\n\n"
        )
    section = intro + "\n---\n\n".join(sections) + "\n"
    _append_top5_section(path, marker, section)
    print(f"Walk-forward top-{len(sections)} +hmm[{mode}] appended → {path}")


def build_leaderboard_variants(cfg: dict, k: int = 5) -> None:
    """Single-split (50/50) leaderboard variants of the top-k models, then re-rank.

    For each current top-k model, train fresh single-split prediction sets that add the
    order-flow / causal-HMM-regime enhancements, and append them to
    ``model_leaderboard.md`` via ``leaderboard``:

      * Linear (logistic, ``algo=="baseline"``): two HMM variants — regime posterior as a
        feature (``ss_hmmfeat_*``) and a high-vol gate (``ss_hmmgate_*``). No order-flow.
      * Non-linear (rf/gbm): order-flow + regime feature (``ss_offeat_*``) and
        order-flow + HMM gate (``ss_ofgate_*``).

    Leakage controls: the HMM + scaler are fit on the **train split only**; the regime is
    assigned by causal filtering (``REGIME_COLS`` are lag-1, so regime at t uses data
    ≤ t-1); flat rows are dropped from training only; the 50/50 split matches
    ``test_flat_mask`` so predictions align 1-to-1 with the leaderboard mask. Each set is
    persisted full-test-length (Rule 7); gate sets also store the high-vol ``gate`` mask.

    Args:
        cfg: Parsed config dict.
        k: Number of top models to enhance (default 5).
    """
    from src.features_orderflow import load_or_build_features_orderflow
    from src.models.regime_hmm import (
        REGIME_COLS,
        canonical_regime_labels,
        filter_regime,
        filter_regime_posterior,
        fit_regime,
    )
    from src.split import split
    from src.tuning import _fit, predict_with_threshold

    train_size = cfg["data"].get("train_size", 0.5)
    df = load_raw(Path(cfg["data"]["path"]))
    raw_align = df.iloc[4:].reset_index(drop=True)
    raw_train, raw_test = split(raw_align, train_size=train_size)
    mid = len(raw_train)
    y_train = build_labels(raw_train)                       # index 0..mid-1
    y_test = build_labels(raw_test).to_numpy()
    move_test = (raw_test["Close"] - raw_test["Open"]).to_numpy()
    keep_train = ~flat_mask(raw_train)
    keep_test = ~flat_mask(raw_test)

    # Regime descriptors (features_v2), aligned to df.iloc[4:].
    features_v2 = load_or_build_features_v2(df)
    X_regime_full = features_v2[REGIME_COLS].to_numpy()
    vol15_idx = REGIME_COLS.index("lag1_vol15")

    feat_cache: dict[str, "pd.DataFrame"] = {}
    X_of_raw: "pd.DataFrame | None" = None
    tmp = _PROC / "_lbvar_tmp" / "model.joblib"
    tmp.parent.mkdir(parents=True, exist_ok=True)

    def _fit_predict_save(algo, params, thr, X_tr_full, X_te, out_stem, gate=None):
        X_tr_nf = X_tr_full[keep_train].reset_index(drop=True)
        y_tr_nf = y_train[keep_train].reset_index(drop=True)
        model = _fit(algo, X_tr_nf, y_tr_nf, params, tmp)
        y_pred = predict_with_threshold(algo, model, X_te, thr)
        kw = dict(y_true=y_test, y_pred=y_pred, move=move_test, keep=keep_test)
        if gate is not None:
            kw["gate"] = gate
        np.savez(_PROC / f"{out_stem}_predictions.npz", **kw)
        print(f"    saved {out_stem} (n_test={len(y_pred):,}"
              + (f", gate cov={float((keep_test & gate).sum()) / keep_test.sum() * 100:.0f}%"
                 if gate is not None else "") + ")")

    for stem, name, *_ in _top5_ranked(cfg, k):
        recipe = _top5_recipe(stem, cfg)
        algo, featset, thr = recipe["algo"], recipe["featset"], recipe["threshold"]
        is_linear = algo == "baseline"

        X_base = feat_cache.get(featset)
        if X_base is None:
            X_base = _featset_builder(featset)(df).reset_index(drop=True)
            feat_cache[featset] = X_base
        if is_linear:
            X_full = X_base
        else:
            if X_of_raw is None:
                X_of_raw = load_or_build_features_orderflow(df, "raw").reset_index(drop=True)
            X_full = pd.concat([X_base, X_of_raw], axis=1)

        # Regime (fit on train split only; causal filtered posterior over full series).
        hmm, scaler = fit_regime(X_regime_full[:mid])
        train_states = filter_regime(hmm, scaler, X_regime_full[:mid])
        remap = canonical_regime_labels(train_states, X_regime_full[:mid], vol15_idx)
        hi_raw = next(raw for raw, canon in remap.items() if canon == 1)
        hi_prob = filter_regime_posterior(hmm, scaler, X_regime_full)[:, hi_raw]
        hi_tr, hi_te = hi_prob[:mid], hi_prob[mid:]
        gate_te = hi_te >= 0.5

        X_tr_full, X_te = split(X_full, train_size=train_size)
        feat_prefix = "ss_hmmfeat_" if is_linear else "ss_offeat_"
        gate_prefix = "ss_hmmgate_" if is_linear else "ss_ofgate_"
        print(f"  {name}: building {feat_prefix}{stem} + {gate_prefix}{stem}")

        # Feature variant: append the regime posterior column.
        X_tr_f = X_tr_full.copy(); X_tr_f["regime_hi_prob"] = hi_tr
        X_te_f = X_te.copy();      X_te_f["regime_hi_prob"] = hi_te
        _fit_predict_save(algo, recipe["params"], thr, X_tr_f, X_te_f,
                          f"{feat_prefix}{stem}")
        # Gate variant: base features, scoring gated to high-vol bars.
        _fit_predict_save(algo, recipe["params"], thr, X_tr_full, X_te,
                          f"{gate_prefix}{stem}", gate=gate_te)

    leaderboard(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute stats for all binary models")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip suites whose .npz predictions already exist",
    )
    parser.add_argument(
        "--sections", nargs="+",
        choices=["a", "c", "d", "nft", "lb", "top5", "top5of", "hmmfeat", "hmmgate",
                 "lbvar"],
        default=["a", "c", "d"],
        help="Which sections to run (a=production, c=no-flat 20-feat + HMM, "
             "d=49-feature binary suites, nft=no-flat-test slice stats, "
             "lb=model_leaderboard.md across all saved prediction sets, "
             "top5=walk-forward evaluation of the top-5 leaderboard models (retrains "
             "per fold), top5of=same top-5 re-run with order-flow features appended, "
             "hmmfeat=top-5 with a causal HMM regime posterior added as a feature, "
             "hmmgate=top-5 scored only on high-vol (gated) bars, "
             "lbvar=single-split order-flow/HMM variants of the top-5 added to "
             "model_leaderboard.md "
             "(run top5 first so the Δ-vs-baseline is available). The lb/nft sections "
             "read existing .npz (no retraining), so run a/c/d first. Default: a c d.",
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
    if "nft" in args.sections:
        section_noflat_test(cfg)
    if "lb" in args.sections:
        leaderboard(cfg)
    if "top5" in args.sections:
        walkforward_top5(cfg)
    if "top5of" in args.sections:
        walkforward_top5_orderflow(cfg)
    if "hmmfeat" in args.sections:
        walkforward_top5_hmm(cfg, "feature")
    if "hmmgate" in args.sections:
        walkforward_top5_hmm(cfg, "gate")
    if "lbvar" in args.sections:
        build_leaderboard_variants(cfg)


if __name__ == "__main__":
    main()
