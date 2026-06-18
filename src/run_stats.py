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

The 3-class / gated-cascade experiments are not part of this driver (they live on
the experimentation branch).

Run with:
    python -m src.run_stats                       # all sections
    python -m src.run_stats --sections a          # production only
    python -m src.run_stats --sections c d --skip-existing
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import src.statistics as statistics
from src.config import load_config
from src.features_v2 import load_or_build_features_v2
from src.labels import flat_mask
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute stats for all binary models")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip suites whose .npz predictions already exist",
    )
    parser.add_argument(
        "--sections", nargs="+", choices=["a", "c", "d", "nft"],
        default=["a", "c", "d"],
        help="Which sections to run (a=production, c=no-flat 20-feat + HMM, "
             "d=49-feature binary suites, nft=no-flat-test slice stats for all "
             "saved prediction sets — reads existing .npz, no retraining, so run "
             "a/c/d first). Default: a c d.",
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


if __name__ == "__main__":
    main()
