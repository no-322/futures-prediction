"""Tests for the no-flat-test evaluation slice in src.run_stats.

These cover the analysis-time slice that drops flat (Open == Close) test bars
from the metrics only — no retraining. They lean on existing prediction
artifacts in data/processed/; tests skip cleanly when those are absent.
"""
from pathlib import Path

import numpy as np
import pytest

import src.statistics as statistics
from src.config import load_config
from src.run_stats import (
    _LEADERBOARD_PATH,
    _NFT_NOFLAT_REPORT_PATH,
    _NFT_REPORT_PATH,
    _NFT_V2_REPORT_PATH,
    _nft_stats,
    leaderboard,
    section_noflat_test,
)
# Aliased so pytest does not collect this `test_`-prefixed helper as a test.
from src.run_stats import test_flat_mask as build_test_flat_mask

_PROC = Path("data/processed")


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def keep(cfg: dict) -> np.ndarray:
    return build_test_flat_mask(cfg)


def test_mask_is_bool_and_drops_some(keep: np.ndarray) -> None:
    assert keep.dtype == bool
    assert keep.ndim == 1
    # Flat bars exist in the TY data, so the mask must drop at least one row
    # but never all of them.
    assert 0 < int((~keep).sum()) < keep.size


def test_mask_matches_move_nonzero(keep: np.ndarray) -> None:
    # flat ⇔ move == 0; the saved binary-suite `move` is the ground truth.
    npz = _PROC / "exp_noflat_rf_predictions.npz"
    if not npz.exists():
        pytest.skip(f"{npz} not present")
    move = np.load(npz)["move"]
    assert move.size == keep.size
    assert np.array_equal(keep, move != 0)


def test_nft_stats_matches_manual_slice(keep: np.ndarray) -> None:
    npz = _PROC / "exp_noflat_rf_predictions.npz"
    if not npz.exists():
        pytest.skip(f"{npz} not present")
    d = np.load(npz)
    res = _nft_stats("exp_noflat_rf", "Random Forest (no-flat)", keep)
    expected = statistics.compute(d["y_true"][keep], d["y_pred"][keep])
    assert res is not None
    assert res["n_samples"] == int(keep.sum())
    assert res["accuracy"] == pytest.approx(expected["accuracy"])
    assert res["mcc"] == pytest.approx(expected["mcc"])


def test_nft_stats_skips_length_mismatch(keep: np.ndarray) -> None:
    # 3-class / two-stage sets have a different length and must be skipped.
    candidates = ["exp_three_class_v1", "exp_two_stage_v1", "exp_regime_v2"]
    stem = next((c for c in candidates
                 if (_PROC / f"{c}_predictions.npz").exists()
                 and len(np.load(_PROC / f"{c}_predictions.npz")["y_pred"]) != keep.size),
                None)
    if stem is None:
        pytest.skip("no length-mismatched prediction set available")
    assert _nft_stats(stem, "mismatch", keep) is None


def test_nft_stats_missing_returns_none(keep: np.ndarray) -> None:
    assert _nft_stats("definitely_not_a_real_stem", "nope", keep) is None


def test_section_writes_reports(cfg: dict) -> None:
    if not (_PROC / "rf_predictions.npz").exists():
        pytest.skip("production predictions not present")
    section_noflat_test(cfg)
    assert _NFT_REPORT_PATH.exists()
    text = _NFT_REPORT_PATH.read_text()
    assert "No-Flat Test Slice" in text
    assert "flat dropped" in text
    # The other two reports exist when their artifacts do.
    if (_PROC / "exp_noflat_rf_predictions.npz").exists():
        assert _NFT_NOFLAT_REPORT_PATH.exists()
    if (_PROC / "exp_v2_rf_predictions.npz").exists():
        assert _NFT_V2_REPORT_PATH.exists()


def test_leaderboard_writes_sorted(cfg: dict) -> None:
    if not (_PROC / "rf_predictions.npz").exists():
        pytest.skip("production predictions not present")
    leaderboard(cfg)
    assert _LEADERBOARD_PATH.exists()
    text = _LEADERBOARD_PATH.read_text()
    assert "| Model | No-flat test acc | Accuracy | MCC |" in text
    # Data rows: start with "|", skip the header and the |---| separator.
    nf_accs = [
        float(line.split("|")[2])
        for line in text.splitlines()
        if line.startswith("|") and "Model" not in line and "---" not in line
    ]
    assert len(nf_accs) > 1
    assert nf_accs == sorted(nf_accs, reverse=True)  # ordered by no-flat acc
