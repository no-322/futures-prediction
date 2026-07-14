"""Tests for the single-test leaderboard in src.run_stats.

Covers the analysis-time functions that read saved prediction sets (no retraining):
display-name decoding, model ranking, and the leaderboard.md writer. Tests that need
the real data file skip cleanly when it is absent.
"""
from pathlib import Path

import pytest

from src.config import load_config
from src.run_stats import (
    _LEADERBOARD_PATH,
    _leaderboard_name,
    leaderboard,
    rank_models,
)

_PROC = Path("data/processed")
_DATA = Path("data/raw/data.csv")


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


def test_leaderboard_name_decoding() -> None:
    reg = {"exp_noflat_logistic": "Logistic Regression (no-flat)"}
    # registry hit
    assert _leaderboard_name("exp_noflat_logistic", reg) == "Logistic Regression (no-flat)"
    # tuned_{feat}_{algo}
    assert _leaderboard_name("tuned_v3_gbm", {}) == "Gradient Boosting (XGBoost) (tuned, v3)"
    # v1-rel variant
    assert _leaderboard_name("exp_noflat_v1rel_logistic", {}) == "Logistic Regression (v1-rel)"
    # unknown → stem passthrough
    assert _leaderboard_name("something_else", {}) == "something_else"


def test_rank_models_returns_sorted(cfg: dict) -> None:
    if not _DATA.exists():
        pytest.skip("raw data file not present")
    rows = rank_models(cfg)
    assert isinstance(rows, list)
    for stem, name, acc, recall, mcc in rows:
        assert isinstance(stem, str) and isinstance(name, str)
        assert 0.0 <= acc <= 1.0
    # Sorted by (accuracy, mcc) descending.
    keys = [(r[2], r[4]) for r in rows]
    assert keys == sorted(keys, reverse=True)


def test_leaderboard_writes_file(cfg: dict) -> None:
    if not _DATA.exists():
        pytest.skip("raw data file not present")
    leaderboard(cfg)
    assert _LEADERBOARD_PATH.exists()
    text = _LEADERBOARD_PATH.read_text()
    assert "| Model | Accuracy | Recall | MCC |" in text
    assert "single test set" in text.lower()
