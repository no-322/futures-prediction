"""Tests for the single-test leaderboard in src.run_stats.

Covers the analysis-time functions that read saved prediction sets (no retraining):
display-name decoding, model ranking, and the leaderboard.md writer. Tests that need
the real data file skip cleanly when it is absent.
"""
from pathlib import Path

import pytest

from src.config import load_config
import numpy as np

from src.run_stats import (
    _LEADERBOARD_PATH,
    _leaderboard_name,
    leaderboard,
    leaderboard_walkforward,
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
    for stem, name, acc, recall, mcc, aum in rows:
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
    assert "| Model | Accuracy | Recall | MCC | AUM % |" in text
    assert "single test set" in text.lower()


def test_leaderboard_walkforward_ranks_and_folds_won(tmp_path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()

    def _save(name, y_true, y_pred, accs):
        np.savez(proc / f"{name}_predictions.npz",
                 y_true=np.array(y_true), y_pred=np.array(y_pred),
                 accuracies=np.array(accs, dtype=float))

    # baseline (always-up) folds + two models with 3 folds each
    _save("wf_baseline_alwaysup", [1, 0, 1, 0], [1, 1, 1, 1], [0.50, 0.50, 0.50])
    _save("wf_v1_logistic", [1, 0, 1, 0], [1, 0, 1, 0], [0.60, 0.55, 0.45])  # wins 2/3
    _save("wf_v3_gbm",      [1, 0, 1, 0], [1, 0, 0, 0], [0.70, 0.40, 0.52])  # wins 2/3

    out = tmp_path / "lbwf.md"
    leaderboard_walkforward({}, proc=proc, out=out)
    text = out.read_text()
    # v3_gbm mean (0.54) > v1_logistic mean (0.5333) → ranked first
    lines = [l for l in text.splitlines() if l.startswith("| ") and "Model" not in l
             and "---" not in l]
    assert lines[0].split("|")[1].strip() == "v3_gbm"
    assert "2/3" in text                       # folds-won computed vs baseline
