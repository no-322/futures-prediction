"""Tests for the single-test leaderboard in src.run_stats.

Covers the analysis-time functions that read saved prediction sets (no retraining):
display-name decoding, model ranking, and the leaderboard.md writer. Tests that need
the real data file skip cleanly when it is absent.
"""
from pathlib import Path

import pytest

from src.config import load_config
import json

import numpy as np
import pandas as pd

import src.run_stats as rs
from src.run_stats import (
    _LEADERBOARD_PATH,
    _leaderboard_name,
    _threshold_predict_fn,
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


def test_threshold_predict_fn() -> None:
    assert _threshold_predict_fn(None) is None
    fn = _threshold_predict_fn(0.6)

    class _M:
        def predict_proba(self, X):
            col = np.array([0.5, 0.7, 0.61])
            return np.column_stack([1 - col, col])
    assert list(fn(_M(), None)) == [0, 1, 1]      # proba >= 0.6


def test_walkforward_curated_tuned_uses_tuned_params(tmp_path, monkeypatch) -> None:
    # tuned spec for v1 only → v1rel/v2/v3 skipped (no spec file).
    proc = tmp_path / "proc"; proc.mkdir()
    (proc / "tuned_params_v1.json").write_text(json.dumps({
        "featset": "v1", "tune_threshold": True,
        "models": {
            "logistic": {"params": {"C": 10.0}, "threshold": 0.52},
            "rf": {"params": {"max_depth": 12}, "threshold": 0.49},
            "gbm": {"params": {"max_depth": 3}, "threshold": 0.50},
        },
    }))
    monkeypatch.setattr(rs, "_PROC", proc)
    monkeypatch.setattr(rs, "_WF_FEATSETS", ("v1", "v2"))   # v2 has no spec → skipped
    monkeypatch.setattr(rs, "_wf_xy", lambda cfg, fs: (
        pd.DataFrame({"a": [0.0, 1.0]}), pd.Series([0, 1]),
        pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"])), np.array([0.0, 0.0])))

    calls = []
    import src.walkforward as wf
    monkeypatch.setattr(wf, "module_factory", lambda m, p, t: ("factory", p))
    monkeypatch.setattr(wf, "walk_forward",
                        lambda *a, **k: calls.append((k["name"], k["predict_fn"], a[3])))

    rs.walkforward_curated_tuned({})
    names = [c[0] for c in calls]
    assert names == ["wf_tuned_v1_logistic", "wf_tuned_v1_rf", "wf_tuned_v1_gbm"]
    assert all(c[1] is not None for c in calls)            # threshold predict_fn applied
    assert all(c[2][1] == params for c, params in           # tuned params forwarded
               zip(calls, [{"C": 10.0}, {"max_depth": 12}, {"max_depth": 3}]))


def test_walkforward_curated_regime_orderflow_recipes(monkeypatch) -> None:
    # Mock the heavy data build + walk_forward; assert the 3 combo recipes are dispatched
    # with a per-fold regime fold_transform and returns forwarded.
    monkeypatch.setattr(rs, "_combo_xy", lambda cfg, base, variant: (
        pd.DataFrame({"a": [0.0, 1.0]}), np.zeros((2, 5)), pd.Series([0, 1]),
        pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"])), np.array([0.1, -0.1])))

    calls = []
    import src.walkforward as wf
    monkeypatch.setattr(wf, "module_factory", lambda m, p, t: ("factory", p))
    monkeypatch.setattr(wf, "walk_forward",
                        lambda *a, **k: calls.append(
                            (k["name"], k["fold_transform"], k["returns"])))

    rs.walkforward_curated_regime_orderflow({})
    names = [c[0] for c in calls]
    assert names == ["wf_ofhmm_v1_logistic", "wf_ofhmm_v3_rf", "wf_ofhmm_v3_gbm"]
    assert all(c[1] is not None for c in calls)                       # regime fold_transform
    assert all(np.array_equal(c[2], np.array([0.1, -0.1])) for c in calls)  # returns forwarded


def test_hmm_fold_transform_feature_appends_causally(monkeypatch) -> None:
    import src.models.regime_hmm as rh
    rng = np.random.RandomState(0)
    X_base = pd.DataFrame(rng.randn(50, 2), columns=["a", "b"])
    X_reg = rng.randn(50, 5)                     # 5 = len(REGIME_COLS)
    train_idx, test_idx = np.arange(40), np.arange(40, 50)

    captured = {}
    real_fit = rh.fit_regime
    def spy(X):
        captured["fit_on"] = np.asarray(X)
        return real_fit(X)
    monkeypatch.setattr(rh, "fit_regime", spy)

    ft = rs._hmm_fold_transform(X_base, X_reg, vol15_idx=0, mode="feature")
    X_tr, X_te, gate = ft(train_idx, test_idx)

    assert gate is None
    assert "regime_hi_prob" in X_tr.columns and "regime_hi_prob" in X_te.columns
    assert len(X_tr) == 40 and len(X_te) == 10
    # Leakage guard: the HMM must be fit on the train block ONLY.
    np.testing.assert_array_equal(captured["fit_on"], X_reg[train_idx])


def test_leaderboard_walkforward_ranks_and_folds_won(tmp_path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()

    def _save(name, y_true, y_pred, accs):
        np.savez(proc / f"{name}_predictions.npz",
                 y_true=np.array(y_true), y_pred=np.array(y_pred),
                 accuracies=np.array(accs, dtype=float))

    # baseline (always-up) folds + two models with 3 folds each (walk_forward prefixes
    # its saved files with "walkforward_"; the curated names then start "wf_").
    _save("walkforward_wf_baseline_alwaysup", [1, 0, 1, 0], [1, 1, 1, 1], [0.50, 0.50, 0.50])
    _save("walkforward_wf_v1_logistic", [1, 0, 1, 0], [1, 0, 1, 0], [0.60, 0.55, 0.45])  # 2/3
    _save("walkforward_wf_v3_gbm",      [1, 0, 1, 0], [1, 0, 0, 0], [0.70, 0.40, 0.52])  # 2/3

    out = tmp_path / "lbwf.md"
    leaderboard_walkforward({}, proc=proc, out=out)
    text = out.read_text()
    # v3_gbm mean (0.54) > v1_logistic mean (0.5333) → ranked first
    lines = [l for l in text.splitlines() if l.startswith("| ") and "Model" not in l
             and "---" not in l]
    assert lines[0].split("|")[1].strip() == "v3_gbm"
    assert "2/3" in text                       # folds-won computed vs baseline
