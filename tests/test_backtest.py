"""Tests for the backtest runner (src/backtest.py)."""
import numpy as np
import pandas as pd
import pytest

import src.backtest as bt


def _synthetic_raw(n: int = 24) -> pd.DataFrame:
    """Minimal raw frame: monotonic timestamps + OHLC; n-4 aligned rows."""
    ts = pd.date_range("2023-01-03 04:00", periods=n, freq="1min")
    opens = np.full(n, 100.0)
    closes = opens + np.where(np.arange(n) % 2 == 0, 0.5, -0.5)  # alternating moves
    return pd.DataFrame({
        "Date and Time": ts, "Open": opens, "High": opens,
        "Low": opens, "Close": closes, "VWAP": opens,
    })


def _wire(monkeypatch, tmp_path, df):
    monkeypatch.setattr(bt, "load_raw", lambda _p: df)
    monkeypatch.setattr(bt, "_PROC", tmp_path)
    monkeypatch.setattr(bt, "_DOCS", tmp_path)
    monkeypatch.setattr(bt, "_REPORT_PATH", tmp_path / "backtest_stats.md")
    cfg = {"data": {"path": "x", "train_size": 0.5}}
    return cfg


def test_run_one_persists_artifacts_and_compounds(monkeypatch, tmp_path) -> None:
    df = _synthetic_raw(24)                     # aligned 20 rows -> test 10
    cfg = _wire(monkeypatch, tmp_path, df)
    ts, bar_returns = bt._reconstruct_test_bars(cfg)
    n = len(bar_returns)

    # A fake binary model: all-long predictions of the right length.
    np.savez(tmp_path / "fake_predictions.npz",
             y_true=np.zeros(n, dtype=int), y_pred=np.ones(n, dtype=int))
    monkeypatch.setitem(bt._REGISTRY, "fake", "Fake model")

    res = bt.run_one("fake", 0.0, ts, bar_returns)

    assert res["n_periods"] == n
    assert res["equity"][0] == pytest.approx(1000.0 * (1.0 + res["net_return"][0]))
    assert (tmp_path / "backtest_fake_predictions.npz").exists()
    assert (tmp_path / "backtest_fake.png").exists()

    saved = np.load(tmp_path / "backtest_fake_predictions.npz")
    for key in ("signal", "position", "bar_return", "net_return", "payoff",
                "equity", "passive_equity", "timestamps"):
        assert len(saved[key]) == n


def test_run_writes_report(monkeypatch, tmp_path) -> None:
    df = _synthetic_raw(24)
    cfg = _wire(monkeypatch, tmp_path, df)
    n = len(bt._reconstruct_test_bars(cfg)[1])
    np.savez(tmp_path / "fake_predictions.npz",
             y_true=np.zeros(n, dtype=int), y_pred=np.ones(n, dtype=int))
    monkeypatch.setitem(bt._REGISTRY, "fake", "Fake model")

    bt.run("fake", transaction_cost=0.0, config=cfg)
    report = (tmp_path / "backtest_stats.md").read_text()
    assert "Fake model" in report
    assert "Annualized Sharpe" in report


def test_length_mismatch_guard_raises(monkeypatch, tmp_path) -> None:
    df = _synthetic_raw(24)
    cfg = _wire(monkeypatch, tmp_path, df)
    ts, bar_returns = bt._reconstruct_test_bars(cfg)
    # Wrong-length predictions (e.g. a 3-class / TSS npz) must be rejected.
    np.savez(tmp_path / "bad_predictions.npz",
             y_pred=np.ones(len(bar_returns) + 5, dtype=int))
    monkeypatch.setitem(bt._REGISTRY, "bad", "Bad model")
    with pytest.raises(ValueError, match="contiguous 50/50"):
        bt.run_one("bad", 0.0, ts, bar_returns)
