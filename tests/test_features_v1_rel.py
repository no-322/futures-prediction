"""Tests for src/features_v1_rel.py — v1 anchored to lag1_Open (log-ratios)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_features
from src.features_v1_rel import N_FEATURES_V1_REL, build_features_v1_rel


def _synthetic_raw(n: int = 80) -> pd.DataFrame:
    ts = pd.date_range("2023-01-03 04:00", periods=n, freq="1min")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.05, n))
    open_ = close + rng.normal(0, 0.02, n)
    return pd.DataFrame({
        "Date and Time": ts,
        "Open": open_,
        "High": np.maximum(open_, close) + 0.05,
        "Low": np.minimum(open_, close) - 0.05,
        "Close": close,
        "VWAP": close,
    })


def test_shape_and_dropped_anchor() -> None:
    raw = _synthetic_raw()
    feats = build_features_v1_rel(raw)
    assert feats.shape == (len(raw) - 4, N_FEATURES_V1_REL)   # 19
    assert list(feats.index) == list(range(len(feats)))
    assert "lag1_Open" not in feats.columns                   # constant anchor dropped
    # Every other v1 column survives.
    v1_cols = [c for c in build_features(raw).columns if c != "lag1_Open"]
    assert list(feats.columns) == v1_cols


def test_finite_no_nan_inf() -> None:
    feats = build_features_v1_rel(_synthetic_raw())
    assert np.isfinite(feats.to_numpy(dtype=float)).all()


def test_values_are_logratio_vs_lag1_open() -> None:
    raw = _synthetic_raw()
    v1 = build_features(raw)
    feats = build_features_v1_rel(raw)
    i = 40
    for col in ("lag4_Close", "lag2_High", "lag1_VWAP", "lag4_Open"):
        expected = np.log(v1[col].iloc[i] / v1["lag1_Open"].iloc[i])
        assert np.isclose(feats[col].iloc[i], expected), col
