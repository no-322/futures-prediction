"""Tests for the stationary v3 feature matrix (src.features_v3)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features_v2 import build_features_v2
from src.features_v3 import (
    N_FEATURES_V3,
    _transform_v3,
    build_features_v3,
    load_or_build_features_v3,
)
from src.load import load_raw

DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "data.csv"
V2_CACHE = Path("data/processed/features_v2.parquet")

_INDICATORS = ["vwap_dev", "bar_range", "body_ratio", "tick_delta", "return",
               "log_return", "rsi5", "rsi15", "vol5", "vol15", "macd_line",
               "macd_signal", "macd_hist"]


def _synthetic_v2(n: int = 50) -> pd.DataFrame:
    """A frame with exactly the 49 v2 columns (base prices kept positive)."""
    rng = np.random.RandomState(42)
    cols: dict[str, np.ndarray] = {}
    for k in (4, 3, 2, 1):
        for c in ("Open", "Close", "High", "Low", "VWAP"):
            cols[f"lag{k}_{c}"] = 100.0 + rng.rand(n) * 5.0  # positive prices
    for lag in (1, 4):
        for name in _INDICATORS:
            cols[f"lag{lag}_{name}"] = rng.randn(n)
    cols["tod_sin"] = rng.randn(n)
    cols["tod_cos"] = rng.randn(n)
    cols["session_min"] = rng.rand(n) * 500
    return pd.DataFrame(cols)


def test_transform_shape_and_drops_ref() -> None:
    v2 = _synthetic_v2()
    v3 = _transform_v3(v2)
    assert v3.shape == (len(v2), N_FEATURES_V3)
    assert "lag1_Close" not in v3.columns  # constant after ratio → dropped


def test_transform_is_log_ratio() -> None:
    v2 = _synthetic_v2()
    v3 = _transform_v3(v2)
    expected = np.log(v2["lag4_Open"] / v2["lag1_Close"]).to_numpy()
    np.testing.assert_allclose(v3["lag4_Open"].to_numpy(), expected, rtol=1e-9)


def test_transform_keeps_derived_unchanged() -> None:
    v2 = _synthetic_v2()
    v3 = _transform_v3(v2)
    for col in ("lag1_rsi5", "lag4_macd_hist", "tod_sin", "session_min"):
        np.testing.assert_array_equal(v3[col].to_numpy(), v2[col].to_numpy())


def test_no_nan_or_inf() -> None:
    v3 = _transform_v3(_synthetic_v2())
    assert not v3.isna().any().any()
    assert np.isfinite(v3.to_numpy()).all()


@pytest.mark.skipif(not V2_CACHE.exists(), reason="features_v2 cache absent")
def test_real_build_is_stationary() -> None:
    df = load_raw(DATA_PATH)
    v3 = load_or_build_features_v3(df)
    assert v3.shape[1] == N_FEATURES_V3
    assert np.isfinite(v3.to_numpy()).all()
    # Base log-ratios should be small and centred near 0 (stationary).
    assert abs(float(v3["lag4_Close"].mean())) < 0.5
    assert "lag1_Close" not in v3.columns
