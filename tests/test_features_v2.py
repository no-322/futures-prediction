"""Tests for the 49-feature v2 builder and its cached loader."""
import numpy as np
import pandas as pd

import src.features_v2 as fv2
from src.models.regime_hmm import REGIME_COLS


def _synthetic_raw(n: int = 60) -> pd.DataFrame:
    ts = pd.date_range("2023-01-03 04:00", periods=n, freq="1min")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.05, n))
    return pd.DataFrame({
        "Date and Time": ts,
        "Open": close + rng.normal(0, 0.01, n),
        "High": close + 0.05,
        "Low": close - 0.05,
        "Close": close,
        "VWAP": close,
        "Volume": rng.integers(1, 100, n),
        "Up Ticks": rng.integers(0, 50, n),
        "Down Ticks": rng.integers(0, 50, n),
        "Same Ticks": rng.integers(0, 10, n),
        "Tick Count": rng.integers(1, 100, n),
    })


def test_build_features_v2_shape_and_regime_cols() -> None:
    df = _synthetic_raw(60)
    feats = fv2.build_features_v2(df)
    assert feats.shape == (len(df) - 4, 49)            # 49 features, drop 4 warm-up
    assert list(feats.index) == list(range(len(feats)))  # reset index
    for col in REGIME_COLS:                            # regime descriptors present
        assert col in feats.columns


def test_load_or_build_uses_cache(monkeypatch, tmp_path) -> None:
    df = _synthetic_raw(60)
    cache = tmp_path / "features_v2.parquet"
    monkeypatch.setattr(fv2, "_FEATURES_V2_CACHE", cache)

    built = fv2.load_or_build_features_v2(df)          # builds + caches
    assert cache.exists()

    # Second call must read the cache, not rebuild.
    monkeypatch.setattr(fv2, "build_features_v2",
                        lambda _df: (_ for _ in ()).throw(AssertionError("rebuilt")))
    cached = fv2.load_or_build_features_v2(df)
    pd.testing.assert_frame_equal(built, cached)
