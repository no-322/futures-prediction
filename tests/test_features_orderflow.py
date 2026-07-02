"""Tests for src/features_orderflow.py.

The headline guarantee is no look-ahead: every order-flow feature at row ``t`` must
depend only on bars strictly before ``t``. We assert this two ways — structurally
(only lagged columns, no lag-0) and behaviourally (perturbing bar ``t``'s own
OHLC/Volume/tick data must not change the feature row whose target is ``t``). All
guarantees are checked for both the ``"raw"`` (tree-model) and ``"linear"``
(scale-stable, logistic-model) variants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.features_orderflow as fof
from src.features_orderflow import build_features_orderflow

# Expected leading warm-up rows (feature-row index): the 60-bar volume z-score plus
# the 4-bar lag shift, minus the 4 rows dropped from the head. lag4_norm_vol becomes
# genuine (non-fill) at feature row 59.
_WARMUP_ROWS = 60 + max([4, 3, 2, 1]) - 4  # = 60


def _synthetic_raw(n: int = 200) -> pd.DataFrame:
    """A gap-free 1-min raw frame with the schema columns the builder needs."""
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
        "Volume": rng.integers(1, 500, n).astype("int64"),
        "Up Ticks": rng.integers(0, 50, n).astype("int64"),
        "Down Ticks": rng.integers(0, 50, n).astype("int64"),
        "Same Ticks": rng.integers(0, 10, n).astype("int64"),
        "Tick Count": rng.integers(1, 100, n).astype("int64"),
    })


@pytest.fixture(scope="module", params=["raw", "linear"])
def variant(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def feats(variant: str) -> pd.DataFrame:
    return build_features_orderflow(_synthetic_raw(), variant=variant)


def test_shape_and_columns(feats: pd.DataFrame) -> None:
    """20 columns, n-4 rows, reset index, exactly the lag{1..4}_{name} columns."""
    raw = _synthetic_raw()
    assert feats.shape == (len(raw) - 4, 20)
    assert list(feats.index) == list(range(len(feats)))

    expected = [f"lag{lag}_{name}" for lag in (4, 3, 2, 1) for name in fof._INDICATORS]
    assert list(feats.columns) == expected


def test_no_lag0_column(feats: pd.DataFrame) -> None:
    """No current-bar column for ANY indicator — guards the signed_vol==label leak."""
    for col in feats.columns:
        assert not col.startswith("lag0_"), col
        # Every column must carry an explicit lag of 1..4.
        assert col.split("_", 1)[0] in {"lag1", "lag2", "lag3", "lag4"}, col
    # signed_vol must only ever appear lagged, never bare.
    assert "signed_vol" not in feats.columns
    assert any(c.endswith("_signed_vol") for c in feats.columns)


def test_finite_no_leakage_nan_inf(feats: pd.DataFrame) -> None:
    """No NaN/inf anywhere after the documented warm-up fill, and real signal past it."""
    arr = feats.to_numpy(dtype=float)
    assert np.isfinite(arr).all(), "NaN/inf present in order-flow features"
    # Beyond the warm-up region the rolling stats are genuine, not fill artefacts:
    # at least one z-score row past warm-up must be non-zero.
    late = feats.iloc[_WARMUP_ROWS + 5:]
    assert late["lag1_norm_vol"].abs().to_numpy().sum() > 0


@pytest.mark.parametrize("variant", ["raw", "linear"])
def test_perturbation_no_lookahead(variant: str) -> None:
    """Perturbing bar p's own data must not change the feature row whose target is p.

    Feature row ``i`` targets raw bar ``i+4`` and references only grid bars
    ``(i+4)-lag`` for lag in 1..4 — strictly before the target. So perturbing raw
    bar ``p`` must leave every feature row up to and including row ``p-4`` untouched,
    while row ``p-3`` (target ``p+1``, which legitimately uses bar ``p`` at lag-1)
    must change. This is the load-bearing leakage check, run for both variants.
    """
    raw = _synthetic_raw()
    base = build_features_orderflow(raw, variant=variant)

    p = 120  # mid-series, well past the 60-bar warm-up
    perturbed = raw.copy()
    for col, val in {
        "Open": 50.0, "High": 60.0, "Low": 40.0, "Close": 55.0,
        "Volume": 999_999, "Up Ticks": 9999, "Down Ticks": 0, "Tick Count": 9999,
    }.items():
        perturbed.loc[p, col] = val
    pert = build_features_orderflow(perturbed, variant=variant)

    # Rows whose target is <= p must be bit-identical (depend only on bars < target).
    pd.testing.assert_frame_equal(base.iloc[: p - 3], pert.iloc[: p - 3])

    # Sanity: the perturbation actually propagated — row p-3 (target p+1) uses bar p.
    assert not base.iloc[p - 3].equals(pert.iloc[p - 3]), (
        "perturbation did not propagate to the lag-1 consumer; test is vacuous"
    )


def test_linear_signed_vol_value_and_scale() -> None:
    """The linear variant's signed_vol is signed relative volume, on an O(1) scale.

    Value: at a mid row, lag1_signed_vol equals the manually computed
    ``sign(Close-Open) * Volume / Volume.rolling(60).mean()`` shifted one bar (the
    synthetic frame is gap-free, so the dense grid coincides with the raw rows).
    Scale: the linear variant's |signed_vol| is far smaller than the raw variant's
    (raw carries the full ~hundreds volume magnitude that dominated the linear fit).
    """
    raw = _synthetic_raw()
    lin = build_features_orderflow(raw, variant="linear")
    rawv = build_features_orderflow(raw, variant="raw")

    vol = raw["Volume"].astype(float)
    sign = np.sign(raw["Close"] - raw["Open"])
    expected = (sign * vol / vol.rolling(fof._VOL_WINDOW).mean()).shift(1)

    i = 120                                   # feature row -> raw bar i+4
    assert np.isclose(lin["lag1_signed_vol"].iloc[i], expected.iloc[i + 4])

    lin_max = lin["lag1_signed_vol"].abs().max()
    raw_max = rawv["lag1_signed_vol"].abs().max()
    assert lin_max < 10.0, lin_max          # signed relative volume is O(1)
    assert raw_max > 50.0                    # raw signed volume carries the magnitude
    assert lin_max < raw_max
