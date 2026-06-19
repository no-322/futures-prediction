"""Feature engineering v3 — stationary base lags on top of the v2 matrix.

Diagnosis behind v3: the v1/v2 base features are **raw price levels** (lag{k}_Open,
lag{k}_Close, …). Across 13 contracts over 3 years those levels are
non-stationary, so a model keying on absolute price cannot transfer from the
2023→mid-2024 train half to the mid-2024→2026 test half.

v3 fixes this by expressing every base OHLCV lag as a **log-ratio versus the most
recent observed close** ``lag1_Close`` (= Close_{t-1}, strictly before the target
bar t — no look-ahead):

    lag{k}_{col}  ->  log( lag{k}_{col} / lag1_Close )

This is stationary (values hover near 0), preserves short-horizon momentum
(e.g. log(Close_{t-4}/Close_{t-1})) and intrabar shape, and is contract-scale
invariant. ``lag1_Close`` itself becomes log(1)=0, a constant, so it is dropped.
The 26 derived v2 indicators (returns, RSI, MACD, vol, vwap_dev, tick_delta, …)
and the 3 target-bar time features are already stationary and kept unchanged.

Total features: 49 (v2) − 1 (dropped lag1_Close) = **48**.

The transform is a pure row-wise function of columns that are already strictly
lagged in v2, so it introduces no look-ahead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features_v2 import build_features_v2, load_or_build_features_v2

_FEATURES_V3_CACHE = Path("data/processed/features_v3.parquet")

_BASE_COLS = ["Open", "Close", "High", "Low", "VWAP"]
_LAG_STEPS_BASE = [4, 3, 2, 1]
_REF_COL = "lag1_Close"
N_FEATURES_V3 = 48  # 49 v2 columns − dropped lag1_Close


def _transform_v3(v2: pd.DataFrame) -> pd.DataFrame:
    """Convert v2's raw base-price lags to log-ratios vs lag1_Close.

    Args:
        v2: A features_v2 matrix (49 columns), as returned by build_features_v2.

    Returns:
        A 48-column matrix: the 20 base lags replaced by
        ``log(value / lag1_Close)`` (with the now-constant lag1_Close dropped),
        all other v2 columns unchanged. Index is reset 0-based.
    """
    ref = v2[_REF_COL]
    out = v2.copy()
    base_cols = [f"lag{k}_{c}" for k in _LAG_STEPS_BASE for c in _BASE_COLS]
    for col in base_cols:
        out[col] = np.log(v2[col] / ref)
    out = out.drop(columns=[_REF_COL]).reset_index(drop=True)

    # Guard against div-by-zero / non-positive ratios at edge bars.
    n_bad = int(np.isinf(out.to_numpy()).sum() + out.isna().sum().sum())
    if n_bad:
        out = out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    assert out.shape[1] == N_FEATURES_V3, (
        f"Expected {N_FEATURES_V3} features, got {out.shape[1]}"
    )
    print(f"features_v3: {out.shape[1]} features, {out.shape[0]:,} rows "
          f"(base lags → log-ratio vs {_REF_COL}; {n_bad} bad cells filled)")
    return out


def build_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 48-dim stationary v3 feature matrix from the raw DataFrame.

    Args:
        df: Raw DataFrame from load_raw(), with 'Date and Time' as a column.

    Returns:
        DataFrame of shape (len(df) - 4, 48) with reset 0-based index.
    """
    return _transform_v3(build_features_v2(df))


def load_or_build_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Return the v3 matrix, using the parquet cache (and the v2 cache) when present.

    Args:
        df: Raw DataFrame as returned by load_raw().

    Returns:
        The 48-feature v3 matrix (same as build_features_v3()).
    """
    if _FEATURES_V3_CACHE.exists():
        print(f"  Loading cached features_v3 from {_FEATURES_V3_CACHE}...")
        return pd.read_parquet(_FEATURES_V3_CACHE)
    print("  Building features_v3 from v2 (will cache for future runs)...")
    v3 = _transform_v3(load_or_build_features_v2(df))
    _FEATURES_V3_CACHE.parent.mkdir(parents=True, exist_ok=True)
    v3.to_parquet(_FEATURES_V3_CACHE)
    print(f"  Cached to {_FEATURES_V3_CACHE}")
    return v3
