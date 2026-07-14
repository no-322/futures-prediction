"""Feature engineering — v1 made stationary by anchoring to the last open.

v1's 20 columns are **raw price levels** (lag{k}_Open/Close/High/Low/VWAP). Across 13
contracts over 3 years those absolute levels drift, so a linear model keying on price
cannot transfer from the 2023→mid-2024 train half to the mid-2024→2026 test half.

This variant expresses every base price lag as a **log-ratio versus the most recent
observed open** ``lag1_Open`` (= Open_{t-1}, strictly before the target bar t — no
look-ahead):

    lag{k}_{col}  ->  log( lag{k}_{col} / lag1_Open )

The result is stationary (values hover near 0), contract-scale invariant, and keeps
both cross-bar drift (e.g. log(Close_{t-4}/Open_{t-1})) and intrabar shape. ``lag1_Open``
itself becomes log(1)=0, a constant, so it is dropped.

This is the v1 analog of features_v3 (which anchors the 49-col v2 matrix on lag1_Close).

Total features: 20 (v1) − 1 (dropped lag1_Open) = **19**.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_features

_BASE_COLS = ["Open", "Close", "High", "Low", "VWAP"]
_LAG_STEPS = [4, 3, 2, 1]
_REF_COL = "lag1_Open"
N_FEATURES_V1_REL = 19  # 20 v1 columns − dropped lag1_Open


def build_features_v1_rel(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 19-dim stationary v1-relative feature matrix from the raw DataFrame.

    Builds v1 (``build_features``) then replaces every raw price lag with
    ``log(value / lag1_Open)`` and drops the now-constant ``lag1_Open``. A pure
    row-wise transform of columns that are already strictly lagged, so it introduces
    no look-ahead.

    Args:
        df: Raw DataFrame from load_raw(), with 'Date and Time' as a column and the
            Open/Close/High/Low/VWAP columns present.

    Returns:
        DataFrame of shape (len(df) - 4, 19) with reset 0-based index.
    """
    v1 = build_features(df)
    ref = v1[_REF_COL]
    out = v1.copy()
    base_cols = [f"lag{k}_{c}" for k in _LAG_STEPS for c in _BASE_COLS]
    for col in base_cols:
        out[col] = np.log(v1[col] / ref)
    out = out.drop(columns=[_REF_COL]).reset_index(drop=True)

    # Guard against div-by-zero / non-positive ratios at edge bars.
    n_bad = int(np.isinf(out.to_numpy()).sum() + out.isna().sum().sum())
    if n_bad:
        out = out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    assert out.shape[1] == N_FEATURES_V1_REL, (
        f"Expected {N_FEATURES_V1_REL} features, got {out.shape[1]}"
    )
    print(f"features_v1_rel: {out.shape[1]} features, {out.shape[0]:,} rows "
          f"(base lags → log-ratio vs {_REF_COL}; {n_bad} bad cells filled)")
    return out
