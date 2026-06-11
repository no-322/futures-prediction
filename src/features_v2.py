"""Feature engineering v2 — extends v1 OHLCV lags with derived signals.

New indicators (computed on the dense 1-min grid, then lagged at 1 and 4):
  vwap_dev    (Close - VWAP) / VWAP
  bar_range   High - Low
  body_ratio  |Close - Open| / (High - Low + eps)
  tick_delta  (Up Ticks - Down Ticks) / Tick Count  — order-flow proxy
  return      arithmetic bar return: Close/Close_prev - 1
  log_return  log(Close / Close_prev)
  rsi5        RSI with 5-period Wilder smoothing
  rsi15       RSI with 15-period Wilder smoothing
  vol5        rolling std of log-returns, window=5
  vol15       rolling std of log-returns, window=15
  macd_line   EMA(6) - EMA(13) of Close  (minute-adapted MACD)
  macd_signal EMA(4) of macd_line
  macd_hist   macd_line - macd_signal

Target-bar features (no lag, computed on original timestamps):
  tod_sin, tod_cos  sin/cos cyclic encoding of minute-of-day
  session_min       minutes elapsed since the current trading session opened
                    (session boundary = gap >= 12 h to the previous bar)

Total features: 20 (base) + 26 (new lagged) + 3 (target-bar) = 49
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_FEATURES_V2_CACHE = Path("data/processed/features_v2.parquet")

_LAG_STEPS_BASE = [4, 3, 2, 1]          # same as v1
_LAG_STEPS_NEW  = [1, 4]                # lags for new indicators
_BASE_COLS      = ["Open", "Close", "High", "Low", "VWAP"]
_TICK_COLS      = ["Up Ticks", "Down Ticks", "Tick Count"]


# ---------------------------------------------------------------------------
# Private indicator helpers (operate on a dense 1-min pd.Series)
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, window: int) -> pd.Series:
    """RSI using Wilder's exponential smoothing (com = window - 1)."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=window - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=window - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _rolling_vol(series: pd.Series, window: int) -> pd.Series:
    """Rolling std of log-returns."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window).std()


def _macd(
    series: pd.Series,
    fast: int = 6,
    slow: int = 13,
    signal: int = 4,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Minute-adapted MACD.  Default (6,13,4) ≈ standard (12,26,9) scaled to
    1-min bars.  Returns (macd_line, signal_line, histogram)."""
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Build a 49-dim feature matrix from the raw DataFrame.

    Extends build_features() with derived signals. Follows the same dense-grid
    pattern: expand to 1-min grid → ffill → compute indicators → lag → project
    back → drop first 4 rows.

    Args:
        df: Raw DataFrame from load_raw(), with 'Date and Time' as a column
            and all schema columns present.

    Returns:
        DataFrame of shape (len(df) - 4, 49) with reset 0-based index.
    """
    all_src_cols = _BASE_COLS + _TICK_COLS
    src = df.set_index("Date and Time")[all_src_cols].copy()

    # -- 1. Dense 1-min grid with forward-fill --------------------------------
    full_grid = pd.date_range(src.index[0], src.index[-1], freq="1min")
    filled    = src.reindex(full_grid).ffill()

    close = filled["Close"]

    # -- 2. Compute indicators on the dense grid ------------------------------
    vwap_dev   = (filled["Close"] - filled["VWAP"]) / filled["VWAP"].replace(0.0, np.nan)
    bar_range  = filled["High"] - filled["Low"]
    body_ratio = (filled["Close"] - filled["Open"]).abs() / (bar_range + 1e-8)
    tick_delta = (
        (filled["Up Ticks"] - filled["Down Ticks"])
        / filled["Tick Count"].replace(0, np.nan)
    )
    ret       = close / close.shift(1) - 1.0
    log_ret   = np.log(close / close.shift(1))
    rsi5      = _rsi(close, 5)
    rsi15     = _rsi(close, 15)
    vol5      = _rolling_vol(close, 5)
    vol15     = _rolling_vol(close, 15)
    macd_line, macd_sig, macd_hist = _macd(close)

    new_indicators: dict[str, pd.Series] = {
        "vwap_dev":    vwap_dev,
        "bar_range":   bar_range,
        "body_ratio":  body_ratio,
        "tick_delta":  tick_delta,
        "return":      ret,
        "log_return":  log_ret,
        "rsi5":        rsi5,
        "rsi15":       rsi15,
        "vol5":        vol5,
        "vol15":       vol15,
        "macd_line":   macd_line,
        "macd_signal": macd_sig,
        "macd_hist":   macd_hist,
    }

    # -- 3. Build lagged columns ----------------------------------------------
    lag_data: dict[str, pd.Series] = {}

    # Base OHLCV lags (identical to v1)
    for lag in _LAG_STEPS_BASE:
        for col in _BASE_COLS:
            lag_data[f"lag{lag}_{col}"] = filled[col].shift(lag)

    # New indicator lags
    for lag in _LAG_STEPS_NEW:
        for name, series in new_indicators.items():
            lag_data[f"lag{lag}_{name}"] = series.shift(lag)

    # -- 4. Build grid DataFrame and project back to original timestamps ------
    features_grid = pd.DataFrame(lag_data, index=full_grid)
    features      = features_grid.reindex(src.index)

    # -- 5. Target-bar time features (computed on original timestamps) --------
    minutes_of_day = pd.Series(
        src.index.hour * 60 + src.index.minute,
        index=src.index,
        dtype=float,
    )
    features["tod_sin"] = np.sin(2.0 * np.pi * minutes_of_day / 1440.0)
    features["tod_cos"] = np.cos(2.0 * np.pi * minutes_of_day / 1440.0)

    # Session minutes: resets whenever gap to prior bar >= 12 h
    ts_series   = pd.Series(src.index)
    new_session = (ts_series.diff() >= pd.Timedelta(hours=12)).fillna(True)
    session_id  = new_session.cumsum().values
    session_starts: dict[int, pd.Timestamp] = {}
    session_min_vals: list[float] = []
    for ts, sid in zip(src.index, session_id):
        if sid not in session_starts:
            session_starts[sid] = ts
        session_min_vals.append(
            (ts - session_starts[sid]).total_seconds() / 60.0
        )
    features["session_min"] = pd.Series(session_min_vals, index=src.index)

    # -- 6. Drop first 4 rows (warm-up for base lags) and reset index --------
    n_drop = len(_LAG_STEPS_BASE)  # 4
    features = features.iloc[n_drop:].reset_index(drop=True)

    # Forward-fill then zero-fill the small number of NaN values remaining
    # from RSI/vol/MACD warm-up at the very start of the dataset.
    n_nan = int(features.isna().sum().sum())
    if n_nan:
        features = features.ffill().fillna(0.0)

    print(
        f"features_v2: {features.shape[1]} features, "
        f"{features.shape[0]:,} rows  "
        f"(dropped {n_drop} warm-up rows, filled {n_nan} NaN cells)"
    )
    return features


def load_or_build_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Return the v2 feature matrix, using the parquet cache when available.

    Building features_v2 over the full dataset is expensive (~12 min), so the
    result is cached at ``data/processed/features_v2.parquet``. This loader
    returns the cache if present, otherwise builds the features and writes the
    cache for subsequent runs.

    Args:
        df: Raw DataFrame as returned by load_raw().

    Returns:
        The 49-feature v2 matrix (same as build_features_v2()).
    """
    if _FEATURES_V2_CACHE.exists():
        print(f"  Loading cached features_v2 from {_FEATURES_V2_CACHE}...")
        return pd.read_parquet(_FEATURES_V2_CACHE)
    print("  Building features_v2 (will cache for future runs)...")
    features = build_features_v2(df)
    _FEATURES_V2_CACHE.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(_FEATURES_V2_CACHE)
    print(f"  Cached to {_FEATURES_V2_CACHE}")
    return features
