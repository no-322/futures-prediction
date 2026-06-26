"""Feature engineering — order-flow signals (volume + tick imbalance).

All base indicators are computed on the dense 1-min grid (the same
``pd.date_range(freq="1min")`` → ``reindex().ffill()`` pattern as
``src/features_v2.py``), then lagged. Five base indicators:

  norm_vol    trailing 60-bar causal z-score of Volume
  signed_vol  variant-dependent (see below):
                "raw"    Volume * sign(Close - Open)   — raw magnitude, for tree models
                "linear" sign(Close - Open) * Volume / Volume.rolling(60).mean()
                         — signed *relative* volume, O(1) scale, for linear models
  cum_td5     trailing 5-bar rolling SUM of tick_delta
  cum_td10    trailing 10-bar rolling SUM of tick_delta
  cum_td15    trailing 15-bar rolling SUM of tick_delta

The ``variant`` only changes ``signed_vol``. The "raw" version's volume magnitude
(~hundreds) dominates an unscaled logistic fit; the "linear" version normalises by
the trailing-mean volume so direction and relative size survive on an O(1) scale.
``norm_vol`` and ``cum_td*`` are identical across variants. Column names are identical
across variants, so the two can be swapped without changing downstream shape.

where ``tick_delta = (Up Ticks - Down Ticks) / Tick Count`` (the order-flow proxy
defined in features_v2).

Each base indicator is lagged by t-1, t-2, t-3, t-4 and **only the lagged columns
are emitted** — there is no lag-0 (current-bar) column for any indicator. This is a
hard leakage rule here: ``signed_vol``'s lag-0 sign equals the up/down label, so the
current bar must never enter any feature. Every feature value at row ``t`` depends
solely on bars strictly before ``t``.

Total features: 5 indicators × 4 lags = 20.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_FEATURES_ORDERFLOW_CACHE = Path("data/processed/features_orderflow.parquet")
_FEATURES_ORDERFLOW_LINEAR_CACHE = Path("data/processed/features_orderflow_linear.parquet")

_VARIANTS = ("raw", "linear")      # "raw" for trees, "linear" (scale-stable) for logistic

_LAG_STEPS = [4, 3, 2, 1]          # same warm-up depth as v1/v2 (drop first 4 rows)
_VOL_WINDOW = 60                   # trailing window for the volume z-score
_TD_WINDOWS = [5, 10, 15]          # trailing windows for cumulative tick_delta
# Raw columns the indicators are built from.
_SRC_COLS = ["Open", "Close", "Volume", "Up Ticks", "Down Ticks", "Tick Count"]
# Base-indicator names, in emission order.
_INDICATORS = ["norm_vol", "signed_vol", "cum_td5", "cum_td10", "cum_td15"]


def _order_flow_indicators(filled: pd.DataFrame, variant: str = "raw") -> dict[str, pd.Series]:
    """Compute the five base order-flow indicators on a dense 1-min frame.

    All windows are trailing (causal) and may include the current bar; leakage
    protection comes from the lag step applied by the caller, never from these
    series directly.

    Args:
        filled: Dense 1-min grid frame (forward-filled) with at least the columns
            in ``_SRC_COLS``.
        variant: ``"raw"`` (signed_vol = Volume·sign(ΔO,C)) or ``"linear"``
            (signed_vol = sign(ΔO,C)·Volume / trailing-mean Volume — O(1) scale).
            Only ``signed_vol`` differs between variants.

    Returns:
        Mapping of indicator name -> dense pd.Series aligned to ``filled.index``.
    """
    vol = filled["Volume"].astype(float)
    roll_mean = vol.rolling(_VOL_WINDOW).mean()
    roll_std = vol.rolling(_VOL_WINDOW).std().replace(0.0, np.nan)
    norm_vol = (vol - roll_mean) / roll_std

    sign = np.sign(filled["Close"] - filled["Open"])
    if variant == "linear":
        # Signed *relative* volume: keeps direction + relative size on an O(1) scale.
        signed_vol = sign * vol / roll_mean.replace(0.0, np.nan)
    else:
        signed_vol = vol * sign

    tick_delta = (
        (filled["Up Ticks"] - filled["Down Ticks"])
        / filled["Tick Count"].replace(0, np.nan)
    )

    indicators: dict[str, pd.Series] = {
        "norm_vol": norm_vol,
        "signed_vol": signed_vol,
    }
    for w in _TD_WINDOWS:
        indicators[f"cum_td{w}"] = tick_delta.rolling(w).sum()
    return indicators


def build_features_orderflow(df: pd.DataFrame, variant: str = "raw") -> pd.DataFrame:
    """Build the 20-dim lagged order-flow feature matrix from the raw DataFrame.

    Follows the dense-grid pattern: expand to a 1-min grid → forward-fill gap
    minutes → compute indicators → lag each by 1..4 → project back to the original
    timestamps → drop the first 4 warm-up rows. Only lagged columns are emitted; no
    current-bar (lag-0) value enters any feature.

    Args:
        df: Raw DataFrame from ``load_raw()``, with ``'Date and Time'`` as a column
            and all schema columns present.
        variant: ``"raw"`` (default; ``signed_vol = Volume·sign(ΔO,C)`` — used by tree
            models) or ``"linear"`` (``signed_vol = sign(ΔO,C)·Volume / trailing-mean
            Volume`` — scale-stable, used by logistic models). Column names are
            identical for both variants; only ``signed_vol`` values differ.

    Returns:
        DataFrame of shape ``(len(df) - 4, 20)`` with a reset 0-based index. Columns
        are ``lag{lag}_{name}`` for lag in 4,3,2,1 and name in
        ``norm_vol, signed_vol, cum_td5, cum_td10, cum_td15``.

    Raises:
        ValueError: If ``variant`` is not in ``{"raw", "linear"}``.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VARIANTS}; got {variant!r}")
    src = df.set_index("Date and Time")[_SRC_COLS].copy()

    # -- 1. Dense 1-min grid with forward-fill ---------------------------------
    full_grid = pd.date_range(src.index[0], src.index[-1], freq="1min")
    filled = src.reindex(full_grid).ffill()

    # -- 2. Compute indicators on the dense grid -------------------------------
    indicators = _order_flow_indicators(filled, variant)

    # -- 3. Build lagged columns (only lagged; never lag-0) --------------------
    lag_data: dict[str, pd.Series] = {}
    for lag in _LAG_STEPS:
        for name in _INDICATORS:
            lag_data[f"lag{lag}_{name}"] = indicators[name].shift(lag)

    # -- 4. Project back to original timestamps --------------------------------
    features_grid = pd.DataFrame(lag_data, index=full_grid)
    features = features_grid.reindex(src.index)

    # -- 5. Drop first 4 rows (align with v1/v2) and reset index ---------------
    n_drop = len(_LAG_STEPS)  # 4
    features = features.iloc[n_drop:].reset_index(drop=True)

    # Forward-fill then zero-fill the leading warm-up NaNs from the rolling
    # windows (z-score needs 60 bars, cum_td needs up to 15) plus lag shift.
    n_nan = int(features.isna().sum().sum())
    if n_nan:
        features = features.ffill().fillna(0.0)

    print(
        f"features_orderflow [{variant}]: {features.shape[1]} features, "
        f"{features.shape[0]:,} rows  "
        f"(dropped {n_drop} warm-up rows, filled {n_nan} NaN cells)"
    )
    return features


def load_or_build_features_orderflow(df: pd.DataFrame, variant: str = "raw") -> pd.DataFrame:
    """Return the order-flow feature matrix, using the parquet cache when available.

    Cached per variant (``features_orderflow.parquet`` for ``"raw"``,
    ``features_orderflow_linear.parquet`` for ``"linear"``) for parity with the other
    feature modules. The build is cheap (vectorised rolling ops), but the cache keeps
    repeated walk-forward runs fast and consistent.

    Args:
        df: Raw DataFrame as returned by ``load_raw()``.
        variant: ``"raw"`` or ``"linear"`` (see ``build_features_orderflow``).

    Returns:
        The 20-feature order-flow matrix (same as ``build_features_orderflow()``).

    Raises:
        ValueError: If ``variant`` is not in ``{"raw", "linear"}``.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VARIANTS}; got {variant!r}")
    cache = (_FEATURES_ORDERFLOW_LINEAR_CACHE if variant == "linear"
             else _FEATURES_ORDERFLOW_CACHE)
    if cache.exists():
        print(f"  Loading cached features_orderflow [{variant}] from {cache}...")
        return pd.read_parquet(cache)
    print(f"  Building features_orderflow [{variant}] (will cache for future runs)...")
    features = build_features_orderflow(df, variant)
    cache.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(cache)
    print(f"  Cached to {cache}")
    return features
