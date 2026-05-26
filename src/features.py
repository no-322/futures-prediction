import pandas as pd

FEATURE_COLS: list[str] = ["Open", "Close", "High", "Low", "VWAP"]
LAG_STEPS: list[int] = [4, 3, 2, 1]
N_FEATURES: int = len(FEATURE_COLS) * len(LAG_STEPS)  # 20


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a 20-dimensional lagged feature matrix from a time-ordered DataFrame.

    For each row t, collects [Open, Close, High, Low, VWAP] from the four
    preceding minutes (t-4, t-3, t-2, t-1). Gaps in the 1-minute grid (overnight
    sessions, weekends) are filled forward with the most-recent available bar so
    that shift arithmetic operates on true clock-minute offsets, not row offsets.
    The first four rows are dropped because they have no full lag history.

    Args:
        df: DataFrame as returned by split(), with a monotonically increasing
            "Date and Time" column and columns Open, Close, High, Low, VWAP.

    Returns:
        DataFrame of shape (len(df) - 4, 20) with columns ordered as
        [lag4_Open, lag4_Close, lag4_High, lag4_Low, lag4_VWAP,
         lag3_Open, ..., lag1_VWAP]. Index is reset to 0-based integers.
    """
    ohlcv = df.set_index("Date and Time")[FEATURE_COLS]

    # Expand to a dense 1-minute grid and forward-fill gap minutes
    full_grid = pd.date_range(ohlcv.index[0], ohlcv.index[-1], freq="1min")
    filled = ohlcv.reindex(full_grid).ffill()

    # Build all lag columns on the dense grid (shift(n) = n calendar minutes)
    lag_data: dict[str, pd.Series] = {}
    for lag in LAG_STEPS:
        for col in FEATURE_COLS:
            lag_data[f"lag{lag}_{col}"] = filled[col].shift(lag)

    features_grid = pd.DataFrame(lag_data, index=full_grid)

    # Project back to original timestamps only
    features = features_grid.reindex(ohlcv.index)

    n_nan_rows = int(features.isna().any(axis=1).sum())
    print(f"Dropped {n_nan_rows} NaN rows from lagging (first {n_nan_rows} original rows)")

    features = features.iloc[4:].reset_index(drop=True)

    assert features.shape[1] == N_FEATURES, (
        f"Expected {N_FEATURES} feature columns, got {features.shape[1]}"
    )
    return features


if __name__ == "__main__":
    from pathlib import Path

    from src.load import load_raw
    from src.split import split

    df = load_raw(Path("data/raw/data.csv"))
    train, _ = split(df)
    features = build_features(train)
    print(f"Feature matrix: {features.shape}")
    print(features.head())
