import pandas as pd


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into train and test halves.

    The first 50% of rows become the training set; the remaining 50% become
    the test set. No shuffling is performed. Accepts either the raw DataFrame
    (with a "Date and Time" column) or a feature matrix (integer index only).
    Timestamp monotonicity is validated only when "Date and Time" is present.

    Args:
        df: Any time-ordered DataFrame — raw rows from load_raw(), an aligned
            raw slice, or a feature matrix from build_features().

    Returns:
        (train, test) — two non-overlapping DataFrames with reset indices.

    Raises:
        ValueError: If "Date and Time" is present but not monotonically
            increasing.
    """
    if "Date and Time" in df.columns:
        if not df["Date and Time"].is_monotonic_increasing:
            raise ValueError("'Date and Time' must be monotonically increasing before splitting.")

    mid = len(df) // 2
    train = df.iloc[:mid].reset_index(drop=True)
    test = df.iloc[mid:].reset_index(drop=True)

    if "Date and Time" in df.columns:
        print(f"Train: {len(train):,} rows | {train['Date and Time'].min()} → {train['Date and Time'].max()}")
        print(f"Test:  {len(test):,} rows | {test['Date and Time'].min()} → {test['Date and Time'].max()}")
    else:
        print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")

    return train, test


if __name__ == "__main__":
    from pathlib import Path

    from src.load import load_raw

    df = load_raw(Path("data/raw/data.csv"))
    split(df)
