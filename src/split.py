import pandas as pd


def split(
    df: pd.DataFrame,
    train_size: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into train and test portions.

    The first `train_size` fraction of rows become the training set; the
    remainder become the test set. No shuffling is performed. Accepts either
    the raw DataFrame (with a "Date and Time" column) or a feature matrix
    (integer index only). Timestamp monotonicity is validated only when
    "Date and Time" is present.

    Args:
        df: Any time-ordered DataFrame — raw rows from load_raw(), an aligned
            raw slice, or a feature matrix from build_features().
        train_size: Fraction of rows for training (default 0.5). Must be in
            (0, 1).

    Returns:
        (train, test) — two non-overlapping DataFrames with reset indices.

    Raises:
        ValueError: If "Date and Time" is present but not monotonically
            increasing, or if train_size is not in (0, 1).
    """
    if not 0 < train_size < 1:
        raise ValueError(f"train_size must be in (0, 1); got {train_size}")

    if "Date and Time" in df.columns:
        if not df["Date and Time"].is_monotonic_increasing:
            raise ValueError("'Date and Time' must be monotonically increasing before splitting.")

    mid = int(len(df) * train_size)
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
