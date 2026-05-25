import pandas as pd


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into train and test halves.

    The first 50% of rows by timestamp become the training set; the
    remaining 50% become the test set. No shuffling is performed.

    Args:
        df: DataFrame with a monotonically increasing "Date and Time" column,
            as returned by load_raw.

    Returns:
        (train, test) — two non-overlapping DataFrames with reset indices.

    Raises:
        ValueError: If "Date and Time" is not monotonically increasing.
    """
    if not df["Date and Time"].is_monotonic_increasing:
        raise ValueError("'Date and Time' must be monotonically increasing before splitting.")

    mid = len(df) // 2
    train = df.iloc[:mid].reset_index(drop=True)
    test = df.iloc[mid:].reset_index(drop=True)

    print(f"Train: {len(train):,} rows | {train['Date and Time'].min()} → {train['Date and Time'].max()}")
    print(f"Test:  {len(test):,} rows | {test['Date and Time'].min()} → {test['Date and Time'].max()}")

    return train, test


if __name__ == "__main__":
    from pathlib import Path

    from src.load import load_raw

    df = load_raw(Path("data/raw/data.csv"))
    split(df)
