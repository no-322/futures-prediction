import pandas as pd


def build_labels(df: pd.DataFrame) -> pd.Series:
    """Compute binary up/down labels aligned with build_features output.

    Args:
        df: DataFrame as returned by split(), with columns Close and Open.

    Returns:
        Series of int (0 or 1) with shape (len(df) - 4,) and reset 0-based
        index. 1 means Close > Open for that row; 0 otherwise. The first
        4 rows are dropped to match the output of build_features().
    """
    labels = (df["Close"] > df["Open"]).astype(int)
    return labels.iloc[4:].reset_index(drop=True)


if __name__ == "__main__":
    from pathlib import Path

    from src.load import load_raw
    from src.split import split

    df = load_raw(Path("data/raw/data.csv"))
    train, _ = split(df)
    labels = build_labels(train)
    print(f"Labels: {len(labels):,} rows")
    print(labels.value_counts().rename({1: "Up (1)", 0: "Down (0)"}))
