import pandas as pd


def build_labels(df: pd.DataFrame) -> pd.Series:
    """Compute binary up/down labels for every row in df.

    Args:
        df: DataFrame with columns Close and Open. In the canonical pipeline
            this is the training slice of the aligned raw DataFrame
            (df.iloc[4:] after build_features has been called), so the index
            aligns directly with the output of build_features().

    Returns:
        Series of int (0 or 1) with the same length as df and a reset 0-based
        index. 1 means Close > Open for that row; 0 otherwise.
    """
    return (df["Close"] > df["Open"]).astype(int).reset_index(drop=True)


if __name__ == "__main__":
    from pathlib import Path

    from src.features import build_features
    from src.load import load_raw
    from src.split import split

    df = load_raw(Path("data/raw/data.csv"))
    features = build_features(df)
    raw_align = df.iloc[4:].reset_index(drop=True)
    _, _ = split(features)
    raw_train, _ = split(raw_align)
    labels = build_labels(raw_train)
    print(f"Labels: {len(labels):,} rows")
    print(labels.value_counts().rename({1: "Up (1)", 0: "Down (0)"}))
