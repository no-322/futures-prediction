from pathlib import Path

import pandas as pd

REQUIRED_COLS: list[str] = [
    "Date and Time",
    "Symbol",
    "Open",
    "High",
    "Low",
    "Close",
    "VWAP",
]


def load_raw(path: Path | str) -> pd.DataFrame:
    """Read the raw CSV and parse the primary timestamp column.

    Args:
        path: Path to the raw CSV file (data/raw/data.csv).

    Returns:
        DataFrame with "Date and Time" as datetime64, all other columns
        unchanged. Rows are not dropped or reordered.
    """
    df = pd.read_csv(path)
    df["Date and Time"] = pd.to_datetime(df["Date and Time"])
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def validate(df: pd.DataFrame) -> None:
    """Assert required columns are present and log NaN counts.

    Args:
        df: DataFrame returned by load_raw.

    Raises:
        ValueError: If any required column is absent.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in REQUIRED_COLS:
        n_nan = int(df[col].isna().sum())
        print(f"  NaN in {col!r}: {n_nan}")

    if not df["Date and Time"].is_monotonic_increasing:
        print("WARNING: 'Date and Time' is not monotonically increasing")


if __name__ == "__main__":
    df = load_raw("data/raw/data.csv")
    validate(df)
