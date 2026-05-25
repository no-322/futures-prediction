from pathlib import Path

import pandas as pd
import pytest

from src.load import load_raw

DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "data.csv"
FEATURE_COLS = ["Open", "Close", "High", "Low", "VWAP"]


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return load_raw(DATA_PATH)


def test_nonempty(raw_df: pd.DataFrame) -> None:
    assert len(raw_df) > 0


def test_required_feature_columns_present(raw_df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_COLS if c not in raw_df.columns]
    assert not missing, f"Missing feature columns: {missing}"


def test_timestamp_column_present(raw_df: pd.DataFrame) -> None:
    assert "Date and Time" in raw_df.columns


def test_symbol_column_present(raw_df: pd.DataFrame) -> None:
    assert "Symbol" in raw_df.columns


def test_timestamp_is_datetime(raw_df: pd.DataFrame) -> None:
    assert pd.api.types.is_datetime64_any_dtype(raw_df["Date and Time"])


def test_feature_columns_are_float(raw_df: pd.DataFrame) -> None:
    for col in FEATURE_COLS:
        assert pd.api.types.is_float_dtype(raw_df[col]), f"{col} is not float64"


def test_no_nan_in_feature_columns(raw_df: pd.DataFrame) -> None:
    for col in FEATURE_COLS:
        assert raw_df[col].isna().sum() == 0, f"{col} has NaN values"


def test_no_nan_in_timestamp(raw_df: pd.DataFrame) -> None:
    assert raw_df["Date and Time"].isna().sum() == 0


def test_timestamps_monotonic(raw_df: pd.DataFrame) -> None:
    assert raw_df["Date and Time"].is_monotonic_increasing
