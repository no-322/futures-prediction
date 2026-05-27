from pathlib import Path

import pandas as pd
import pytest

from src.features import FEATURE_COLS, LAG_STEPS, N_FEATURES, build_features
from src.load import load_raw
from src.split import split

DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "data.csv"


@pytest.fixture(scope="session")
def raw_and_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_raw(DATA_PATH)
    features = build_features(df)
    return df, features


def test_row_count(raw_and_features: tuple) -> None:
    df, features = raw_and_features
    assert len(features) + 4 == len(df)


def test_column_count(raw_and_features: tuple) -> None:
    _, features = raw_and_features
    assert features.shape[1] == N_FEATURES


def test_no_nulls(raw_and_features: tuple) -> None:
    _, features = raw_and_features
    assert features.isna().sum().sum() == 0


def test_column_names(raw_and_features: tuple) -> None:
    _, features = raw_and_features
    expected = [f"lag{lag}_{col}" for lag in LAG_STEPS for col in FEATURE_COLS]
    assert list(features.columns) == expected


def test_index_reset(raw_and_features: tuple) -> None:
    _, features = raw_and_features
    assert list(features.index) == list(range(len(features)))
