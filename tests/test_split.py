from pathlib import Path

import pandas as pd
import pytest

from src.load import load_raw
from src.split import split

DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "data.csv"


@pytest.fixture(scope="session")
def dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_raw(DATA_PATH)
    train, test = split(df)
    return df, train, test


def test_row_counts_sum_to_total(dataframes: tuple) -> None:
    df, train, test = dataframes
    assert len(train) + len(test) == len(df)


def test_train_is_first_half(dataframes: tuple) -> None:
    df, train, test = dataframes
    assert len(train) == len(df) // 2


def test_test_is_second_half(dataframes: tuple) -> None:
    df, train, test = dataframes
    assert len(test) == len(df) - len(df) // 2


def test_no_overlapping_timestamps(dataframes: tuple) -> None:
    _, train, test = dataframes
    assert train["Date and Time"].max() < test["Date and Time"].min()


def test_train_before_test(dataframes: tuple) -> None:
    _, train, test = dataframes
    assert train["Date and Time"].max() < test["Date and Time"].min()


def test_indices_reset(dataframes: tuple) -> None:
    _, train, test = dataframes
    assert list(train.index) == list(range(len(train)))
    assert list(test.index) == list(range(len(test)))


def test_reproducible(dataframes: tuple) -> None:
    df, train1, test1 = dataframes
    train2, test2 = split(df)
    assert train1.equals(train2)
    assert test1.equals(test2)


def test_split_feature_matrix() -> None:
    from src.features import build_features
    df = load_raw(DATA_PATH)
    features = build_features(df)
    f_train, f_test = split(features)
    assert len(f_train) + len(f_test) == len(features)
    assert list(f_train.index) == list(range(len(f_train)))
    assert list(f_test.index) == list(range(len(f_test)))
