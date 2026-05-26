from pathlib import Path

import pandas as pd
import pytest

from src.features import build_features
from src.labels import build_labels
from src.load import load_raw
from src.split import split

DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "data.csv"


@pytest.fixture(scope="session")
def train_and_labels() -> tuple[pd.DataFrame, pd.Series]:
    df = load_raw(DATA_PATH)
    train, _ = split(df)
    return train, build_labels(train)


def test_row_count(train_and_labels: tuple) -> None:
    train, labels = train_and_labels
    assert len(labels) + 4 == len(train)


def test_binary_values(train_and_labels: tuple) -> None:
    _, labels = train_and_labels
    assert set(labels.unique()) <= {0, 1}


def test_no_nulls(train_and_labels: tuple) -> None:
    _, labels = train_and_labels
    assert labels.isna().sum() == 0


def test_index_reset(train_and_labels: tuple) -> None:
    _, labels = train_and_labels
    assert list(labels.index) == list(range(len(labels)))


def test_label_correctness(train_and_labels: tuple) -> None:
    train, labels = train_and_labels
    # Check first 10 labels against the original rows (offset by 4)
    for i in range(10):
        row = train.iloc[i + 4]
        expected = 1 if row["Close"] > row["Open"] else 0
        assert labels.iloc[i] == expected, f"Label mismatch at index {i}"


def test_alignment_with_features(train_and_labels: tuple) -> None:
    train, labels = train_and_labels
    features = build_features(train)
    assert len(features) == len(labels)
