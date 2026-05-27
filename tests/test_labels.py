from pathlib import Path

import pandas as pd
import pytest

from src.features import build_features
from src.labels import build_labels
from src.load import load_raw
from src.split import split

DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "data.csv"


@pytest.fixture(scope="session")
def pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = load_raw(DATA_PATH)
    features = build_features(df)
    raw_align = df.iloc[4:].reset_index(drop=True)
    X_train, _ = split(features)
    raw_train, _ = split(raw_align)
    labels = build_labels(raw_train)
    return X_train, raw_train, labels


def test_row_count(pipeline: tuple) -> None:
    _, raw_train, labels = pipeline
    assert len(labels) == len(raw_train)


def test_binary_values(pipeline: tuple) -> None:
    _, _, labels = pipeline
    assert set(labels.unique()) <= {0, 1}


def test_no_nulls(pipeline: tuple) -> None:
    _, _, labels = pipeline
    assert labels.isna().sum() == 0


def test_index_reset(pipeline: tuple) -> None:
    _, _, labels = pipeline
    assert list(labels.index) == list(range(len(labels)))


def test_label_correctness(pipeline: tuple) -> None:
    _, raw_train, labels = pipeline
    # labels[i] corresponds to raw_train.iloc[i] (no offset — alignment is caller's responsibility)
    for i in range(10):
        row = raw_train.iloc[i]
        expected = 1 if row["Close"] > row["Open"] else 0
        assert labels.iloc[i] == expected, f"Label mismatch at index {i}"


def test_alignment_with_features(pipeline: tuple) -> None:
    X_train, _, labels = pipeline
    assert len(X_train) == len(labels)
