from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import build_features
from src.labels import build_labels
from src.load import load_raw
from src.models.rf import predict, train
from src.split import split

DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "data.csv"


@pytest.fixture(scope="session")
def rf_results() -> tuple:
    df = load_raw(DATA_PATH)
    features = build_features(df)
    raw_align = df.iloc[4:].reset_index(drop=True)
    X_train, X_test = split(features)
    raw_train, raw_test = split(raw_align)
    y_train = build_labels(raw_train)
    y_test = build_labels(raw_test)
    model = train(X_train, y_train)
    return X_train, X_test, y_train, y_test, model


def test_model_is_fitted(rf_results: tuple) -> None:
    _, _, _, _, model = rf_results
    assert hasattr(model, "estimators_")


def test_oob_score_available(rf_results: tuple) -> None:
    _, _, _, _, model = rf_results
    assert 0.0 <= model.oob_score_ <= 1.0


def test_predict_shape(rf_results: tuple) -> None:
    _, X_test, _, _, model = rf_results
    assert len(predict(model, X_test)) == len(X_test)


def test_predict_binary(rf_results: tuple) -> None:
    _, X_test, _, _, model = rf_results
    assert set(predict(model, X_test)) <= {0, 1}
