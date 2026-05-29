from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import build_features
from src.labels import build_labels
from src.load import load_raw
from src.models.baseline import (
    load,
    predict,
    predict_always_up,
    predict_last_direction,
    save,
    train,
)
from src.split import split

DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "data.csv"


@pytest.fixture(scope="session")
def baseline_results() -> tuple:
    df = load_raw(DATA_PATH)
    features = build_features(df)
    raw_align = df.iloc[4:].reset_index(drop=True)
    X_train, X_test = split(features)
    raw_train, raw_test = split(raw_align)
    y_train = build_labels(raw_train)
    y_test = build_labels(raw_test)
    model = train(X_train, y_train)
    return X_train, X_test, y_train, y_test, model


def test_model_is_fitted(baseline_results: tuple) -> None:
    _, _, _, _, model = baseline_results
    assert hasattr(model, "classes_")


def test_predict_shape(baseline_results: tuple) -> None:
    _, X_test, _, _, model = baseline_results
    assert len(predict(model, X_test)) == len(X_test)


def test_predict_binary(baseline_results: tuple) -> None:
    _, X_test, _, _, model = baseline_results
    assert set(predict(model, X_test)) <= {0, 1}


def test_predict_always_up_shape(baseline_results: tuple) -> None:
    _, X_test, _, _, _ = baseline_results
    assert len(predict_always_up(len(X_test))) == len(X_test)


def test_predict_always_up_values(baseline_results: tuple) -> None:
    _, X_test, _, _, _ = baseline_results
    preds = predict_always_up(len(X_test))
    assert np.all(preds == 1)


def test_predict_last_direction_shape(baseline_results: tuple) -> None:
    _, _, y_train, y_test, _ = baseline_results
    assert len(predict_last_direction(y_train, y_test)) == len(y_test)


def test_predict_last_direction_first_value(baseline_results: tuple) -> None:
    _, _, y_train, y_test, _ = baseline_results
    preds = predict_last_direction(y_train, y_test)
    assert preds[0] == int(y_train.iloc[-1])


def test_predict_last_direction_follows_previous(baseline_results: tuple) -> None:
    _, _, y_train, y_test, _ = baseline_results
    preds = predict_last_direction(y_train, y_test)
    for i in range(1, 11):
        assert preds[i] == int(y_test.iloc[i - 1]), f"Mismatch at index {i}"


def test_save_load_roundtrip(baseline_results: tuple, tmp_path) -> None:
    _, X_test, _, _, model = baseline_results
    path = tmp_path / "baseline_model.joblib"
    save(model, path)
    loaded = load(path)
    np.testing.assert_array_equal(predict(model, X_test.iloc[:20]), predict(loaded, X_test.iloc[:20]))
