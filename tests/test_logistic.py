from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import build_features
from src.labels import build_labels
from src.load import load_raw
from src.models.logistic import load, predict, save, train
from src.split import split

DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "data.csv"


@pytest.fixture(scope="session")
def logistic_results() -> tuple:
    df = load_raw(DATA_PATH)
    features = build_features(df)
    raw_align = df.iloc[4:].reset_index(drop=True)
    X_train, X_test = split(features)
    raw_train, _ = split(raw_align)
    y_train = build_labels(raw_train)
    model = train(X_train, y_train)
    return X_train, X_test, y_train, model


def test_model_is_fitted(logistic_results: tuple) -> None:
    _, _, _, model = logistic_results
    assert hasattr(model, "classes_")


def test_predict_shape(logistic_results: tuple) -> None:
    _, X_test, _, model = logistic_results
    assert len(predict(model, X_test)) == len(X_test)


def test_predict_binary(logistic_results: tuple) -> None:
    _, X_test, _, model = logistic_results
    assert set(predict(model, X_test)) <= {0, 1}


def test_save_load_roundtrip(logistic_results: tuple, tmp_path) -> None:
    _, X_test, _, model = logistic_results
    path = tmp_path / "logistic_model.joblib"
    save(model, path)
    loaded = load(path)
    np.testing.assert_array_equal(
        predict(model, X_test.iloc[:20]), predict(loaded, X_test.iloc[:20]))


def test_train_accepts_sample_weight(tmp_path) -> None:
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(80, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X["f0"] + 0.1 * rng.randn(80) > 0).astype(int))
    w = np.abs(rng.randn(80)) + 0.1
    model = train(X, y, save_path=tmp_path / "m.joblib", sample_weight=w)
    preds = predict(model, X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)) <= {0, 1}
