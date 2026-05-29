from pathlib import Path

import numpy as np
import pytest
from xgboost import XGBClassifier

from src.features import build_features
from src.labels import build_labels
from src.load import load_raw
from src.models.gbm import load, predict, save, train
from src.split import split

DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "data.csv"


@pytest.fixture(scope="session")
def gbm_results() -> tuple:
    df = load_raw(DATA_PATH)
    features = build_features(df)
    raw_align = df.iloc[4:].reset_index(drop=True)
    X_train, X_test = split(features)
    raw_train, raw_test = split(raw_align)
    y_train = build_labels(raw_train)
    y_test = build_labels(raw_test)
    model = train(X_train, y_train)
    return X_train, X_test, y_train, y_test, model


def test_model_is_fitted(gbm_results: tuple) -> None:
    _, _, _, _, model = gbm_results
    assert hasattr(model, "feature_importances_")


def test_predict_shape(gbm_results: tuple) -> None:
    _, X_test, _, y_test, model = gbm_results
    assert predict(model, X_test).shape == y_test.shape


def test_predict_binary(gbm_results: tuple) -> None:
    _, X_test, _, _, model = gbm_results
    assert set(predict(model, X_test)) <= {0, 1}


def test_predict_reproducible(gbm_results: tuple) -> None:
    """Predictions are deterministic given random_state=42."""
    _, X_test, _, _, model = gbm_results
    preds_a = predict(model, X_test.iloc[:500])
    preds_b = predict(model, X_test.iloc[:500])
    np.testing.assert_array_equal(preds_a, preds_b)


def test_save_load_roundtrip(gbm_results: tuple, tmp_path) -> None:
    _, X_test, _, _, model = gbm_results
    path = tmp_path / "gbm_model.joblib"
    save(model, path)
    loaded: XGBClassifier = load(path)
    np.testing.assert_array_equal(
        predict(model, X_test.iloc[:20]),
        predict(loaded, X_test.iloc[:20]),
    )
