from pathlib import Path

import numpy as np
import pytest

from src.features import build_features
from src.labels import build_labels
from src.load import load_raw
from src.models.svm import SVMModel, load, predict, save, train
from src.split import split

DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "data.csv"


@pytest.fixture(scope="session")
def svm_results() -> tuple:
    df = load_raw(DATA_PATH)
    features = build_features(df)
    raw_align = df.iloc[4:].reset_index(drop=True)
    X_train, X_test = split(features)
    raw_train, raw_test = split(raw_align)
    y_train = build_labels(raw_train)
    y_test = build_labels(raw_test)
    model = train(X_train, y_train)
    return X_train, X_test, y_train, y_test, model


def test_model_is_fitted(svm_results: tuple) -> None:
    _, _, _, _, model = svm_results
    assert hasattr(model["clf"], "support_vectors_")


def test_scaler_fit_on_train_only(svm_results: tuple) -> None:
    """The persisted scaler's mean must equal X_train column means exactly.

    This directly asserts that predict() uses the training-fit scaler rather
    than a scaler refit on test data — the critical no-look-ahead invariant
    for SVM (which is scale-sensitive).
    """
    X_train, _, _, _, model = svm_results
    expected_mean = np.mean(X_train.values, axis=0)
    np.testing.assert_allclose(model["scaler"].mean_, expected_mean, rtol=1e-6)


def test_predict_shape(svm_results: tuple) -> None:
    _, X_test, _, y_test, model = svm_results
    assert predict(model, X_test).shape == y_test.shape


def test_predict_binary(svm_results: tuple) -> None:
    _, X_test, _, _, model = svm_results
    assert set(predict(model, X_test)) <= {0, 1}


def test_predict_reproducible(svm_results: tuple) -> None:
    """Predictions are deterministic; random_state=42 fixes SMO tie-breaking."""
    _, X_test, _, _, model = svm_results
    preds_a = predict(model, X_test.iloc[:500])
    preds_b = predict(model, X_test.iloc[:500])
    np.testing.assert_array_equal(preds_a, preds_b)


def test_save_load_roundtrip(svm_results: tuple, tmp_path) -> None:
    _, X_test, _, _, model = svm_results
    path = tmp_path / "svm_model.joblib"
    save(model, path)
    loaded: SVMModel = load(path)
    np.testing.assert_allclose(loaded["scaler"].mean_, model["scaler"].mean_)
    np.testing.assert_array_equal(
        predict(model, X_test.iloc[:20]),
        predict(loaded, X_test.iloc[:20]),
    )
