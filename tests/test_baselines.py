import numpy as np
import pandas as pd

from src.baselines import predict_always_up, predict_last_direction


def test_predict_always_up_shape_and_values() -> None:
    preds = predict_always_up(50)
    assert len(preds) == 50
    assert np.all(preds == 1)


def test_predict_last_direction_shape() -> None:
    y_train = pd.Series([0, 1, 1, 0])
    y_test = pd.Series([1, 0, 1])
    assert len(predict_last_direction(y_train, y_test)) == len(y_test)


def test_predict_last_direction_first_value() -> None:
    y_train = pd.Series([0, 1, 1, 0])          # last train label = 0
    y_test = pd.Series([1, 0, 1])
    preds = predict_last_direction(y_train, y_test)
    assert preds[0] == 0


def test_predict_last_direction_follows_previous() -> None:
    y_train = pd.Series([0, 1])
    y_test = pd.Series([1, 0, 1, 1, 0])
    preds = predict_last_direction(y_train, y_test)
    # each row = the previous test row's actual label
    for i in range(1, len(y_test)):
        assert preds[i] == int(y_test.iloc[i - 1]), f"mismatch at {i}"
