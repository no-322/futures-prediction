from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_DEFAULT_PATH = Path("data/processed/baseline_model.joblib")


def train(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """Fit a logistic regression classifier on the training feature matrix.

    Args:
        X: Feature matrix from build_features(), shape (n_samples, 20).
        y: Binary labels from build_labels(), shape (n_samples,).

    Returns:
        Fitted LogisticRegression instance.
    """
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    return model


def predict(model: LogisticRegression, X: pd.DataFrame) -> np.ndarray:
    """Predict class labels for the test feature matrix.

    Args:
        model: Fitted LogisticRegression returned by train().
        X: Feature matrix, shape (n_samples, 20).

    Returns:
        Integer ndarray of predicted labels (0 or 1), shape (n_samples,).
    """
    return model.predict(X)


def save(model: LogisticRegression, path: Path = _DEFAULT_PATH) -> None:
    """Serialize a fitted LogisticRegression to disk with joblib.

    Args:
        model: Fitted LogisticRegression returned by train().
        path: Destination file path; parent directories are created if absent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: Path = _DEFAULT_PATH) -> LogisticRegression:
    """Deserialize a LogisticRegression previously saved by save().

    Args:
        path: Path to the .joblib file written by save().

    Returns:
        Fitted LogisticRegression identical in state to the original train() output.
    """
    return joblib.load(Path(path))


def predict_always_up(n: int) -> np.ndarray:
    """Baseline: predict up (1) for every row.

    Args:
        n: Number of predictions to return.

    Returns:
        Integer ndarray of ones, shape (n,).
    """
    return np.ones(n, dtype=int)


def predict_last_direction(y_train: pd.Series, y_test: pd.Series) -> np.ndarray:
    """Baseline: predict the last observed direction for each test row.

    For test row 0, predicts the last label seen in training. For each
    subsequent row i, predicts the actual label of the previous test row
    (i-1). This simulates a naive lag-1 forecaster with no look-ahead.

    Args:
        y_train: Training labels from build_labels(), shape (n_train,).
        y_test: Test labels from build_labels(), shape (n_test,).

    Returns:
        Integer ndarray of predicted labels (0 or 1), shape (n_test,).
    """
    preds = np.empty(len(y_test), dtype=int)
    preds[0] = int(y_train.iloc[-1])
    preds[1:] = y_test.values[:-1]
    return preds


if __name__ == "__main__":
    from pathlib import Path

    from src.features import build_features
    from src.labels import build_labels
    from src.load import load_raw
    from src.split import split

    df = load_raw(Path("data/raw/data.csv"))
    features = build_features(df)
    raw_align = df.iloc[4:].reset_index(drop=True)
    X_train, X_test = split(features)
    raw_train, raw_test = split(raw_align)
    y_train = build_labels(raw_train)
    y_test = build_labels(raw_test)

    model = train(X_train, y_train)

    lr_preds = predict(model, X_test)
    always_up = predict_always_up(len(X_test))
    last_dir = predict_last_direction(y_train, y_test)

    print(f"Logistic regression — predicted up: {lr_preds.sum():,} / {len(lr_preds):,}")
    print(f"Always up           — predicted up: {always_up.sum():,} / {len(always_up):,}")
    print(f"Last direction      — predicted up: {last_dir.sum():,} / {len(last_dir):,}")
