"""Logistic Regression — the project's primary classifier.

L2 logistic regression on the lagged feature matrix. Promoted from the former
``baseline`` module: it is the strongest linear model here and the reference the
tree models must beat. Scale features first when a pipeline's columns are not
already O(1) (fit the scaler on train only).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_DEFAULT_PATH = Path("data/processed/logistic_model.joblib")


def train(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    save_path: Path | None = None,
    sample_weight: np.ndarray | None = None,
) -> LogisticRegression:
    """Fit a logistic regression classifier on the training feature matrix.

    Args:
        X: Feature matrix (rows = decisive bars, columns = a feature pipeline's lags).
        y: Binary labels from build_labels(), shape (n_samples,).
        params: Optional hyperparameter overrides from config.model_params().
            random_state is always 42.
        save_path: Where to persist the fitted model. Defaults to
            data/processed/logistic_model.joblib.
        sample_weight: Optional per-row training weights. None gives the standard
            unweighted fit.

    Returns:
        Fitted LogisticRegression instance.
    """
    p: dict = {"max_iter": 1000}
    if params:
        p.update(params)
    p["random_state"] = 42
    model = LogisticRegression(**p)
    model.fit(X, y, sample_weight=sample_weight)
    save(model, save_path or _DEFAULT_PATH)
    return model


def predict(model: LogisticRegression, X: pd.DataFrame) -> np.ndarray:
    """Predict class labels for the feature matrix.

    Args:
        model: Fitted LogisticRegression returned by train().
        X: Feature matrix.

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
