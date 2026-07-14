"""Naive comparison baselines (not learned models).

Two trivial forecasters every model is measured against:
  - always-up: predict 1 (up) for every bar.
  - last-direction: predict the previous bar's realised direction (lag-1 persistence).
Both are look-ahead free.
"""
import numpy as np
import pandas as pd


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

    For test row 0, predicts the last label seen in training. For each subsequent
    row i, predicts the actual label of the previous test row (i-1) — a naive lag-1
    forecaster with no look-ahead.

    Args:
        y_train: Training labels from build_labels(), shape (n_train,).
        y_test: Test labels from build_labels(), shape (n_test,).

    Returns:
        Integer ndarray of predicted labels (0 or 1), shape (n_test,).
    """
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)
    preds = np.empty(len(y_test), dtype=int)
    preds[0] = int(y_train.iloc[-1])
    preds[1:] = y_test.values[:-1]
    return preds
