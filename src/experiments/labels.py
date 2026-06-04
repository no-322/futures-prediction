"""Label factories for feature-engineering experiments.

All functions accept `raw_align` — the raw DataFrame after dropping the first 4
rows (`df.iloc[4:].reset_index(drop=True)`), whose index aligns 1-to-1 with the
feature matrix returned by build_features().
"""
import numpy as np
import pandas as pd


def move_series(raw_align: pd.DataFrame) -> pd.Series:
    """Return the signed intrabar move (Close − Open) for each bar.

    Args:
        raw_align: Raw DataFrame aligned to the feature matrix (df.iloc[4:]).

    Returns:
        Float Series of Close − Open, same length and index as raw_align.
    """
    return (raw_align["Close"] - raw_align["Open"]).reset_index(drop=True)


def three_class_labels(raw_align: pd.DataFrame) -> pd.Series:
    """Assign a three-class direction label to each bar.

    Classes:
        0 — down  (Close < Open)
        1 — up    (Close > Open)
        2 — flat  (Close == Open)

    Note: flat bars are extremely rare for TY futures (tick size 1/32);
    expect heavy class imbalance on class 2.

    Args:
        raw_align: Raw DataFrame aligned to the feature matrix (df.iloc[4:]).

    Returns:
        Int Series of labels {0, 1, 2}, same length and index as raw_align.
    """
    move = move_series(raw_align)
    labels = pd.Series(np.where(move > 0, 1, np.where(move < 0, 0, 2)),
                       dtype=int)
    return labels


def gate_labels(move: pd.Series, threshold: float) -> pd.Series:
    """Label each bar as tradeable (1) or not (0) based on absolute move size.

    Args:
        move: Signed move series from move_series().
        threshold: Minimum absolute move (in points) to be considered tradeable.

    Returns:
        Int Series of {0, 1}, same length and index as move.
    """
    return (move.abs() > threshold).astype(int).reset_index(drop=True)


def direction_labels(raw_align: pd.DataFrame) -> pd.Series:
    """Binary direction label: 1 if Close > Open, 0 otherwise.

    Used as the Stage-2 target in the two-stage cascade (applied only to
    bars that pass the gate).

    Args:
        raw_align: Raw DataFrame aligned to the feature matrix (df.iloc[4:]).

    Returns:
        Int Series of {0, 1}, same length and index as raw_align.
    """
    return (raw_align["Close"] > raw_align["Open"]).astype(int).reset_index(drop=True)
