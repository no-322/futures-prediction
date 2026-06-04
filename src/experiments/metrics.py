"""Evaluation metrics for the feature-engineering experiments.

Experiment 1 (three-class):  mcc, macro_f1, per_class_recall
Experiment 2 (two-stage):    coverage, conditional_hit_rate
Debug diagnostics:           gate_recall_debug, direction_mcc_debug
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    accuracy_score,
)


# ---------------------------------------------------------------------------
# Experiment 1 — three-class metrics
# ---------------------------------------------------------------------------

def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews Correlation Coefficient — handles binary and multiclass.

    MCC is robust to class imbalance; ranges from -1 (worst) to +1 (perfect).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        MCC as a float in [-1, 1].
    """
    return float(matthews_corrcoef(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 across all classes (unweighted mean per class).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Macro-F1 as a float in [0, 1].
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, float]:
    """Recall for every class present in y_true.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Dict mapping class int → recall float.
    """
    labels = sorted(set(y_true))
    _, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {int(lbl): float(r) for lbl, r in zip(labels, rec)}


# ---------------------------------------------------------------------------
# Experiment 2 — two-stage cascade metrics
# ---------------------------------------------------------------------------

def coverage(gate_pred: np.ndarray) -> float:
    """Fraction of bars the gate chose to trade.

    Args:
        gate_pred: Binary array of gate predictions (1 = trade, 0 = skip).

    Returns:
        Coverage as a float in [0, 1].
    """
    return float(gate_pred.mean())


def conditional_hit_rate(
    direction_true: np.ndarray,
    direction_pred: np.ndarray,
) -> float:
    """Directional accuracy among bars that the gate chose to trade.

    Args:
        direction_true: True direction labels for the gated subset.
        direction_pred: Predicted direction labels for the gated subset.

    Returns:
        Conditional hit rate as a float in [0, 1]; 0.0 if no bars traded.
    """
    if len(direction_true) == 0:
        return 0.0
    return float(accuracy_score(direction_true, direction_pred))


# ---------------------------------------------------------------------------
# Debug diagnostics
# ---------------------------------------------------------------------------

def gate_recall_debug(
    move_true: np.ndarray,
    threshold: float,
    gate_pred: np.ndarray,
) -> dict[str, float]:
    """Recall and precision of the gate on bars that genuinely moved.

    The oracle gate label is 1 where |move_true| > threshold. Comparing that
    to gate_pred gives a direct measure of how well the gate identifies
    truly tradeable bars.

    Args:
        move_true: True signed move (Close − Open) for the test fold.
        threshold: Threshold used to define tradeable bars.
        gate_pred: Gate model predictions (1 = trade, 0 = skip).

    Returns:
        Dict with keys: recall, precision, f1, n_truly_tradeable.
    """
    oracle = (np.abs(move_true) > threshold).astype(int)
    _, rec, f1, _ = precision_recall_fscore_support(
        oracle, gate_pred, labels=[1], zero_division=0
    )
    prec_arr, _, _, _ = precision_recall_fscore_support(
        oracle, gate_pred, labels=[1], zero_division=0
    )
    return {
        "recall":            float(rec[0]),
        "precision":         float(prec_arr[0]),
        "f1":                float(f1[0]),
        "n_truly_tradeable": int(oracle.sum()),
    }


def direction_mcc_debug(
    direction_true: np.ndarray,
    direction_pred_gated: np.ndarray,
    genuinely_moved_mask: np.ndarray,
    gated_mask: np.ndarray,
) -> float:
    """MCC for the direction model on bars that are BOTH gated AND genuinely moved.

    Isolates direction classifier quality from gate errors by intersecting the
    bars the gate chose to trade with bars that actually had a real move to
    predict. This gives the direction MCC on the hardest-to-fake subset.

    Args:
        direction_true: True direction labels for the full test fold.
        direction_pred_gated: Direction predictions for gate_pred==1 bars only
            (length == gated_mask.sum()).
        genuinely_moved_mask: Boolean mask where |move_true| > threshold.
        gated_mask: Boolean mask where gate_pred == 1.

    Returns:
        MCC as a float in [-1, 1]; 0.0 if intersection is empty.
    """
    if direction_pred_gated is None or len(direction_pred_gated) == 0:
        return 0.0

    # Intersection: bars the gate traded that also genuinely moved
    intersection_mask = gated_mask & genuinely_moved_mask
    if intersection_mask.sum() == 0:
        return 0.0

    # Map intersection back to indices within the gated subset
    truly_moved_within_gated = genuinely_moved_mask[gated_mask]
    true_subset = direction_true[intersection_mask]
    pred_subset = direction_pred_gated[truly_moved_within_gated]

    if len(true_subset) != len(pred_subset) or len(true_subset) == 0:
        return 0.0

    return float(matthews_corrcoef(true_subset, pred_subset))
