"""Standardised evaluation statistics for all classifiers.

Works for binary and multi-class predictions. Returns a structured
StatsResult TypedDict covering accuracy, macro/weighted F1, MCC, and
full per-class precision/recall/F1/support.

Usage::

    from src.statistics import compute, format_markdown, write_results
    result = compute(y_true, y_pred, name="Random Forest")
    print(format_markdown(result))
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class ClassMetrics(TypedDict):
    """Metrics for a single class."""

    precision: float
    recall:    float
    f1:        float
    support:   int


class StatsResult(TypedDict):
    """Complete evaluation statistics for one model run."""

    name:             str
    n_samples:        int
    accuracy:         float
    macro_f1:         float        # unweighted mean F1 across classes
    weighted_f1:      float        # support-weighted mean F1
    mcc:              float        # Matthews Correlation Coefficient
    classes:          list[int]    # sorted unique class labels
    per_class:        dict[int, ClassMetrics]
    confusion_matrix: list[list[int]]   # rows=actual, cols=predicted


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "",
    labels: list[int] | None = None,
) -> StatsResult:
    """Compute the full statistics report for one model.

    Works for binary and multi-class predictions. Uses only data from y_true
    and y_pred — no model loading is performed here.

    Args:
        y_true: Ground-truth class labels, shape (n,).
        y_pred: Predicted class labels, shape (n,).
        name: Display name for this model (used in formatted reports).
        labels: Explicit list of class labels. If None, uses sorted unique
            values from y_true. Useful when a class may be absent from
            a fold's test set.

    Returns:
        StatsResult TypedDict with all scalar and per-class metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    class_labels: list[int] = (
        labels if labels is not None
        else sorted(int(c) for c in np.unique(y_true))
    )

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred,
        labels=class_labels,
        zero_division=0,
    )

    per_class: dict[int, ClassMetrics] = {
        int(lbl): ClassMetrics(
            precision=float(prec[i]),
            recall=float(rec[i]),
            f1=float(f1[i]),
            support=int(sup[i]),
        )
        for i, lbl in enumerate(class_labels)
    }

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    return StatsResult(
        name=name,
        n_samples=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro",
                                labels=class_labels, zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted",
                                   labels=class_labels, zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        classes=class_labels,
        per_class=per_class,
        confusion_matrix=cm.tolist(),
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_markdown(result: StatsResult) -> str:
    """Format a StatsResult as a self-contained markdown section.

    Args:
        result: StatsResult returned by compute().

    Returns:
        Markdown string with scalar metrics table, per-class table, and
        confusion matrix.
    """
    lines: list[str] = [
        f"## {result['name']}\n\n",
        f"Samples evaluated: {result['n_samples']:,}\n\n",
        "### Scalar metrics\n\n",
        "| Metric | Value |\n",
        "|--------|-------|\n",
        f"| Accuracy | {result['accuracy']:.4f} |\n",
        f"| Macro F1 | {result['macro_f1']:.4f} |\n",
        f"| Weighted F1 | {result['weighted_f1']:.4f} |\n",
        f"| MCC | {result['mcc']:.4f} |\n\n",
        "### Per-class metrics\n\n",
        "| Class | Precision | Recall | F1 | Support |\n",
        "|-------|-----------|--------|----|---------|\n",
    ]
    for cls in result["classes"]:
        m = result["per_class"][cls]
        lines.append(
            f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m['support']:,} |\n"
        )

    # Confusion matrix header
    n = len(result["classes"])
    cm = result["confusion_matrix"]
    pred_headers = " | ".join(f"Pred {c}" for c in result["classes"])
    sep          = "|".join(["--"] * (n + 1))

    lines += [
        "\n### Confusion matrix (rows = actual, cols = predicted)\n\n",
        f"|  | {pred_headers} |\n",
        f"|{sep}|\n",
    ]
    for i, cls in enumerate(result["classes"]):
        row_vals = " | ".join(f"{cm[i][j]:,}" for j in range(n))
        lines.append(f"| **Actual {cls}** | {row_vals} |\n")

    return "".join(lines)


def write_results(results: list[StatsResult], path: Path) -> None:
    """Write a list of StatsResults to a single markdown file.

    Args:
        results: List of StatsResult dicts, one per model.
        path: Destination file path; parent directories are created if absent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Model Evaluation Statistics\n\n"
        "All metrics computed on the held-out test set.\n\n"
    )
    body = "\n---\n\n".join(format_markdown(r) for r in results)
    path.write_text(header + body + "\n")
    print(f"Statistics written to {path}")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def to_dict(result: StatsResult) -> dict:
    """Return a JSON-serialisable dict.

    The confusion_matrix is already list[list[int]], so json.dumps works
    directly on the returned dict.

    Args:
        result: StatsResult returned by compute().

    Returns:
        Plain dict safe to pass to json.dumps().
    """
    return dict(result)
