from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
)


def _select(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    keep: np.ndarray | None,
    include_flat: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict predictions to the non-flat test rows unless full set is requested.

    Flat bars (Close == Open) are ambiguous up/down targets; by default they are
    excluded from every metric when a keep mask is supplied. Passing
    ``include_flat=True`` — or omitting ``keep`` entirely — evaluates the whole set.

    Args:
        y_true: Ground-truth binary labels, shape (n,).
        y_pred: Predicted binary labels, shape (n,).
        keep: Boolean mask, True where the row is non-flat (Close != Open), same
            length/order as y_true/y_pred. None means flatness is unknown → full set.
        include_flat: If True, ignore ``keep`` and use the full arrays.

    Returns:
        (y_true, y_pred) sliced to the evaluated rows.
    """
    if include_flat or keep is None:
        return y_true, y_pred
    keep = np.asarray(keep, dtype=bool)
    return y_true[keep], y_pred[keep]


def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    keep: np.ndarray | None = None,
    *,
    include_flat: bool = False,
) -> float:
    """Compute accuracy (fraction of correct predictions).

    By default flat bars are dropped when a ``keep`` mask is supplied (see
    ``_select``); pass ``include_flat=True`` to score the whole set.

    Args:
        y_true: Ground-truth binary labels, shape (n,).
        y_pred: Predicted binary labels, shape (n,).
        keep: Boolean non-flat mask (True = Close != Open) aligned to y_*; None → full.
        include_flat: If True, score the full set regardless of ``keep``.

    Returns:
        Accuracy as a float in [0, 1].
    """
    yt, yp = _select(y_true, y_pred, keep, include_flat)
    return float(accuracy_score(yt, yp))


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    keep: np.ndarray | None = None,
    *,
    include_flat: bool = False,
) -> float:
    """Compute recall for class 1 (up direction).

    By default flat bars are dropped when a ``keep`` mask is supplied; pass
    ``include_flat=True`` to score the whole set.

    Args:
        y_true: Ground-truth binary labels, shape (n,).
        y_pred: Predicted binary labels, shape (n,).
        keep: Boolean non-flat mask (True = Close != Open) aligned to y_*; None → full.
        include_flat: If True, score the full set regardless of ``keep``.

    Returns:
        Recall for the positive (up) class as a float in [0, 1].
    """
    yt, yp = _select(y_true, y_pred, keep, include_flat)
    return float(recall_score(yt, yp, zero_division=0))


def confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    keep: np.ndarray | None = None,
    *,
    include_flat: bool = False,
) -> np.ndarray:
    """Compute the 2×2 confusion matrix.

    By default flat bars are dropped when a ``keep`` mask is supplied; pass
    ``include_flat=True`` to use the whole set.

    Args:
        y_true: Ground-truth binary labels, shape (n,).
        y_pred: Predicted binary labels, shape (n,).
        keep: Boolean non-flat mask (True = Close != Open) aligned to y_*; None → full.
        include_flat: If True, use the full set regardless of ``keep``.

    Returns:
        ndarray of shape (2, 2): [[TN, FP], [FN, TP]].
    """
    yt, yp = _select(y_true, y_pred, keep, include_flat)
    return confusion_matrix(yt, yp, labels=[0, 1])


def report(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    keep: np.ndarray | None = None,
    *,
    include_flat: bool = False,
) -> str:
    """Format accuracy, recall, and confusion matrix for one model as markdown.

    By default metrics are computed on the non-flat test slice when a ``keep`` mask
    is supplied; pass ``include_flat=True`` (or omit ``keep``) for the full set. A
    one-line note records which slice was used.

    Args:
        name: Display name for the model (used as the section heading).
        y_true: Ground-truth binary labels, shape (n,).
        y_pred: Predicted binary labels, shape (n,).
        keep: Boolean non-flat mask (True = Close != Open) aligned to y_*; None → full.
        include_flat: If True, report on the full set regardless of ``keep``.

    Returns:
        Markdown string containing a slice note, a metrics table, and a confusion matrix.
    """
    yt, yp = _select(y_true, y_pred, keep, include_flat)
    acc = float(accuracy_score(yt, yp))
    rec = float(recall_score(yt, yp, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    if include_flat or keep is None:
        note = f"_Evaluated on {len(yt):,} test bars (full set)._"
    else:
        note = (
            f"_Evaluated on {len(yt):,} non-flat test bars "
            f"(flat `Close == Open` excluded)._"
        )
    return (
        f"## {name}\n\n"
        f"{note}\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Accuracy | {acc:.4f} |\n"
        f"| Recall (Up) | {rec:.4f} |\n\n"
        f"**Confusion Matrix** (rows = actual, cols = predicted):\n\n"
        f"|  | Predicted Down | Predicted Up |\n"
        f"|--|----------------|--------------|\n"
        f"| **Actual Down** | {tn:,} | {fp:,} |\n"
        f"| **Actual Up** | {fn:,} | {tp:,} |\n"
    )


def write_results(reports: list[str], path: Path, intro: str | None = None) -> None:
    """Write formatted model reports to a markdown file.

    Args:
        reports: List of markdown sections, one per model, as returned by
            report(). Written in order, separated by horizontal rules.
        path: Destination file path (e.g. docs/results.md). Parent
            directories are created if they do not exist.
        intro: Optional header block (title + framing) prepended to the file. When
            None, a default header documenting the full held-out test set is used;
            callers reporting a non-flat slice should pass an accurate intro.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = intro if intro is not None else (
        "# Model Evaluation Results\n\n"
        "All metrics computed on the held-out test set "
        "(second 50% of data by timestamp).\n\n"
    )
    content = header + "\n---\n\n".join(reports) + "\n"
    path.write_text(content)
    print(f"Results written to {path}")


if __name__ == "__main__":
    from src.features import build_features
    from src.labels import build_labels, move_series
    from src.load import load_raw
    from src.models import baseline, rf
    from src.models.gbm import train as gbm_train
    from src.models.gbm import predict as gbm_predict
    from src.models.svm import train as svm_train
    from src.models.svm import predict as svm_predict
    from src.split import split

    df = load_raw(Path("data/raw/data.csv"))
    features = build_features(df)
    raw_align = df.iloc[4:].reset_index(drop=True)
    X_train, X_test = split(features)
    raw_train, raw_test = split(raw_align)
    y_train = build_labels(raw_train)
    y_test = build_labels(raw_test)

    y_true = y_test.to_numpy()
    # Default to the no-flat slice: drop flat (Close == Open) test bars from metrics.
    keep = move_series(raw_test).to_numpy() != 0

    print("Training Logistic Regression...")
    lr_model = baseline.train(X_train, y_train)
    print("Training Random Forest...")
    rf_model = rf.train(X_train, y_train)
    print("Training GBM...")
    gbm_model = gbm_train(X_train, y_train)
    print("Training SVM (slow — O(n²))...")
    svm_model = svm_train(X_train, y_train)
    print("Done. Writing results...")

    reports = [
        report("Always Up (baseline)", y_true, baseline.predict_always_up(len(y_true)), keep),
        report("Last Direction (baseline)", y_true, baseline.predict_last_direction(y_train, y_test), keep),
        report("Logistic Regression", y_true, baseline.predict(lr_model, X_test), keep),
        report("Random Forest", y_true, rf.predict(rf_model, X_test), keep),
        report("Gradient Boosting (XGBoost)", y_true, gbm_predict(gbm_model, X_test), keep),
        report("SVM (RBF kernel)", y_true, svm_predict(svm_model, X_test), keep),
    ]

    intro = (
        "# Model Evaluation Results\n\n"
        "Metrics computed on the **no-flat** held-out test set (second 50% of data by "
        "timestamp, flat `Close == Open` bars excluded from evaluation only).\n\n"
    )
    write_results(reports, Path("docs/results.md"), intro=intro)
