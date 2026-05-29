from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy (fraction of correct predictions).

    Args:
        y_true: Ground-truth binary labels, shape (n,).
        y_pred: Predicted binary labels, shape (n,).

    Returns:
        Accuracy as a float in [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute recall for class 1 (up direction).

    Args:
        y_true: Ground-truth binary labels, shape (n,).
        y_pred: Predicted binary labels, shape (n,).

    Returns:
        Recall for the positive (up) class as a float in [0, 1].
    """
    return float(recall_score(y_true, y_pred, zero_division=0))


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the 2×2 confusion matrix.

    Args:
        y_true: Ground-truth binary labels, shape (n,).
        y_pred: Predicted binary labels, shape (n,).

    Returns:
        ndarray of shape (2, 2): [[TN, FP], [FN, TP]].
    """
    return confusion_matrix(y_true, y_pred)


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Format accuracy, recall, and confusion matrix for one model as markdown.

    Args:
        name: Display name for the model (used as the section heading).
        y_true: Ground-truth binary labels, shape (n,).
        y_pred: Predicted binary labels, shape (n,).

    Returns:
        Markdown string containing a metrics table and confusion matrix.
    """
    acc = accuracy(y_true, y_pred)
    rec = recall(y_true, y_pred)
    cm = confusion(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return (
        f"## {name}\n\n"
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


def write_results(reports: list[str], path: Path) -> None:
    """Write formatted model reports to a markdown file.

    Args:
        reports: List of markdown sections, one per model, as returned by
            report(). Written in order, separated by horizontal rules.
        path: Destination file path (e.g. docs/results.md). Parent
            directories are created if they do not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Model Evaluation Results\n\n"
        "All metrics computed on the held-out test set "
        "(second 50% of data by timestamp).\n\n"
    )
    content = header + "\n---\n\n".join(reports) + "\n"
    path.write_text(content)
    print(f"Results written to {path}")


if __name__ == "__main__":
    from src.features import build_features
    from src.labels import build_labels
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
        report("Always Up (baseline)", y_true, baseline.predict_always_up(len(y_true))),
        report("Last Direction (baseline)", y_true, baseline.predict_last_direction(y_train, y_test)),
        report("Logistic Regression", y_true, baseline.predict(lr_model, X_test)),
        report("Random Forest", y_true, rf.predict(rf_model, X_test)),
        report("Gradient Boosting (XGBoost)", y_true, gbm_predict(gbm_model, X_test)),
        report("SVM (RBF kernel)", y_true, svm_predict(svm_model, X_test)),
    ]

    write_results(reports, Path("docs/results.md"))
