"""SVM classifier for futures price direction prediction.

Hyperparameter rationale (ESL Chapter 12):
- kernel='rbf': General-purpose Gaussian kernel (ESL 12.3.1). Handles nonlinear
  decision boundaries without manual feature engineering.
- C=1.0: Sklearn default; controls the margin/error trade-off (ESL 12.2.1).
  Log-spaced grid search over [0.01, 100] is deferred — given the 50/50
  time-ordered split, cross-validation must respect temporal order (no k-fold
  shuffle), and the one-week project scope makes tuning impractical.
- gamma='scale': Sets γ = 1 / (n_features * X.var()), adapting the RBF
  bandwidth to feature variance (ESL 12.3.2). Avoids manual bandwidth
  selection and is robust to feature scale after StandardScaler.
- class_weight='balanced': Compensates for class imbalance in up/down labels.
- probability=False: Avoids Platt scaling overhead (additional O(n²) inner CV);
  only hard class labels are needed for accuracy comparison.
- random_state=42: Project-wide seed for reproducibility.
- cache_size=500: Kernel matrix cache in MB; reduces recomputation during SMO.

Complexity note: SVC training is O(n²)–O(n³) in n_train rows (SMO algorithm,
ESL 12.2). For n_train ≈ 275 k rows this will be slow (potentially hours).
Documented fallback options (not implemented):
  - sklearn.svm.LinearSVC: O(n) training via liblinear; valid if RBF advantage
    is marginal on this dataset.
  - Subsample training set (e.g., 50 k rows stratified by label) and accept the
    accuracy/variance trade-off.
"""
from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_DEFAULT_PATH = Path("data/processed/svm_model.joblib")


class SVMModel(TypedDict):
    """Bundled training-fit scaler and fitted SVC classifier."""

    scaler: StandardScaler
    clf: SVC


def train(X: pd.DataFrame, y: pd.Series) -> SVMModel:
    """Fit a StandardScaler and SVC on the training feature matrix.

    The scaler is fit exclusively on X (no test-set information leaks in).
    Both objects are returned together so predict() applies the same transform
    at test time without refitting.

    Args:
        X: Feature matrix from build_features(), shape (n_samples, 20).
        y: Binary labels from build_labels(), shape (n_samples,).

    Returns:
        SVMModel with keys 'scaler' (fitted StandardScaler) and 'clf' (fitted SVC).
    """
    scaler: StandardScaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=False,
        random_state=42,
        cache_size=500,
    )
    clf.fit(X_scaled, y)
    return SVMModel(scaler=scaler, clf=clf)


def predict(model: SVMModel, X: pd.DataFrame) -> np.ndarray:
    """Predict class labels using the training-fit scaler and SVC.

    The scaler in model was fit on X_train only; no information from X leaks
    into the transform, satisfying the no-look-ahead constraint.

    Args:
        model: SVMModel returned by train(), with 'scaler' and 'clf'.
        X: Feature matrix, shape (n_samples, 20).

    Returns:
        Integer ndarray of predicted labels (0 or 1), shape (n_samples,).
    """
    X_scaled = model["scaler"].transform(X)
    return model["clf"].predict(X_scaled)


def save(model: SVMModel, path: Path = _DEFAULT_PATH) -> None:
    """Serialize the SVMModel (scaler + classifier) to disk with joblib.

    Args:
        model: SVMModel returned by train(); both scaler and clf are preserved.
        path: Destination file path; parent directories are created if absent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: Path = _DEFAULT_PATH) -> SVMModel:
    """Deserialize an SVMModel previously saved by save().

    Args:
        path: Path to the .joblib file written by save().

    Returns:
        SVMModel dict with 'scaler' and 'clf' identical to train() output.
    """
    return joblib.load(Path(path))


if __name__ == "__main__":
    from src.features import build_features
    from src.labels import build_labels
    from src.load import load_raw
    from src.split import split

    df = load_raw(Path("data/raw/data.csv"))
    features = build_features(df)
    raw_align = df.iloc[4:].reset_index(drop=True)
    X_train, X_test = split(features)
    raw_train, _ = split(raw_align)
    y_train = build_labels(raw_train)

    print("Training SVM (O(n²)–O(n³) — may take a while)…")
    model = train(X_train, y_train)
    print(f"Support vectors: {model['clf'].n_support_}")

    preds = predict(model, X_test)
    print(f"Predicted up: {preds.sum():,} / {len(preds):,}")

    save(model)
    print(f"Saved to {_DEFAULT_PATH}")
