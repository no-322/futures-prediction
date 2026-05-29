"""Gradient Boosted Machine classifier for futures price direction prediction.

Hyperparameter rationale (ESL Chapter 10):
- n_estimators=500: Number of boosting rounds (ESL 10.12). Fixed; early
  stopping via a validation split is deferred because the 50/50 time-ordered
  design leaves no held-out slice that is both representative and unseen.
- learning_rate=0.05: Shrinkage applied to each tree's contribution (ESL
  10.12.1). Smaller shrinkage with more trees is a reliable recipe for
  better generalisation (Friedman 2001).
- max_depth=4: Shallow trees act as weak learners (ESL 10.11). Depth 4
  captures up to 4-way feature interactions; boosting's additive structure
  provides global model complexity so individual trees need not be deep.
- subsample=0.8: Stochastic gradient boosting — subsample 80% of training
  rows per round (ESL 10.12.2, Friedman 1999). Reduces variance and speeds
  training.
- colsample_bytree=0.8: Subsample 80% of features per tree, analogous to
  Random Forest's max_features (ESL 10.12.2).
- reg_lambda=1.0: L2 regularisation on leaf weights (XGBoost manual; related
  to ESL 10.12.3 ridge penalisation). Default value.
- min_child_weight=1: Minimum sum of instance weights required in a child
  node; XGBoost default.
- objective='binary:logistic': Binary cross-entropy loss (log-loss), the
  natural choice for a two-class direction prediction task.
- eval_metric='logloss': Consistent with the training objective.
- random_state=42: Project-wide seed for reproducibility.
- n_jobs=-1: Use all available cores for training.

Scale invariance: GBM splits on rank order, not absolute feature values;
no StandardScaler is needed (contrast with SVM, ESL 12.3.2).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

_DEFAULT_PATH = Path("data/processed/gbm_model.joblib")


def train(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """Fit an XGBClassifier on the training feature matrix.

    Args:
        X: Feature matrix from build_features(), shape (n_samples, 20).
        y: Binary labels from build_labels(), shape (n_samples,).

    Returns:
        Fitted XGBClassifier instance.
    """
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Predict class labels for the test feature matrix.

    Args:
        model: Fitted XGBClassifier returned by train().
        X: Feature matrix, shape (n_samples, 20).

    Returns:
        Integer ndarray of predicted labels (0 or 1), shape (n_samples,).
    """
    return model.predict(X)


def save(model: XGBClassifier, path: Path = _DEFAULT_PATH) -> None:
    """Serialize a fitted XGBClassifier to disk with joblib.

    Args:
        model: Fitted XGBClassifier returned by train().
        path: Destination file path; parent directories are created if absent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: Path = _DEFAULT_PATH) -> XGBClassifier:
    """Deserialize an XGBClassifier previously saved by save().

    Args:
        path: Path to the .joblib file written by save().

    Returns:
        Fitted XGBClassifier identical to train() output.
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

    model = train(X_train, y_train)

    preds = predict(model, X_test)
    print(f"Predicted up: {preds.sum():,} / {len(preds):,}")

    save(model)
    print(f"Saved to {_DEFAULT_PATH}")
