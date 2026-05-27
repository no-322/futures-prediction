import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Fit a Random Forest classifier on the training feature matrix.

    Args:
        X: Feature matrix from build_features(), shape (n_samples, 20).
        y: Binary labels from build_labels(), shape (n_samples,).

    Returns:
        Fitted RandomForestClassifier instance with oob_score_ attribute set.
    """
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        oob_score=True,
        bootstrap=True,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    """Predict class labels for the test feature matrix.

    Args:
        model: Fitted RandomForestClassifier returned by train().
        X: Feature matrix, shape (n_samples, 20).

    Returns:
        Integer ndarray of predicted labels (0 or 1), shape (n_samples,).
    """
    return model.predict(X)


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
    raw_train, _ = split(raw_align)
    y_train = build_labels(raw_train)

    model = train(X_train, y_train)
    print(f"OOB accuracy: {model.oob_score_:.4f}")

    preds = predict(model, X_test)
    print(f"Predicted up: {preds.sum():,} / {len(preds):,}")
