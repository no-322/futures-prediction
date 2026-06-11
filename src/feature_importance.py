"""Feature importance analysis on the v2 49-feature matrix.

Trains a RandomForestClassifier(200 trees) on the 50% time-ordered training
split with binary direction labels and reports MDI feature importances.

Run with:
    python -m src.feature_importance
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import load_config
from src.features_v2 import build_features_v2
from src.labels import direction_labels
from src.load import load_raw
from src.split import split

_OUT_PATH = Path("docs/notes/feature_importance.md")


def run() -> pd.DataFrame:
    """Train one RF and return a DataFrame of feature importances ranked desc.

    Returns:
        DataFrame with columns ['feature', 'importance'] sorted by importance.
    """
    cfg       = load_config()
    data_path = Path(cfg["data"]["path"])
    train_size = cfg["data"].get("train_size", 0.5)

    df        = load_raw(data_path)
    features  = build_features_v2(df)
    raw_align = df.iloc[4:].reset_index(drop=True)

    X_train, _ = split(features, train_size=train_size)
    raw_train, _ = split(raw_align, train_size=train_size)
    y_train    = direction_labels(raw_train)

    print(f"Training RF (200 trees) on {len(X_train):,} rows × "
          f"{X_train.shape[1]} features...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    importance_df = (
        pd.DataFrame({
            "feature":    X_train.columns,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.index += 1  # 1-based rank
    return importance_df


def write_report(df: pd.DataFrame) -> None:
    """Write full importance ranking to docs/notes/feature_importance.md."""
    lines = [
        "# Feature Importance — v2 Features (49 features, RF 200 trees)\n\n",
        "Model: RandomForestClassifier, MDI (mean decrease in impurity)\n",
        "Training set: 50% time-ordered split, binary direction labels\n\n",
        "## Top 10\n\n",
        "| Rank | Feature | Importance |\n",
        "|------|---------|------------|\n",
    ]
    for rank, row in df.head(10).iterrows():
        lines.append(f"| {rank} | `{row['feature']}` | {row['importance']:.5f} |\n")

    lines += [
        "\n## Full ranking (49 features)\n\n",
        "| Rank | Feature | Importance |\n",
        "|------|---------|------------|\n",
    ]
    for rank, row in df.iterrows():
        lines.append(f"| {rank} | `{row['feature']}` | {row['importance']:.5f} |\n")

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text("".join(lines))
    print(f"Full ranking written to {_OUT_PATH}")


if __name__ == "__main__":
    df = run()

    print("\nTop 10 features by MDI importance:")
    print(f"{'Rank':<6}{'Feature':<30}{'Importance':<12}")
    print("-" * 48)
    for rank, row in df.head(10).iterrows():
        print(f"{rank:<6}{row['feature']:<30}{row['importance']:.5f}")

    write_report(df)
