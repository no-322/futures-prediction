"""Experiment 1 — Three-class direction classifier with TimeSeriesSplit CV.

Labels: 0 = down (Close < Open), 1 = up (Close > Open), 2 = flat (Close == Open).

Run with:
    python -m src.experiments.three_class
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

from src.config import load_config, model_params
from src.features import build_features
from src.load import load_raw
from src.experiments.labels import three_class_labels
from src.experiments.metrics import mcc, macro_f1, per_class_recall

_CLASS_NAMES = {0: "down", 1: "up", 2: "flat"}
_RESULTS_PATH = Path("docs/exp_three_class_results.md")


def _build_rf(params: dict) -> RandomForestClassifier:
    p = dict(params)
    p["random_state"] = 42
    p["class_weight"] = "balanced"
    return RandomForestClassifier(**p)


def run(
    n_splits: int = 5,
    config: dict | None = None,
) -> list[dict[str, Any]]:
    """Run TimeSeriesSplit CV for the three-class labelling scheme.

    Args:
        n_splits: Number of TimeSeriesSplit folds.
        config: Parsed config dict; RF params read from config['models']['rf'].
                Defaults applied if None.

    Returns:
        List of per-fold result dicts with keys:
            fold, n_train, n_test, accuracy, mcc, macro_f1, per_class_recall,
            class_counts_train, class_counts_test.
    """
    cfg = config or load_config()
    rf_params = model_params(cfg, "rf")

    data_path = Path(cfg["data"]["path"])
    df = load_raw(data_path)
    features = build_features(df)
    raw_align = df.iloc[4:].reset_index(drop=True)

    X = features.values
    y = three_class_labels(raw_align).values

    tss = TimeSeriesSplit(n_splits=n_splits)
    results: list[dict[str, Any]] = []

    for fold_i, (train_idx, test_idx) in enumerate(tss.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = _build_rf(rf_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "fold":               fold_i,
            "n_train":            int(len(train_idx)),
            "n_test":             int(len(test_idx)),
            "accuracy":           float(accuracy_score(y_test, y_pred)),
            "mcc":                mcc(y_test, y_pred),
            "macro_f1":           macro_f1(y_test, y_pred),
            "per_class_recall":   per_class_recall(y_test, y_pred),
            "class_counts_train": {int(k): int(v) for k, v in
                                   zip(*np.unique(y_train, return_counts=True))},
            "class_counts_test":  {int(k): int(v) for k, v in
                                   zip(*np.unique(y_test,  return_counts=True))},
        })
        print(f"  Fold {fold_i}: acc={results[-1]['accuracy']:.4f}  "
              f"mcc={results[-1]['mcc']:.4f}  "
              f"macro_f1={results[-1]['macro_f1']:.4f}")

    return results


def summarise(results: list[dict[str, Any]]) -> str:
    """Format per-fold results as a markdown report.

    Args:
        results: List returned by run().

    Returns:
        Markdown string ready to write to docs/exp_three_class_results.md.
    """
    scalar_keys = ["accuracy", "mcc", "macro_f1"]
    all_classes = sorted({k for r in results for k in r["per_class_recall"]})

    lines = [
        "# Experiment 1 — Three-Class Direction Classifier\n\n",
        f"Folds: {len(results)}  |  Model: Random Forest  |  "
        "Labels: 0=down, 1=up, 2=flat\n\n",
        "## Scalar metrics (mean ± std across folds)\n\n",
        "| Metric | Mean | Std |\n",
        "|--------|------|-----|\n",
    ]
    for key in scalar_keys:
        vals = np.array([r[key] for r in results])
        lines.append(f"| {key} | {vals.mean():.4f} | {vals.std():.4f} |\n")

    lines += [
        "\n## Per-class recall (mean ± std across folds)\n\n",
        "| Class | Mean recall | Std |\n",
        "|-------|-------------|-----|\n",
    ]
    for cls in all_classes:
        vals = np.array([r["per_class_recall"].get(cls, 0.0) for r in results])
        lines.append(f"| {cls} ({_CLASS_NAMES.get(cls, '?')}) | "
                     f"{vals.mean():.4f} | {vals.std():.4f} |\n")

    lines += [
        "\n## Per-fold detail\n\n",
        "| Fold | n_train | n_test | accuracy | mcc | macro_f1 | "
        "recall_down | recall_up | recall_flat |\n",
        "|------|---------|--------|----------|-----|----------|"
        "------------|-----------|-------------|\n",
    ]
    for r in results:
        pcr = r["per_class_recall"]
        lines.append(
            f"| {r['fold']} | {r['n_train']:,} | {r['n_test']:,} | "
            f"{r['accuracy']:.4f} | {r['mcc']:.4f} | {r['macro_f1']:.4f} | "
            f"{pcr.get(0, 0):.4f} | {pcr.get(1, 0):.4f} | {pcr.get(2, 0):.4f} |\n"
        )

    return "".join(lines)


if __name__ == "__main__":
    print("Running Experiment 1 — Three-class RF with TimeSeriesSplit CV")
    results = run()
    report = summarise(results)
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(report)
    print(f"\nResults written to {_RESULTS_PATH}")
    print(report)
