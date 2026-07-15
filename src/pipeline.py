"""Automated end-to-end pipeline: data → features → train/load → evaluate.

Usage:
    python -m src.pipeline --algo rf
    python -m src.pipeline --algo logistic
    python -m src.pipeline --algo all
    python -m src.pipeline --algo all --force      # retrain even if joblib exists
    python -m src.pipeline --config my_config.yaml # use a custom config file

Hyperparameters and train_size are read from config.yaml at startup. Edit that
file (or point a GUI at it) to change any setting without touching source code.
If a saved model exists at data/processed/<algo>_model.joblib, it is loaded
instead of retrained. Pass --force to override and retrain unconditionally.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd

import src.evaluate as evaluate
from src.config import load_config, model_params
from src.features import build_features
from src.labels import build_labels
from src.load import load_raw
from src.split import split

_ALGO_CHOICES = ("logistic", "rf", "gbm")

METADATA_PATH = Path("data/processed/training_metadata.json")

_DEFAULT_PATHS: dict[str, Path] = {
    "logistic": Path("data/processed/logistic_model.joblib"),
    "rf":       Path("data/processed/rf_model.joblib"),
    "gbm":      Path("data/processed/gbm_model.joblib"),
}

_DISPLAY_NAMES: dict[str, str] = {
    "logistic": "Logistic Regression",
    "rf":       "Random Forest",
    "gbm":      "Gradient Boosting (XGBoost)",
}


class PipelineResult(TypedDict):
    """Outputs produced by a single algorithm run."""

    algo: str
    model: Any
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, Any]


def _build_dataset(
    data_path: Path,
    train_size: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load raw data and return (X_train, X_test, y_train, y_test).

    Args:
        data_path: Path to the raw CSV (data/raw/data.csv).
        train_size: Fraction of rows for training; passed to split().

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) with reset indices.
    """
    from src.labels import drop_flat
    df = load_raw(data_path)
    features = build_features(df)
    raw_align = df.iloc[4:].reset_index(drop=True)
    features, raw_align = drop_flat(features, raw_align)   # binary 0/1 modelling set
    X_train, X_test = split(features, train_size=train_size)
    raw_train, raw_test = split(raw_align, train_size=train_size)
    y_train = build_labels(raw_train)
    y_test = build_labels(raw_test)
    return X_train, X_test, y_train, y_test


def _write_training_metadata(
    algo: str,
    result: "PipelineResult",
    data_path: Path,
    train_size: float,
    params: dict | None,
) -> None:
    """Write (or update) the per-model entry in training_metadata.json.

    Reads the existing file (if present), updates only the entry for algo,
    and writes back — other models' entries are preserved.

    Args:
        algo: Model key ('logistic', 'rf', 'gbm').
        result: PipelineResult returned by run().
        data_path: Path to the training CSV used.
        train_size: Fraction of data used for training.
        params: Hyperparameter dict passed to train(); None if defaults used.
    """
    all_meta: dict = {}
    if METADATA_PATH.exists():
        try:
            all_meta = json.loads(METADATA_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            all_meta = {}

    all_meta[algo] = {
        "display_name": _DISPLAY_NAMES[algo],
        "data_path":    str(data_path),
        "train_size":   train_size,
        "n_test":       int(result["y_true"].shape[0]),
        "params":       params or {},
        "metrics": {
            "accuracy": float(result["metrics"]["accuracy"]),
            "recall":   float(result["metrics"]["recall"]),
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(all_meta, indent=2))


def _load_or_train(
    algo: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    force_retrain: bool,
    params: dict | None = None,
) -> tuple[Any, bool]:
    """Return a fitted model and whether training actually occurred.

    Args:
        algo: One of 'logistic', 'rf', 'gbm'.
        X_train: Training feature matrix.
        y_train: Training labels.
        force_retrain: If True, retrain even when a saved model exists.
        params: Hyperparameter dict from config; passed to train() when
            retraining. Ignored when loading from disk.

    Returns:
        (model, was_trained) — model artifact and True if trained, False if loaded.
    """
    from src.models import logistic, rf
    from src.models.gbm import load as gbm_load
    from src.models.gbm import train as gbm_train

    path = _DEFAULT_PATHS[algo]

    if path.exists() and not force_retrain:
        print(f"  [{algo}] Loading saved model from {path}")
        if algo == "logistic":
            return logistic.load(path), False
        if algo == "rf":
            return rf.load(path), False
        if algo == "gbm":
            return gbm_load(path), False

    print(f"  [{algo}] Training...")
    if algo == "logistic":
        return logistic.train(X_train, y_train, params=params), True
    if algo == "rf":
        return rf.train(X_train, y_train, params=params), True
    if algo == "gbm":
        return gbm_train(X_train, y_train, params=params), True

    raise ValueError(f"Unknown algo: {algo!r}")


def _get_predictions(algo: str, model: Any, X_test: pd.DataFrame) -> np.ndarray:
    """Return predictions from a fitted model.

    Args:
        algo: One of 'logistic', 'rf', 'gbm'.
        model: Fitted model artifact returned by _load_or_train().
        X_test: Test feature matrix.

    Returns:
        Integer ndarray of predicted labels (0 or 1), shape (n_test,).
    """
    from src.models import logistic, rf
    from src.models.gbm import predict as gbm_predict

    if algo == "logistic":
        return logistic.predict(model, X_test)
    if algo == "rf":
        return rf.predict(model, X_test)
    if algo == "gbm":
        return gbm_predict(model, X_test)
    raise ValueError(f"Unknown algo: {algo!r}")


def run(
    algo: str,
    data_path: Path = Path("data/raw/data.csv"),
    force_retrain: bool = False,
    config: dict | None = None,
    *,
    _dataset: tuple | None = None,
) -> PipelineResult:
    """Run the full pipeline for one algorithm.

    Loads data (or reuses a pre-loaded dataset), loads or trains the model,
    predicts on the test set, and returns metrics.

    Args:
        algo: One of 'logistic', 'rf', 'gbm'.
        data_path: Path to raw CSV; ignored when _dataset is provided.
        force_retrain: Retrain even if a saved joblib exists.
        config: Parsed config dict from load_config(); used to extract
            train_size and model hyperparameters. If None, defaults are used.
        _dataset: Optional pre-loaded (X_train, X_test, y_train, y_test) tuple
            to avoid redundant I/O when running multiple algos in sequence.

    Returns:
        PipelineResult with algo name, model artifact, predictions, and metrics.
    """
    train_size = (config or {}).get("data", {}).get("train_size", 0.5)
    params = model_params(config, algo) if config else None

    if _dataset is not None:
        X_train, X_test, y_train, y_test = _dataset
    else:
        X_train, X_test, y_train, y_test = _build_dataset(data_path, train_size)

    model, was_trained = _load_or_train(algo, X_train, y_train, force_retrain, params=params)
    y_pred = _get_predictions(algo, model, X_test)
    y_true = y_test.to_numpy()

    metrics: dict[str, Any] = {
        "accuracy": evaluate.accuracy(y_true, y_pred),
        "recall": evaluate.recall(y_true, y_pred),
        "confusion": evaluate.confusion(y_true, y_pred),
    }
    result = PipelineResult(algo=algo, model=model, y_true=y_true, y_pred=y_pred, metrics=metrics)

    if was_trained:
        effective_data_path = data_path if _dataset is None else Path("(pre-loaded dataset)")
        _write_training_metadata(algo, result, effective_data_path, train_size, params)

    return result


def _print_result(result: PipelineResult) -> None:
    m = result["metrics"]
    cm = m["confusion"]
    tn, fp, fn, tp = cm.ravel()
    print(f"    Accuracy : {m['accuracy']:.4f}")
    print(f"    Recall   : {m['recall']:.4f}")
    print(f"    Confusion: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Futures price direction pipeline")
    parser.add_argument(
        "--algo",
        choices=[*_ALGO_CHOICES, "all"],
        default="all",
        help="Algorithm to run (default: all)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to raw CSV (overrides config.yaml data.path)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when a saved joblib exists",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_path = args.data or Path(cfg["data"]["path"])

    if args.algo == "all":
        train_size = cfg["data"].get("train_size", 0.5)
        print(f"Loading dataset (train_size={train_size})...")
        dataset = _build_dataset(data_path, train_size)
        _, X_test, y_train, y_test = dataset
        y_true = y_test.to_numpy()

        results: list[PipelineResult] = []
        for algo in _ALGO_CHOICES:
            print(f"\n[{algo}]")
            result = run(algo, force_retrain=args.force, config=cfg, _dataset=dataset)
            _print_result(result)
            results.append(result)

        from src.baselines import predict_always_up, predict_last_direction

        reports = [
            evaluate.report("Always Up (baseline)", y_true, predict_always_up(len(y_true))),
            evaluate.report("Last Direction (baseline)", y_true, predict_last_direction(y_train, y_test)),
        ]
        for result in results:
            reports.append(evaluate.report(_DISPLAY_NAMES[result["algo"]], y_true, result["y_pred"]))

        evaluate.write_results(reports, Path("docs/results.md"))

    else:
        print("Loading dataset...")
        result = run(args.algo, data_path=data_path, force_retrain=args.force, config=cfg)
        print(f"\nResults for {_DISPLAY_NAMES[result['algo']]}:")
        _print_result(result)
