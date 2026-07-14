"""Tests for the no-flat binary classification suite."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import src.binary_suite as bn


def _synthetic_raw(n: int = 12) -> pd.DataFrame:
    """Minimal raw frame: monotonic timestamps + OHLC needed downstream."""
    ts = pd.date_range("2023-01-03 04:00", periods=n, freq="1min")
    return pd.DataFrame({
        "Date and Time": ts,
        "Open":  np.full(n, 100.0),
        "High":  np.full(n, 100.0),
        "Low":   np.full(n, 100.0),
        "Close": np.full(n, 100.0),
        "VWAP":  np.full(n, 100.0),
    })


def test_build_noflat_dataset_drops_only_training_flats(monkeypatch) -> None:
    # 12 raw rows -> aligned raw is rows 4..11 (8 rows) -> 4 train / 4 test.
    df = _synthetic_raw(12)
    # Aligned rows are df.iloc[4:]; set Close so two TRAIN rows are flat and one
    # TEST row is flat (must survive untouched).
    closes = df["Close"].to_numpy().copy()
    # aligned index 0..7 == df index 4..11
    closes[4] = 100.5   # train row 0: up   (not flat)
    closes[5] = 100.0   # train row 1: flat -> dropped
    closes[6] = 100.0   # train row 2: flat -> dropped
    closes[7] = 99.0    # train row 3: down (not flat)
    closes[8] = 100.0   # test  row 0: flat -> kept, label 0
    closes[9] = 100.5   # test  row 1: up
    closes[10] = 99.0   # test  row 2: down
    closes[11] = 100.0  # test  row 3: flat -> kept, label 0
    df["Close"] = closes

    features = pd.DataFrame(np.arange(8 * 2).reshape(8, 2),
                            columns=["f0", "f1"], dtype=float)

    monkeypatch.setattr(bn, "load_raw", lambda _p: df)
    monkeypatch.setattr(bn, "build_features", lambda _d: features)

    cfg = {"data": {"path": "x", "train_size": 0.5}}
    X_train, X_test, y_train, y_test, move_test = bn._build_dataset(cfg)

    # Test split untouched: 4 rows, including the two flat ones.
    assert len(X_test) == 4
    assert len(y_test) == 4
    assert len(move_test) == 4
    assert list(y_test) == [0, 1, 0, 0]          # flat -> down(0)

    # Two flat training rows removed -> 2 remain (the up and the down).
    assert len(X_train) == 2
    assert len(y_train) == 2
    assert list(y_train) == [1, 0]


def test_build_dataset_keep_flat_retains_all_training_rows(monkeypatch) -> None:
    # Same setup, but drop_flat=False must keep every training row (incl. flats).
    df = _synthetic_raw(12)
    closes = df["Close"].to_numpy().copy()
    closes[4] = 100.5   # up
    closes[5] = 100.0   # flat (kept when drop_flat=False)
    closes[6] = 100.0   # flat
    closes[7] = 99.0    # down
    df["Close"] = closes
    features = pd.DataFrame(np.arange(8 * 2).reshape(8, 2),
                            columns=["f0", "f1"], dtype=float)

    monkeypatch.setattr(bn, "load_raw", lambda _p: df)
    # Pass an explicit builder to exercise the build_features_fn path too.
    cfg = {"data": {"path": "x", "train_size": 0.5}}
    X_train, X_test, y_train, y_test, _ = bn._build_dataset(
        cfg, build_features_fn=lambda _d: features, drop_flat=False
    )

    assert len(X_train) == 4          # nothing dropped
    assert len(y_train) == 4
    assert list(y_train) == [1, 0, 0, 0]   # flats labelled down(0), still present
    assert len(X_test) == 4           # test untouched


def test_feature_importance_rf_and_baseline_sorted() -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(int)

    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    imp = bn._feature_importance("rf", rf, list(X.columns))
    assert list(imp.columns) == ["feature", "importance"]
    assert len(imp) == 3
    assert imp["importance"].is_monotonic_decreasing

    lr = LogisticRegression(max_iter=200).fit(X, y)
    imp_b = bn._feature_importance("baseline", lr, list(X.columns))
    assert len(imp_b) == 3
    assert (imp_b["importance"] >= 0).all()       # abs(coef)


def test_feature_importance_unknown_returns_none() -> None:
    assert bn._feature_importance("unknown", object(), ["a", "b"]) is None
