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


def test_build_dataset_drops_flat_globally(monkeypatch) -> None:
    # 12 raw rows -> aligned raw is rows 4..11 (8 rows). Flat rows are dropped from
    # the whole modelling set (train AND test) before the 50/50 split.
    df = _synthetic_raw(12)
    closes = df["Close"].to_numpy().copy()
    # aligned index 0..7 == df index 4..11 (Open is 100 everywhere)
    closes[4] = 100.5   # aligned 0: up
    closes[5] = 100.0   # aligned 1: flat -> dropped
    closes[6] = 100.0   # aligned 2: flat -> dropped
    closes[7] = 99.0    # aligned 3: down
    closes[8] = 100.0   # aligned 4: flat -> dropped
    closes[9] = 100.5   # aligned 5: up
    closes[10] = 99.0   # aligned 6: down
    closes[11] = 100.0  # aligned 7: flat -> dropped
    df["Close"] = closes

    features = pd.DataFrame(np.arange(8 * 2).reshape(8, 2),
                            columns=["f0", "f1"], dtype=float)

    monkeypatch.setattr(bn, "load_raw", lambda _p: df)
    monkeypatch.setattr(bn, "build_features", lambda _d: features)

    cfg = {"data": {"path": "x", "train_size": 0.5}}
    X_train, X_test, y_train, y_test, move_test = bn._build_dataset(cfg)

    # 4 non-flat rows survive [up, down, up, down] → 2 train / 2 test, strictly binary.
    assert len(X_train) == 2 and len(X_test) == 2
    assert list(y_train) == [1, 0]
    assert list(y_test) == [1, 0]
    assert len(move_test) == 2
    assert 0 not in np.abs(move_test)          # no flat (zero-move) bars remain
    # Feature rows follow the surviving aligned indices [0, 3] (train) and [5, 6] (test).
    assert X_train["f0"].tolist() == [0.0, 6.0]
    assert X_test["f0"].tolist() == [10.0, 12.0]


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
    imp_b = bn._feature_importance("logistic", lr, list(X.columns))
    assert len(imp_b) == 3
    assert (imp_b["importance"] >= 0).all()       # abs(coef)


def test_feature_importance_unknown_returns_none() -> None:
    assert bn._feature_importance("unknown", object(), ["a", "b"]) is None
