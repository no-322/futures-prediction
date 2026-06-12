"""Tests for the binary HMM-regime direction model."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import src.models.regime_binary as rb
from src.models.regime_hmm import REGIME_COLS


def _synthetic(n: int = 420, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw frame + a features_v2-shaped frame (len n-4) with the regime cols.

    The vol15 descriptor is drawn from two interleaved clusters so the HMM can
    recover two regimes in both the train and test halves of the split.
    """
    rng = np.random.default_rng(seed)
    ts  = pd.date_range("2023-01-03 04:00", periods=n, freq="1min")
    opens  = np.full(n, 100.0)
    closes = opens + rng.choice([-0.5, 0.5], size=n)
    closes[::50] = opens[::50]                      # a few flat bars
    df = pd.DataFrame({
        "Date and Time": ts, "Open": opens, "High": opens,
        "Low": opens, "Close": closes, "VWAP": opens,
    })

    m       = n - 4
    cluster = rng.integers(0, 2, size=m)
    vol15   = np.where(cluster == 0,
                       rng.normal(0.1, 0.01, m),
                       rng.normal(1.0, 0.01, m))
    feats = {"lag1_vol15": vol15}
    for col in REGIME_COLS:
        if col != "lag1_vol15":
            feats[col] = rng.normal(size=m)
    feats["extra0"] = rng.normal(size=m)
    feats["extra1"] = rng.normal(size=m)
    return df, pd.DataFrame(feats)


def test_run_persists_binary_predictions_with_regime_and_move(monkeypatch, tmp_path) -> None:
    df, fr = _synthetic()
    monkeypatch.setattr(rb, "load_raw", lambda _p: df)
    monkeypatch.setattr(rb, "build_features_v2", lambda _d: fr)
    monkeypatch.setattr(rb, "_PROC", tmp_path)
    monkeypatch.setattr(rb, "_NPZ", tmp_path / "exp_regime_binary_predictions.npz")

    cfg = {"data": {"path": "x", "train_size": 0.5},
           "models": {"rf": {"n_estimators": 20}}}
    summary = rb.run(config=cfg)

    d = np.load(tmp_path / "exp_regime_binary_predictions.npz")
    n = len(d["y_true"])
    # All four arrays present and aligned.
    assert len(d["y_pred"]) == n
    assert len(d["regime"]) == n
    assert len(d["move"]) == n
    # Binary predictions; every bar received one (no -1 sentinels left).
    assert set(np.unique(d["y_pred"]).tolist()) <= {0, 1}
    assert set(np.unique(d["y_true"]).tolist()) <= {0, 1}
    assert set(np.unique(d["regime"]).tolist()) <= {0, 1}

    # Artifacts written under the patched processed dir.
    assert (tmp_path / "exp_regime_binary_hmm.joblib").exists()
    assert (tmp_path / "exp_regime_binary_scaler.joblib").exists()
    assert (tmp_path / "exp_regime_binary_remap.joblib").exists()
    assert (tmp_path / "exp_regime_binary_dir_r0.joblib").exists()
    assert (tmp_path / "exp_regime_binary_dir_r1.joblib").exists()
    assert summary["n_test"] == n


def test_load_bundle_and_predict_roundtrip(monkeypatch, tmp_path) -> None:
    df, fr = _synthetic()
    monkeypatch.setattr(rb, "load_raw", lambda _p: df)
    monkeypatch.setattr(rb, "build_features_v2", lambda _d: fr)
    monkeypatch.setattr(rb, "_PROC", tmp_path)
    monkeypatch.setattr(rb, "_NPZ", tmp_path / "exp_regime_binary_predictions.npz")
    cfg = {"data": {"path": "x", "train_size": 0.5},
           "models": {"rf": {"n_estimators": 20}}}
    rb.run(config=cfg)

    bundle = rb.load_bundle(tmp_path)
    assert set(bundle) == {"hmm", "scaler", "remap", "dir_models"}

    preds = rb.predict(fr, bundle)                 # predict on the full feature frame
    assert len(preds) == len(fr)
    assert set(np.unique(preds).tolist()) <= {0, 1}


def test_save_importance_writes_ranked_csv(tmp_path) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    out = tmp_path / "imp.csv"
    rb._save_importance(model, ["a", "b", "c"], out)
    df = pd.read_csv(out)
    assert list(df.columns) == ["feature", "importance"]
    assert len(df) == 3
    assert df["importance"].is_monotonic_decreasing
