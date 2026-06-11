"""Tests for the Gaussian-HMM regime utilities."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.regime_hmm import (
    MIN_REGIME_ROWS,
    REGIME_COLS,
    assign_regime,
    build_rf,
    canonical_regime_labels,
    fit_regime,
)


def test_constants() -> None:
    assert REGIME_COLS[0] == "lag1_vol15"
    assert len(REGIME_COLS) == 5
    assert MIN_REGIME_ROWS == 200


def test_build_rf_enforces_seed_and_balance() -> None:
    m = build_rf({"n_estimators": 10})
    assert isinstance(m, RandomForestClassifier)
    assert m.random_state == 42
    assert m.class_weight == "balanced"


def test_fit_assign_and_canonicalise_two_regimes() -> None:
    rng = np.random.default_rng(0)
    n = 400
    cluster = rng.integers(0, 2, size=n)
    vol15 = np.where(cluster == 0, rng.normal(0.1, 0.01, n),
                     rng.normal(1.0, 0.01, n))
    X = np.column_stack([vol15] + [rng.normal(size=n) for _ in range(4)])

    hmm, scaler = fit_regime(X)
    raw = assign_regime(hmm, scaler, X)
    assert set(np.unique(raw).tolist()) <= {0, 1}

    remap = canonical_regime_labels(raw, X, vol15_col_idx=0)
    canon = np.array([remap[r] for r in raw])
    # Canonical regime 1 must have the higher mean vol15.
    assert X[canon == 1, 0].mean() > X[canon == 0, 0].mean()
