"""Tests for the Gaussian-HMM regime utilities."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.regime_hmm import (
    MIN_REGIME_ROWS,
    REGIME_COLS,
    assign_regime,
    build_rf,
    canonical_regime_labels,
    filter_regime,
    filter_regime_posterior,
    fit_regime,
)


def _two_regime_X(n: int = 400, seed: int = 0) -> np.ndarray:
    """Synthetic descriptors with a clean low-vol / high-vol split in column 0."""
    rng = np.random.default_rng(seed)
    cluster = rng.integers(0, 2, size=n)
    vol15 = np.where(cluster == 0, rng.normal(0.1, 0.01, n),
                     rng.normal(1.0, 0.01, n))
    return np.column_stack([vol15] + [rng.normal(size=n) for _ in range(4)])


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


def test_filter_posterior_is_a_valid_distribution() -> None:
    X = _two_regime_X()
    hmm, scaler = fit_regime(X)
    post = filter_regime_posterior(hmm, scaler, X)
    assert post.shape == (len(X), 2)
    assert np.isfinite(post).all()
    assert (post >= -1e-9).all() and (post <= 1 + 1e-9).all()
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-9)
    # filter_regime is the argmax of the filtered posterior.
    assert np.array_equal(filter_regime(hmm, scaler, X), post.argmax(axis=1))
    assert set(np.unique(filter_regime(hmm, scaler, X)).tolist()) <= {0, 1}


def test_filter_is_causal_no_lookahead() -> None:
    """Filtering: posterior at row t depends only on rows 0..t.

    Perturbing the descriptor at row p must leave every earlier row's filtered
    posterior bit-identical, and change row p. (Viterbi smoothing would not satisfy
    this — that is exactly the look-ahead this filter removes.)
    """
    X = _two_regime_X()
    hmm, scaler = fit_regime(X)                 # params fixed; isolate the decode
    base = filter_regime_posterior(hmm, scaler, X)

    p = 200
    Xp = X.copy()
    Xp[p] = X[p] + 50.0                          # large perturbation at row p only
    pert = filter_regime_posterior(hmm, scaler, Xp)

    # Rows strictly before p are untouched (no look-ahead into the future).
    np.testing.assert_array_equal(base[:p], pert[:p])
    # The perturbed row itself changes (sanity: the test is not vacuous).
    assert not np.allclose(base[p], pert[p])
