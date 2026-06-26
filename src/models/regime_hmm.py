"""Gaussian-HMM market-regime utilities (shared by regime-based models).

A 2-state Gaussian HMM detects latent market regimes (low-vol / high-vol) from a
small set of regime-descriptor features. These helpers are model-agnostic: they
fit the HMM, decode the regime sequence, and canonicalise the raw state labels so
regime 0 is always the calmer (lower-vol) regime. Used by
`src.models.regime_binary` (and, on the experimental branch, the gated cascade).
"""
from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Features used to characterise the market regime (columns of features_v2).
REGIME_COLS = [
    "lag1_vol15",
    "lag1_macd_hist",
    "lag1_rsi15",
    "lag1_tick_delta",
    "lag1_return",
]

# Minimum bars in a regime before training a dedicated direction model for it.
MIN_REGIME_ROWS = 200


def build_rf(params: dict) -> RandomForestClassifier:
    """Build a RandomForest with the project seed and balanced class weights.

    Args:
        params: Hyperparameter dict (e.g. from config.model_params(cfg, "rf")).
            random_state and class_weight are always enforced.

    Returns:
        An unfitted RandomForestClassifier.
    """
    p = dict(params)
    p["random_state"] = 42
    p["class_weight"] = "balanced"
    return RandomForestClassifier(**p)


def fit_regime(X_regime: np.ndarray) -> tuple[GaussianHMM, StandardScaler]:
    """Fit a Gaussian HMM (2 states) on the regime-descriptor features.

    Args:
        X_regime: Array of shape (n_train, len(REGIME_COLS)) of descriptor values.

    Returns:
        (hmm, scaler) — fitted HMM and the StandardScaler used to normalise
        inputs (fit on training data only).
    """
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_regime)
    hmm = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=100,
        random_state=42,
    )
    hmm.fit(X_sc)
    return hmm, scaler


def assign_regime(
    hmm: GaussianHMM,
    scaler: StandardScaler,
    X_regime: np.ndarray,
) -> np.ndarray:
    """Decode the hidden-state sequence (Viterbi) for new descriptor rows.

    WARNING: this is the **smoothing** decode — Viterbi picks the most-likely state
    at bar t using the *entire* sequence, including bars after t, so the regime label
    at t peeks at the future (look-ahead bias). Retained for the legacy
    ``regime_binary`` path; prefer ``filter_regime`` / ``filter_regime_posterior`` for
    causal, no-look-ahead assignment.

    Args:
        hmm: Fitted GaussianHMM from fit_regime().
        scaler: Fitted StandardScaler from fit_regime().
        X_regime: Descriptor array, shape (n, len(REGIME_COLS)).

    Returns:
        Integer raw-state array of shape (n,).
    """
    return hmm.predict(scaler.transform(X_regime))


def filter_regime_posterior(
    hmm: GaussianHMM,
    scaler: StandardScaler,
    X_regime: np.ndarray,
) -> np.ndarray:
    """Causal forward-filtered posterior P(state_t | descriptors_0..t) per row.

    Runs the HMM forward algorithm (filtering) instead of Viterbi (smoothing): the
    posterior at row t uses only rows 0..t, never the future. Because the regime
    descriptors are themselves lag-1 (built from data <= t-1), the filtered posterior
    at row t depends only on data <= t-1 — no look-ahead.

    Implemented in log-space from the fitted HMM's own parameters (startprob, transmat,
    Gaussian emissions) so it does not depend on private hmmlearn internals.

    Args:
        hmm: Fitted GaussianHMM from fit_regime() (covariance_type="full").
        scaler: Fitted StandardScaler from fit_regime().
        X_regime: Descriptor array, shape (n, len(REGIME_COLS)).

    Returns:
        Float array of shape (n, hmm.n_components): each row is the filtered posterior
        over raw states (non-negative, sums to 1).
    """
    Xs = scaler.transform(X_regime)
    n = Xs.shape[0]
    k = int(hmm.n_components)

    # Per-state Gaussian emission log-probabilities, shape (n, k).
    log_b = np.column_stack([
        multivariate_normal.logpdf(
            Xs, mean=hmm.means_[s], cov=hmm.covars_[s], allow_singular=True
        )
        for s in range(k)
    ])
    # Zeros in startprob/transmat → -inf in log-space, which logsumexp handles
    # correctly (those states are simply excluded); silence the benign warning.
    with np.errstate(divide="ignore"):
        log_start = np.log(hmm.startprob_)
        log_trans = np.log(hmm.transmat_)

    log_alpha = np.empty((n, k))
    log_alpha[0] = log_start + log_b[0]
    for t in range(1, n):
        # log_alpha[t, j] = logsumexp_i(log_alpha[t-1, i] + log_trans[i, j]) + log_b[t, j]
        log_alpha[t] = logsumexp(
            log_alpha[t - 1][:, None] + log_trans, axis=0
        ) + log_b[t]

    # Normalise each row to a filtered posterior (softmax over states).
    log_post = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_post)


def filter_regime(
    hmm: GaussianHMM,
    scaler: StandardScaler,
    X_regime: np.ndarray,
) -> np.ndarray:
    """Causal forward-filtered raw-state assignment (argmax of the filtered posterior).

    The no-look-ahead replacement for ``assign_regime``: state_t uses only data <= t-1.

    Args:
        hmm: Fitted GaussianHMM from fit_regime().
        scaler: Fitted StandardScaler from fit_regime().
        X_regime: Descriptor array, shape (n, len(REGIME_COLS)).

    Returns:
        Integer raw-state array of shape (n,).
    """
    return filter_regime_posterior(hmm, scaler, X_regime).argmax(axis=1)


def canonical_regime_labels(
    regime_train: np.ndarray,
    X_regime_train: np.ndarray,
    vol15_col_idx: int,
) -> dict[int, int]:
    """Map raw HMM state labels to canonical labels (0=low-vol, 1=high-vol).

    Args:
        regime_train: Raw decoded states for the training rows.
        X_regime_train: Training descriptor array.
        vol15_col_idx: Index of the vol15 descriptor within the regime columns.

    Returns:
        Remapping dict {raw_label → canonical_label}; the state with higher mean
        vol15 becomes canonical 1.
    """
    mean_vol = {
        s: float(X_regime_train[regime_train == s, vol15_col_idx].mean())
        for s in [0, 1]
    }
    if mean_vol[0] > mean_vol[1]:
        return {0: 1, 1: 0}
    return {0: 0, 1: 1}
