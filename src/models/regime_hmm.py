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

    Args:
        hmm: Fitted GaussianHMM from fit_regime().
        scaler: Fitted StandardScaler from fit_regime().
        X_regime: Descriptor array, shape (n, len(REGIME_COLS)).

    Returns:
        Integer raw-state array of shape (n,).
    """
    return hmm.predict(scaler.transform(X_regime))


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
