"""Tests for the no-flat model-selection harness (src.tuning).

The data-split tests use the real dataset (load + v1 features); the search /
threshold tests use small synthetic folds so they stay fast and deterministic.
The key guarantee is **no test leakage**: the validation fold lives entirely
inside the training half.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features import build_features
from src.labels import flat_mask
from src.tuning import (
    _GRIDS,
    SelectionSplit,
    _full_train_test,
    _load_splits,
    build_selection_split,
    grid_search,
    predict_with_threshold,
    select_features,
    tune_threshold,
)
from src.models import baseline

VAL_FRAC = 0.2


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def sel(cfg: dict) -> SelectionSplit:
    return build_selection_split(cfg, build_features, val_frac=VAL_FRAC)


def test_selection_shapes_align(sel: SelectionSplit) -> None:
    assert len(sel.X_inner) == len(sel.y_inner) == len(sel.move_inner) > 0
    assert len(sel.X_val) == len(sel.y_val) > 0
    assert set(np.unique(sel.y_val)) <= {0, 1}


def test_no_test_leakage(cfg: dict, sel: SelectionSplit) -> None:
    """Every validation timestamp must precede the test-half start."""
    _, _, _, raw_test = _load_splits(cfg, build_features)
    test_start = pd.Timestamp(raw_test["Date and Time"].iloc[0])
    assert sel.val_start < test_start


def test_validation_is_flat_free(cfg: dict, sel: SelectionSplit) -> None:
    """The validation fold must contain exactly the non-flat val rows."""
    X_train, raw_train, _, _ = _load_splits(cfg, build_features)
    cut = int(len(X_train) * (1.0 - VAL_FRAC))
    raw_val = raw_train.iloc[cut:]
    expected = int((~flat_mask(raw_val)).sum())
    assert len(sel.X_val) == expected


def test_full_train_test_masks(cfg: dict) -> None:
    X_tr, y_tr, move_tr, X_te, y_te, move_te, keep_te = _full_train_test(
        cfg, build_features)
    assert len(X_tr) == len(y_tr) == len(move_tr) > 0
    assert len(keep_te) == len(X_te) == len(y_te) == len(move_te)
    assert 0 < int(keep_te.sum()) < len(keep_te)  # some flats dropped, not all
    assert set(np.unique(y_te)) <= {0, 1}


def test_tune_threshold_separable() -> None:
    rng = np.random.RandomState(42)
    y = rng.randint(0, 2, size=2000)
    scores = y + 0.10 * rng.randn(2000)  # well-separated around 0.5
    thr = tune_threshold(scores, y)
    acc = float(((scores >= thr).astype(int) == y).mean())
    assert acc > 0.95


def _synthetic_split() -> SelectionSplit:
    rng = np.random.RandomState(0)
    cols = [f"f{i}" for i in range(4)]
    Xi = pd.DataFrame(rng.randn(400, 4), columns=cols)
    yi = pd.Series((Xi["f0"] + 0.2 * rng.randn(400) > 0).astype(int))
    Xv = pd.DataFrame(rng.randn(150, 4), columns=cols)
    yv = (Xv["f0"] > 0).astype(int).to_numpy()
    move = np.abs(rng.randn(400)) + 0.1
    return SelectionSplit(Xi, yi, move, Xv, yv, pd.Timestamp("2024-01-01"))


def test_grid_search_picks_learnable() -> None:
    sel = _synthetic_split()
    best, acc, results = grid_search("baseline", sel)
    assert best in [p for p, _ in results]
    assert acc > 0.6  # f0-driven label is learnable
    assert len(results) == len(_GRIDS["baseline"])


def test_predict_with_threshold(tmp_path) -> None:
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(120, 4), columns=[f"f{i}" for i in range(4)])
    y = pd.Series((X["f0"] > 0).astype(int))
    model = baseline.train(X, y, save_path=tmp_path / "m.joblib")
    # threshold=None reproduces the plain label prediction
    np.testing.assert_array_equal(
        predict_with_threshold("baseline", model, X, None),
        baseline.predict(model, X),
    )
    # an impossibly high probability threshold forces every prediction to 0
    assert set(np.unique(predict_with_threshold("baseline", model, X, 1.01))) == {0}


def test_select_features_returns_subset() -> None:
    # f0 is the only informative feature; selection should keep it and prune.
    sel = _synthetic_split()
    cols, results = select_features(sel, k_grid=[1, 2, 4])
    assert "f0" in cols
    assert set(cols) <= set(sel.X_inner.columns)
    assert [k for k, _ in results] == [1, 2, 4]
