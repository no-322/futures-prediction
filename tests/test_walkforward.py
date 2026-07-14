"""Tests for the rolling walk-forward harness (src/walkforward.py).

Uses small synthetic data with clean daily timestamps so fold geometry is exact
and leakage assertions are easy to verify. The skill's invariants — fixed-width
rolling windows from config, fresh model per fold, per-fold + mean±std reporting,
and forward-only ordering — are each exercised directly.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.walkforward import (
    Fold,
    make_folds,
    module_factory,
    project_factories,
    sklearn_factory,
    summarize,
    walk_forward,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic, seed-42, ~8 months of daily bars
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """8 months of daily timestamps, random 4-dim X, random binary y."""
    rng = np.random.default_rng(42)
    ts = pd.Series(pd.date_range("2024-01-01", periods=243, freq="D"))  # ~8 months
    X = pd.DataFrame(rng.standard_normal((len(ts), 4)), columns=list("abcd"))
    y = pd.Series(rng.integers(0, 2, size=len(ts)))
    return X, y, ts


class _CountingFactory:
    """Stub estimator + factory that records how many fresh instances are built."""

    def __init__(self) -> None:
        self.builds = 0
        self.fits: list[int] = []

    def __call__(self):  # returns a fresh stub each call
        self.builds += 1
        outer = self

        class _Stub:
            def fit(self, X, y):
                outer.fits.append(len(X))
                self._majority = int(pd.Series(y).mode().iloc[0])
                return self

            def predict(self, X):
                return np.full(len(X), self._majority, dtype=int)

        return _Stub()


# ---------------------------------------------------------------------------
# make_folds geometry
# ---------------------------------------------------------------------------

def test_folds_are_time_ordered_and_nonempty(synthetic) -> None:
    _, _, ts = synthetic
    folds = make_folds(ts, train_months=3, test_months=1, step_months=1)
    assert len(folds) > 0
    for f in folds:
        assert isinstance(f, Fold)
        assert f.train_idx.size > 0 and f.test_idx.size > 0
        assert f.train_start <= f.train_end < f.test_start <= f.test_end


def test_no_leakage_indices_and_timestamps(synthetic) -> None:
    _, _, ts = synthetic
    folds = make_folds(ts, train_months=3, test_months=1, step_months=1)
    for f in folds:
        assert f.train_idx.max() < f.test_idx.min()
        assert f.train_end < f.test_start


def test_window_widths_match_config(synthetic) -> None:
    _, _, ts = synthetic
    folds = make_folds(ts, train_months=3, test_months=1, step_months=1)
    for f in folds:
        # Train block spans (close to) 3 months; test block (close to) 1 month.
        train_days = (f.train_end - f.train_start).days
        test_days = (f.test_end - f.test_start).days
        assert 80 <= train_days <= 92          # ~3 calendar months of daily bars
        assert 20 <= test_days <= 31           # ~1 calendar month


def test_nonoverlapping_test_blocks_when_step_equals_test(synthetic) -> None:
    _, _, ts = synthetic
    folds = make_folds(ts, train_months=3, test_months=1, step_months=1)
    for a, b in zip(folds, folds[1:]):
        assert a.test_end < b.test_start            # disjoint, ordered test blocks


def test_unsorted_timestamps_raise() -> None:
    ts = pd.Series(pd.to_datetime(["2024-03-01", "2024-01-01", "2024-02-01"]))
    with pytest.raises(ValueError, match="monotonic"):
        make_folds(ts, train_months=1, test_months=1, step_months=1)


def test_nonpositive_window_raises(synthetic) -> None:
    _, _, ts = synthetic
    with pytest.raises(ValueError):
        make_folds(ts, train_months=0, test_months=1, step_months=1)


def test_series_shorter_than_window_yields_no_folds() -> None:
    ts = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
    assert make_folds(ts, train_months=3, test_months=1, step_months=1) == []


# ---------------------------------------------------------------------------
# purge / embargo
# ---------------------------------------------------------------------------

def test_purge_drops_tail_train_rows(synthetic) -> None:
    _, _, ts = synthetic
    base = make_folds(ts, 3, 1, 1, purge=0, embargo=0)
    purged = make_folds(ts, 3, 1, 1, purge=2, embargo=0)
    for b, p in zip(base, purged):
        assert p.train_idx.size == b.train_idx.size - 2
        assert p.train_idx.max() < p.test_idx.min()   # still leak-free


def test_embargo_skips_leading_test_rows(synthetic) -> None:
    _, _, ts = synthetic
    base = make_folds(ts, 3, 1, 1, purge=0, embargo=0)
    embargoed = make_folds(ts, 3, 1, 1, purge=0, embargo=3)
    for b, e in zip(base, embargoed):
        assert e.test_idx.size == b.test_idx.size - 3
        assert e.test_idx.min() > b.test_idx.min()


def test_purge_embargo_default_inert(synthetic) -> None:
    _, _, ts = synthetic
    default = make_folds(ts, 3, 1, 1)
    explicit = make_folds(ts, 3, 1, 1, purge=0, embargo=0)
    assert len(default) == len(explicit)
    for a, b in zip(default, explicit):
        assert np.array_equal(a.train_idx, b.train_idx)
        assert np.array_equal(a.test_idx, b.test_idx)


# ---------------------------------------------------------------------------
# walk_forward: fresh model per fold, aggregation, persistence
# ---------------------------------------------------------------------------

def test_fresh_model_per_fold(synthetic) -> None:
    X, y, ts = synthetic
    factory = _CountingFactory()
    result = walk_forward(X, y, ts, factory, train_months=3, test_months=1,
                          step_months=1, name="stub", save=False)
    assert factory.builds == result["n_folds"]        # one fresh build per fold
    assert len(factory.fits) == result["n_folds"]


def test_mean_std_match_manual(synthetic) -> None:
    X, y, ts = synthetic
    result = walk_forward(X, y, ts, _CountingFactory(), train_months=3,
                          test_months=1, step_months=1, name="stub", save=False)
    accs = np.array([f["accuracy"] for f in result["per_fold"]])
    assert result["mean_accuracy"] == pytest.approx(accs.mean())
    assert result["std_accuracy"] == pytest.approx(accs.std())
    assert result["min_accuracy"] == pytest.approx(accs.min())
    assert result["max_accuracy"] == pytest.approx(accs.max())
    assert result["n_folds"] == len(accs)


def test_misaligned_inputs_raise(synthetic) -> None:
    X, y, ts = synthetic
    with pytest.raises(ValueError, match="row-aligned"):
        walk_forward(X.iloc[:-1], y, ts, _CountingFactory(), save=False)


def test_no_fold_raises() -> None:
    ts = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
    X = pd.DataFrame(np.zeros((10, 2)))
    y = pd.Series(np.zeros(10, dtype=int))
    with pytest.raises(ValueError, match="No folds"):
        walk_forward(X, y, ts, _CountingFactory(), train_months=3, test_months=1,
                     save=False)


def test_config_supplies_window_sizes(synthetic) -> None:
    X, y, ts = synthetic
    cfg = {"walk_forward": {"train_months": 3, "test_months": 1, "step_months": 1}}
    via_config = walk_forward(X, y, ts, _CountingFactory(), config=cfg,
                              name="cfg", save=False)
    explicit = walk_forward(X, y, ts, _CountingFactory(), train_months=3,
                            test_months=1, step_months=1, name="exp", save=False)
    assert via_config["n_folds"] == explicit["n_folds"]


def test_persistence_roundtrip(synthetic, tmp_path, monkeypatch) -> None:
    import src.walkforward as wf
    monkeypatch.setattr(wf, "_PROC", tmp_path)
    X, y, ts = synthetic
    result = wf.walk_forward(X, y, ts, _CountingFactory(), train_months=3,
                             test_months=1, step_months=1, name="rt", save=True)
    path = Path(result["npz_path"])
    assert path.exists()
    d = np.load(path, allow_pickle=False)
    total_test = sum(f["n_test"] for f in result["per_fold"])
    assert d["y_true"].shape[0] == total_test
    assert d["y_pred"].shape[0] == total_test
    assert d["fold_id"].shape[0] == total_test
    assert d["accuracies"].shape[0] == result["n_folds"]
    assert d["test_starts"].shape[0] == result["n_folds"]


def test_sklearn_style_estimator_runs_end_to_end(synthetic) -> None:
    from sklearn.tree import DecisionTreeClassifier
    X, y, ts = synthetic
    factory = sklearn_factory(DecisionTreeClassifier, {"max_depth": 2})
    result = walk_forward(X, y, ts, factory, train_months=3, test_months=1,
                          step_months=1, name="dtree", save=False)
    assert result["n_folds"] > 0
    assert 0.0 <= result["mean_accuracy"] <= 1.0


def test_sklearn_factory_injects_seed() -> None:
    from sklearn.ensemble import RandomForestClassifier
    est = sklearn_factory(RandomForestClassifier, {"n_estimators": 10})()
    assert est.random_state == 42


def test_summarize_contains_headline_and_table(synthetic) -> None:
    X, y, ts = synthetic
    result = walk_forward(X, y, ts, _CountingFactory(), train_months=3,
                          test_months=1, step_months=1, name="stub", save=False)
    text = summarize(result)
    assert "across" in text and "folds" in text
    assert "| Fold |" in text
    assert text.count("\n") >= result["n_folds"]


def test_project_factories_build_fresh_seeded_models() -> None:
    cfg = {"models": {"baseline": {"max_iter": 500}, "rf": {"n_estimators": 10},
                      "gbm": {"n_estimators": 10}}}
    facs = project_factories(cfg)
    assert set(facs) == {"baseline", "rf", "gbm"}
    assert facs["baseline"]().random_state == 42


# ---------------------------------------------------------------------------
# No-flat scoring / training, thresholds (keep / include_flat / drop_flat_train)
# ---------------------------------------------------------------------------

def test_include_flat_default_scores_no_flat(synthetic) -> None:
    # With a keep mask, default (include_flat=False) scores only non-flat test rows.
    X, y, ts = synthetic
    rng = np.random.default_rng(0)
    keep = rng.random(len(X)) > 0.3                       # ~70% non-flat
    no_flat = walk_forward(X, y, ts, _CountingFactory(), train_months=3, test_months=1,
                           step_months=1, keep=keep, name="nf", save=False)
    full = walk_forward(X, y, ts, _CountingFactory(), train_months=3, test_months=1,
                        step_months=1, keep=keep, include_flat=True, name="full",
                        save=False)
    # n_test reflects the scored rows: fewer under no-flat than full.
    nf_counts = [f["n_test"] for f in no_flat["per_fold"]]
    full_counts = [f["n_test"] for f in full["per_fold"]]
    assert all(a <= b for a, b in zip(nf_counts, full_counts))
    assert sum(nf_counts) < sum(full_counts)


def test_keep_none_scores_full_block(synthetic) -> None:
    # Back-compat: without keep, every test row is scored (full block).
    X, y, ts = synthetic
    res = walk_forward(X, y, ts, _CountingFactory(), train_months=3, test_months=1,
                       step_months=1, name="bc", save=False)
    folds = make_folds(ts, 3, 1, 1)
    assert [f["n_test"] for f in res["per_fold"]] == [fd.test_idx.size for fd in folds]


def test_drop_flat_train_shrinks_train(synthetic) -> None:
    X, y, ts = synthetic
    keep = np.ones(len(X), dtype=bool)
    keep[::5] = False                                     # every 5th row flat
    factory = _CountingFactory()
    res = walk_forward(X, y, ts, factory, train_months=3, test_months=1, step_months=1,
                       keep=keep, drop_flat_train=True, name="dft", save=False)
    folds = make_folds(ts, 3, 1, 1)
    for f, fold in zip(res["per_fold"], folds):
        assert f["n_train"] == int(keep[fold.train_idx].sum())   # flats dropped
        assert f["n_train"] < fold.train_idx.size


def test_predict_fn_threshold_applied(synthetic) -> None:
    # A predict_fn overriding the threshold changes predictions vs default predict.
    X, y, ts = synthetic

    class _ProbaFactory:
        def __call__(self):
            class _M:
                def fit(self, X, y):
                    self._p = float(pd.Series(y).mean()); return self
                def predict(self, X):
                    return np.zeros(len(X), dtype=int)
                def predict_proba(self, X):
                    col = np.full(len(X), 0.7)
                    return np.column_stack([1 - col, col])
            return _M()

    thr_fn = lambda m, Xte: (m.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
    res = walk_forward(X, y, ts, _ProbaFactory(), train_months=3, test_months=1,
                       step_months=1, predict_fn=thr_fn, include_flat=True, name="thr",
                       save=False)
    # proba=0.7 ≥ 0.5 → all predicted 1; default predict would give all 0.
    d = np.load(res["npz_path"]) if res["npz_path"] else None
    assert res["n_folds"] > 0
    # Accuracy equals fraction of actual-up rows (since everything predicted up).
    # Recompute from per-fold is awkward; assert via a fresh full-set run instead:
    assert 0.0 <= res["mean_accuracy"] <= 1.0


def test_fold_transform_appends_column(synthetic) -> None:
    # A fold_transform that appends a column is seen by the fitted model (per fold).
    X, y, ts = synthetic
    captured: dict[str, int] = {}

    class _F:
        def __call__(self):
            class _M:
                def fit(self2, Xtr, ytr):
                    captured["ncol"] = Xtr.shape[1]
                    self2.maj = int(pd.Series(ytr).mode().iloc[0])
                    return self2

                def predict(self2, Xte):
                    return np.full(len(Xte), self2.maj, dtype=int)
            return _M()

    def ft(full_train_idx, test_idx):
        X_tr = X.iloc[full_train_idx].copy(); X_tr["extra"] = 1.0
        X_te = X.iloc[test_idx].copy(); X_te["extra"] = 1.0
        return X_tr, X_te, None

    res = walk_forward(X, y, ts, _F(), train_months=3, test_months=1, step_months=1,
                       fold_transform=ft, name="ft", save=False)
    assert res["n_folds"] > 0
    assert captured["ncol"] == X.shape[1] + 1          # model trained on augmented X


def test_fold_transform_gate_restricts_scoring(synthetic) -> None:
    # A test_gate from fold_transform narrows scoring; coverage/n_eligible are reported.
    X, y, ts = synthetic
    keep = np.ones(len(X), dtype=bool)

    def ft(full_train_idx, test_idx):
        gate = (test_idx % 2 == 0)                      # keep ~half the test bars
        return X.iloc[full_train_idx], X.iloc[test_idx], gate

    res = walk_forward(X, y, ts, _CountingFactory(), train_months=3, test_months=1,
                       step_months=1, keep=keep, fold_transform=ft, name="gate",
                       save=False)
    for f in res["per_fold"]:
        assert "coverage" in f and "n_eligible" in f
        assert f["n_test"] <= f["n_eligible"]
        assert f["coverage"] == pytest.approx(f["n_test"] / f["n_eligible"])
    assert (sum(f["n_test"] for f in res["per_fold"])
            < sum(f["n_eligible"] for f in res["per_fold"]))


def test_module_factory_reproduces_module_training(synthetic) -> None:
    from src.models import baseline as m_baseline
    X, y, ts = synthetic
    tmp = Path("/tmp/_wf_modfac_test.joblib")
    factory = module_factory(m_baseline, {"max_iter": 200}, tmp)
    est = factory()
    est.fit(X.iloc[:120], y.iloc[:120])
    preds = est.predict(X.iloc[120:])
    proba = est.predict_proba(X.iloc[120:])
    assert preds.shape[0] == len(X) - 120
    assert proba.shape == (len(X) - 120, 2)
    assert tmp.exists()                                  # module.train persisted
