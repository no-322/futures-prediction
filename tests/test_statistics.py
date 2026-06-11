import json

import numpy as np
import pytest

from src.statistics import ClassMetrics, StatsResult, compute, format_markdown, to_dict, write_results

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_result() -> StatsResult:
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    return compute(y_true, y_pred, name="Binary test")


@pytest.fixture
def multiclass_result() -> StatsResult:
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 2, 0, 1, 1, 2, 1, 2])
    return compute(y_true, y_pred, name="Three-class test")


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def test_compute_binary_accuracy(binary_result: StatsResult) -> None:
    # 5 correct out of 8
    assert binary_result["accuracy"] == pytest.approx(5 / 8)


def test_compute_accuracy_range(binary_result: StatsResult) -> None:
    assert 0.0 <= binary_result["accuracy"] <= 1.0


def test_compute_binary_n_samples(binary_result: StatsResult) -> None:
    assert binary_result["n_samples"] == 8


def test_compute_binary_classes(binary_result: StatsResult) -> None:
    assert binary_result["classes"] == [0, 1]


def test_compute_binary_per_class_keys(binary_result: StatsResult) -> None:
    assert set(binary_result["per_class"].keys()) == {0, 1}


def test_compute_binary_per_class_fields(binary_result: StatsResult) -> None:
    for cls in [0, 1]:
        m = binary_result["per_class"][cls]
        assert "precision" in m
        assert "recall" in m
        assert "f1" in m
        assert "support" in m
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"]    <= 1.0
        assert 0.0 <= m["f1"]        <= 1.0
        assert m["support"] >= 0


def test_compute_multiclass_classes(multiclass_result: StatsResult) -> None:
    assert multiclass_result["classes"] == [0, 1, 2]


def test_compute_multiclass_per_class_count(multiclass_result: StatsResult) -> None:
    assert len(multiclass_result["per_class"]) == 3


def test_compute_confusion_matrix_shape_binary(binary_result: StatsResult) -> None:
    cm = binary_result["confusion_matrix"]
    assert len(cm) == 2
    assert all(len(row) == 2 for row in cm)


def test_compute_confusion_matrix_shape_multiclass(multiclass_result: StatsResult) -> None:
    cm = multiclass_result["confusion_matrix"]
    assert len(cm) == 3
    assert all(len(row) == 3 for row in cm)


def test_compute_confusion_matrix_sum(binary_result: StatsResult) -> None:
    total = sum(v for row in binary_result["confusion_matrix"] for v in row)
    assert total == binary_result["n_samples"]


def test_compute_macro_f1_range(binary_result: StatsResult) -> None:
    assert 0.0 <= binary_result["macro_f1"] <= 1.0


def test_compute_weighted_f1_range(binary_result: StatsResult) -> None:
    assert 0.0 <= binary_result["weighted_f1"] <= 1.0


def test_compute_mcc_range(binary_result: StatsResult) -> None:
    assert -1.0 <= binary_result["mcc"] <= 1.0


def test_compute_perfect_prediction() -> None:
    y = np.array([0, 0, 1, 1])
    r = compute(y, y, name="perfect")
    assert r["accuracy"] == pytest.approx(1.0)
    assert r["mcc"] == pytest.approx(1.0)
    for cls in r["per_class"].values():
        assert cls["precision"] == pytest.approx(1.0)
        assert cls["recall"]    == pytest.approx(1.0)
        assert cls["f1"]        == pytest.approx(1.0)


def test_compute_explicit_labels() -> None:
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1, 0])
    r = compute(y_true, y_pred, name="explicit", labels=[0, 1, 2])
    assert r["classes"] == [0, 1, 2]
    assert 2 in r["per_class"]


def test_compute_name_preserved(binary_result: StatsResult) -> None:
    assert binary_result["name"] == "Binary test"


# ---------------------------------------------------------------------------
# format_markdown()
# ---------------------------------------------------------------------------

def test_format_markdown_contains_name(binary_result: StatsResult) -> None:
    md = format_markdown(binary_result)
    assert "Binary test" in md


def test_format_markdown_contains_scalar_headers(binary_result: StatsResult) -> None:
    md = format_markdown(binary_result)
    assert "Accuracy" in md
    assert "Macro F1" in md
    assert "MCC" in md


def test_format_markdown_contains_per_class_section(binary_result: StatsResult) -> None:
    md = format_markdown(binary_result)
    assert "Per-class" in md
    assert "Precision" in md
    assert "Recall" in md


def test_format_markdown_contains_confusion_section(binary_result: StatsResult) -> None:
    md = format_markdown(binary_result)
    assert "Confusion" in md


def test_format_markdown_is_string(binary_result: StatsResult) -> None:
    assert isinstance(format_markdown(binary_result), str)


# ---------------------------------------------------------------------------
# to_dict() / serialisation
# ---------------------------------------------------------------------------

def test_to_dict_json_serialisable(binary_result: StatsResult) -> None:
    d = to_dict(binary_result)
    # Should not raise
    json.dumps(d)


def test_to_dict_contains_expected_keys(binary_result: StatsResult) -> None:
    d = to_dict(binary_result)
    for key in ("name", "accuracy", "macro_f1", "mcc", "per_class",
                "confusion_matrix", "classes", "n_samples"):
        assert key in d


# ---------------------------------------------------------------------------
# write_results()
# ---------------------------------------------------------------------------

def test_write_results_creates_file(binary_result: StatsResult,
                                    tmp_path: pytest.TempPathFactory) -> None:
    out = tmp_path / "stats.md"
    write_results([binary_result], out)
    assert out.exists()
    content = out.read_text()
    assert "Binary test" in content


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

from src.statistics import (  # noqa: E402
    annualized_sharpe,
    backtest,
    max_drawdown,
    _positions_from_signals,
)


def test_positions_from_signals_binary() -> None:
    pos = _positions_from_signals(np.array([1, 0, 1, 0]), "binary")
    assert pos.tolist() == [1.0, -1.0, 1.0, -1.0]


def test_positions_from_signals_three_class() -> None:
    pos = _positions_from_signals(np.array([0, 1, 2]), "three_class")
    assert pos.tolist() == [0.0, 1.0, -1.0]   # HOLD, LONG, SHORT


def test_backtest_compounds_and_signs() -> None:
    # long, short, long against +10%, +10%, -10% bars (cost 0).
    r = backtest(np.array([1, 0, 1]), np.array([0.1, 0.1, -0.1]),
                 transaction_cost=0.0, name="t")
    assert np.allclose(r["equity"], [1100.0, 990.0, 891.0])
    assert np.allclose(r["payoff"], [100.0, -110.0, -99.0])   # prev_equity * net
    assert r["final_equity"] == pytest.approx(891.0)
    assert r["total_return"] == pytest.approx(-0.109)
    assert r["max_drawdown"] == pytest.approx(0.19)           # 1100 -> 891
    assert r["n_periods"] == 3


def test_backtest_passive_benchmark() -> None:
    bar = np.array([0.1, -0.05, 0.2])
    r = backtest(np.array([1, 0, 1]), bar, transaction_cost=0.0)
    # Passive = always-long, frictionless = initial * cumprod(1 + bar).
    assert np.allclose(r["passive_equity"], 1000.0 * np.cumprod(1.0 + bar))
    assert r["passive_final_equity"] == pytest.approx(1000.0 * np.prod(1.0 + bar))
    assert r["passive_total_return"] == pytest.approx(np.prod(1.0 + bar) - 1.0)


def test_backtest_always_long_matches_passive() -> None:
    bar = np.array([0.1, -0.05, 0.2, -0.1])
    r = backtest(np.ones(4, dtype=int), bar, transaction_cost=0.0)
    # An all-long, cost-free strategy is exactly the passive benchmark.
    assert np.allclose(r["equity"], r["passive_equity"])
    assert r["final_equity"] == pytest.approx(r["passive_final_equity"])


def test_backtest_transaction_cost_reduces_net() -> None:
    # Flat market, always long: net return == -cost each bar.
    free = backtest(np.array([1, 1]), np.array([0.0, 0.0]), transaction_cost=0.0)
    paid = backtest(np.array([1, 1]), np.array([0.0, 0.0]), transaction_cost=0.01)
    assert np.allclose(free["equity"], [1000.0, 1000.0])
    assert np.allclose(paid["net_return"], [-0.01, -0.01])
    assert np.allclose(paid["equity"], [990.0, 980.1])


def test_backtest_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        backtest(np.array([1, 0]), np.array([0.1]))


def test_max_drawdown_known_series() -> None:
    # peak 1100 -> trough 880 => 20% drawdown
    assert max_drawdown(np.array([1000.0, 1100.0, 880.0, 1050.0])) == pytest.approx(0.2)
    assert max_drawdown(np.array([1000.0, 1100.0, 1200.0])) == 0.0   # monotonic up


def test_annualized_sharpe_formula_and_guard() -> None:
    r = np.array([0.01, -0.01, 0.02])
    expected = np.sqrt(252) * r.mean() / r.std(ddof=1)
    assert annualized_sharpe(r, 252) == pytest.approx(expected)
    assert annualized_sharpe(np.zeros(3), 252) == 0.0            # zero-vol guard
    assert annualized_sharpe(np.array([0.01]), 252) == 0.0       # too short
