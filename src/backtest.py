"""Backtest runner — select a model, simulate its trading P&L on the test set.

Thin I/O glue around the backtest computation in `src.statistics` (which holds
the equity-curve / drawdown / Sharpe math and the plot). This module reconstructs
the contiguous second-50% test slice (prices + timestamps), maps a saved model's
per-record signals into the simulation, persists the per-record results, and
writes the equity-curve PNG + markdown report.

Scope: binary up/down models, whose predictions are saved in the same order as
the 50/50 `raw_test` slice (production + no-flat + v2 + HMM-regime binary). A
length-mismatch guard rejects the 3-class / TimeSeriesSplit prediction sets.

Run with::

    python -m src.backtest --algo rf                  # frictionless
    python -m src.backtest --algo exp_noflat_v2_rf --cost 0.0001
    python -m src.backtest --algo all --cost 0.0      # every binary model
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import src.statistics as statistics
from src.config import load_config
from src.binary_suite import BASE_DISPLAY
from src.load import load_raw
from src.split import split

_PROC = Path("data/processed")
_DOCS = Path("docs/notes")
_REPORT_PATH = _DOCS / "backtest_stats.md"

_ALGOS = ("baseline", "rf", "gbm", "svm")


def _build_registry() -> dict[str, str]:
    """Ordered map of backtestable binary prediction-set stems → display names."""
    reg: dict[str, str] = {}
    for a in _ALGOS:
        reg[a] = BASE_DISPLAY[a]
    for a in _ALGOS:
        reg[f"exp_noflat_{a}"] = f"{BASE_DISPLAY[a]} (no-flat)"
    for a in _ALGOS:
        reg[f"exp_noflat_v2_{a}"] = f"{BASE_DISPLAY[a]} (no-flat, v2)"
    for a in _ALGOS:
        reg[f"exp_v2_{a}"] = f"{BASE_DISPLAY[a]} (v2, flat-incl)"
    reg["exp_regime_binary"] = "HMM-regime binary"
    return reg


_REGISTRY = _build_registry()


def _reconstruct_test_bars(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the 50/50 test slice → (timestamps, per-bar returns).

    Returns:
        (timestamps, bar_returns) for the test set, aligned 1-to-1 with the
        binary models' saved predictions. bar_return = (Close - Open) / Open.
    """
    train_size = cfg["data"].get("train_size", 0.5)
    data_path  = Path(cfg["data"]["path"])

    df        = load_raw(data_path)
    raw_align = df.iloc[4:].reset_index(drop=True)
    _, raw_test = split(raw_align, train_size=train_size)

    timestamps  = raw_test["Date and Time"].to_numpy()
    bar_returns = ((raw_test["Close"] - raw_test["Open"])
                   / raw_test["Open"]).to_numpy()
    return timestamps, bar_returns


def run_one(
    algo: str,
    transaction_cost: float,
    timestamps: np.ndarray,
    bar_returns: np.ndarray,
) -> statistics.BacktestResult:
    """Backtest one model and persist its per-record results + equity PNG.

    Args:
        algo: Prediction-set stem (a key of the registry).
        transaction_cost: Proportional per-bar cost (0 = frictionless).
        timestamps: Test-set timestamps from `_reconstruct_test_bars`.
        bar_returns: Test-set per-bar returns from `_reconstruct_test_bars`.

    Returns:
        The BacktestResult (with `plot_path` set to the saved PNG).
    """
    npz_path = _PROC / f"{algo}_predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"predictions not found: {npz_path}")

    y_pred = np.load(npz_path)["y_pred"]
    if len(y_pred) != len(bar_returns):
        raise ValueError(
            f"{algo}: predictions length {len(y_pred):,} != test bars "
            f"{len(bar_returns):,}. Backtest supports only contiguous 50/50 "
            "binary models (not the 3-class / TimeSeriesSplit experiments)."
        )

    name   = _REGISTRY.get(algo, algo)
    result = statistics.backtest(
        y_pred, bar_returns, timestamps=timestamps,
        transaction_cost=transaction_cost, name=name,
    )

    # Persist per-record results (rule 7: always persist).
    out_npz = _PROC / f"backtest_{algo}_predictions.npz"
    np.savez(
        out_npz,
        signal=y_pred,
        position=result["position"],
        bar_return=result["bar_return"],
        net_return=result["net_return"],
        payoff=result["payoff"],
        equity=result["equity"],
        passive_equity=result["passive_equity"],
        timestamps=np.asarray(timestamps, dtype="datetime64[ns]"),
    )

    png = _DOCS / f"backtest_{algo}.png"
    statistics.plot_equity_curve(result, png)
    result["plot_path"] = str(png)

    print(f"  [{algo}] {name}: final=${result['final_equity']:,.2f}  "
          f"return={result['total_return']:.2%}  "
          f"maxDD={result['max_drawdown']:.2%}  "
          f"Sharpe={result['annualized_sharpe']:.4f}  "
          f"(cost={transaction_cost:g})")
    print(f"    results → {out_npz}")
    print(f"    plot    → {png}")
    return result


def run(
    algo: str,
    transaction_cost: float = 0.0,
    config: dict | None = None,
) -> list[statistics.BacktestResult]:
    """Backtest one model or all binary models, then write the report.

    Args:
        algo: A registry stem, or "all" for every binary model with predictions.
        transaction_cost: Proportional per-bar cost (default 0).
        config: Parsed config dict; defaults to load_config().

    Returns:
        List of BacktestResults (one per model run).
    """
    cfg = config or load_config()
    timestamps, bar_returns = _reconstruct_test_bars(cfg)

    if algo == "all":
        algos = [a for a in _REGISTRY
                 if (_PROC / f"{a}_predictions.npz").exists()]
    else:
        algos = [algo]

    results: list[statistics.BacktestResult] = []
    for a in algos:
        results.append(run_one(a, transaction_cost, timestamps, bar_returns))

    statistics.write_backtest_results(results, _REPORT_PATH)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Backtest a model's signals into a compounding equity curve"
    )
    parser.add_argument(
        "--algo", choices=[*_REGISTRY.keys(), "all"], default="rf",
        help="Prediction-set stem to backtest, or 'all' (default: rf).",
    )
    parser.add_argument(
        "--cost", type=float, default=0.0,
        help="Proportional transaction cost per bar (default 0; e.g. 0.0001 = 1bp).",
    )
    args = parser.parse_args()

    print(f"Backtesting [{args.algo}]  transaction_cost={args.cost:g}  "
          "initial=$1,000")
    run(args.algo, transaction_cost=args.cost)
