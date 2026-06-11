"""Standardised evaluation statistics for all classifiers.

Works for binary and multi-class predictions. Returns a structured
StatsResult TypedDict covering accuracy, macro/weighted F1, MCC, and
full per-class precision/recall/F1/support.

Usage::

    from src.statistics import compute, format_markdown, write_results
    result = compute(y_true, y_pred, name="Random Forest")
    print(format_markdown(result))
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class ClassMetrics(TypedDict):
    """Metrics for a single class."""

    precision: float
    recall:    float
    f1:        float
    support:   int


class StatsResult(TypedDict):
    """Complete evaluation statistics for one model run."""

    name:             str
    n_samples:        int
    accuracy:         float
    macro_f1:         float        # unweighted mean F1 across classes
    weighted_f1:      float        # support-weighted mean F1
    mcc:              float        # Matthews Correlation Coefficient
    classes:          list[int]    # sorted unique class labels
    per_class:        dict[int, ClassMetrics]
    confusion_matrix: list[list[int]]   # rows=actual, cols=predicted


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "",
    labels: list[int] | None = None,
) -> StatsResult:
    """Compute the full statistics report for one model.

    Works for binary and multi-class predictions. Uses only data from y_true
    and y_pred — no model loading is performed here.

    Args:
        y_true: Ground-truth class labels, shape (n,).
        y_pred: Predicted class labels, shape (n,).
        name: Display name for this model (used in formatted reports).
        labels: Explicit list of class labels. If None, uses sorted unique
            values from y_true. Useful when a class may be absent from
            a fold's test set.

    Returns:
        StatsResult TypedDict with all scalar and per-class metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    class_labels: list[int] = (
        labels if labels is not None
        else sorted(int(c) for c in np.unique(y_true))
    )

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred,
        labels=class_labels,
        zero_division=0,
    )

    per_class: dict[int, ClassMetrics] = {
        int(lbl): ClassMetrics(
            precision=float(prec[i]),
            recall=float(rec[i]),
            f1=float(f1[i]),
            support=int(sup[i]),
        )
        for i, lbl in enumerate(class_labels)
    }

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    return StatsResult(
        name=name,
        n_samples=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro",
                                labels=class_labels, zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted",
                                   labels=class_labels, zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        classes=class_labels,
        per_class=per_class,
        confusion_matrix=cm.tolist(),
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_markdown(result: StatsResult) -> str:
    """Format a StatsResult as a self-contained markdown section.

    Args:
        result: StatsResult returned by compute().

    Returns:
        Markdown string with scalar metrics table, per-class table, and
        confusion matrix.
    """
    lines: list[str] = [
        f"## {result['name']}\n\n",
        f"Samples evaluated: {result['n_samples']:,}\n\n",
        "### Scalar metrics\n\n",
        "| Metric | Value |\n",
        "|--------|-------|\n",
        f"| Accuracy | {result['accuracy']:.4f} |\n",
        f"| Macro F1 | {result['macro_f1']:.4f} |\n",
        f"| Weighted F1 | {result['weighted_f1']:.4f} |\n",
        f"| MCC | {result['mcc']:.4f} |\n\n",
        "### Per-class metrics\n\n",
        "| Class | Precision | Recall | F1 | Support |\n",
        "|-------|-----------|--------|----|---------|\n",
    ]
    for cls in result["classes"]:
        m = result["per_class"][cls]
        lines.append(
            f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m['support']:,} |\n"
        )

    # Confusion matrix header
    n = len(result["classes"])
    cm = result["confusion_matrix"]
    pred_headers = " | ".join(f"Pred {c}" for c in result["classes"])
    sep          = "|".join(["--"] * (n + 1))

    lines += [
        "\n### Confusion matrix (rows = actual, cols = predicted)\n\n",
        f"|  | {pred_headers} |\n",
        f"|{sep}|\n",
    ]
    for i, cls in enumerate(result["classes"]):
        row_vals = " | ".join(f"{cm[i][j]:,}" for j in range(n))
        lines.append(f"| **Actual {cls}** | {row_vals} |\n")

    return "".join(lines)


def write_results(results: list[StatsResult], path: Path) -> None:
    """Write a list of StatsResults to a single markdown file.

    Args:
        results: List of StatsResult dicts, one per model.
        path: Destination file path; parent directories are created if absent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Model Evaluation Statistics\n\n"
        "All metrics computed on the held-out test set.\n\n"
    )
    body = "\n---\n\n".join(format_markdown(r) for r in results)
    path.write_text(header + body + "\n")
    print(f"Statistics written to {path}")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def to_dict(result: StatsResult) -> dict:
    """Return a JSON-serialisable dict.

    The confusion_matrix is already list[list[int]], so json.dumps works
    directly on the returned dict.

    Args:
        result: StatsResult returned by compute().

    Returns:
        Plain dict safe to pass to json.dumps().
    """
    return dict(result)


# ===========================================================================
# Backtest — translate model signals into a compounding P&L simulation
# ===========================================================================
#
# Per test bar t (enter at Open, exit at Close — matches the Close>Open label):
#   position_t    = +1 long / -1 short / 0 hold
#   bar_return_t  = (Close_t - Open_t) / Open_t
#   gross_t       = position_t * bar_return_t
#   net_t         = gross_t - transaction_cost * |position_t|
#   equity_t      = equity_{t-1} * (1 + net_t),  equity_0 = initial (reinvest all)
#   payoff_t ($)  = equity_{t-1} * net_t
# Default transaction_cost is 0; it is charged on each bar a position is taken.


class BacktestResult(TypedDict):
    """Outputs of a single backtest run (summary scalars + per-bar arrays)."""

    name:               str
    initial:            float
    transaction_cost:   float
    n_periods:          int
    periods_per_year:   float
    final_equity:       float
    total_return:       float        # final/initial - 1
    max_drawdown:       float        # fraction in [0, 1]
    annualized_sharpe:  float
    # passive (always-long, frictionless) buy-&-hold benchmark
    passive_final_equity:  float
    passive_total_return:  float
    passive_max_drawdown:  float
    passive_sharpe:        float
    start:              str          # first timestamp (minute) or ""
    end:                str          # last timestamp (minute) or ""
    plot_path:          str          # PNG path once plotted, else ""
    # per-bar arrays (length n_periods)
    position:           np.ndarray
    bar_return:         np.ndarray
    net_return:         np.ndarray
    payoff:             np.ndarray
    equity:             np.ndarray
    passive_equity:     np.ndarray
    timestamps:         np.ndarray | None


def _positions_from_signals(
    signals: np.ndarray,
    encoding: str = "binary",
) -> np.ndarray:
    """Map model signals to positions in {-1, 0, +1}.

    Args:
        signals: Predicted labels.
        encoding: "binary" — 1→long(+1), 0→short(-1); "three_class" —
            0→hold(0), 1→long(+1), 2→short(-1).

    Returns:
        Float position array, same length as signals.
    """
    s = np.asarray(signals)
    if encoding == "binary":
        return np.where(s == 1, 1.0, -1.0)
    if encoding == "three_class":
        pos = np.zeros(len(s), dtype=float)
        pos[s == 1] = 1.0
        pos[s == 2] = -1.0
        return pos
    raise ValueError(f"Unknown signal encoding: {encoding!r}")


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough decline of an equity curve, as a fraction.

    Args:
        equity: Equity values over time (should include the starting capital
            as the first element so an early dip is measured from the start).

    Returns:
        Max drawdown in [0, 1]; 0.0 for an empty or never-declining curve.
    """
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / peak
    return float(np.max(drawdown))


def annualized_sharpe(net_returns: np.ndarray, periods_per_year: float) -> float:
    """Annualised Sharpe ratio of a per-period net-return series (rf = 0).

    Args:
        net_returns: Per-bar net returns.
        periods_per_year: Number of bars per year (annualisation factor).

    Returns:
        sqrt(periods_per_year) * mean / std (ddof=1); 0.0 if std is 0 or
        the inputs are degenerate.
    """
    r = np.asarray(net_returns, dtype=float)
    if len(r) < 2 or not np.isfinite(periods_per_year) or periods_per_year <= 0:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / sd)


def _periods_per_year(timestamps: np.ndarray, n: int) -> float:
    """Empirical bars-per-year from the first/last timestamp (gap-robust)."""
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    if len(ts) < 2:
        return float("nan")
    elapsed_s = (ts[-1] - ts[0]) / np.timedelta64(1, "s")
    elapsed_years = float(elapsed_s) / (365.25 * 24 * 3600)
    if elapsed_years <= 0:
        return float("nan")
    return n / elapsed_years


def backtest(
    signals: np.ndarray,
    bar_returns: np.ndarray,
    timestamps: np.ndarray | None = None,
    initial: float = 1000.0,
    transaction_cost: float = 0.0,
    periods_per_year: float | None = None,
    signal_encoding: str = "binary",
    name: str = "",
) -> BacktestResult:
    """Run a compounding long/short backtest of model signals.

    For every record a position is taken from the signal and earns that bar's
    return; net of (optional) proportional transaction cost, equity compounds
    from `initial` (all gains reinvested). Computes the equity curve, max
    drawdown, and annualised Sharpe, plus per-bar payoff/PnL arrays. Also
    computes a passive always-long, frictionless buy-&-hold benchmark
    (`passive_equity` and its summary metrics) on the same per-bar returns.

    Args:
        signals: Model predictions (encoding per `signal_encoding`).
        bar_returns: Per-bar market return (Close - Open) / Open, same length.
        timestamps: Per-bar timestamps; used for Sharpe annualisation and the
            equity-curve x-axis. If None, `periods_per_year` must be given for
            a non-zero Sharpe.
        initial: Starting capital (default $1000).
        transaction_cost: Proportional cost charged per bar a position is taken
            (default 0 = frictionless). E.g. 0.0001 = 1 bp.
        periods_per_year: Annualisation factor; if None, derived from timestamps.
        signal_encoding: "binary" or "three_class".
        name: Display name for the run.

    Returns:
        BacktestResult with summary scalars and per-bar arrays.
    """
    signals     = np.asarray(signals)
    bar_returns = np.asarray(bar_returns, dtype=float)
    if len(signals) != len(bar_returns):
        raise ValueError(
            f"signals ({len(signals)}) and bar_returns ({len(bar_returns)}) "
            "must have the same length"
        )

    position = _positions_from_signals(signals, signal_encoding)
    gross    = position * bar_returns
    net      = gross - transaction_cost * np.abs(position)
    # Cap a single-period loss at -100% so full-reinvestment equity stays >= 0.
    net      = np.maximum(net, -1.0)

    equity      = initial * np.cumprod(1.0 + net)
    prev_equity = np.concatenate(([initial], equity[:-1])) if len(equity) else equity
    payoff      = prev_equity * net

    # Passive benchmark: always-long, frictionless ("what holding would do").
    passive_net    = np.maximum(bar_returns, -1.0)
    passive_equity = initial * np.cumprod(1.0 + passive_net)

    if periods_per_year is None:
        periods_per_year = (
            _periods_per_year(timestamps, len(net))
            if timestamps is not None else float("nan")
        )

    final_equity = float(equity[-1]) if len(equity) else float(initial)
    # Prepend the starting capital so an early dip is measured from `initial`.
    mdd = max_drawdown(np.concatenate(([initial], equity)))

    passive_final = float(passive_equity[-1]) if len(passive_equity) else float(initial)
    passive_mdd   = max_drawdown(np.concatenate(([initial], passive_equity)))

    start = end = ""
    if timestamps is not None and len(timestamps) > 0:
        ts = np.asarray(timestamps, dtype="datetime64[ns]")
        start = np.datetime_as_string(ts[0], unit="m")
        end   = np.datetime_as_string(ts[-1], unit="m")

    return BacktestResult(
        name=name,
        initial=float(initial),
        transaction_cost=float(transaction_cost),
        n_periods=int(len(net)),
        periods_per_year=float(periods_per_year),
        final_equity=final_equity,
        total_return=float(final_equity / initial - 1.0),
        max_drawdown=mdd,
        annualized_sharpe=annualized_sharpe(net, periods_per_year),
        passive_final_equity=passive_final,
        passive_total_return=float(passive_final / initial - 1.0),
        passive_max_drawdown=passive_mdd,
        passive_sharpe=annualized_sharpe(passive_net, periods_per_year),
        start=start,
        end=end,
        plot_path="",
        position=position,
        bar_return=bar_returns,
        net_return=net,
        payoff=payoff,
        equity=equity,
        passive_equity=passive_equity,
        timestamps=(np.asarray(timestamps) if timestamps is not None else None),
    )


def plot_equity_curve(result: BacktestResult, path: Path) -> Path:
    """Plot the equity curve ($ over time) and save it as a PNG.

    Matplotlib is imported lazily (it is an optional dependency).

    Args:
        result: BacktestResult from backtest().
        path: Destination PNG path; parent dirs are created.

    Returns:
        The path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Render '$' literally (don't treat $...$ as a TeX math span in labels/title).
    plt.rcParams["text.parse_math"] = False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    equity  = result["equity"]
    passive = result["passive_equity"]
    ts      = result["timestamps"]
    x       = (np.asarray(ts, dtype="datetime64[ns]")
               if ts is not None else np.arange(len(equity)))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, equity, lw=1.0, color="#1f77b4",
            label=f"Strategy (final ${result['final_equity']:,.0f})")
    ax.plot(x, passive, lw=1.0, color="#ff7f0e", alpha=0.85,
            label=f"Passive buy & hold (final ${result['passive_final_equity']:,.0f})")
    ax.axhline(result["initial"], ls="--", lw=0.8, color="grey",
               label=f"Initial ${result['initial']:,.0f}")
    ax.set_title(
        f"{result['name']} — equity curve "
        f"(cost={result['transaction_cost']:g}); "
        f"final=${result['final_equity']:,.2f}, "
        f"maxDD={result['max_drawdown']:.1%}, "
        f"Sharpe={result['annualized_sharpe']:.2f}  "
        f"| passive=${result['passive_final_equity']:,.2f}"
    )
    ax.set_xlabel("Time" if ts is not None else "Bar")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def format_backtest_markdown(result: BacktestResult) -> str:
    """Format a BacktestResult as a markdown section (scalars + optional plot)."""
    lines = [
        f"## {result['name']}\n\n",
        f"Periods: {result['n_periods']:,}",
    ]
    if result["start"]:
        lines.append(f"  |  {result['start']} → {result['end']}")
    lines += [
        "\n\n",
        "| Metric | Value |\n",
        "|--------|-------|\n",
        f"| Initial capital | ${result['initial']:,.2f} |\n",
        f"| Final equity | ${result['final_equity']:,.2f} |\n",
        f"| Total return | {result['total_return']:.2%} |\n",
        f"| Max drawdown | {result['max_drawdown']:.2%} |\n",
        f"| Annualized Sharpe | {result['annualized_sharpe']:.4f} |\n",
        f"| Transaction cost (per bar) | {result['transaction_cost']:g} |\n",
        f"| Bars / year | {result['periods_per_year']:,.0f} |\n",
        f"| Passive final equity | ${result['passive_final_equity']:,.2f} |\n",
        f"| Passive total return | {result['passive_total_return']:.2%} |\n",
        f"| Passive max drawdown | {result['passive_max_drawdown']:.2%} |\n",
        f"| Passive Sharpe | {result['passive_sharpe']:.4f} |\n",
        f"| Strategy − Passive (return) | "
        f"{(result['total_return'] - result['passive_total_return']):+.2%} |\n",
    ]
    if result["plot_path"]:
        rel = Path(result["plot_path"]).name
        lines.append(f"\n![{result['name']} equity curve]({rel})\n")
    return "".join(lines)


def write_backtest_results(results: list[BacktestResult], path: Path) -> None:
    """Write a list of BacktestResults to one markdown file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Backtest — Strategy Equity Statistics\n\n"
        "Long/short, fully-invested, compounding from the initial capital on the "
        "50/50 time-ordered test set. Position is +1 (predict up) / -1 (predict "
        "down); per-bar return = (Close - Open) / Open; net return deducts the "
        "per-bar transaction cost. Risk-free rate = 0 for the Sharpe ratio.\n\n"
    )
    body = "\n---\n\n".join(format_backtest_markdown(r) for r in results)
    path.write_text(header + body + "\n")
    print(f"Backtest statistics written to {path}")
