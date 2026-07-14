# Futures Price Prediction — Project Guide

## What this project does
Predict whether a futures contract's price goes **up or down** in the next minute, using
lagged minute-bar features. Train and compare three classifiers — **Logistic Regression,
Random Forest, Gradient Boosting** — against simple baselines, across several parallel
feature pipelines, using rolling walk-forward validation.

## Directory layout
```
.
├── CLAUDE.md
├── .claude/
│   ├── settings.json        # hooks (prompt log, post-edit tests, doc regen)
│   ├── settings.local.json
│   ├── scripts/             # hook scripts (log-prompt.sh, run-test.sh, gen-docs.sh†)
│   ├── skills/              # auto-triggered procedural knowledge (evaluation, feature-engineering)
│   └── agents/              # subagents (leakage-auditor)
├── data/
│   ├── raw/                 # original data, read-only, never modified
│   └── processed/           # generated artifacts (models, predictions), gitignored
├── src/
│   ├── load.py              # load + validate raw data
│   ├── split.py             # time-ordered train/test split
│   ├── labels.py            # binary up/down labels + move/flat helpers
│   ├── features.py          # v1 pipeline — raw OHLCV lags
│   ├── features_v1_rel.py   # v1-rel pipeline — v1 log-ratios vs lag1_Open (stationary)
│   ├── features_v2.py       # v2 pipeline — OHLCV lags + derived indicators + time (cached)
│   ├── features_v3.py       # v3 pipeline — v2 log-ratios vs lag1_Close (stationary, cached)
│   ├── features_orderflow.py# order-flow pipeline — volume/tick-flow lags (raw + linear)
│   ├── evaluate.py          # accuracy/recall/confusion + baselines → results.md
│   ├── statistics.py        # standardised metrics + backtest (equity / P&L / AUM)
│   ├── backtest.py          # backtest runner (equity curve vs passive)
│   ├── walkforward.py       # rolling walk-forward harness (the iteration metric)
│   ├── binary_suite.py      # single-split binary trainer (per feature-set)
│   ├── tuning.py            # hyperparameter + threshold tuning on a validation fold
│   ├── run_stats.py         # batch stats / leaderboards / walk-forward orchestrator
│   ├── pipeline.py          # end-to-end driver for one production model
│   ├── feature_importance.py# per-model importance report
│   ├── config.py            # config.yaml loader + model_params
│   ├── gen_module_docs.py†  # AST → module_diagram.md + MODULES.md tables (Stage 2)
│   └── models/{logistic,rf,gbm,regime_hmm,regime_binary}.py   # logistic ← was baseline; svm removed
├── app.py                   # Streamlit GUI (Train / Predict / Backtest Explorer)
├── config.yaml              # data path, walk-forward windows, model hyperparameters
├── tests/                   # one test file per src module
├── prompts/log.md           # auto-logged by hook
├── docs/
│   ├── MODULES.md           # function reference — signature + description + pseudocode
│   ├── INTERFACES.md        # per-module data contracts (swap any stage)
│   ├── FEATURES.md          # feature-pipeline registry (source of truth for features)
│   ├── results.md           # per-model evaluation report (walk-forward)
│   └── notes/               # generated reports + module_diagram.md + the two leaderboards
│       ├── leaderboard.md†             # single test-set leaderboard (Stage 2 rename)
│       ├── leaderboard-walk-forward.md†# walk-forward leaderboard (Stage 2)
│       ├── module_diagram.md           # ASCII module diagram + inventory (auto-updated)
│       └── ...                         # top5_evaluation, tuning/backtest stats, etc.
└── pyproject.toml

† = introduced/renamed by the Stage-2 refactor (see "Roadmap" below); not yet present.
```

## Data

**File:** `data/raw/data.csv` — read-only, never modified.

**Shape:** 551,521 rows × 14 columns.

**Schema:**

| Column        | dtype        | Notes                                           |
|---------------|--------------|-------------------------------------------------|
| Date          | str          | M/D/YYYY — redundant with "Date and Time"       |
| Time          | str          | H:MM:SS — redundant with "Date and Time"        |
| Date and Time | str→datetime | Parse as the primary timestamp index            |
| Symbol        | str          | TY (10-Year Treasury Note) futures contract symbol |
| Open          | float64      | Minute-bar open price                           |
| High          | float64      | Minute-bar high price                           |
| Low           | float64      | Minute-bar low price                            |
| Close         | float64      | Minute-bar close price                          |
| VWAP          | float64      | Volume-weighted average price for the bar       |
| Volume        | int64        | Share/contract volume; min=1, no zero rows      |
| Up Ticks      | int64        |                                                 |
| Down Ticks    | int64        |                                                 |
| Same Ticks    | int64        |                                                 |
| Tick Count    | int64        |                                                 |

**Sort order:** Monotonically increasing by "Date and Time" — no backward jumps, no
duplicate timestamps. Rows are already sorted; do not re-sort.

**Data quality:**
- No NaN values in any column.
- No zero-volume rows.
- Features must handle time gaps correctly: **do not assume consecutive rows are
  consecutive minutes.** Use timestamp arithmetic, not row offsets, when computing lags
  (build features on a dense 1-minute grid, forward-filling gap minutes).

## Non-negotiable rules

1. **Random seed is `42`** everywhere randomness appears.
2. **No look-ahead bias.** Features for minute `t` use only data from minutes strictly
   less than `t`. Never include minute `t` itself.
3. **The test set is sacred.** Never fit transformers, compute statistics, or tune
   hyperparameters using test data.
4. **Split is time-ordered, not shuffled.** The first fraction of rows by timestamp is
   training; the remainder is test (`train_size` in `config.yaml`).
5. **Every `src/` module has a matching `tests/test_*.py`.**
6. **`docs/MODULES.md` documents everything with pseudocode.** Each function gets one
   entry: type-hinted **signature**, a one-line **description**, and a **pseudocode block
   of its logic**. This applies to *everything* MODULES.md covers — ETL / data-engineering
   steps and the construction logic of **each feature created**. Signatures + the module
   diagram/inventory are kept in sync by `gen_module_docs.py` (Stage 2); the pseudocode
   block is author-written and its presence is enforced by the post-edit hook.
7. **Drop flat bars.** Rows where `Open == Close` are removed from the dataset **before
   modelling** (train and test). Labels are therefore strictly binary; flat bars do not
   exist downstream. No "flat-included" metric is produced.
8. **Always persist predictions.** Any run that produces predictions (`y_true`, `y_pred`)
   MUST save them to `data/processed/*.npz` before returning. Never discard predictions in
   memory only — stats, leaderboards, and reproducibility depend on reloading them.
9. **Keep the module diagram current.** After any change to `src/`, `module_diagram.md`
   (ASCII diagram + module inventory) is auto-updated by `gen_module_docs.py` via the
   post-edit hook.

## Feature pipelines

Feature engineering is **not a single fixed vector** — the project runs several parallel
feature pipelines. **`docs/FEATURES.md` is the source of truth**: it registers every
pipeline (v1, v1-rel, v2, v3, order-flow, HMM-regime) with its column list and per-feature
pseudocode.

**Every feature pipeline must satisfy this contract:**
- Built on the **dense 1-minute grid** (reindex to continuous minutes, forward-fill gaps),
  so lags are true clock-minute offsets, not row offsets.
- **Causal only:** every emitted column is a `shift(k)` with **k ≥ 1** — no lag-0 / current
  bar, no `shift(-1)`. A feature for minute `t` depends only on data ≤ `t-1`.
- **Warm-up rows dropped** (and any residual NaN filled + logged), index reset 0-based,
  row-aligned with labels and timestamps.
- **Registered** in `docs/FEATURES.md` and documented in `docs/MODULES.md` with pseudocode.

New feature work adds a pipeline (or a variant) under this contract rather than editing a
single canonical vector.

## Label specification
After flat bars are dropped (rule 7), the label is exactly two classes:
`label(t) = 1 if Close(t) > Open(t)`; `label(t) = 0 if Close(t) < Open(t)`.
There is no third (flat) class — `Close(t) == Open(t)` rows are not in the dataset.

## Models
Three classifiers on the same labels, compared across feature pipelines:
- **Logistic Regression** (`src/models/logistic.py`) — the current best performer,
  promoted to its own module (was `baseline.py`). Scale features when a pipeline's columns
  are not already O(1) (fit the scaler on train only).
- **Random Forest** (`src/models/rf.py`).
- **Gradient Boosting** (`src/models/gbm.py`) — XGBoost.

(SVM has been removed.) Naive baselines for comparison:
- Always predict 1 (up).
- Predict last observed direction.

## Evaluation format
The iteration metric is **rolling walk-forward** (see the `evaluation` skill). Every model
report — written to `docs/results.md` — states results as the **mean ± std across folds**:
- **Accuracy** (flat already dropped, so this is the decisive-bar accuracy).
- **Recall.**
- **Confusion matrix.**
- **Δ vs the naive baselines** (improvement/decrease vs always-up and predict-last-direction).
- **Backtest: % increase in AUM** (from the compounding backtest's total return), plus
  drawdown / Sharpe vs passive buy-and-hold.

## Leaderboards
Every new model's performance is tracked in **two** leaderboards:
- **`docs/notes/leaderboard.md`** — single time-ordered test-set performance.
- **`docs/notes/leaderboard-walk-forward.md`** — walk-forward performance, **ranked by
  mean accuracy across folds**, with a **"folds won"** column (how many windows the model
  beats the baseline) as the stability signal.

Both rank on accuracy and also show the backtest (AUM %). There is no flat-included column.

## Roadmap (Stage 2 — pending)
CLAUDE.md now describes the target conventions; the code refactor that makes the repo match
is a separate, gated step: remove SVM; split logistic out of `baseline.py` → `logistic.py`
(migrate the `baseline` algo key + regenerate stems); drop flat at load time and simplify
the walk-forward keep-mask harness; rework `statistics`/`evaluate` to the walk-forward
report format with AUM%; build `gen_module_docs.py` + the doc-regen hook and the two
leaderboard generators; then regenerate `results.md`, `MODULES.md`, `module_diagram.md`,
and both leaderboards. Items marked † above arrive in this stage.

## Commands
```bash
pytest                           # all tests
pytest tests/test_features.py    # one test file
python -m src.load               # load + validate data
python -m src.walkforward --algo logistic --featset v1   # walk-forward one model
python -m src.run_stats --sections lb                    # rebuild leaderboards
```

## Coding conventions
- Python 3.11+, type hints required, docstrings required.
- Pure functions where possible; isolate I/O to `load.py` and `evaluate.py`.
- Notebooks are for exploration only — production code lives in `src/`.

## Prompt logging
Every prompt I submit is auto-appended to `prompts/log.md` by a UserPromptSubmit hook.
Commit it alongside the code it produced — this is the project's version-controlled
reasoning trail.

## Never do
- Shuffle rows before splitting.
- Fit a scaler on the full dataset (fit on train only, inside each fold).
- Keep flat (`Open == Close`) rows in the modelling dataset.
- Modify `data/raw/`.
- Delete or rewrite `prompts/log.md`.
- Silently drop NaN rows — log the count.
- Discard `y_true` / `y_pred` arrays at the end of a run without saving them to `data/processed/`.
