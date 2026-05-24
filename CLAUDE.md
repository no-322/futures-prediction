# Futures Price Prediction — Project Guide

## What this project does
Predict whether a futures contract's price goes up or down in the next minute, using lagged minute-bar features. Train and compare three classifiers (SVM, Gradient Boosting, Random Forest) against simple baselines.

## Directory layout
```
.
├── CLAUDE.md
├── .claude/
│   ├── settings.json        # hooks
│   ├── scripts/             # hook scripts
│   └── skills/              # auto-triggered procedural knowledge
├── data/
│   ├── raw/                 # original data, read-only, never modified
│   └── processed/           # generated artifacts, gitignored
├── src/
│   ├── load.py              # load + validate raw data
│   ├── split.py             # 50/50 time-ordered split
│   ├── features.py          # 20-dim lagged feature vector
│   ├── labels.py            # up/down labels
│   ├── evaluate.py          # accuracy, confusion matrix, baselines
│   └── models/{rf,gbm,svm}.py
├── tests/                   # one test file per src module
├── prompts/log.md           # auto-logged by hook
├── docs/
│   ├── MODULES.md           # function table for the professor
│   └── notes/               # ESL reading notes
└── pyproject.toml
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
| Symbol        | str          | 13 TY (10-Year Treasury Note) futures contracts |
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

**Time range:** 2023-01-03 04:01 → 2026-01-09 16:00 (~3 years of minute bars).

**Sort order:** Monotonically increasing by "Date and Time" — no backward jumps, no duplicate timestamps. Rows are already sorted; do not re-sort.

**Symbols:** 13 TY futures contracts (delivery months H/M/U/Z, years 2023–2026). Each symbol covers roughly one quarterly contract period (~44,000–46,500 rows each); TYH23 and TYH26 are shorter (contract boundaries).

**Data quality:**
- No NaN values in any column.
- No zero-volume rows.
- ~0.76% of rows (4,201) have a gap > 1 minute to the next row:
  - 2-minute gaps: 3,086 — normal tick consolidation or thin-market minutes.
  - Overnight/session-boundary gaps (≥12 h): 598 — Treasury futures close and reopen.
  - 3-minute gaps: 277.
  - Multi-day gaps: 145 — weekends and holidays.
- Features must handle these gaps correctly: **do not assume consecutive rows are consecutive minutes**. Use timestamp arithmetic, not row offsets, when computing lags.

## Non-negotiable rules

1. **Random seed is `42`** everywhere randomness appears.
2. **No look-ahead bias.** Features for minute `t` use only data from minutes strictly less than `t`. Never include minute `t` itself.
3. **The test set is sacred.** Never fit transformers, compute statistics, or tune hyperparameters using test data.
4. **Split is time-ordered, not shuffled.** First 50% of rows by timestamp = training; second 50% = test.
5. **Every `src/` module has a matching `tests/test_*.py`.**
6. **One pseudo-code line = one function = one entry in `docs/MODULES.md`.** Each function has type hints and a docstring with Args/Returns.

## Feature specification
For each target minute `t`, build a 20-dim vector from minutes `t-4, t-3, t-2, t-1`. For each of those four minutes, extract:
- Open
- Close
- High
- Low
- VWAP

Order: `[m-4 features, m-3 features, m-2 features, m-1 features]`.

## Label specification
`label(t) = 1 if close(t) > Open(t) else 0`.

## Models
Three classifiers on the same features and labels:
- Random Forest (`src/models/rf.py`)
- Gradient Boosting (`src/models/gbm.py`) — XGBoost or sklearn's GradientBoostingClassifier
- SVM (`src/models/svm.py`) — RBF kernel; scale features first (fit scaler on train only)

Baselines for comparison:
- Always predict 1 (up)
- Predict last observed direction

## Evaluation format
For every model write to `docs/results.md`:
- Accuracy on test set
- Recall on test set
- Confusion matrix
- Comparison against both baselines

## Commands
```bash
pytest                           # all tests
pytest tests/test_features.py    # one test file
python -m src.load               # load + validate data
python -m src.models.rf          # train and evaluate RF
```

## Coding conventions
- Python 3.11+, type hints required, docstrings required.
- Pure functions where possible; isolate I/O to `load.py` and `evaluate.py`.
- Notebooks are for exploration only — production code lives in `src/`.

## Prompt logging
Every prompt I submit is auto-appended to `prompts/log.md` by a UserPromptSubmit hook. Commit it alongside the code it produced — this is the project's version-controlled reasoning trail.

## Never do
- Shuffle rows before splitting.
- Fit a scaler on the full dataset.
- Modify `data/raw/`.
- Delete or rewrite `prompts/log.md`.
- Silently drop NaN rows — log the count.