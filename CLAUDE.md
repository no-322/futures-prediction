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