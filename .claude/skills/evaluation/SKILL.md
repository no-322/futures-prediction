---
name: evaluation
description: Out-of-sample validation methodology for this Treasury-futures prediction project. Use this whenever building or running model validation, comparing models or feature sets, reporting accuracy, selecting a confidence threshold, or evaluating by regime — even when the request is just "test the model", "check if X beats Y", or "is this better". Defines the rolling walk-forward harness, the leakage rules, and the reporting conventions that every evaluation in this repo must follow.
---

# Evaluation & Validation

The iteration metric is rolling walk-forward. The final held-out test set is touch-once. Apply these rules to every model comparison, threshold choice, and accuracy report in this repo.

## Walk-forward harness (the iteration metric)
- Rolling fixed-width window, not expanding: a train block then a test block, stepped forward. Rolling because edge here is regime-dependent and a fixed window tracks regime shifts and contract rolls.
- Window sizes come from config (default: 3 months train / 1 month test). Never hardcode them in the function.
- Test-block timestamps are always strictly greater than train-block timestamps. Assert per fold: `train_idx.max() < test_idx.min()`.
- Assert the time index is monotonically sorted before splitting. Contract-roll concatenation can silently break the forward-only assumption.
- Fresh model instance per fold. Never reuse a fitted estimator across folds.
- Report per-fold accuracy AND mean ± std. Per-fold is mandatory: the between-period spread is the regime signal, and the only way to tell a stable improvement (wins in every fold) from a single-period fluke (wins in some, loses in others).

## Leakage rules
- Fit ALL scaling, normalization, and feature statistics inside the fold, on the training block only. Never fit on the full series. Covers scalers, rolling z-scores, and any window statistic.
- Rolling and lagged features use past data only (`shift(1)`, etc.). No `shift(-1)`. No full-series statistic that sees the test period.
- The final held-out test set is touch-once. Never tune, threshold-select, or make any keep/drop decision against it. All interim comparisons run on walk-forward. A number obtained by repeatedly reading the test set is selection bias, not out-of-sample performance.

## purge & embargo
- Expose `purge` and `embargo` as parameters; default both to 0 (inert).
- For the current single-bar label (`close(t) > open(t)`), purge is a no-op: labels do not overlap in time, so there is nothing to purge. Keep the parameter for later.
- Purge becomes load-bearing only under a forward-horizon or triple-barrier label: then purge the tail bars of each train block whose label window crosses into the test block.
- Embargo guards cross-sided folds (train after test); in a forward-only walk it is near-inert. A 1–2 bar embargo is cheap insurance for adjacent-boundary effects.

## CPCV — verdict stage, not iteration
- Do not use CPCV as the day-to-day iteration metric. It costs C(N,k)× the training time and is too slow for rapid comparison.
- Reserve CPCV for final consolidation, where the distribution of out-of-sample paths feeds probability of backtest overfitting / deflated Sharpe. It pairs with the forward-horizon label, where purge also starts to matter.

## Reporting & decisions
- State results as "X% ± y% across N folds (range a–b)", never a bare point estimate.
- "Better" means significantly better, not numerically better. Confirm interim walk-forward improvements with McNemar's test at consolidation.
- Model complexity that does not beat the linear baseline (logistic regression) under walk-forward is overfitting surface — do not ship it.
- Prefer the conditional framing for the accuracy target: accuracy on a calibrated high-confidence subset, reported with its coverage, over a forced unconditional number.
