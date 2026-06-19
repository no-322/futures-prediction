# Optimizing the No-Flat Test Score — Analysis & Roadmap

*Status: active research direction. Optimization target = **no-flat test accuracy**
(report MCC alongside). Last updated after the no-flat-test slice landed.*

## Why we're doing this

We added a **no-flat test slice**: a view of the held-out test set with flat
(`Open == Close`) bars removed from the **metrics only** — predictions are
unchanged (they were generated on the whole test set, blind to flatness), so this
is a reporting slice, not a refit, and it does not violate the "test set is
sacred" rule. Flat bars are ~**49%** of the test set and are ambiguous (neither up
nor down, yet forced to class 0 by the label spec), so including them inflates the
"always-down" portion of accuracy and hides whether a model has any real edge.

With flat bars removed, the scores tell a clear story and give us a clean metric
to optimize.

## What the no-flat scores reveal

No-flat test slice = 140,613 of 275,759 rows (135,146 flat dropped, 49.01%).
Class balance: 69,494 down (49.4%) / 71,119 up (50.6%) → "always-up" baseline ≈ **0.506**.

| Model (no-flat trained) | Feature set | Accuracy | MCC |
|---|---|---|---|
| **Logistic Regression** | **v1 (20 raw)** | **0.5472** | **0.0942** |
| Gradient Boosting (XGBoost) | v2 (49) | 0.5335 | 0.0660 |
| SVM (RBF) | v2 (49) | 0.5326 | 0.0649 |
| Logistic Regression | v2 (49) | 0.5285 | 0.0558 |
| HMM-regime binary | — | 0.5195 | 0.0385 |
| Random Forest | v2 (49) | 0.5215 | 0.0424 |
| Random Forest | v1 (20) | 0.5071 | 0.0141 |
| SVM (RBF) | v1 (20) | 0.5027 | 0.0048 |
| Gradient Boosting | v1 (20) | 0.5028 | 0.0020 |

(Flat-*included*-trained models collapse to ~0.494 on this slice — worse than the
baseline — because they mostly predict the class-0 flat majority. Only no-flat
training is viable; that's settled.)

### Three diagnoses

1. **The simplest model wins.** A plain logistic regression on 20 raw features
   beats every SVM/RF/GBM and every 49-feature variant. The predictable component
   is **weak and roughly linear**, and the complex models **overfit**.
2. **RF is the worst-regularized.** `config.yaml` sets `rf.max_depth: null` →
   trees grow to pure leaves → near-random on the no-flat test. Prime regularization target.
3. **More features hurt the linear model** (v1 LogReg 0.5472 > v2 LogReg 0.5285).
   The 29 extra v2 features add more noise than signal for the model that
   generalizes best → a feature-selection / denoising opportunity.

Plus a structural concern: the base 20 features are **raw price levels** spanning
13 contracts over 3 years. Raw levels are **non-stationary**, so a model keying on
absolute price cannot transfer cleanly from the 2023→mid-2024 train half to the
mid-2024→2026 test half.

### Reality check

One-minute direction is near-efficient-market hard. MCC in the **0.06–0.10** range
is already a real (if small) edge. Expect **incremental** gains, and keep the
backtest (equity / Sharpe net of cost) as a secondary, economic sanity check.

## Directions considered

| Direction | Verdict | Rationale |
|---|---|---|
| **HPO + regularization harness** | **Chosen** | Includes the SVM/RF regularization lean; RF's unbounded depth is the clearest overfit. Must be done inside a proper validation harness. |
| **Feature selection + stationarity** | **Chosen** | Linear-wins + fewer-features-better + non-stationary raw levels all point here. Likely the biggest lever. |
| **Threshold tuning, calibration & \|move\| weighting** | **Chosen** | Cheap multipliers; \|move\| weighting also aligns training with backtest P&L. |
| Ensembling / stacking | Deferred | Reliable small boost, but only worthwhile after individual models are tuned. |

**Selection metric:** no-flat **test accuracy** is the headline objective; models
are *selected* on no-flat **validation** accuracy (see guardrail). MCC is reported
alongside because it's free and robust to slight imbalance.

## The critical guardrail

To "optimize the no-flat test score" without fooling ourselves, **the test half is
touched once, at the very end.** All hyperparameter search, feature selection, and
threshold tuning happen on a **no-flat validation fold carved from the *training*
half** (time-ordered: last ~20% of train). Repeatedly tuning against the test score
*is* test leakage. This is the enabling discipline for every direction above.

## Roadmap (implementation)

**W1 — Model-selection harness (`src/tuning.py`), the backbone.**
Carve inner-train / no-flat-validation from the train half; grid-search each model
on no-flat val accuracy; retrain the best config on the full train half; evaluate
**once** on the real no-flat test; persist predictions + best params + report.
Curated, seed-42 grids — RF: cap `max_depth`, raise `min_samples_leaf`; GBM:
shallower trees, lower LR + early stopping, tune `reg_alpha`/`reg_lambda`; SVM:
sweep `C`/`gamma` (subsampled for tractability); LogReg: `C` + L1/L2/elastic-net.

**W2 — Stationary features + selection.** New `src/features_v3.py` expressing the
base lags as log-returns / deviations vs `t-1` close (keeping v2's already-
stationary indicators), built under the `feature-engineering` skill. Plus L1/
importance-based feature selection chosen on the val fold. Compare v1/v2/v3/
v3-selected through the harness.

**W3 — Threshold + |move| weighting.** Tune the decision threshold on the val fold;
add an optional `sample_weight` (= normalized `|Close − Open|`) to each model's
`train()` so decisive bars dominate. Calibration is an optional stretch.

**The bar to beat:** current champion = LogReg v1 no-flat, **acc 0.5472**.

## Results — first tuning pass (baseline/rf/gbm × v1/v2/v3, threshold-tuned)

Selection on the no-flat validation fold (val_frac=0.2 → inner-train 120,575 /
no-flat val 27,338); final fit on the full no-flat train (147,913); evaluated
once on the no-flat test (140,613). SVM and |move|-weighting/selection not yet run.

| Featset | Model | Val acc | Test acc (no-flat) | Test MCC |
|---|---|---|---|---|
| v1 | **Logistic Regression** | 0.5346 | **0.5461** | 0.0950 |
| v1 | Random Forest | 0.5095 | 0.5010 | 0.0050 |
| v1 | Gradient Boosting | 0.5106 | 0.5060 | 0.0056 |
| v2 | Logistic Regression | 0.5309 | 0.5310 | 0.0611 |
| v2 | Random Forest | 0.5240 | 0.5256 | 0.0513 |
| v2 | Gradient Boosting | 0.5245 | 0.5335 | 0.0665 |
| v3 | Logistic Regression | 0.5256 | 0.5203 | 0.0390 |
| v3 | **Random Forest** | 0.5344 | **0.5424** | 0.0841 |
| v3 | **Gradient Boosting** | 0.5376 | **0.5442** | 0.0890 |

Findings:
1. **Harness validated.** Tuned v1 LogReg reproduces the known champion
   (0.5461 / MCC 0.0950 ≈ the prior 0.5472 / 0.0942) — the validation→test
   pipeline is correct and not leaking.
2. **Regularization + stationarity transformed the tree models.** RF went from
   0.5010 (v1, unbounded depth) to **0.5424** (v3, capped depth); GBM from 0.5060
   to **0.5442**. RF MCC rose ~17× (0.005 → 0.084). This confirms the
   over-fitting diagnosis and the value of v3's stationary features for trees.
3. **The linear/nonlinear gap collapsed** from ~0.04 to ~0.002. By honest
   validation-based selection, **GBM-v3 is now the pick** (val 0.5376 → test
   0.5442) — a richer model essentially tying the linear champion, with threshold
   tuning, |move| weighting, SVM, and ensembling still untried.
4. The log-ratio transform *hurt* the linear model (LogReg v1 0.5461 → v3 0.5203);
   trees love v3, the linear model preferred raw levels. Keep both tracks.

Still to run: SVM on v2/v3; `--move-weight`; `--select`; then revisit ensembling.

## Verification

- New/changed tests pass, including a **leakage test**: `build_selection_split`
  never returns test-half rows (val timestamps all precede the test-half start).
- `python -m src.tuning ...` runs end to end and writes `tuned_*_predictions.npz`,
  `tuned_params.json`, and `docs/notes/tuning_stats.md`.
- Tuned RF no-flat test acc should rise above its current 0.5215 once depth is
  capped; overall best should challenge 0.5472.
- Determinism: re-running a config reproduces identical predictions (seed 42).
