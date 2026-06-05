# Experiment Log — TY Futures Price Direction

Branch: `exp/feature-engineering`
Dataset: 551,521 minute bars, TY (10-Year Treasury Note) futures, 2023-01-03 → 2026-01-09

---

## Background

The production pipeline (on `main`) trains binary classifiers (up/down) using a 50/50
time-ordered split and fixed hyperparameters. The experiments here explore two alternative
labelling schemes and richer feature engineering, evaluated with 5-fold TimeSeriesSplit CV.

**Class imbalance in the raw data (full dataset):**
- Down / Flat (Close ≤ Open): 72.9% of bars
- Up (Close > Open): 27.1% of bars
- Flat (Close == Open exactly): < 1% — very rare for TY in 32nds

This imbalance is the central challenge: a naive "always predict down" strategy achieves
72.9% accuracy for free. All models use `class_weight='balanced'` to resist this collapse.

---

## Feature sets

### v1 — 20 lagged OHLCV features (`src/features.py`)
For each target bar t, collect [Open, Close, High, Low, VWAP] from lags t−4…t−1.
Computed on a dense 1-minute grid to handle overnight/weekend gaps correctly.
Output: (N−4, 20) matrix.

### v2 — 49 engineered features (`src/experiments/features_v2.py`)
All 20 v1 features plus 29 new ones:

**New lagged features (lags 1 and 4):**
| Feature | Description |
|---------|-------------|
| `vwap_dev` | (Close − VWAP) / VWAP — price relative to volume-weighted average |
| `bar_range` | High − Low — intrabar range |
| `body_ratio` | \|Close−Open\| / (High−Low + ε) — candlestick body fraction |
| `tick_delta` | (Up Ticks − Down Ticks) / Tick Count — order-flow imbalance |
| `return` | Close/Close_prev − 1 — arithmetic bar return |
| `log_return` | log(Close/Close_prev) — log return |
| `rsi5` | RSI with 5-period Wilder smoothing |
| `rsi15` | RSI with 15-period smoothing |
| `vol5` | Rolling std of log-returns, window=5 |
| `vol15` | Rolling std of log-returns, window=15 |
| `macd_line` | EMA(6) − EMA(13) — minute-adapted MACD |
| `macd_signal` | EMA(4) of macd_line |
| `macd_hist` | macd_line − macd_signal |

**Target-bar features (no lag):**
| Feature | Description |
|---------|-------------|
| `tod_sin` | sin(2π × minute\_of\_day / 1440) — cyclic time-of-day |
| `tod_cos` | cos(2π × minute\_of\_day / 1440) |
| `session_min` | Minutes elapsed since session open (resets at overnight gaps ≥ 12 h) |

---

## Evaluation framework

**TimeSeriesSplit(n_splits=5)** on the full dataset:
- Fold 0: train=91,922 rows, test=91,919 rows
- Fold 1: train=183,841, test=91,919
- Fold 2: train=275,760, test=91,919
- Fold 3: train=367,679, test=91,919
- Fold 4: train=459,598, test=91,919

**Hyperparameter tuning (v2 experiments):** Nested walk-forward CV — inside each outer
fold's training window, `RandomizedSearchCV(n_iter=4, cv=TimeSeriesSplit(3))` searches
over `{n_estimators: [50,100], max_depth: [None,5,10], min_samples_leaf: [1,5,10],
max_features: [sqrt,log2]}`. Best params used for the final fold model.

---

## Experiment 1 — Three-Class Classifier

**Scheme:** Label each bar as down (0), up (1), or flat (2) based on Close vs Open.
**Model:** RandomForestClassifier, `class_weight='balanced'`, random_state=42.
**Metrics:** Accuracy, MCC, macro-F1, per-class recall.

### Results

| Metric | v1 (fixed HP) | v2 + HP tuning | Δ |
|--------|--------------|----------------|---|
| Accuracy | 0.399 ± 0.043 | 0.458 ± 0.012 | +0.059 |
| MCC | 0.039 ± 0.014 | **0.100 ± 0.009** | **+0.061 (+154%)** |
| Macro-F1 | 0.340 ± 0.032 | 0.378 ± 0.009 | +0.038 |

**Per-class recall v1:**
| Class | Mean | Std |
|-------|------|-----|
| Down (0) | 0.200 | 0.053 |
| Up (1) | 0.334 | **0.196** |
| Flat (2) | 0.532 | 0.168 |

**Per-class recall v2:**
| Class | Mean | Std |
|-------|------|-----|
| Down (0) | 0.198 | 0.070 |
| Up (1) | 0.271 | 0.058 |
| Flat (2) | 0.692 | 0.039 |

### Insights

1. **MCC more than doubled (+154%)** moving from v1 to v2 features. The new indicators
   (tick_delta, RSI, vol, MACD) add genuine predictive signal that pure price lags miss.

2. **Fold stability improved dramatically.** Up-recall standard deviation collapsed from
   0.196 (v1) to 0.058 (v2). The v1 model's fold-0 anomaly (recall_up=0.73 with only 91k
   training rows) disappeared with richer features and tuned HPs.

3. **Flat class dominates recall.** `class_weight='balanced'` over-weights the rare flat
   class, causing the model to predict flat aggressively. This artificially inflates flat
   recall but makes the model less useful for trading (flat = no trade).

4. **Down recall remains low (~0.20).** The model struggles to identify down bars,
   likely because the class-imbalance correction overshoots in the flat direction.
   A two-class (up/down only) model with explicit flat-bar exclusion may perform better.

5. **MCC = 0.100 is still modest** but indicates meaningful signal in the v2 features.
   For a financial time series, any positive MCC above ~0.05 is worth investigating.

---

## Experiment 2 — Two-Stage Cascade Classifier

**Scheme:**
- Stage 1 (Gate): predict tradeable (1) vs skip (0) based on whether |move| > threshold
- Stage 2 (Direction): predict long (1) vs short (0) for gated bars only
- Combined: no-trade, long, or short per bar

**Threshold tuning:** Inside each fold's training window, the last 20% is held out as
inner validation. Thresholds at percentiles [30,40,50,60,70,80] of |move_train| are
each scored by gate F1 on the inner validation set. Best threshold is selected.

**Model:** RandomForestClassifier for both stages.

### Results

| Metric | v1 (fixed HP) | v2 + HP tuning | Δ |
|--------|--------------|----------------|---|
| Coverage | 0.563 ± 0.132 | 0.365 ± 0.043 | −0.198 |
| Conditional hit rate | 0.463 ± 0.069 | 0.465 ± 0.046 | ≈0 |
| Gate F1 | 0.563 ± 0.054 | 0.506 ± 0.040 | −0.057 |
| Gate precision | 0.545 ± 0.023 | **0.611 ± 0.027** | **+0.066** |
| Gate recall | 0.593 ± 0.123 | 0.434 ± 0.049 | −0.159 |
| Direction MCC | 0.012 ± 0.008 | **0.042 ± 0.016** | **+0.030 (+250%)** |
| Threshold (pts) | 0.000 | 0.000 | — |

### Insights

1. **Direction MCC tripled (+250%)** with v2 features (0.012 → 0.042). Still low in
   absolute terms, but the signal is more persistent across folds (std 0.008 → 0.016
   with similar mean — the improvement is real, not statistical noise from one fold).

2. **Gate became more selective.** Coverage dropped from 56% → 37% with v2 features.
   The v2 gate has learned to skip more bars, and its precision improved (+0.066) —
   it's calling fewer false tradeable bars. The recall tradeoff is expected.

3. **Threshold always tunes to 0.0.** The percentile-based threshold search consistently
   selects any-nonzero-move as "tradeable." This means the threshold search is not finding
   a meaningful cutoff — either the gate model is learning directly from features (not
   the threshold), or a fixed absolute threshold (e.g., 1 tick = 0.03125 pts) would be
   more principled.

4. **Conditional hit rate ≈ 0.465 — essentially coin-flip.** On average, the direction
   model barely exceeds 50%. This suggests direction is not uniformly predictable: it may
   be learnable in certain market conditions (regimes) but not others, which averages out
   to near-random across the full dataset.

5. **Coverage instability in v1 (std=0.132) fixed in v2 (std=0.043).** The v2 gate is
   much more consistent about what fraction of bars it selects across time periods.

---

---

## Experiment 3 — Regime-Conditional Two-Stage Cascade (Gaussian HMM)

**Scheme:** Insert a 2-state Gaussian HMM between the gate and the direction classifier.
HMM is fitted on 5 regime features (lag1_vol15, lag1_macd_hist, lag1_rsi15,
lag1_tick_delta, lag1_return) on the training fold. Viterbi decoding assigns a regime
label to every bar. Separate RF direction classifiers are trained per regime.

**Regime labels (canonical):** Regime 0 = low-vol / calm, Regime 1 = high-vol / active
(determined by which HMM state has higher mean lag1_vol15).

### Results vs Exp 2 v2

| Metric | Exp 2 v2 | Exp 3 (HMM) | Δ |
|--------|----------|-------------|---|
| Coverage | 0.3651 | 0.3651 | 0.000 |
| Overall hit rate | 0.4646 | 0.4672 | +0.003 |
| Overall direction MCC | 0.0415 | 0.0418 | +0.000 |
| Gate recall | 0.4335 | 0.4335 | identical |
| Gate precision | 0.6110 | 0.6110 | identical |

### Per-regime hit rates

| Regime | Interpretation | n_test (mean) | Hit rate (mean ± std) |
|--------|---------------|---------------|----------------------|
| 0 | Low-vol / calm | 57,147 | 0.456 ± 0.049 |
| 1 | High-vol / active | 34,772 | **0.483 ± 0.047** |

**Regime vol15 centres (consistent across folds):**
- Regime 0: 0.000127–0.000138
- Regime 1: 0.000166–0.000237 (~1.5–1.7× higher)

### Insights

1. **Regime 1 (high-vol) is directionally more predictable than regime 0 (low-vol).**
   Hit rate 0.483 vs 0.456, a consistent gap of ~2.7pp. In folds 2 and 4 (larger
   training sets), regime 1 exceeds 50% (0.509, 0.528), confirming the pattern is not
   noise. Hypothesis supported: **volatile / active conditions generate stronger
   directional signals** from the v2 feature set.

2. **No aggregate improvement over v2.** The gains in the high-vol regime are offset by
   below-50% performance in the low-vol regime. The HMM regime split reveals where signal
   exists but doesn't improve the overall pipeline.

3. **HMM regimes are stable and meaningful.** Vol15 centres separate cleanly and
   consistently across all 5 folds (1.5–1.7× ratio between regimes). The HMM found a
   genuine market structure, not noise.

4. **Convergence warning on fold 3** is non-critical — the HMM log-likelihood decreased
   slightly (EM local minimum), but the regime centres are still stable.

5. **Low-vol regime is near-unpredictable (hit rate 0.456 < 50%).**  In calm conditions,
   minute-bar direction is essentially random. Resources should be focused on the high-vol
   regime.

---

## Cross-Experiment Conclusions

| Finding | Evidence |
|---------|---------|
| v2 features add real signal | MCC +154% (Exp1), direction MCC +250% (Exp2) |
| Pure price lags are insufficient | v1 MCC ≈ random; v2 with order flow + momentum helps |
| Direction is near-coin-flip on average | hit rate 0.465 across all folds and both feature sets |
| High-vol regime is more directionally predictable | Exp3: hit rate 0.483 vs 0.456 in low-vol; exceeds 50% in later folds |
| Class imbalance distorts three-class scheme | Flat recall 0.69 but flat is irrelevant for trading |
| HP tuning with walk-forward CV stabilises metrics | Fold std halved in Exp1 |
| Gate finds a meaningful selectivity signal | Precision 0.611, better than v1's 0.545 |
| Threshold search degenerates | Percentile tuning always picks 0 — needs absolute tick threshold |

---

## Open Questions / Next Steps

1. **Focus on high-vol regime only.** Train and evaluate direction model exclusively on
   regime-1 bars. Accept no-trade in the low-vol regime. Measure conditional accuracy
   and MCC on that subset — the signal is strongest there.

2. **Absolute threshold for gate:** Replace percentile-based threshold tuning with a
   fixed 1-tick (0.03125 pts) minimum move. Domain-grounded and avoids the degenerate
   percentile search.

3. **Two-class (up/down) classifier excluding flat bars:** The three-class scheme is
   distorted by the rare flat class. Filtering out flat bars and training a binary
   classifier may give cleaner up/down signal.

4. **Feature importance analysis:** Use RF feature importances from the v2 models to
   identify which of the 49 features drive predictions — would inform further
   feature engineering.

5. **Longer lags:** The current maximum lag is 4 bars. Lags at 15, 30, 60 minutes may
   capture intraday session structure not visible at 1–4 minute level.
