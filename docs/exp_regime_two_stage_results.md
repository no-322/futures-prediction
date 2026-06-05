# Experiment 3 — Regime-Conditional Two-Stage Cascade (Gaussian HMM)

Folds: 5  |  Regime model: GaussianHMM(2 states)  |  Direction: per-regime RF

## Overall metrics (mean ± std — comparable to Exp 2 v2)

| Metric | Mean | Std | Exp 2 v2 baseline |
|--------|------|-----|-------------------|
| coverage | 0.3651 | 0.0434 | 0.3651 |
| conditional_hit_rate | 0.4672 | 0.0492 | 0.4646 |
| direction_mcc | 0.0418 | 0.0185 | 0.0415 |
| gate_recall | 0.4335 | 0.0488 | 0.4335 |
| gate_precision | 0.6110 | 0.0270 | 0.6110 |

## Per-regime metrics (mean ± std across folds)

| Regime | Interpretation | n_test (mean) | n_gated (mean) | hit_rate | direction_mcc |
|--------|---------------|---------------|----------------|---------|---------------|
| 0 (low-vol / calm) | 57147 | 18837 | 0.4558 ± 0.0489 | 0.0418 ± 0.0185 |
| 1 (high-vol / active) | 34772 | 14725 | 0.4827 ± 0.0467 | 0.0418 ± 0.0185 |

## Regime vol centres (mean lag1_vol15 per regime, across folds)

| Fold | Regime 0 vol15 | Regime 1 vol15 |
|------|----------------|----------------|
| 0 | 0.000138 | 0.000237 |
| 1 | 0.000136 | 0.000208 |
| 2 | 0.000126 | 0.000172 |
| 3 | 0.000126 | 0.000166 |
| 4 | 0.000128 | 0.000187 |

## Per-fold detail

| Fold | threshold | coverage | hit_rate | dir_mcc | r0_hit | r1_hit | r0_mcc | r1_mcc |
|------|-----------|----------|----------|---------|--------|--------|--------|--------|
| 0 | 0.0000 | 0.3871 | 0.3698 | 0.0136 | 0.3606 | 0.3933 | 0.0136 | 0.0136 |
| 1 | 0.0000 | 0.3064 | 0.4793 | 0.0510 | 0.4763 | 0.4864 | 0.0510 | 0.0510 |
| 2 | 0.0000 | 0.3775 | 0.4921 | 0.0394 | 0.4609 | 0.5089 | 0.0394 | 0.0394 |
| 3 | 0.0000 | 0.4274 | 0.4942 | 0.0354 | 0.4880 | 0.4972 | 0.0354 | 0.0354 |
| 4 | 0.0000 | 0.3273 | 0.5008 | 0.0698 | 0.4932 | 0.5275 | 0.0698 | 0.0698 |
