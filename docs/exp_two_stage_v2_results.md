# Experiment 2 — Two-Stage Cascade Classifier [v2 features + walk-forward HP tuning]

Stage 1: Gate (tradeable vs skip) | Stage 2: Direction (long vs short)
Folds: 5  |  Model: Random Forest (both stages)

## Primary metrics (mean ± std across folds)

| Metric | Mean | Std | Description |
|--------|------|-----|-------------|
| coverage | 0.3651 | 0.0434 | fraction of bars the gate trades |
| conditional_hit_rate | 0.4646 | 0.0460 | directional accuracy on traded bars |

## Debug diagnostics (mean ± std across folds)

| Metric | Mean | Std | Description |
|--------|------|-----|-------------|
| gate_recall | 0.4335 | 0.0488 | recall on bars that genuinely moved > threshold |
| gate_precision | 0.6110 | 0.0270 | precision of gate predictions |
| gate_f1 | 0.5062 | 0.0404 | F1 of gate model on test fold |
| direction_mcc | 0.0415 | 0.0156 | MCC on bars that genuinely moved (regardless of gate) |
| threshold (pts) | 0.0000 | 0.0000 | tuned threshold per fold |

## Per-fold detail

| Fold | n_train | n_test | threshold | coverage | hit_rate | gate_recall | dir_mcc |
|------|---------|--------|-----------|----------|----------|------------|----------|
| 0 | 91,922 | 91,919 | 0.0000 | 0.3871 | 0.3742 | 0.4633 | 0.0206 |
| 1 | 183,841 | 91,919 | 0.0000 | 0.3064 | 0.4723 | 0.3714 | 0.0534 |
| 2 | 275,760 | 91,919 | 0.0000 | 0.3775 | 0.4883 | 0.4458 | 0.0332 |
| 3 | 367,679 | 91,919 | 0.0000 | 0.4274 | 0.4901 | 0.5018 | 0.0354 |
| 4 | 459,598 | 91,919 | 0.0000 | 0.3273 | 0.4980 | 0.3852 | 0.0647 |
