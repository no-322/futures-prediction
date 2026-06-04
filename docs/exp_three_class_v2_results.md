# Experiment 1 — Three-Class Direction Classifier [v2 features + walk-forward HP tuning]

Folds: 5  |  Model: Random Forest  |  Labels: 0=down, 1=up, 2=flat

## Scalar metrics (mean ± std across folds)

| Metric | Mean | Std |
|--------|------|-----|
| accuracy | 0.4577 | 0.0121 |
| mcc | 0.0998 | 0.0085 |
| macro_f1 | 0.3777 | 0.0090 |

## Per-class recall (mean ± std across folds)

| Class | Mean recall | Std |
|-------|-------------|-----|
| 0 (down) | 0.1975 | 0.0698 |
| 1 (up) | 0.2709 | 0.0576 |
| 2 (flat) | 0.6923 | 0.0386 |

## Per-fold detail

| Fold | n_train | n_test | accuracy | mcc | macro_f1 | recall_down | recall_up | recall_flat |
|------|---------|--------|----------|-----|----------|------------|-----------|-------------|
| 0 | 91,922 | 91,919 | 0.4558 | 0.1132 | 0.3652 | 0.0826 | 0.3811 | 0.7125 |
| 1 | 183,841 | 91,919 | 0.4703 | 0.0963 | 0.3716 | 0.1583 | 0.2440 | 0.7441 |
| 2 | 275,760 | 91,919 | 0.4514 | 0.0965 | 0.3825 | 0.2526 | 0.2327 | 0.6730 |
| 3 | 367,679 | 91,919 | 0.4393 | 0.1046 | 0.3913 | 0.2762 | 0.2735 | 0.6299 |
| 4 | 459,598 | 91,919 | 0.4717 | 0.0883 | 0.3781 | 0.2180 | 0.2232 | 0.7018 |
