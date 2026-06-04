# Experiment 1 — Three-Class Direction Classifier

Folds: 5  |  Model: Random Forest  |  Labels: 0=down, 1=up, 2=flat

## Scalar metrics (mean ± std across folds)

| Metric | Mean | Std |
|--------|------|-----|
| accuracy | 0.3990 | 0.0429 |
| mcc | 0.0393 | 0.0144 |
| macro_f1 | 0.3400 | 0.0324 |

## Per-class recall (mean ± std across folds)

| Class | Mean recall | Std |
|-------|-------------|-----|
| 0 (down) | 0.1995 | 0.0528 |
| 1 (up) | 0.3339 | 0.1964 |
| 2 (flat) | 0.5315 | 0.1678 |

## Per-fold detail

| Fold | n_train | n_test | accuracy | mcc | macro_f1 | recall_down | recall_up | recall_flat |
|------|---------|--------|----------|-----|----------|------------|-----------|-------------|
| 0 | 91,922 | 91,919 | 0.3155 | 0.0160 | 0.2759 | 0.0993 | 0.7254 | 0.1990 |
| 1 | 183,841 | 91,919 | 0.4176 | 0.0385 | 0.3517 | 0.2128 | 0.2347 | 0.6152 |
| 2 | 275,760 | 91,919 | 0.4093 | 0.0437 | 0.3566 | 0.2438 | 0.2463 | 0.5835 |
| 3 | 367,679 | 91,919 | 0.4143 | 0.0609 | 0.3648 | 0.2412 | 0.2544 | 0.6071 |
| 4 | 459,598 | 91,919 | 0.4384 | 0.0377 | 0.3508 | 0.2004 | 0.2085 | 0.6528 |
