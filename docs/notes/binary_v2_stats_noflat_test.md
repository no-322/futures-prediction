# 49-Feature Binary Suites — Stats on the No-Flat Test Slice

**No-flat test slice:** flat (`Open == Close`) bars removed from **evaluation only** — predictions are unchanged (they were generated on the whole test set, blind to flatness). Kept 140,613 of 275,759 test rows (135,146 flat dropped, 49.01%). Per-label metrics: class 0 = down, class 1 = up.

## Flat-included (49 features)

## Logistic Regression (v2, flat-incl)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4943 |
| Macro F1 | 0.3321 |
| Weighted F1 | 0.3283 |
| MCC | 0.0014 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.4942 | 0.9986 | 0.6612 | 69,494 |
| 1 | 0.5240 | 0.0015 | 0.0031 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 69,395 | 99 |
| **Actual 1** | 71,010 | 109 |

---

## Random Forest (v2, flat-incl)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4942 |
| Macro F1 | 0.3573 |
| Weighted F1 | 0.3539 |
| MCC | -0.0022 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.4940 | 0.9669 | 0.6539 | 69,494 |
| 1 | 0.4997 | 0.0323 | 0.0607 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 67,194 | 2,300 |
| **Actual 1** | 68,822 | 2,297 |

---

## Gradient Boosting (XGBoost) (v2, flat-incl)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4943 |
| Macro F1 | 0.3319 |
| Weighted F1 | 0.3281 |
| MCC | 0.0006 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.4942 | 0.9988 | 0.6613 | 69,494 |
| 1 | 0.5146 | 0.0012 | 0.0025 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 69,411 | 83 |
| **Actual 1** | 71,031 | 88 |


---

## No-flat (49 features)

## Logistic Regression (no-flat, v2)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5285 |
| Macro F1 | 0.5224 |
| Weighted F1 | 0.5230 |
| MCC | 0.0558 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5289 | 0.4206 | 0.4685 | 69,494 |
| 1 | 0.5282 | 0.6339 | 0.5763 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 29,227 | 40,267 |
| **Actual 1** | 26,035 | 45,084 |

---

## Random Forest (no-flat, v2)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5215 |
| Macro F1 | 0.5210 |
| Weighted F1 | 0.5211 |
| MCC | 0.0424 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5166 | 0.4948 | 0.5055 | 69,494 |
| 1 | 0.5259 | 0.5475 | 0.5365 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 34,389 | 35,105 |
| **Actual 1** | 32,183 | 38,936 |

---

## Gradient Boosting (XGBoost) (no-flat, v2)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5335 |
| Macro F1 | 0.5291 |
| Weighted F1 | 0.5297 |
| MCC | 0.0660 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5338 | 0.4425 | 0.4839 | 69,494 |
| 1 | 0.5333 | 0.6224 | 0.5744 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 30,754 | 38,740 |
| **Actual 1** | 26,858 | 44,261 |

