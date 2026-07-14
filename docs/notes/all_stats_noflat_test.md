# Production Models — Stats on the No-Flat Test Slice

**No-flat test slice:** flat (`Open == Close`) bars removed from **evaluation only** — predictions are unchanged (they were generated on the whole test set, blind to flatness). Kept 140,613 of 275,759 test rows (135,146 flat dropped, 49.01%). Per-label metrics: class 0 = down, class 1 = up.

## Logistic Regression

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4944 |
| Macro F1 | 0.3346 |
| Weighted F1 | 0.3308 |
| MCC | 0.0022 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.4943 | 0.9960 | 0.6607 | 69,494 |
| 1 | 0.5232 | 0.0043 | 0.0085 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 69,217 | 277 |
| **Actual 1** | 70,815 | 304 |

---

## Random Forest

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4981 |
| Macro F1 | 0.4275 |
| Weighted F1 | 0.4252 |
| MCC | 0.0062 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.4955 | 0.8590 | 0.6285 | 69,494 |
| 1 | 0.5134 | 0.1453 | 0.2265 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 59,698 | 9,796 |
| **Actual 1** | 60,784 | 10,335 |

---

## Gradient Boosting (XGBoost)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4942 |
| Macro F1 | 0.3308 |
| Weighted F1 | 0.3270 |
| MCC | 0.0008 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.4942 | 0.9999 | 0.6615 | 69,494 |
| 1 | 0.5556 | 0.0001 | 0.0001 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 69,490 | 4 |
| **Actual 1** | 71,114 | 5 |

