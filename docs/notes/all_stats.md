# Model Evaluation Statistics

All metrics computed on the held-out test set.

## Logistic Regression

Samples evaluated: 144,263

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5474 |
| Macro F1 | 0.5406 |
| Weighted F1 | 0.5412 |
| MCC | 0.0947 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5543 | 0.4306 | 0.4847 | 71,312 |
| 1 | 0.5431 | 0.6616 | 0.5965 | 72,951 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 30,705 | 40,607 |
| **Actual 1** | 24,687 | 48,264 |

---

## Random Forest

Samples evaluated: 144,263

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5072 |
| Macro F1 | 0.5071 |
| Weighted F1 | 0.5072 |
| MCC | 0.0143 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5016 | 0.4994 | 0.5005 | 71,312 |
| 1 | 0.5127 | 0.5149 | 0.5138 | 72,951 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 35,614 | 35,698 |
| **Actual 1** | 35,390 | 37,561 |

---

## Gradient Boosting (XGBoost)

Samples evaluated: 144,263

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5038 |
| Macro F1 | 0.4854 |
| Weighted F1 | 0.4865 |
| MCC | 0.0036 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.4970 | 0.3187 | 0.3884 | 71,312 |
| 1 | 0.5069 | 0.6847 | 0.5825 | 72,951 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 22,728 | 48,584 |
| **Actual 1** | 23,005 | 49,946 |

