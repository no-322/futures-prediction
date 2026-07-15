# No-Flat Binary Suite — Model Evaluation Statistics

Binary up/down classification on the 50/50 time-ordered test set, with flat bars (Close == Open) removed from the **training** set only (test set whole). Per-label metrics: class 0 = down, class 1 = up.

## Logistic Regression (no-flat)

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

## Random Forest (no-flat)

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

## Gradient Boosting (XGBoost) (no-flat)

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

---

## HMM-regime binary

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4956 |
| Macro F1 | 0.4712 |
| Weighted F1 | 0.5262 |
| MCC | 0.0200 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7514 | 0.4788 | 0.5849 | 204,640 |
| 1 | 0.2662 | 0.5441 | 0.3575 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 97,976 | 106,664 |
| **Actual 1** | 32,423 | 38,696 |


---

## HMM-regime binary — per-regime breakdown

## Regime 0 (low-vol / calm)

Samples evaluated: 209,811

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4933 |
| Macro F1 | 0.4680 |
| Weighted F1 | 0.5253 |
| MCC | 0.0169 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7543 | 0.4765 | 0.5841 | 156,632 |
| 1 | 0.2604 | 0.5428 | 0.3520 | 53,179 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 74,642 | 81,990 |
| **Actual 1** | 24,312 | 28,867 |

## Regime 1 (high-vol / active)

Samples evaluated: 65,948

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5029 |
| Macro F1 | 0.4811 |
| Weighted F1 | 0.5296 |
| MCC | 0.0302 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7421 | 0.4860 | 0.5874 | 48,008 |
| 1 | 0.2849 | 0.5479 | 0.3748 | 17,940 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 23,334 | 24,674 |
| **Actual 1** | 8,111 | 9,829 |

