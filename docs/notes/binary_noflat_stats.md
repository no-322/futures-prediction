# No-Flat Binary Suite — Model Evaluation Statistics

Binary up/down classification on the 50/50 time-ordered test set, with flat bars (Close == Open) removed from the **training** set only (test set whole). Per-label metrics: class 0 = down, class 1 = up.

## Logistic Regression (no-flat)

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4648 |
| Macro F1 | 0.4560 |
| Weighted F1 | 0.4895 |
| MCC | 0.0480 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7686 | 0.3989 | 0.5252 | 204,640 |
| 1 | 0.2745 | 0.6545 | 0.3868 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 81,627 | 123,013 |
| **Actual 1** | 24,569 | 46,550 |

---

## Random Forest (no-flat)

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5013 |
| Macro F1 | 0.4718 |
| Weighted F1 | 0.5322 |
| MCC | 0.0093 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7462 | 0.4971 | 0.5967 | 204,640 |
| 1 | 0.2619 | 0.5135 | 0.3469 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 101,718 | 102,922 |
| **Actual 1** | 34,597 | 36,522 |

---

## Gradient Boosting (XGBoost) (no-flat)

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4259 |
| Macro F1 | 0.4215 |
| Weighted F1 | 0.4457 |
| MCC | 0.0030 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7439 | 0.3452 | 0.4715 | 204,640 |
| 1 | 0.2588 | 0.6581 | 0.3715 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 70,634 | 134,006 |
| **Actual 1** | 24,319 | 46,800 |

---

## SVM (RBF kernel) (no-flat)

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4973 |
| Macro F1 | 0.4708 |
| Weighted F1 | 0.5281 |
| MCC | 0.0144 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7486 | 0.4856 | 0.5891 | 204,640 |
| 1 | 0.2640 | 0.5308 | 0.3526 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 99,375 | 105,265 |
| **Actual 1** | 33,368 | 37,751 |

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

