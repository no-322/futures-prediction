# No-Flat Binary Suite — Stats on the No-Flat Test Slice

**No-flat test slice:** flat (`Open == Close`) bars removed from **evaluation only** — predictions are unchanged (they were generated on the whole test set, blind to flatness). Kept 140,613 of 275,759 test rows (135,146 flat dropped, 49.01%). Per-label metrics: class 0 = down, class 1 = up.

## Logistic Regression (no-flat)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5472 |
| Macro F1 | 0.5412 |
| Weighted F1 | 0.5418 |
| MCC | 0.0942 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5530 | 0.4374 | 0.4885 | 69,494 |
| 1 | 0.5435 | 0.6545 | 0.5939 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 30,399 | 39,095 |
| **Actual 1** | 24,569 | 46,550 |

---

## Random Forest (no-flat)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5071 |
| Macro F1 | 0.5070 |
| Weighted F1 | 0.5071 |
| MCC | 0.0141 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5014 | 0.5006 | 0.5010 | 69,494 |
| 1 | 0.5127 | 0.5135 | 0.5131 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 34,786 | 34,708 |
| **Actual 1** | 34,597 | 36,522 |

---

## Gradient Boosting (XGBoost) (no-flat)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5028 |
| Macro F1 | 0.4892 |
| Weighted F1 | 0.4902 |
| MCC | 0.0020 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.4956 | 0.3438 | 0.4060 | 69,494 |
| 1 | 0.5065 | 0.6581 | 0.5724 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 23,895 | 45,599 |
| **Actual 1** | 24,319 | 46,800 |

---

## HMM-regime binary

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5195 |
| Macro F1 | 0.5191 |
| Weighted F1 | 0.5192 |
| MCC | 0.0385 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5145 | 0.4944 | 0.5042 | 69,494 |
| 1 | 0.5241 | 0.5441 | 0.5339 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 34,357 | 35,137 |
| **Actual 1** | 32,423 | 38,696 |


---

## HMM-regime binary — per-regime breakdown

## Regime 0 (low-vol / calm)

Samples evaluated: 105,140

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5182 |
| Macro F1 | 0.5178 |
| Weighted F1 | 0.5179 |
| MCC | 0.0359 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5131 | 0.4930 | 0.5029 | 51,961 |
| 1 | 0.5229 | 0.5428 | 0.5327 | 53,179 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 25,618 | 26,343 |
| **Actual 1** | 24,312 | 28,867 |

## Regime 1 (high-vol / active)

Samples evaluated: 35,473

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5234 |
| Macro F1 | 0.5230 |
| Weighted F1 | 0.5232 |
| MCC | 0.0464 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5186 | 0.4984 | 0.5083 | 17,533 |
| 1 | 0.5278 | 0.5479 | 0.5376 | 17,940 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 8,739 | 8,794 |
| **Actual 1** | 8,111 | 9,829 |

