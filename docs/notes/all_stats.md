# Model Evaluation Statistics

All metrics computed on the held-out test set.

## Logistic Regression

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.7420 |
| Macro F1 | 0.4301 |
| Weighted F1 | 0.6342 |
| MCC | 0.0235 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7426 | 0.9983 | 0.8517 | 204,640 |
| 1 | 0.4699 | 0.0043 | 0.0085 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 204,297 | 343 |
| **Actual 1** | 70,815 | 304 |

---

## Random Forest

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.6873 |
| Macro F1 | 0.4997 |
| Weighted F1 | 0.6480 |
| MCC | 0.0273 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7467 | 0.8756 | 0.8061 | 204,640 |
| 1 | 0.2888 | 0.1453 | 0.1934 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 179,193 | 25,447 |
| **Actual 1** | 60,784 | 10,335 |

---

## Gradient Boosting (XGBoost)

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.7421 |
| Macro F1 | 0.4260 |
| Weighted F1 | 0.6323 |
| MCC | 0.0039 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7421 | 1.0000 | 0.8520 | 204,640 |
| 1 | 0.5556 | 0.0001 | 0.0001 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 204,636 | 4 |
| **Actual 1** | 71,114 | 5 |

---

## SVM (RBF kernel)

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5680 |
| Macro F1 | 0.4904 |
| Weighted F1 | 0.5867 |
| MCC | -0.0088 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7393 | 0.6456 | 0.6893 | 204,640 |
| 1 | 0.2527 | 0.3447 | 0.2916 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 132,123 | 72,517 |
| **Actual 1** | 46,601 | 24,518 |

---

## Three-class v1 (20 features)

Samples evaluated: 459,595

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.3976 |
| Macro F1 | 0.3544 |
| Weighted F1 | 0.3937 |
| MCC | 0.0419 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.2757 | 0.2009 | 0.2324 | 116,636 |
| 1 | 0.2783 | 0.3390 | 0.3057 | 119,680 |
| 2 | 0.5189 | 0.5317 | 0.5252 | 223,279 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 | Pred 2 |
|--|--|--|--|
| **Actual 0** | 23,428 | 38,806 | 54,402 |
| **Actual 1** | 23,407 | 40,576 | 55,697 |
| **Actual 2** | 38,153 | 66,399 | 118,727 |

---

## Three-class v2 (49 features)

Samples evaluated: 459,595

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4699 |
| Macro F1 | 0.3651 |
| Weighted F1 | 0.4256 |
| MCC | 0.0947 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.3173 | 0.1609 | 0.2136 | 116,636 |
| 1 | 0.3268 | 0.2055 | 0.2524 | 119,680 |
| 2 | 0.5307 | 0.7729 | 0.6293 | 223,279 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 | Pred 2 |
|--|--|--|--|
| **Actual 0** | 18,771 | 22,433 | 75,432 |
| **Actual 1** | 17,915 | 24,596 | 77,169 |
| **Actual 2** | 22,472 | 28,225 | 172,582 |

---

## Two-stage v1 (20 features)

Samples evaluated: 202,625

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4966 |
| Macro F1 | 0.4741 |
| Weighted F1 | 0.5223 |
| MCC | 0.0071 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7250 | 0.4873 | 0.5829 | 146,234 |
| 1 | 0.2814 | 0.5205 | 0.3653 | 56,391 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 71,264 | 74,970 |
| **Actual 1** | 27,037 | 29,354 |

---

## Two-stage v2 (49 features)

Samples evaluated: 235,873

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4734 |
| Macro F1 | 0.4665 |
| Weighted F1 | 0.4917 |
| MCC | 0.0271 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7222 | 0.4152 | 0.5273 | 166,823 |
| 1 | 0.3030 | 0.6141 | 0.4058 | 69,050 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 69,266 | 97,557 |
| **Actual 1** | 26,647 | 42,403 |

---

## Regime cascade v2 (HMM)

Samples evaluated: 235,873

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4743 |
| Macro F1 | 0.4666 |
| Weighted F1 | 0.4932 |
| MCC | 0.0230 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7197 | 0.4205 | 0.5309 | 166,823 |
| 1 | 0.3015 | 0.6043 | 0.4023 | 69,050 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 70,155 | 96,668 |
| **Actual 1** | 27,322 | 41,728 |

