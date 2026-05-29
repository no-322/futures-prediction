# Model Evaluation Results

All metrics computed on the held-out test set (second 50% of data by timestamp).

## Always Up (baseline)

| Metric | Value |
|--------|-------|
| Accuracy | 0.2579 |
| Recall (Up) | 1.0000 |

**Confusion Matrix** (rows = actual, cols = predicted):

|  | Predicted Down | Predicted Up |
|--|----------------|--------------|
| **Actual Down** | 0 | 204,640 |
| **Actual Up** | 0 | 71,119 |

---

## Last Direction (baseline)

| Metric | Value |
|--------|-------|
| Accuracy | 0.6284 |
| Recall (Up) | 0.2796 |

**Confusion Matrix** (rows = actual, cols = predicted):

|  | Predicted Down | Predicted Up |
|--|----------------|--------------|
| **Actual Down** | 153,407 | 51,233 |
| **Actual Up** | 51,233 | 19,886 |

---

## Logistic Regression

| Metric | Value |
|--------|-------|
| Accuracy | 0.7420 |
| Recall (Up) | 0.0043 |

**Confusion Matrix** (rows = actual, cols = predicted):

|  | Predicted Down | Predicted Up |
|--|----------------|--------------|
| **Actual Down** | 204,297 | 343 |
| **Actual Up** | 70,815 | 304 |

---

## Random Forest

| Metric | Value |
|--------|-------|
| Accuracy | 0.6873 |
| Recall (Up) | 0.1453 |

**Confusion Matrix** (rows = actual, cols = predicted):

|  | Predicted Down | Predicted Up |
|--|----------------|--------------|
| **Actual Down** | 179,193 | 25,447 |
| **Actual Up** | 60,784 | 10,335 |

---

## Gradient Boosting (XGBoost)

| Metric | Value |
|--------|-------|
| Accuracy | 0.7421 |
| Recall (Up) | 0.0001 |

**Confusion Matrix** (rows = actual, cols = predicted):

|  | Predicted Down | Predicted Up |
|--|----------------|--------------|
| **Actual Down** | 204,636 | 4 |
| **Actual Up** | 71,114 | 5 |

---

## SVM (RBF kernel)

| Metric | Value |
|--------|-------|
| Accuracy | 0.5680 |
| Recall (Up) | 0.3447 |

**Confusion Matrix** (rows = actual, cols = predicted):

|  | Predicted Down | Predicted Up |
|--|----------------|--------------|
| **Actual Down** | 132,123 | 72,517 |
| **Actual Up** | 46,601 | 24,518 |

