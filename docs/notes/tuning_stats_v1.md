# Tuned Models — No-Flat Test Slice (v1)

Hyperparameters selected on a **no-flat validation fold from the training half** (the test set is touched once). Feature set: v1; |move| weighting: False; threshold tuned: True. No-flat test rows: 140,613.

| Model | Val acc | Test acc (no-flat) | Test MCC | Threshold | Params |
|---|---|---|---|---|---|
| Logistic Regression | 0.5346 | 0.5461 | 0.0950 | 0.509 | `{'l1_ratio': 0.0, 'C': 1.0, 'solver': 'lbfgs', 'max_iter': 2000}` |
| Gradient Boosting (XGBoost) | 0.5106 | 0.5060 | 0.0056 | 0.498 | `{'n_estimators': 600, 'max_depth': 3, 'learning_rate': 0.02, 'reg_lambda': 10.0, 'reg_alpha': 1.0, 'min_child_weight': 20}` |
| Random Forest | 0.5095 | 0.5010 | 0.0050 | 0.501 | `{'n_estimators': 300, 'max_depth': 6, 'min_samples_leaf': 200, 'max_features': 0.3}` |

---

## Logistic Regression (tuned, v1)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5461 |
| Macro F1 | 0.5438 |
| Weighted F1 | 0.5435 |
| MCC | 0.0950 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5350 | 0.6229 | 0.5756 | 69,494 |
| 1 | 0.5611 | 0.4709 | 0.5121 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 43,291 | 26,203 |
| **Actual 1** | 37,627 | 33,492 |

---

## Random Forest (tuned, v1)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5010 |
| Macro F1 | 0.4942 |
| Weighted F1 | 0.4935 |
| MCC | 0.0050 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.4962 | 0.6241 | 0.5528 | 69,494 |
| 1 | 0.5090 | 0.3808 | 0.4356 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 43,368 | 26,126 |
| **Actual 1** | 44,039 | 27,080 |

---

## Gradient Boosting (XGBoost) (tuned, v1)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5060 |
| Macro F1 | 0.4385 |
| Weighted F1 | 0.4408 |
| MCC | 0.0056 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5007 | 0.1612 | 0.2439 | 69,494 |
| 1 | 0.5070 | 0.8429 | 0.6332 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 11,204 | 58,290 |
| **Actual 1** | 11,174 | 59,945 |

