# Tuned Models — No-Flat Test Slice (v1)

Hyperparameters selected on a **no-flat validation fold from the training half** (the test set is touched once). Feature set: v1; |move| weighting: False; threshold tuned: True. No-flat test rows: 144,263.

| Model | Val acc | Test acc (no-flat) | Test MCC | Threshold | Params |
|---|---|---|---|---|---|
| Logistic Regression | 0.5313 | 0.5454 | 0.0928 | 0.493 | `{'l1_ratio': 0.0, 'C': 10.0, 'solver': 'lbfgs', 'max_iter': 2000}` |
| Gradient Boosting (XGBoost) | 0.5088 | 0.5061 | 0.0045 | 0.494 | `{'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.03, 'reg_lambda': 5.0, 'reg_alpha': 0.0, 'min_child_weight': 5}` |
| Random Forest | 0.5108 | 0.5016 | 0.0037 | 0.500 | `{'n_estimators': 300, 'max_depth': 6, 'min_samples_leaf': 200, 'max_features': 0.3}` |

---

## Logistic Regression (tuned, v1)

Samples evaluated: 144,263

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5454 |
| Macro F1 | 0.5292 |
| Weighted F1 | 0.5302 |
| MCC | 0.0928 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5619 | 0.3642 | 0.4420 | 71,312 |
| 1 | 0.5376 | 0.7224 | 0.6164 | 72,951 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 25,975 | 45,337 |
| **Actual 1** | 20,251 | 52,700 |

---

## Random Forest (tuned, v1)

Samples evaluated: 144,263

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5016 |
| Macro F1 | 0.5014 |
| Weighted F1 | 0.5014 |
| MCC | 0.0037 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.4961 | 0.5248 | 0.5100 | 71,312 |
| 1 | 0.5076 | 0.4789 | 0.4929 | 72,951 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 37,423 | 33,889 |
| **Actual 1** | 38,012 | 34,939 |

---

## Gradient Boosting (XGBoost) (tuned, v1)

Samples evaluated: 144,263

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5061 |
| Macro F1 | 0.3853 |
| Weighted F1 | 0.3884 |
| MCC | 0.0045 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5031 | 0.0636 | 0.1129 | 71,312 |
| 1 | 0.5063 | 0.9386 | 0.6578 | 72,951 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 4,533 | 66,779 |
| **Actual 1** | 4,478 | 68,473 |

