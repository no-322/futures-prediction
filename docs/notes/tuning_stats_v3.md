# Tuned Models — No-Flat Test Slice (v3)

Hyperparameters selected on a **no-flat validation fold from the training half** (the test set is touched once). Feature set: v3; |move| weighting: False; threshold tuned: True. No-flat test rows: 144,263.

| Model | Val acc | Test acc (no-flat) | Test MCC | Threshold | Params |
|---|---|---|---|---|---|
| Gradient Boosting (XGBoost) | 0.5383 | 0.5470 | 0.0934 | 0.500 | `{'n_estimators': 600, 'max_depth': 3, 'learning_rate': 0.02, 'reg_lambda': 10.0, 'reg_alpha': 1.0, 'min_child_weight': 20}` |
| Random Forest | 0.5323 | 0.5420 | 0.0832 | 0.496 | `{'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 200, 'max_features': 0.3}` |
| Logistic Regression | 0.5263 | 0.5207 | 0.0398 | 0.501 | `{'l1_ratio': 0.0, 'C': 0.1, 'solver': 'lbfgs', 'max_iter': 2000}` |

---

## Logistic Regression (tuned, v3)

Samples evaluated: 144,263

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5207 |
| Macro F1 | 0.5117 |
| Weighted F1 | 0.5124 |
| MCC | 0.0398 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5203 | 0.3892 | 0.4453 | 71,312 |
| 1 | 0.5209 | 0.6492 | 0.5780 | 72,951 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 27,755 | 43,557 |
| **Actual 1** | 25,592 | 47,359 |

---

## Random Forest (tuned, v3)

Samples evaluated: 144,263

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5420 |
| Macro F1 | 0.5402 |
| Weighted F1 | 0.5406 |
| MCC | 0.0832 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5409 | 0.4855 | 0.5117 | 71,312 |
| 1 | 0.5428 | 0.5973 | 0.5688 | 72,951 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 34,620 | 36,692 |
| **Actual 1** | 29,381 | 43,570 |

---

## Gradient Boosting (XGBoost) (tuned, v3)

Samples evaluated: 144,263

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5470 |
| Macro F1 | 0.5437 |
| Weighted F1 | 0.5442 |
| MCC | 0.0934 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5491 | 0.4674 | 0.5050 | 71,312 |
| 1 | 0.5455 | 0.6248 | 0.5825 | 72,951 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 33,331 | 37,981 |
| **Actual 1** | 27,370 | 45,581 |

