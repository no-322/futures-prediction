# Tuned Models — No-Flat Test Slice (v2)

Hyperparameters selected on a **no-flat validation fold from the training half** (the test set is touched once). Feature set: v2; |move| weighting: False; threshold tuned: True. No-flat test rows: 140,613.

| Model | Val acc | Test acc (no-flat) | Test MCC | Threshold | Params |
|---|---|---|---|---|---|
| Gradient Boosting (XGBoost) | 0.5245 | 0.5335 | 0.0665 | 0.505 | `{'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.05, 'reg_lambda': 1.0, 'reg_alpha': 0.0, 'min_child_weight': 1}` |
| Logistic Regression | 0.5309 | 0.5310 | 0.0611 | 0.503 | `{'l1_ratio': 0.0, 'C': 10.0, 'solver': 'lbfgs', 'max_iter': 2000}` |
| Random Forest | 0.5240 | 0.5256 | 0.0513 | 0.489 | `{'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 50, 'max_features': 'sqrt'}` |

---

## Logistic Regression (tuned, v2)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5310 |
| Macro F1 | 0.5296 |
| Weighted F1 | 0.5299 |
| MCC | 0.0611 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5278 | 0.4836 | 0.5047 | 69,494 |
| 1 | 0.5336 | 0.5773 | 0.5546 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 33,604 | 35,890 |
| **Actual 1** | 30,062 | 41,057 |

---

## Random Forest (tuned, v2)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5256 |
| Macro F1 | 0.5005 |
| Weighted F1 | 0.5018 |
| MCC | 0.0513 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5352 | 0.3049 | 0.3885 | 69,494 |
| 1 | 0.5218 | 0.7413 | 0.6125 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 21,189 | 48,305 |
| **Actual 1** | 18,402 | 52,717 |

---

## Gradient Boosting (XGBoost) (tuned, v2)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5335 |
| Macro F1 | 0.5330 |
| Weighted F1 | 0.5332 |
| MCC | 0.0665 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5293 | 0.5074 | 0.5181 | 69,494 |
| 1 | 0.5373 | 0.5591 | 0.5480 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 35,258 | 34,236 |
| **Actual 1** | 31,356 | 39,763 |

