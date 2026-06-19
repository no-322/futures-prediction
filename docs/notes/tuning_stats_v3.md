# Tuned Models — No-Flat Test Slice (v3)

Hyperparameters selected on a **no-flat validation fold from the training half** (the test set is touched once). Feature set: v3; |move| weighting: False; threshold tuned: True. No-flat test rows: 140,613.

| Model | Val acc | Test acc (no-flat) | Test MCC | Threshold | Params |
|---|---|---|---|---|---|
| Gradient Boosting (XGBoost) | 0.5376 | 0.5442 | 0.0890 | 0.493 | `{'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.03, 'reg_lambda': 5.0, 'reg_alpha': 0.0, 'min_child_weight': 5}` |
| Random Forest | 0.5344 | 0.5424 | 0.0841 | 0.493 | `{'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 50, 'max_features': 0.3}` |
| Logistic Regression | 0.5256 | 0.5203 | 0.0390 | 0.499 | `{'l1_ratio': 0.0, 'C': 10.0, 'solver': 'lbfgs', 'max_iter': 2000}` |

---

## Logistic Regression (tuned, v3)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5203 |
| Macro F1 | 0.5061 |
| Weighted F1 | 0.5071 |
| MCC | 0.0390 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5216 | 0.3551 | 0.4225 | 69,494 |
| 1 | 0.5197 | 0.6817 | 0.5898 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 24,677 | 44,817 |
| **Actual 1** | 22,634 | 48,485 |

---

## Random Forest (tuned, v3)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5424 |
| Macro F1 | 0.5375 |
| Weighted F1 | 0.5380 |
| MCC | 0.0841 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5454 | 0.4447 | 0.4899 | 69,494 |
| 1 | 0.5403 | 0.6378 | 0.5850 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 30,902 | 38,592 |
| **Actual 1** | 25,757 | 45,362 |

---

## Gradient Boosting (XGBoost) (tuned, v3)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5442 |
| Macro F1 | 0.5329 |
| Weighted F1 | 0.5337 |
| MCC | 0.0890 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5548 | 0.3933 | 0.4603 | 69,494 |
| 1 | 0.5384 | 0.6916 | 0.6055 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 27,331 | 42,163 |
| **Actual 1** | 21,932 | 49,187 |

