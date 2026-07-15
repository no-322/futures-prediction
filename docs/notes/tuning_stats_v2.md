# Tuned Models — No-Flat Test Slice (v2)

Hyperparameters selected on a **no-flat validation fold from the training half** (the test set is touched once). Feature set: v2; |move| weighting: False; threshold tuned: True. No-flat test rows: 144,263.

| Model | Val acc | Test acc (no-flat) | Test MCC | Threshold | Params |
|---|---|---|---|---|---|
| Logistic Regression | 0.5300 | 0.5312 | 0.0614 | 0.501 | `{'l1_ratio': 0.0, 'C': 1.0, 'solver': 'lbfgs', 'max_iter': 2000}` |
| Gradient Boosting (XGBoost) | 0.5245 | 0.5292 | 0.0604 | 0.487 | `{'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.05, 'reg_lambda': 1.0, 'reg_alpha': 0.0, 'min_child_weight': 1}` |
| Random Forest | 0.5239 | 0.5260 | 0.0507 | 0.497 | `{'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 50, 'max_features': 'sqrt'}` |

---

## Logistic Regression (tuned, v2)

Samples evaluated: 144,263

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5312 |
| Macro F1 | 0.5274 |
| Weighted F1 | 0.5279 |
| MCC | 0.0614 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5307 | 0.4461 | 0.4847 | 71,312 |
| 1 | 0.5316 | 0.6144 | 0.5700 | 72,951 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 31,812 | 39,500 |
| **Actual 1** | 28,129 | 44,822 |

---

## Random Forest (tuned, v2)

Samples evaluated: 144,263

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5260 |
| Macro F1 | 0.5208 |
| Weighted F1 | 0.5213 |
| MCC | 0.0507 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5253 | 0.4267 | 0.4709 | 71,312 |
| 1 | 0.5264 | 0.6230 | 0.5706 | 72,951 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 30,432 | 40,880 |
| **Actual 1** | 27,506 | 45,445 |

---

## Gradient Boosting (XGBoost) (tuned, v2)

Samples evaluated: 144,263

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5292 |
| Macro F1 | 0.4986 |
| Weighted F1 | 0.5000 |
| MCC | 0.0604 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5455 | 0.2853 | 0.3747 | 71,312 |
| 1 | 0.5235 | 0.7676 | 0.6225 | 72,951 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 20,348 | 50,964 |
| **Actual 1** | 16,956 | 55,995 |

