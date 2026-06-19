# Tuned Models — No-Flat Test Slice (v1)

Hyperparameters selected on a **no-flat validation fold from the training half** (the test set is touched once). Feature set: v1; |move| weighting: False; threshold tuned: True. No-flat test rows: 140,613.

| Model | Val acc | Test acc (no-flat) | Test MCC | Threshold | Params |
|---|---|---|---|---|---|
| Logistic Regression | 0.5109 | 0.5306 | 0.0601 | 0.505 | `{'l1_ratio': 0.0, 'C': 0.5, 'solver': 'saga', 'max_iter': 2000}` |

---

## Logistic Regression (tuned, v1)

Samples evaluated: 140,613

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5306 |
| Macro F1 | 0.5250 |
| Weighted F1 | 0.5256 |
| MCC | 0.0601 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5313 | 0.4264 | 0.4731 | 69,494 |
| 1 | 0.5302 | 0.6325 | 0.5768 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 29,631 | 39,863 |
| **Actual 1** | 26,139 | 44,980 |

