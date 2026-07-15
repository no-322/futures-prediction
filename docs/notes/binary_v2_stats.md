# 49-Feature Binary Suites — Model Evaluation Statistics

Binary up/down classification on the 50/50 time-ordered test set using the **49-feature v2 matrix**. Two variants: flat-included (all training rows) and no-flat (flat `Close == Open` rows removed from **training** only; test whole). Per-label metrics: class 0 = down, class 1 = up.

## No-flat (49 features)

## Logistic Regression (no-flat, v2)

Samples evaluated: 144,263

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5310 |
| Macro F1 | 0.5243 |
| Weighted F1 | 0.5249 |
| MCC | 0.0611 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5328 | 0.4165 | 0.4676 | 71,312 |
| 1 | 0.5299 | 0.6429 | 0.5810 | 72,951 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 29,705 | 41,607 |
| **Actual 1** | 26,049 | 46,902 |

---

## Random Forest (no-flat, v2)

Samples evaluated: 144,263

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5231 |
| Macro F1 | 0.5226 |
| Weighted F1 | 0.5227 |
| MCC | 0.0456 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5184 | 0.4959 | 0.5069 | 71,312 |
| 1 | 0.5273 | 0.5496 | 0.5382 | 72,951 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 35,363 | 35,949 |
| **Actual 1** | 32,854 | 40,097 |

---

## Gradient Boosting (XGBoost) (no-flat, v2)

Samples evaluated: 144,263

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5338 |
| Macro F1 | 0.5295 |
| Weighted F1 | 0.5300 |
| MCC | 0.0666 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.5343 | 0.4431 | 0.4844 | 71,312 |
| 1 | 0.5334 | 0.6224 | 0.5745 | 72,951 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 31,596 | 39,716 |
| **Actual 1** | 27,543 | 45,408 |

