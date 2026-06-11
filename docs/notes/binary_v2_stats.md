# 49-Feature Binary Suites — Model Evaluation Statistics

Binary up/down classification on the 50/50 time-ordered test set using the **49-feature v2 matrix**. Two variants: flat-included (all training rows) and no-flat (flat `Close == Open` rows removed from **training** only; test whole). Per-label metrics: class 0 = down, class 1 = up.

## Flat-included (49 features)

## Logistic Regression (v2, flat-incl)

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.7421 |
| Macro F1 | 0.4275 |
| Weighted F1 | 0.6330 |
| MCC | 0.0148 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7423 | 0.9994 | 0.8519 | 204,640 |
| 1 | 0.4844 | 0.0015 | 0.0031 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 204,524 | 116 |
| **Actual 1** | 71,010 | 109 |

---

## Random Forest (v2, flat-incl)

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.7363 |
| Macro F1 | 0.4530 |
| Weighted F1 | 0.6436 |
| MCC | 0.0390 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7447 | 0.9809 | 0.8466 | 204,640 |
| 1 | 0.3705 | 0.0323 | 0.0594 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 200,738 | 3,902 |
| **Actual 1** | 68,822 | 2,297 |

---

## Gradient Boosting (XGBoost) (v2, flat-incl)

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.7420 |
| Macro F1 | 0.4272 |
| Weighted F1 | 0.6328 |
| MCC | 0.0112 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7422 | 0.9995 | 0.8518 | 204,640 |
| 1 | 0.4400 | 0.0012 | 0.0025 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 204,528 | 112 |
| **Actual 1** | 71,031 | 88 |

---

## SVM (RBF kernel) (v2, flat-incl)

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.6060 |
| Macro F1 | 0.5401 |
| Weighted F1 | 0.6244 |
| MCC | 0.0949 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7735 | 0.6634 | 0.7142 | 204,640 |
| 1 | 0.3128 | 0.4409 | 0.3660 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 135,766 | 68,874 |
| **Actual 1** | 39,766 | 31,353 |


---

## No-flat (49 features)

## Logistic Regression (no-flat, v2)

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4550 |
| Macro F1 | 0.4459 |
| Weighted F1 | 0.4802 |
| MCC | 0.0240 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7553 | 0.3928 | 0.5168 | 204,640 |
| 1 | 0.2662 | 0.6339 | 0.3750 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 80,381 | 124,259 |
| **Actual 1** | 26,035 | 45,084 |

---

## Random Forest (no-flat, v2)

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4963 |
| Macro F1 | 0.4721 |
| Weighted F1 | 0.5268 |
| MCC | 0.0227 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7526 | 0.4785 | 0.5850 | 204,640 |
| 1 | 0.2673 | 0.5475 | 0.3592 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 97,912 | 106,728 |
| **Actual 1** | 32,183 | 38,936 |

---

## Gradient Boosting (XGBoost) (no-flat, v2)

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4672 |
| Macro F1 | 0.4555 |
| Weighted F1 | 0.4941 |
| MCC | 0.0317 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7590 | 0.4132 | 0.5351 | 204,640 |
| 1 | 0.2693 | 0.6224 | 0.3760 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 84,566 | 120,074 |
| **Actual 1** | 26,858 | 44,261 |

---

## SVM (RBF kernel) (no-flat, v2)

Samples evaluated: 275,759

### Scalar metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5048 |
| Macro F1 | 0.4788 |
| Weighted F1 | 0.5352 |
| MCC | 0.0317 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.7565 | 0.4906 | 0.5952 | 204,640 |
| 1 | 0.2713 | 0.5456 | 0.3624 | 71,119 |

### Confusion matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 |
|--|--|--|
| **Actual 0** | 100,405 | 104,235 |
| **Actual 1** | 32,319 | 38,800 |

