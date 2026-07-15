# Model Evaluation Results — walk-forward

Rolling walk-forward (3-month train / 1-month test). Each model reports the mean ± std across folds; Δ is vs the always-up baseline; AUM % is the compounding backtest total return.

---

## tuned_v1_gbm

- **Accuracy:** 50.5% ± 0.9% across 34 folds (range 48.2–52.5%)
- **Recall (up):** 0.6291
- **Δ vs always-up baseline:** -0.1 pp
- **Backtest AUM %:** +50.2%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 48,726 | 80,374 |
| **Actual 1** | 49,101 | 83,295 |

---

## tuned_v1_logistic

- **Accuracy:** 53.6% ± 2.2% across 34 folds (range 51.0–60.1%)
- **Recall (up):** 0.7401
- **Δ vs always-up baseline:** +3.0 pp
- **Backtest AUM %:** +2464.6%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 41,640 | 87,460 |
| **Actual 1** | 34,409 | 97,987 |

---

## tuned_v1_rf

- **Accuracy:** 50.4% ± 0.8% across 34 folds (range 48.6–52.0%)
- **Recall (up):** 0.5743
- **Δ vs always-up baseline:** -0.1 pp
- **Backtest AUM %:** +65.4%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 55,982 | 73,118 |
| **Actual 1** | 56,355 | 76,041 |

---

## tuned_v2_gbm

- **Accuracy:** 52.0% ± 1.2% across 34 folds (range 50.1–55.1%)
- **Recall (up):** 0.6372
- **Δ vs always-up baseline:** +1.4 pp
- **Backtest AUM %:** +418.2%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 51,233 | 77,867 |
| **Actual 1** | 48,032 | 84,364 |

---

## tuned_v2_logistic

- **Accuracy:** 53.0% ± 1.7% across 34 folds (range 50.4–57.1%)
- **Recall (up):** 0.6074
- **Δ vs always-up baseline:** +2.4 pp
- **Backtest AUM %:** +1295.2%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 57,631 | 71,469 |
| **Actual 1** | 51,975 | 80,421 |

---

## tuned_v2_rf

- **Accuracy:** 51.8% ± 0.9% across 34 folds (range 50.3–54.2%)
- **Recall (up):** 0.5826
- **Δ vs always-up baseline:** +1.2 pp
- **Backtest AUM %:** +438.1%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 58,250 | 70,850 |
| **Actual 1** | 55,262 | 77,134 |

---

## tuned_v3_gbm

- **Accuracy:** 53.3% ± 1.8% across 34 folds (range 50.6–58.0%)
- **Recall (up):** 0.5927
- **Δ vs always-up baseline:** +2.7 pp
- **Backtest AUM %:** +1685.2%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 60,391 | 68,709 |
| **Actual 1** | 53,921 | 78,475 |

---

## tuned_v3_logistic

- **Accuracy:** 51.7% ± 0.9% across 34 folds (range 49.6–53.6%)
- **Recall (up):** 0.6083
- **Δ vs always-up baseline:** +1.2 pp
- **Backtest AUM %:** +413.3%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 54,701 | 74,399 |
| **Actual 1** | 51,857 | 80,539 |

---

## tuned_v3_rf

- **Accuracy:** 53.2% ± 1.7% across 34 folds (range 50.7–57.3%)
- **Recall (up):** 0.5894
- **Δ vs always-up baseline:** +2.6 pp
- **Backtest AUM %:** +1551.0%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 60,570 | 68,530 |
| **Actual 1** | 54,358 | 78,038 |

---

## v1_gbm

- **Accuracy:** 50.5% ± 0.9% across 34 folds (range 47.8–52.4%)
- **Recall (up):** 0.5647
- **Δ vs always-up baseline:** -0.1 pp
- **Backtest AUM %:** +51.8%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 57,172 | 71,928 |
| **Actual 1** | 57,636 | 74,760 |

---

## v1_logistic

- **Accuracy:** 53.4% ± 2.1% across 34 folds (range 50.5–59.0%)
- **Recall (up):** 0.7390
- **Δ vs always-up baseline:** +2.8 pp
- **Backtest AUM %:** +1910.6%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 41,292 | 87,808 |
| **Actual 1** | 34,561 | 97,835 |

---

## v1_rf

- **Accuracy:** 50.6% ± 0.8% across 34 folds (range 49.0–53.6%)
- **Recall (up):** 0.5388
- **Δ vs always-up baseline:** -0.0 pp
- **Backtest AUM %:** +62.5%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 60,686 | 68,414 |
| **Actual 1** | 61,065 | 71,331 |

---

## v1rel_gbm

- **Accuracy:** 52.9% ± 1.8% across 34 folds (range 49.6–57.7%)
- **Recall (up):** 0.5651
- **Δ vs always-up baseline:** +2.4 pp
- **Backtest AUM %:** +1241.8%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 63,115 | 65,985 |
| **Actual 1** | 57,578 | 74,818 |

---

## v1rel_logistic

- **Accuracy:** 50.5% ± 0.8% across 34 folds (range 47.9–51.9%)
- **Recall (up):** 0.9740
- **Δ vs always-up baseline:** -0.1 pp
- **Backtest AUM %:** +55.9%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 3,165 | 125,935 |
| **Actual 1** | 3,437 | 128,959 |

---

## v1rel_rf

- **Accuracy:** 52.7% ± 1.9% across 34 folds (range 50.0–57.4%)
- **Recall (up):** 0.5450
- **Δ vs always-up baseline:** +2.1 pp
- **Backtest AUM %:** +841.8%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 65,018 | 64,082 |
| **Actual 1** | 60,243 | 72,153 |

---

## v2_gbm

- **Accuracy:** 52.0% ± 1.3% across 34 folds (range 49.8–55.7%)
- **Recall (up):** 0.5691
- **Δ vs always-up baseline:** +1.4 pp
- **Backtest AUM %:** +427.4%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 60,183 | 68,917 |
| **Actual 1** | 57,047 | 75,349 |

---

## v2_logistic

- **Accuracy:** 52.4% ± 1.3% across 34 folds (range 50.4–55.3%)
- **Recall (up):** 0.6326
- **Δ vs always-up baseline:** +1.9 pp
- **Backtest AUM %:** +767.1%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 53,109 | 75,991 |
| **Actual 1** | 48,642 | 83,754 |

---

## v2_rf

- **Accuracy:** 51.5% ± 1.0% across 34 folds (range 49.8–53.8%)
- **Recall (up):** 0.5531
- **Δ vs always-up baseline:** +0.9 pp
- **Backtest AUM %:** +282.1%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 61,207 | 67,893 |
| **Actual 1** | 59,163 | 73,233 |

---

## v3_gbm

- **Accuracy:** 53.0% ± 1.8% across 34 folds (range 50.7–57.9%)
- **Recall (up):** 0.5702
- **Δ vs always-up baseline:** +2.4 pp
- **Backtest AUM %:** +1340.0%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 62,580 | 66,520 |
| **Actual 1** | 56,910 | 75,486 |

---

## v3_logistic

- **Accuracy:** 51.8% ± 0.9% across 34 folds (range 49.8–53.7%)
- **Recall (up):** 0.6220
- **Δ vs always-up baseline:** +1.2 pp
- **Backtest AUM %:** +414.6%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 52,932 | 76,168 |
| **Actual 1** | 50,047 | 82,349 |

---

## v3_rf

- **Accuracy:** 52.8% ± 1.6% across 34 folds (range 50.9–57.4%)
- **Recall (up):** 0.5552
- **Δ vs always-up baseline:** +2.2 pp
- **Backtest AUM %:** +1036.2%

Confusion matrix (rows=actual, cols=predicted):

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 64,016 | 65,084 |
| **Actual 1** | 58,890 | 73,506 |
