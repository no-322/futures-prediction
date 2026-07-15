# Model Leaderboard — single test set

Every binary up/down model on the flat-free 50/50 time-ordered test set (144,263 decisive bars). **Sorted by accuracy, then MCC.**

- Excluded (length mismatch / non-binary): `exp_regime_binary`, `walkforward_wf_baseline_alwaysup`, `walkforward_wf_tuned_v1_logistic`, `walkforward_wf_tuned_v1_rf`, `walkforward_wf_v1_gbm`, `walkforward_wf_v1_logistic`, `walkforward_wf_v1_rf`, `walkforward_wf_v1rel_gbm`, `walkforward_wf_v1rel_logistic`, `walkforward_wf_v1rel_rf`, `walkforward_wf_v2_gbm`, `walkforward_wf_v2_logistic`, `walkforward_wf_v2_rf`, `walkforward_wf_v3_gbm`, `walkforward_wf_v3_logistic`, `walkforward_wf_v3_rf`.
*AUM %* = total return of the compounding long/short backtest on the test bars.

| Model | Accuracy | Recall | MCC | AUM % |
|-------|----------|--------|-----|-------|
| Logistic Regression (no-flat) | 0.5474 | 0.6616 | 0.0947 | +1153.8% |
| Logistic Regression | 0.5474 | 0.6616 | 0.0947 | +1153.8% |
| Gradient Boosting (XGBoost) (tuned, v3) | 0.5470 | 0.6248 | 0.0934 | +1049.0% |
| Logistic Regression (tuned, v1) | 0.5454 | 0.7224 | 0.0928 | +1034.8% |
| Random Forest (tuned, v3) | 0.5420 | 0.5973 | 0.0832 | +780.1% |
| Gradient Boosting (XGBoost) (no-flat, v2) | 0.5338 | 0.6224 | 0.0666 | +449.6% |
| Logistic Regression (tuned, v2) | 0.5312 | 0.6144 | 0.0614 | +440.0% |
| Logistic Regression (no-flat, v2) | 0.5310 | 0.6429 | 0.0611 | +428.8% |
| Gradient Boosting (XGBoost) (tuned, v2) | 0.5292 | 0.7676 | 0.0604 | +322.5% |
| Random Forest (tuned, v2) | 0.5260 | 0.6230 | 0.0507 | +282.8% |
| Random Forest (no-flat, v2) | 0.5231 | 0.5496 | 0.0456 | +222.4% |
| Logistic Regression (tuned, v3) | 0.5207 | 0.6492 | 0.0398 | +214.4% |
| Random Forest (no-flat) | 0.5072 | 0.5149 | 0.0143 | +52.6% |
| Random Forest | 0.5072 | 0.5149 | 0.0143 | +52.6% |
| Gradient Boosting (XGBoost) (tuned, v1) | 0.5061 | 0.9386 | 0.0045 | +33.7% |
| Gradient Boosting (XGBoost) (no-flat) | 0.5038 | 0.6847 | 0.0036 | +28.7% |
| Gradient Boosting (XGBoost) | 0.5038 | 0.6847 | 0.0036 | +28.7% |
| Random Forest (tuned, v1) | 0.5016 | 0.4789 | 0.0037 | +18.2% |
