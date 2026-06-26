# Model Leaderboard

Every binary up/down model on the 50/50 time-ordered test set, compared in one place. **Sorted by no-flat test accuracy, then full-test MCC.**

- *No-flat test accuracy*: accuracy on the 140,613 of 275,759 test bars where `Close != Open` (flat bars dropped from evaluation only).
- *Accuracy* and *MCC*: computed on the full test set.
- `+ HMM gate` rows are scored **only on high-vol bars** (the rest are not traded); their `(cov NN%)` is the fraction of non-flat bars traded, and all three metrics are over that gated subset.
- Excluded (non-binary / different length): `exp_regime_v2`, `exp_three_class_v1`, `exp_three_class_v2`, `exp_two_stage_v1`, `exp_two_stage_v2`, `walkforward_baseline_v1`, `walkforward_top5_exp_noflat_baseline`, `walkforward_top5_tuned_v1_baseline`, `walkforward_top5_tuned_v2_gbm`, `walkforward_top5_tuned_v3_gbm`, `walkforward_top5_tuned_v3_rf`, `walkforward_top5hmmfeat_exp_noflat_baseline`, `walkforward_top5hmmfeat_tuned_v1_baseline`, `walkforward_top5hmmfeat_tuned_v2_gbm`, `walkforward_top5hmmfeat_tuned_v3_gbm`, `walkforward_top5hmmfeat_tuned_v3_rf`, `walkforward_top5hmmgate_exp_noflat_baseline`, `walkforward_top5hmmgate_tuned_v1_baseline`, `walkforward_top5hmmgate_tuned_v2_gbm`, `walkforward_top5hmmgate_tuned_v3_gbm`, `walkforward_top5hmmgate_tuned_v3_rf`, `walkforward_top5of_exp_noflat_baseline`, `walkforward_top5of_tuned_v1_baseline`, `walkforward_top5of_tuned_v2_gbm`, `walkforward_top5of_tuned_v3_gbm`, `walkforward_top5of_tuned_v3_rf`.

| Model | No-flat test acc | Accuracy | MCC |
|-------|------------------|----------|-----|
| Logistic Regression (no-flat) + HMM gate (high-vol) (cov 56%) | 0.5496 | 0.4729 | 0.0504 |
| Logistic Regression (tuned, v1) + HMM gate (high-vol) (cov 56%) | 0.5482 | 0.5562 | 0.0556 |
| Logistic Regression (no-flat) + HMM regime feature | 0.5473 | 0.4642 | 0.0477 |
| Logistic Regression (no-flat) | 0.5472 | 0.4648 | 0.0480 |
| Gradient Boosting (XGBoost) (tuned, v3) + order-flow + HMM gate (cov 56%) | 0.5463 | 0.4670 | 0.0537 |
| Logistic Regression (tuned, v1) | 0.5461 | 0.5619 | 0.0570 |
| Logistic Regression (tuned, v1) + HMM regime feature | 0.5459 | 0.5606 | 0.0562 |
| Gradient Boosting (XGBoost) (tuned, v3) + order-flow + regime feature | 0.5448 | 0.4472 | 0.0444 |
| Gradient Boosting (XGBoost) (tuned, v3) | 0.5442 | 0.4424 | 0.0436 |
| Random Forest (tuned, v3) + order-flow + HMM gate (cov 56%) | 0.5437 | 0.4863 | 0.0497 |
| Random Forest (tuned, v3) + order-flow + regime feature | 0.5435 | 0.4708 | 0.0429 |
| Random Forest (tuned, v3) | 0.5424 | 0.4680 | 0.0419 |
| Gradient Boosting (XGBoost) (tuned, v2) + order-flow + HMM gate (cov 56%) | 0.5422 | 0.5224 | 0.0470 |
| Gradient Boosting (XGBoost) (tuned, v2) + order-flow + regime feature | 0.5420 | 0.5149 | 0.0441 |
| Gradient Boosting (XGBoost) (tuned, v2) | 0.5335 | 0.5004 | 0.0342 |
| Gradient Boosting (XGBoost) (no-flat, v2) | 0.5335 | 0.4672 | 0.0317 |
| SVM (RBF kernel) (no-flat, v2) | 0.5326 | 0.5048 | 0.0317 |
| Logistic Regression (tuned, v2) | 0.5310 | 0.4808 | 0.0217 |
| Logistic Regression (no-flat, v2) | 0.5285 | 0.4550 | 0.0240 |
| Random Forest (tuned, v2) | 0.5256 | 0.3981 | 0.0197 |
| Random Forest (no-flat, v2) | 0.5215 | 0.4963 | 0.0227 |
| Logistic Regression (tuned, v3) | 0.5203 | 0.4257 | 0.0172 |
| HMM-regime binary | 0.5195 | 0.4956 | 0.0200 |
| SVM (RBF kernel) (v2, flat-incl) | 0.5157 | 0.6060 | 0.0949 |
| Random Forest (no-flat) | 0.5071 | 0.5013 | 0.0093 |
| Gradient Boosting (XGBoost) (tuned, v1) | 0.5060 | 0.3370 | 0.0048 |
| Gradient Boosting (XGBoost) (no-flat) | 0.5028 | 0.4259 | 0.0030 |
| SVM (RBF kernel) (no-flat) | 0.5027 | 0.4973 | 0.0144 |
| Random Forest (tuned, v1) | 0.5010 | 0.5676 | 0.0121 |
| Random Forest | 0.4981 | 0.6873 | 0.0273 |
| Logistic Regression | 0.4944 | 0.7420 | 0.0235 |
| Logistic Regression (v2, flat-incl) | 0.4943 | 0.7421 | 0.0148 |
| Gradient Boosting (XGBoost) (v2, flat-incl) | 0.4943 | 0.7420 | 0.0112 |
| Gradient Boosting (XGBoost) | 0.4942 | 0.7421 | 0.0039 |
| Random Forest (v2, flat-incl) | 0.4942 | 0.7363 | 0.0390 |
| SVM (RBF kernel) | 0.4941 | 0.5680 | -0.0088 |
