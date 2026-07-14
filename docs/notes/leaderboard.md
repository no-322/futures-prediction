# Model Leaderboard — single test set

Every binary up/down model on the flat-free 50/50 time-ordered test set (144,263 decisive bars). **Sorted by accuracy, then MCC.**

- Excluded (length mismatch / non-binary): `exp_noflat_gbm`, `exp_noflat_rf`, `exp_noflat_svm`, `exp_noflat_v2_gbm`, `exp_noflat_v2_rf`, `exp_noflat_v2_svm`, `exp_v2_gbm`, `exp_v2_rf`, `exp_v2_svm`, `gbm`, `rf`, `tuned_v1_gbm`, `tuned_v1_rf`, `tuned_v2_gbm`, `tuned_v2_rf`, `tuned_v3_gbm`, `tuned_v3_rf`.
*AUM %* = total return of the compounding long/short backtest on the test bars.

| Model | Accuracy | Recall | MCC | AUM % |
|-------|----------|--------|-----|-------|
