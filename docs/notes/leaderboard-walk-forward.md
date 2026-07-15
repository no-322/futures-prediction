# Walk-Forward Leaderboard

Rolling walk-forward (3-month train / 1-month test, from `config.yaml`) over the flat-free modelling set. **Ranked by mean fold accuracy.** *Folds won* = folds beating the always-up baseline; *AUM %* = total return of the compounding long/short backtest over the concatenated walk-forward test bars.

| Model | Mean acc ± std | Folds won | Recall | AUM % |
|-------|----------------|-----------|--------|-------|
| v1_logistic | 53.4% ± 2.1% | 34/34 | 0.7390 | +1910.6% |
| v3_gbm | 53.0% ± 1.8% | 33/34 | 0.5702 | +1340.0% |
| v1rel_gbm | 52.9% ± 1.8% | 32/34 | 0.5651 | +1241.8% |
| v3_rf | 52.8% ± 1.6% | 32/34 | 0.5552 | +1036.2% |
| v1rel_rf | 52.7% ± 1.9% | 29/34 | 0.5450 | +841.8% |
| v2_logistic | 52.4% ± 1.3% | 31/34 | 0.6326 | +767.1% |
| v2_gbm | 52.0% ± 1.3% | 29/34 | 0.5691 | +427.4% |
| v3_logistic | 51.8% ± 0.9% | 31/34 | 0.6220 | +414.6% |
| v2_rf | 51.5% ± 1.0% | 25/34 | 0.5531 | +282.1% |
| v1_rf | 50.6% ± 0.8% | 14/34 | 0.5388 | +62.5% |
| v1_gbm | 50.5% ± 0.9% | 15/34 | 0.5647 | +51.8% |
| v1rel_logistic | 50.5% ± 0.8% | 0/34 | 0.9740 | +55.9% |
