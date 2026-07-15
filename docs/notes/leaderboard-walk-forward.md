# Walk-Forward Leaderboard

Rolling walk-forward (3-month train / 1-month test, from `config.yaml`) over the flat-free modelling set. **Ranked by mean fold accuracy.** *Folds won* = folds beating the always-up baseline; *AUM %* = total return of the compounding long/short backtest over the concatenated walk-forward test bars.

| Model | Mean acc ± std | Folds won | Recall | AUM % |
|-------|----------------|-----------|--------|-------|
| tuned_v1_logistic | 53.6% ± 2.2% | 34/34 | 0.7401 | +2464.6% |
| v1_logistic | 53.4% ± 2.1% | 34/34 | 0.7390 | +1910.6% |
| tuned_v3_gbm | 53.3% ± 1.8% | 33/34 | 0.5927 | +1685.2% |
| ofhmm_v3_gbm | 53.2% ± 1.8% | 33/34 | 0.5668 | +1648.2% |
| tuned_v3_rf | 53.2% ± 1.7% | 32/34 | 0.5894 | +1551.0% |
| v3_gbm | 53.0% ± 1.8% | 33/34 | 0.5702 | +1340.0% |
| tuned_v2_logistic | 53.0% ± 1.7% | 33/34 | 0.6074 | +1295.2% |
| v1rel_gbm | 52.9% ± 1.8% | 32/34 | 0.5651 | +1241.8% |
| ofhmm_v3_rf | 52.9% ± 1.7% | 32/34 | 0.5611 | +1273.5% |
| v3_rf | 52.8% ± 1.6% | 32/34 | 0.5552 | +1036.2% |
| v1rel_rf | 52.7% ± 1.9% | 29/34 | 0.5450 | +841.8% |
| v2_logistic | 52.4% ± 1.3% | 31/34 | 0.6326 | +767.1% |
| ofhmm_v1_logistic | 52.1% ± 1.5% | 29/34 | 0.6657 | +593.5% |
| tuned_v2_gbm | 52.0% ± 1.2% | 30/34 | 0.6372 | +418.2% |
| v2_gbm | 52.0% ± 1.3% | 29/34 | 0.5691 | +427.4% |
| tuned_v2_rf | 51.8% ± 0.9% | 27/34 | 0.5826 | +438.1% |
| v3_logistic | 51.8% ± 0.9% | 31/34 | 0.6220 | +414.6% |
| tuned_v3_logistic | 51.7% ± 0.9% | 30/34 | 0.6083 | +413.3% |
| v2_rf | 51.5% ± 1.0% | 25/34 | 0.5531 | +282.1% |
| v1_rf | 50.6% ± 0.8% | 14/34 | 0.5388 | +62.5% |
| v1_gbm | 50.5% ± 0.9% | 15/34 | 0.5647 | +51.8% |
| tuned_v1_gbm | 50.5% ± 0.9% | 18/34 | 0.6291 | +50.2% |
| v1rel_logistic | 50.5% ± 0.8% | 0/34 | 0.9740 | +55.9% |
| tuned_v1_rf | 50.4% ± 0.8% | 17/34 | 0.5743 | +65.4% |
