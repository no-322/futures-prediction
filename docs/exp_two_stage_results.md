# Experiment 2 — Two-Stage Cascade Classifier

Stage 1: Gate (tradeable vs skip) | Stage 2: Direction (long vs short)
Folds: 5  |  Model: Random Forest (both stages)

## Primary metrics (mean ± std across folds)

| Metric | Mean | Std | Description |
|--------|------|-----|-------------|
| coverage | 0.5628 | 0.1321 | fraction of bars the gate trades |
| conditional_hit_rate | 0.4631 | 0.0691 | directional accuracy on traded bars |

## Debug diagnostics (mean ± std across folds)

| Metric | Mean | Std | Description |
|--------|------|-----|-------------|
| gate_recall | 0.5932 | 0.1233 | recall on bars that genuinely moved > threshold |
| gate_precision | 0.5445 | 0.0232 | precision of gate predictions |
| gate_f1 | 0.5627 | 0.0544 | F1 of gate model on test fold |
| direction_mcc | 0.0118 | 0.0078 | MCC on bars that genuinely moved (regardless of gate) |
| threshold (pts) | 0.0000 | 0.0000 | tuned threshold per fold |

## Per-fold detail

| Fold | n_train | n_test | threshold | coverage | hit_rate | gate_recall | dir_mcc |
|------|---------|--------|-----------|----------|----------|------------|----------|
| 0 | 91,922 | 91,919 | 0.0000 | 0.8210 | 0.3251 | 0.8314 | 0.0029 |
| 1 | 183,841 | 91,919 | 0.0000 | 0.4810 | 0.4936 | 0.5131 | 0.0063 |
| 2 | 275,760 | 91,919 | 0.0000 | 0.5271 | 0.5007 | 0.5616 | 0.0087 |
| 3 | 367,679 | 91,919 | 0.0000 | 0.5290 | 0.4993 | 0.5739 | 0.0164 |
| 4 | 459,598 | 91,919 | 0.0000 | 0.4561 | 0.4971 | 0.4862 | 0.0246 |
