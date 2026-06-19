# Pipeline Module Diagram

End-to-end architecture of the Futures Price Direction Predictor, drawn in plain
ASCII (so it renders in any monospace view). Nodes are source files; arrows show
**execution / data-flow order** — read top to bottom, where `A --> B` (or
`A | v B`) means *A runs or feeds data before B*.

```text
  FUTURES PRICE DIRECTION PREDICTOR  —  module execution order

  config.yaml                          data/raw/data.csv
     |                                       |
     | load_config / model_params            | load_raw + validate
     v                                       v
  +-----------+                        +-----------+
  | config.py |                        | load.py   |
  +-----+-----+                        +-----+-----+
        |                                    |
        | params / train_size                | raw OHLCV (+ tick cols)
        |                                    |
        |             .----------------------+----------------------.
        |             v                      v                      v
        |       +-----------+          +-----------+          +-----------+
        |       |features.py|          |features_v2|          | labels.py |
        |       |  v1 (20)  |          |   (49)    |          | y / flat /|
        |       +-----+-----+          +-----+-----+          |   move    |
        |             |                      |                +-----+-----+
        |             |                      v                      |
        |             |                +-----------+                |
        |             |                |features_v3|                |
        |             |                | (48, stat)|                |
        |             |                +-----+-----+                |
        |             '----------------------+----------------------'
        |                                    |  X (features) + y (labels)
        |                                    v
        |                              +-----------+
        |                              | split.py  |   50/50 time-ordered
        |                              +-----+-----+   (no shuffle)
        |                                    |
        |        X_train / y_train           |          X_test / y_test
        |                                    v
        |   +--------------------------------------------------------+
        |   | src/models/                                            |
        |   |   baseline    rf    gbm    svm                         |
        |   |   regime_hmm  -->  regime_binary                       |
        |   +-------------------------------+------------------------+
        |                                   |  fitted model + y_pred
        |                                   v
        |   ===== ORCHESTRATORS  (drive the stages above) ============
        '-> pipeline.py      one production model end-to-end + metadata
            binary_suite.py  v1/v2 x flat-toggle variants  -> exp_* npz
            tuning.py        HPO on a no-flat VALIDATION fold (carved from
                             TRAIN); saves tuned_{feat}_{algo} + threshold
                                   |
                                   |  joblib / npz / parquet
                                   v
                            +----------------+
                            | data/processed |<--------------------------.
                            +-------+--------+                           |
                                    |                                    |
                  .-----------------+------------------.                 |
                  v                 v                  v                 |
            +-----------+     +-----------+      +-----------+           |
            |statistics |     |backtest.py|      |run_stats  |           |
            |.py compute|     | equity &  |      |.py  batch |           |
            |(+bt math) |     | P&L vs    |      | + no-flat |           |
            +-----+-----+     | passive   |      | test slice|           |
                  |           +-----+-----+      +-----+-----+           |
                  '-----------------+------------------'                 |
                                    |  markdown reports                  |
                                    v                                    |
                            +----------------+      +-----------------+  |
                            | docs/notes/*.md|      | app.py (GUI)    |  |
                            | docs/results.md|<reads| Train / Predict |--'
                            +----------------+      +-----------------+
                                                    loads model, applies
                                                    tuned threshold

  ----------------------------------------------------------------------------
  Non-negotiables (apply throughout):  seed = 42  |  no look-ahead (lags < t)
  |  test set is sacred (fit/tune on TRAIN only)  |  persist y_true/y_pred .npz
  ----------------------------------------------------------------------------
```

## Module inventory

| Module / functions | File | Purpose |
|--------------------|------|---------|
| `load_config`, `model_params` | `src/config.py` | Read `config.yaml` (train_size, per-model hyperparameters). |
| `load_raw`, `validate` | `src/load.py` | Parse raw CSV, validate schema/ordering. |
| `build_features` | `src/features.py` | v1 — 20-dim lagged OHLCV feature matrix. |
| `build_features_v2`, `load_or_build_features_v2` | `src/features_v2.py` | v2 — 49 features (OHLCV lags + VWAP-dev, tick-delta, RSI, MACD, vol, time-of-day); parquet cache. |
| `build_features_v3`, `load_or_build_features_v3` | `src/features_v3.py` | v3 — 48 **stationary** features (base lags → `log(price / Close_{t-1})`, v2 indicators kept). |
| `build_labels`, `flat_mask`, `move_series`, `direction_labels` | `src/labels.py` | Binary up/down label (Close > Open); flat-bar mask; signed move. |
| `split` | `src/split.py` | 50/50 time-ordered train/test split (no shuffle). |
| `train`, `predict`, `save`, `load` (+ `sample_weight`) | `src/models/{baseline,rf,gbm,svm}.py` | Logistic Regression, Random Forest, XGBoost, RBF-SVM (scaler fit on train only). |
| `run`, `predict`, `load_bundle` | `src/models/regime_hmm.py`, `src/models/regime_binary.py` | 2-state Gaussian HMM regime + per-regime binary RF. |
| `run` | `src/pipeline.py` | End-to-end driver for one production model (joblib cache + `training_metadata.json`). |
| `run`, `_build_dataset` | `src/binary_suite.py` | Binary suite over feature-set × flat-toggle variants → `exp_*` artifacts. |
| `build_selection_split`, `grid_search`, `tune_threshold`, `select_features`, `predict_with_threshold`, `run_tuning` | `src/tuning.py` | HPO/regularization on a **no-flat validation fold** carved from TRAIN; saves `tuned_{feat}_{algo}` models + thresholds. |
| `compute`, `format_markdown`, `write_results`, `backtest`, `plot_equity_curve` | `src/statistics.py` | Standardised `StatsResult` metrics + backtest math (equity / drawdown / Sharpe). |
| `run`, `run_one` | `src/backtest.py` | Simulate a model's signals into a compounding equity curve vs passive. |
| `section_a/c/d`, `test_flat_mask`, `section_noflat_test` | `src/run_stats.py` | Batch stats for all model variants + the no-flat-test evaluation slice. |
| `accuracy`, `recall`, `report`, `write_results` | `src/evaluate.py` | Legacy binary evaluation helpers. |
| feature-importance helpers | `src/feature_importance.py` | Rank/print model feature importances. |
| Training tab, Prediction tab | `app.py` | Streamlit GUI: train variants; predict (loads `data/processed/` models, applies tuned threshold; v1/v2/v3 + tuned/no-flat toggles). |

