# Pipeline Interfaces & Module Contracts

This document specifies the **data contracts** between every stage of the futures
direction-prediction pipeline so that **any single module can be replaced by an
independent implementation** (e.g. a collaborator's) without touching the rest.
If your module honours the input/output contract of the stage it replaces, it
drops in. It is the canonical reference for an LLM/agent (Claude, Codex, …)
generating interoperable code against this repo.

Scope: the **binary up/down** classification pipeline on `main`. (3-class and
gated-cascade experiments live on a separate branch and are out of scope here.)

---

## 1. Repo structure

```
src/
  load.py          # read + validate the raw CSV
  features.py      # build_features        → 20-dim v1 lagged matrix
  features_v2.py   # build_features_v2     → 49-dim v2 matrix (+ cached loader)
  split.py         # split                 → 50/50 time-ordered train/test
  labels.py        # build_labels / direction_labels / move_series / flat_mask
  config.py        # load_config, model_params  (reads config.yaml)
  models/
    baseline.py rf.py gbm.py svm.py   # train/predict/save/load per model
    regime_hmm.py                     # Gaussian-HMM regime utilities
    regime_binary.py                  # binary HMM-regime model
  binary_suite.py  # train baseline/rf/gbm/svm with feature-set + flat toggle
  pipeline.py      # end-to-end driver for one production model
  statistics.py    # compute (classification metrics) + backtest (equity/P&L)
  backtest.py      # selection/I-O runner around statistics.backtest
  run_stats.py     # batch stats orchestrator (Sections A/C/D + nft no-flat-test)
  feature_importance.py
  evaluate.py      # legacy accuracy/recall/confusion helpers
app.py             # Streamlit GUI (train / predict / statistics+backtest)
data/raw/data.csv  # read-only input        data/processed/  generated artifacts (gitignored)
docs/notes/        # generated reports (*.md) and equity-curve graphs (*.png)
config.yaml        # single source of tuneable params
```

**Non-negotiable invariants** (every replacement module must respect):
`random_state = 42`; **no look-ahead** (features for minute *t* use only *< t*);
the **test set is sacred** (never fit/tune on it); the split is **50/50
time-ordered, not shuffled**; **persist predictions** to `data/processed/*.npz`.

---

## 2. End-to-end call sequence (what to call after each step)

Each step's return value is the next step's input. A replacement module must
produce the same output type/shape for the step it replaces.

1. `df = load.load_raw(path)` → raw `DataFrame` (timestamps parsed).
2. `features = features.build_features(df)` *(or `features_v2.build_features_v2(df)`)*
   → feature matrix, **rows aligned to `df.iloc[4:]`** (first 4 warm-up rows dropped).
3. `raw_align = df.iloc[4:].reset_index(drop=True)` → row-aligned raw slice (for labels/returns).
4. `X_train, X_test = split.split(features, train_size=0.5)` and
   `raw_train, raw_test = split.split(raw_align, train_size=0.5)`.
5. `y_train = labels.build_labels(raw_train)` (and `y_test = build_labels(raw_test)`).
   *Binary-suite only:* drop flats with `keep = ~labels.flat_mask(raw_train)` before fitting.
6. `model = models.<algo>.train(X_train, y_train, params=…, save_path=…)`.
7. `y_pred = models.<algo>.predict(model, X_test)` → `ndarray` of `{0,1}`.
8. **Persist:** `np.savez("data/processed/<tag>_predictions.npz", y_true=y_test.to_numpy(), y_pred=y_pred[, move=…])`.
9. `result = statistics.compute(y_true, y_pred, name=…)` → `StatsResult` (accuracy, MCC, …).
10. Backtest: `bar_returns = (raw_test.Close - raw_test.Open) / raw_test.Open`;
    `bt = statistics.backtest(y_pred, bar_returns, timestamps=raw_test["Date and Time"], transaction_cost=c)`.
11. `statistics.plot_equity_curve(bt, "docs/notes/backtest_<tag>.png")` → equity graph (strategy + passive).
12. Report: `statistics.write_results([result], …)` / `write_backtest_results([bt], …)`.

**Pre-wired drivers** (already chain the above — call these, or swap a module they use):
`pipeline.run(algo)` (steps 1–9 for one model) · `run_stats.main()` (batch stats →
`docs/notes/*.md`) · `backtest.run(algo, transaction_cost)` (steps 1,3–4,8,10–12) ·
`tuning.run_tuning(cfg, …)` (model selection → `docs/notes/tuning_stats.md`).

**Model-selection harness (`src.tuning`).** Optimises **no-flat test accuracy**
while keeping the test set sacred: hyperparameters, threshold, and feature subset
are chosen on a **no-flat validation fold carved from the training half**
(`build_selection_split`, the last `val_frac` of train, time-ordered — every
validation timestamp precedes the test-half start). Only the final retrained model
touches the test set, once. Model `train()` functions accept an optional
`sample_weight` (the harness uses `|Close − Open|` when `use_move_weight=True`).
Artifacts: `tuned_{featset}_{algo}_predictions.npz` (`y_true, y_pred, move, keep`),
`tuned_params_{featset}.json`, `selected_{featset}.json`. The Streamlit **Predict**
tab can load these tuned models (a "Use tuned (regularized) model" checkbox, plus
the v1/v2/**v3** feature radio) and applies the stored threshold via
`tuning.predict_with_threshold`.

```mermaid
flowchart LR
    A[load_raw] --> B[build_features / build_features_v2]
    A --> R[df.iloc[4:] raw_align]
    B --> S[split → X_train/X_test]
    R --> S2[split → raw_train/raw_test]
    S2 --> L[build_labels / flat_mask]
    S --> T[model.train]
    L --> T
    T --> P[model.predict → y_pred]
    P --> NPZ[(save *_predictions.npz)]
    NPZ --> ST[statistics.compute → StatsResult]
    R --> BR[bar_returns = Close-Open / Open]
    P --> BT[statistics.backtest → BacktestResult]
    BR --> BT
    BT --> PLT[plot_equity_curve → PNG]
    ST --> RPT[write_results → docs/notes/*.md]
    BT --> RPT
```

---

## 3. Per-boundary data contracts

### 3.1 Raw input — `data/raw/data.csv` → `load_raw(path) -> pd.DataFrame`
551,521 rows × 14 columns, monotonically increasing by `Date and Time`, no NaN,
no zero-volume rows.

| Column | dtype | Notes |
|---|---|---|
| `Date`, `Time` | str | redundant with `Date and Time` |
| `Date and Time` | datetime64[ns] | **primary index**, parsed by `load_raw` |
| `Symbol` | str | 13 TY contracts stitched (gaps at rolls) |
| `Open`,`High`,`Low`,`Close`,`VWAP` | float64 | minute-bar prices |
| `Volume`,`Up Ticks`,`Down Ticks`,`Same Ticks`,`Tick Count` | int64 | volume/flow |

A replacement loader must return a DataFrame with **these columns and dtypes**,
sorted by `Date and Time`, with `Date and Time` parsed to `datetime64`.

### 3.2 Features — `build_features(df) -> DataFrame` / `build_features_v2(df) -> DataFrame`
- Input: full raw DataFrame from `load_raw`.
- Output: float matrix, **shape `(len(df) - 4, K)`**, integer index reset to
  `0..n-1`, **aligned to `df.iloc[4:]`** (so row *i* describes raw row *i+4*).
  `K = 20` for v1, `K = 49` for v2.
- v1 columns: `[m-4,m-3,m-2,m-1] × [Open,Close,High,Low,VWAP]` (lags via timestamp
  arithmetic on a 1-min grid — **not row offsets**).
- v2 adds derived indicators (returns, RSI, vol, MACD, tick-flow, time-of-day…);
  the regime-descriptor subset is `regime_hmm.REGIME_COLS`
  (`lag1_vol15, lag1_macd_hist, lag1_rsi15, lag1_tick_delta, lag1_return`).
- **Contract for a swap:** accept the raw df, return an `(n-4, K)` numeric matrix
  aligned to `df.iloc[4:]`, no NaN, reset index. Column names are free unless a
  downstream regime model needs `REGIME_COLS`.

### 3.3 Split — `split(df, train_size=0.5) -> (train, test)`
First `train_size` fraction of rows → train, remainder → test; **no shuffle**;
both returned with reset index. Works on either the feature matrix or a raw
slice. Validates timestamp monotonicity when a `Date and Time` column is present.

### 3.4 Labels — `src/labels.py`
| Function | Output | Meaning |
|---|---|---|
| `build_labels(raw)` / `direction_labels(raw)` | int Series `{0,1}` | `1` if `Close > Open` else `0` |
| `move_series(raw)` | float Series | `Close − Open` (signed intrabar move) |
| `flat_mask(raw)` | bool ndarray | `True` where `Close == Open` (drop from **training only**, or use as a test-set *evaluation* slice — see below) |

Input is the row-aligned raw slice (`df.iloc[4:]` or its train/test split). Length
and order match the feature matrix.

**No-flat test slice.** `run_stats.test_flat_mask(cfg)` returns the test-set keep
mask (`~flat_mask(raw_test)`), and `run_stats.section_noflat_test(cfg)` recomputes
stats for every saved prediction set on that slice → `*_noflat_test.md` reports.
This drops flat bars from the **metrics only** (predictions were already made on
the whole test set, blind to flatness); it is not a refit and does not violate
the "test set is sacred" rule. Equivalently, `flat ⇔ move == 0 ⇔ bar_return == 0`.

### 3.5 Models — `src/models/<algo>.py` (baseline, rf, gbm, svm)
Uniform interface:
- `train(X: DataFrame, y: Series, params: dict|None=None, save_path: Path|None=None) -> model`
  — fits with `random_state=42`; persists to `save_path` (default
  `data/processed/<algo>_model.joblib`).
- `predict(model, X: DataFrame) -> np.ndarray[int]` in `{0,1}`, length `len(X)`.
- `save(model, path)` / `load(path)`.
- Artifact type: `baseline`→`LogisticRegression`, `rf`→`RandomForestClassifier`,
  `gbm`→`XGBClassifier`, `svm`→`{"scaler": StandardScaler, "clf": SVC}` (dict).
- **Swap contract:** any object with a matching `train`/`predict` (mapping
  `(X_train,y_train)`→fitted model, `(model,X_test)`→`{0,1}` array) is a drop-in.

`models/regime_binary.run(config) -> dict` trains a per-regime binary RF on
`features_v2`; saves `exp_regime_binary_{hmm,scaler,dir_r0,dir_r1}.joblib` and the
predictions npz (see §4).

### 3.6 Statistics — `statistics.compute(y_true, y_pred, name="", labels=None) -> StatsResult`
Pure function of the two label arrays. `StatsResult` (TypedDict):
`name, n_samples, accuracy, macro_f1, weighted_f1, mcc, classes: list[int],
per_class: {cls: {precision,recall,f1,support}}, confusion_matrix: list[list[int]]`.
Works for binary and multiclass.

### 3.7 Backtest — `statistics.backtest(signals, bar_returns, timestamps=None, initial=1000.0, transaction_cost=0.0, periods_per_year=None, signal_encoding="binary", name="") -> BacktestResult`
- `signals`: `{0,1}` predictions (binary) — mapped to position `+1/-1` (and `0` for
  hold under `three_class`).
- `bar_returns`: per-bar `(Close-Open)/Open`, same length/order as signals.
- Compounds equity from `initial` (reinvest all); deducts `transaction_cost` per bar
  a position is taken. Returns per-bar `position/bar_return/net_return/payoff/equity/
  passive_equity` arrays + scalars `final_equity, total_return, max_drawdown,
  annualized_sharpe` and the **passive always-long benchmark** (`passive_*`).

---

## 4. Persisted artifact formats (`data/processed/`)

| File | Keys / type |
|---|---|
| `<algo>_model.joblib` | the model artifact above (joblib) |
| `<algo>_predictions.npz` | `y_true, y_pred` (int `{0,1}`); binary-suite variants add `move` (float) |
| `exp_regime_binary_predictions.npz` | `y_true, y_pred, regime ({0,1}), move` |
| `backtest_<algo>_predictions.npz` | `signal, position, bar_return, net_return, payoff, equity, passive_equity, timestamps(datetime64)` |
| `features_v2.parquet` | cached 49-feature matrix |
| `training_metadata.json` | `{algo: {display_name, data_path, train_size, n_test, params, metrics:{accuracy,recall}, timestamp}}` |

Binary-suite prefixes: `exp_noflat_<algo>` (no-flat, 20-feat), `exp_v2_<algo>`
(flat-incl, 49-feat), `exp_noflat_v2_<algo>` (no-flat, 49-feat).

### `config.yaml` schema
```yaml
data:   { path: data/raw/data.csv, train_size: 0.5 }
models: { baseline: {max_iter}, rf: {n_estimators, max_depth, …},
          gbm: {n_estimators, learning_rate, …}, svm: {C, kernel, gamma, …} }
```
`config.load_config(path) -> dict`; `config.model_params(cfg, algo) -> dict`.

---

## 5. How to replace a stage
Implement the function(s) for that stage with the **exact input → output contract**
above, keep the artifact filenames/keys identical, and the rest of the pipeline (and
the GUI) keeps working. Quick checks: `pytest` (each module has a `tests/test_*.py`),
then `python -m src.pipeline --algo rf`, `python -m src.run_stats`, and
`python -m src.backtest --algo rf` should run end-to-end on your module.
