# Module Reference

One entry per function: a type-hinted **signature**, a one-line **description**, and a
**pseudocode block** of its logic (CLAUDE.md rule 6). Private helpers (leading `_`) are
documented where load-bearing but are exempt from the completeness check enforced by
`tests/test_gen_module_docs.py`. The module diagram + inventory in
`docs/notes/module_diagram.md` is auto-generated; this file's pseudocode is author-written.

## src.load

### `load_raw`
`(path: Path | str) -> pd.DataFrame`

Read the raw CSV and parse `"Date and Time"` as `datetime64`, sorting if needed.

```
df = pd.read_csv(path)
df["Date and Time"] = to_datetime(df["Date and Time"])
if not df["Date and Time"].is_monotonic_increasing:
    warn; df = df.sort_values("Date and Time").reset_index(drop=True)
return df
```

### `validate`
`(df: pd.DataFrame) -> None`

Assert every required column is present and log the NaN count per column.

```
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing: raise ValueError(missing)
for col in REQUIRED_COLS: print(col, df[col].isna().sum())
if not df["Date and Time"].is_monotonic_increasing: warn
```

## src.split

### `split`
`(df: pd.DataFrame, train_size: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]`

Time-ordered split: first `train_size` fraction → train, remainder → test (no shuffle).

```
if not 0 < train_size < 1: raise ValueError
if "Date and Time" in df and not monotonic: raise ValueError
mid = int(len(df) * train_size)
return df.iloc[:mid].reset_index(drop=True), df.iloc[mid:].reset_index(drop=True)
```

## src.features

### `build_features`
`(df: pd.DataFrame) -> pd.DataFrame`

Build the 20-dim v1 lag matrix: [Open, Close, High, Low, VWAP] from the 4 prior clock-minutes, on a dense forward-filled 1-min grid; drop the 4 warm-up rows.

```
ohlcv = df.set_index("Date and Time")[FEATURE_COLS]
grid  = date_range(first, last, freq="1min")
filled = ohlcv.reindex(grid).ffill()               # gap minutes forward-filled
for lag in [4,3,2,1]:
    for col in FEATURE_COLS:
        lag_data[f"lag{lag}_{col}"] = filled[col].shift(lag)   # k>=1, causal
features = DataFrame(lag_data, grid).reindex(ohlcv.index)  # back to real timestamps
return features.iloc[4:].reset_index(drop=True)    # assert 20 columns
```

## src.labels

### `build_labels`
`(df: pd.DataFrame) -> pd.Series`

Binary up/down label per row: 1 if `Close > Open`, else 0. No rows dropped.

```
return (df["Close"] > df["Open"]).astype(int).reset_index(drop=True)
```

### `direction_labels`
`(raw_align: pd.DataFrame) -> pd.Series`

Alias of `build_labels` — the "direction" target name used by the binary/regime suites.

```
return build_labels(raw_align)
```

### `move_series`
`(raw_align: pd.DataFrame) -> pd.Series`

Signed intrabar move `Close − Open` per bar (float, reset index).

```
return (raw_align["Close"] - raw_align["Open"]).reset_index(drop=True)
```

### `flat_mask`
`(raw_align: pd.DataFrame) -> np.ndarray`

Boolean mask, True where the bar is flat (`Close == Open`). Computed after features are built, so it never alters another row's lags.

```
return raw_align["Close"].values == raw_align["Open"].values
```

### `drop_flat`
`(features: pd.DataFrame, raw_align: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]`

Remove flat target rows from the aligned modelling set (features + raw) before splitting, so labels are strictly binary. No look-ahead: features were already built on the full series.

```
keep = ~flat_mask(raw_align)          # True = decisive bar
features  = features[keep].reset_index(drop=True)
raw_align = raw_align[keep].reset_index(drop=True)
log removed count
return features, raw_align
```

## src.features_v2

### `build_features_v2`
`(df: pd.DataFrame) -> pd.DataFrame`

Build the 49-dim v2 matrix: v1's 20 base OHLCV lags + 13 derived indicators lagged at {1,4} (26 cols) + 3 target-bar time features. Same dense-grid, forward-fill, drop-4-warm-up pattern.

```
filled = df.set_index("Date and Time")[base+tick].reindex(1min_grid).ffill()
indicators = {vwap_dev, bar_range, body_ratio, tick_delta, return, log_return,
              rsi5, rsi15, vol5, vol15, macd_line, macd_signal, macd_hist}  # on filled
for lag in [4,3,2,1]: base OHLCV lag columns = filled[col].shift(lag)
for lag in [1,4]:     new-indicator lag columns = series.shift(lag)
features = grid frame reindexed to real timestamps
add target-bar tod_sin/tod_cos (minute-of-day) + session_min (resets on >=12h gap)
features = features.iloc[4:].reset_index(drop=True); ffill().fillna(0) residual warm-up NaN
return features                       # 49 columns
```

### `load_or_build_features_v2`
`(df: pd.DataFrame) -> pd.DataFrame`

Return the v2 matrix from the parquet cache if present (build is ~12 min), else build and cache.

```
if features_v2.parquet exists: return read_parquet(cache)
features = build_features_v2(df); features.to_parquet(cache); return features
```

## src.features_v3

### `_transform_v3`
`(v2: pd.DataFrame) -> pd.DataFrame`

Pure row-wise transform: replace v2's 20 raw base-price lags with `log(value / lag1_Close)` and drop the now-constant `lag1_Close` (→ 48 cols). Stationary, no look-ahead.

```
ref = v2["lag1_Close"]                      # Close_{t-1}, strictly prior
for col in base 20 lag cols: out[col] = log(v2[col] / ref)
out = out.drop(columns="lag1_Close").reset_index(drop=True)
replace inf/NaN edge cells via ffill().fillna(0); assert 48 cols
return out
```

### `build_features_v3`
`(df: pd.DataFrame) -> pd.DataFrame`

Build the 48-dim stationary v3 matrix = `_transform_v3(build_features_v2(df))`.

```
return _transform_v3(build_features_v2(df))
```

### `load_or_build_features_v3`
`(df: pd.DataFrame) -> pd.DataFrame`

Return the v3 matrix from cache if present, else build from the cached v2 matrix and cache.

```
if features_v3.parquet exists: return read_parquet(cache)
v3 = _transform_v3(load_or_build_features_v2(df)); v3.to_parquet(cache); return v3
```

## src.features_v1_rel

### `build_features_v1_rel`
`(df: pd.DataFrame) -> pd.DataFrame`

Build the 19-dim stationary v1-relative matrix: v1's 20 raw lags → `log(value / lag1_Open)`, drop the constant `lag1_Open`. The v1 analog of v3 (anchored on the prior open). No look-ahead.

```
v1 = build_features(df); ref = v1["lag1_Open"]     # Open_{t-1}, strictly prior
for col in base 20 lag cols: out[col] = log(v1[col] / ref)
out = out.drop(columns="lag1_Open").reset_index(drop=True)
fix inf/NaN via ffill().fillna(0); assert 19 cols
return out
```

## src.features_orderflow

### `_order_flow_indicators`
`(filled: pd.DataFrame, variant: str = "raw") -> dict[str, pd.Series]`

Compute the 5 base order-flow indicators on a dense 1-min frame; trailing/causal windows (leakage protection deferred to the caller's lag step). Only `signed_vol` differs by variant.

```
norm_vol   = (Volume - Volume.roll(60).mean()) / Volume.roll(60).std()
sign       = sign(Close - Open)
signed_vol = sign * Volume / Volume.roll(60).mean()  if variant=="linear" (O(1) scale)
             else Volume * sign                        # "raw", tree models
tick_delta = (UpTicks - DownTicks) / TickCount
cum_td{w}  = tick_delta.rolling(w).sum() for w in [5,10,15]
return {norm_vol, signed_vol, cum_td5, cum_td10, cum_td15}
```

### `build_features_orderflow`
`(df: pd.DataFrame, variant: str = "raw") -> pd.DataFrame`

Build the 20-dim order-flow matrix: 5 indicators × 4 lags (t-1…t-4). Only lagged columns emitted — no lag-0 (its sign equals the label). Dense-grid, drop-4-warm-up pattern.

```
if variant not in {"raw","linear"}: raise ValueError
filled = df.set_index("Date and Time")[src_cols].reindex(1min_grid).ffill()
ind = _order_flow_indicators(filled, variant)
for lag in [4,3,2,1]:
    for name in indicators: lag_data[f"lag{lag}_{name}"] = ind[name].shift(lag)  # k>=1
features = grid frame reindexed to real timestamps
features = features.iloc[4:].reset_index(drop=True); ffill().fillna(0) warm-up NaN
return features                       # 20 columns
```

### `load_or_build_features_orderflow`
`(df: pd.DataFrame, variant: str = "raw") -> pd.DataFrame`

Return the order-flow matrix from the variant-specific parquet cache if present, else build and cache.

```
if variant not in {"raw","linear"}: raise ValueError
cache = linear_cache if variant=="linear" else raw_cache
if cache exists: return read_parquet(cache)
features = build_features_orderflow(df, variant); features.to_parquet(cache); return features
```

## src.models.logistic

The project's primary classifier (promoted from the former `baseline` module).

### `train`
`(X, y, params=None, save_path=None, sample_weight=None) -> LogisticRegression`

Fit L2 logistic regression; `max_iter=1000` default, `random_state=42` always enforced; auto-saves.

```
p = {"max_iter": 1000} | (params or {}); p["random_state"] = 42
model = LogisticRegression(**p).fit(X, y, sample_weight=sample_weight)
save(model, save_path or default); return model
```

### `predict`
`(model, X) -> np.ndarray`

Class-label predictions (0/1).

```
return model.predict(X)
```

### `save`
`(model, path="data/processed/logistic_model.joblib") -> None`

Serialize the fitted model to disk (creating parent dirs).

```
path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(model, path)
```

### `load`
`(path="data/processed/logistic_model.joblib") -> LogisticRegression`

Deserialize a model saved by `save`.

```
return joblib.load(path)
```

## src.models.rf

### `train`
`(X, y, params=None, save_path=None, sample_weight=None) -> RandomForestClassifier`

Fit Random Forest; defaults 500 trees / `sqrt` features / `min_samples_leaf=5` / `class_weight="balanced"` / `oob_score=True`; `random_state=42` enforced; auto-saves.

```
p = {n_estimators:500, max_depth:None, min_samples_leaf:5, max_features:"sqrt",
     oob_score:True, bootstrap:True, class_weight:"balanced", n_jobs:-1} | (params or {})
p["random_state"] = 42
model = RandomForestClassifier(**p).fit(X, y, sample_weight=sample_weight)
save(model, save_path or default); return model
```

### `predict`
`(model, X) -> np.ndarray`

Class-label predictions (0/1).

```
return model.predict(X)
```

### `save`
`(model, path="data/processed/rf_model.joblib") -> None`

Serialize the fitted forest (preserving `oob_score_`), creating parent dirs.

```
path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(model, path)
```

### `load`
`(path="data/processed/rf_model.joblib") -> RandomForestClassifier`

Deserialize a forest saved by `save`, `oob_score_` intact.

```
return joblib.load(path)
```

## src.models.gbm

### `train`
`(X, y, params=None, save_path=None, sample_weight=None) -> XGBClassifier`

Fit XGBoost; defaults 500 trees / lr 0.05 / depth 4 / subsample 0.8 / colsample 0.8 / `reg_lambda=1` / `binary:logistic`; `random_state=42` enforced; auto-saves.

```
p = {n_estimators:500, learning_rate:0.05, max_depth:4, subsample:0.8,
     colsample_bytree:0.8, reg_lambda:1.0, min_child_weight:1,
     objective:"binary:logistic", eval_metric:"logloss", n_jobs:-1} | (params or {})
p["random_state"] = 42
model = XGBClassifier(**p).fit(X, y, sample_weight=sample_weight)
save(model, save_path or default); return model
```

### `predict`
`(model, X) -> np.ndarray`

Class-label predictions (0/1).

```
return model.predict(X)
```

### `save`
`(model, path="data/processed/gbm_model.joblib") -> None`

Serialize the fitted booster, creating parent dirs.

```
path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(model, path)
```

### `load`
`(path="data/processed/gbm_model.joblib") -> XGBClassifier`

Deserialize a booster saved by `save`.

```
return joblib.load(path)
```

## src.baselines

Naive comparison baselines (not learned models); both look-ahead free.

### `predict_always_up`
`(n: int) -> np.ndarray`

Predict up (1) for every one of `n` bars.

```
return np.ones(n, dtype=int)
```

### `predict_last_direction`
`(y_train, y_test) -> np.ndarray`

Lag-1 persistence: predict the previous bar's realised direction; test row 0 uses the last train label.

```
preds[0]  = y_train.iloc[-1]
preds[1:] = y_test.values[:-1]        # previous test row's actual label
return preds
```

## src.config

### `load_config`
`(path: Path = "config.yaml") -> dict`

Parse the YAML config and return the full dict.

```
with path.open() as f: return yaml.safe_load(f)
```

### `model_params`
`(config: dict, algo: str) -> dict`

Return a copy of `config["models"][algo]`; empty dict if missing.

```
return dict(config.get("models", {}).get(algo, {}))
```

## src.evaluate

### `_select`
`(y_true, y_pred, keep, include_flat) -> tuple[np.ndarray, np.ndarray]`

Shared no-flat gate: restrict to non-flat rows unless `include_flat` or `keep is None`.

```
if include_flat or keep is None: return y_true, y_pred
keep = asarray(keep, bool); return y_true[keep], y_pred[keep]
```

### `accuracy`
`(y_true, y_pred, keep=None, *, include_flat=False) -> float`

Fraction correct; no-flat by default when a `keep` mask is supplied.

```
yt, yp = _select(y_true, y_pred, keep, include_flat); return accuracy_score(yt, yp)
```

### `recall`
`(y_true, y_pred, keep=None, *, include_flat=False) -> float`

Recall for class 1 (up), `zero_division=0`; no-flat by default when `keep` given.

```
yt, yp = _select(...); return recall_score(yt, yp, zero_division=0)
```

### `confusion`
`(y_true, y_pred, keep=None, *, include_flat=False) -> np.ndarray`

2×2 confusion matrix `[[TN,FP],[FN,TP]]`; no-flat by default when `keep` given.

```
yt, yp = _select(...); return confusion_matrix(yt, yp, labels=[0,1])
```

### `report`
`(name, y_true, y_pred, keep=None, *, include_flat=False) -> str`

Format accuracy + recall + confusion as a markdown section, with a note recording the slice.

```
yt, yp = _select(...); acc, rec = accuracy_score, recall_score; tn,fp,fn,tp = confusion.ravel()
note = "full set" if include_flat or keep is None else "non-flat N bars"
return markdown(name, note, acc, rec, confusion table)
```

### `write_results`
`(reports: list[str], path: Path, intro: str | None = None) -> None`

Write markdown report sections to a file (default header documents the full test set; callers reporting a slice pass their own `intro`).

```
path.parent.mkdir(parents=True, exist_ok=True)
header = intro or default_full_test_header
path.write_text(header + "\n---\n".join(reports) + "\n")
```

## src.statistics

### `compute`
`(y_true, y_pred, name="", labels=None) -> StatsResult`

Full evaluation stats (binary or multi-class): accuracy, macro/weighted F1, MCC, per-class precision/recall/F1/support, confusion matrix.

```
class_labels = labels or sorted(unique(y_true))
prec,rec,f1,sup = precision_recall_fscore_support(y_true, y_pred, labels=class_labels)
per_class = {lbl: ClassMetrics(prec,rec,f1,sup) per label}
return StatsResult(accuracy, macro_f1, weighted_f1, mcc, per_class, confusion_matrix)
```

### `format_markdown`
`(result: StatsResult) -> str`

Render a `StatsResult` as a markdown section: scalar table, per-class table, confusion matrix.

```
lines = [name, n_samples, scalar-metrics table, per-class table, confusion table]
return "".join(lines)
```

### `write_results`
`(results: list[StatsResult], path: Path) -> None`

Write a list of `StatsResult` sections to one markdown file.

```
path.parent.mkdir(...); path.write_text(header + "\n---\n".join(format_markdown(r)) + "\n")
```

### `to_dict`
`(result: StatsResult) -> dict`

Return a JSON-serialisable plain dict (confusion matrix already list[list[int]]).

```
return dict(result)
```

### `backtest`
`(signals, bar_returns, timestamps=None, initial=1000.0, transaction_cost=0.0, periods_per_year=None, signal_encoding="binary", name="") -> BacktestResult`

Compounding long/short backtest: signals→positions, net each bar's return minus cost, compound equity from `initial`; also a passive always-long frictionless benchmark. Returns per-bar arrays + summary scalars (final equity, total return, max drawdown, annualized Sharpe).

```
if len(signals) != len(bar_returns): raise ValueError
position = _positions_from_signals(signals, encoding)     # +1/-1/0
net      = max(position*bar_returns - cost*|position|, -1) # cap loss at -100%
equity   = initial * cumprod(1+net); payoff = prev_equity * net
passive_equity = initial * cumprod(1 + max(bar_returns,-1))
periods_per_year = given or _periods_per_year(timestamps, n)
return BacktestResult(final_equity, total_return, max_drawdown(equity),
                      annualized_sharpe(net,ppy), passive_* metrics, per-bar arrays)
```

### `max_drawdown`
`(equity: np.ndarray) -> float`

Largest peak-to-trough decline of an equity curve, as a fraction in [0,1].

```
if len(equity)==0: return 0.0
peak = maximum.accumulate(equity); return max((peak - equity) / peak)
```

### `annualized_sharpe`
`(net_returns: np.ndarray, periods_per_year: float) -> float`

`sqrt(periods_per_year) * mean/std` (ddof=1, rf=0); 0.0 on degenerate/zero-vol input.

```
if len(r)<2 or ppy<=0: return 0.0
sd = r.std(ddof=1); if sd==0: return 0.0
return sqrt(ppy) * r.mean() / sd
```

### `plot_equity_curve`
`(result: BacktestResult, path: Path) -> Path`

Plot strategy equity ($) vs time with the passive buy-&-hold line + `$initial` reference; save PNG (lazy matplotlib import).

```
import matplotlib(Agg); x = timestamps or arange(n)
plot strategy equity, passive equity, initial reference line; title with metrics
fig.savefig(path); return path
```

### `format_backtest_markdown`
`(result: BacktestResult) -> str`

Markdown section: initial/final equity, total return, drawdown, Sharpe, cost, bars/year, passive rows + strategy−passive delta, optional equity image.

```
lines = [name, periods+dates, metrics table, passive rows, delta row]
if plot_path: append image link
return "".join(lines)
```

### `write_backtest_results`
`(results: list[BacktestResult], path: Path) -> None`

Write a list of `BacktestResult` sections to one markdown file.

```
path.write_text(header + "\n---\n".join(format_backtest_markdown(r)) + "\n")
```

## src.backtest

Selection + I/O glue for backtesting saved binary models (math lives in `src.statistics`).

### `run_one`
`(algo, transaction_cost, timestamps, bar_returns) -> BacktestResult`

Backtest one prediction-set stem: load its `.npz`, guard length vs test bars, run `statistics.backtest`, persist per-record results + equity PNG.

```
y_pred = load(f"{algo}_predictions.npz")["y_pred"]
if len(y_pred) != len(bar_returns): raise ValueError (non-contiguous set)
result = statistics.backtest(y_pred, bar_returns, timestamps, transaction_cost, name)
np.savez(f"backtest_{algo}_predictions.npz", signal/position/equity/... )   # Rule 7
statistics.plot_equity_curve(result, png); result["plot_path"] = png; return result
```

### `run`
`(algo, transaction_cost=0.0, config=None) -> list[BacktestResult]`

Backtest one registry stem or `"all"` binary models, then write `docs/notes/backtest_stats.md`.

```
timestamps, bar_returns = _reconstruct_test_bars(cfg)     # 50/50 test slice
algos = [all registry stems with predictions] if algo=="all" else [algo]
results = [run_one(a, cost, timestamps, bar_returns) for a in algos]
statistics.write_backtest_results(results, report_path); return results
```

## src.walkforward

Rolling walk-forward validation — the iteration metric. Fixed-width calendar windows stepped forward; fresh model per fold; mean ± std across folds.

### `make_folds`
`(timestamps, train_months, test_months, step_months, purge=0, embargo=0) -> list[Fold]`

Build rolling calendar-time folds via `pd.DateOffset`: train `[t0, t0+train)`, test `[t0+train, +test)`, advance `t0` by `step`. Emits a fold only when both blocks are non-empty; asserts no train/test index overlap.

```
if any size <= 0: raise ValueError
ts = to_datetime(timestamps); if not monotonic: raise ValueError
t0 = ts[0]
while t0 + train_off <= last:
    train_pos = where(t0 <= ts < t0+train_off)
    test_pos  = where(t0+train_off <= ts < t0+train_off+test_off)
    test_pos  = test_pos[embargo:]; train_pos = train_pos[:-purge]   # if set
    if both non-empty: assert train_pos.max() < test_pos.min(); append Fold(...)
    t0 += step_off
return folds
```

### `walk_forward`
`(X, y, timestamps, model_factory, *, config=None, train_months=None, test_months=None, step_months=None, purge=None, embargo=None, predict_fn=None, fold_transform=None, returns=None, name="wf", save=True) -> WalkForwardResult`

Run walk-forward for one sklearn-style model. Sizes resolve kwargs → `config['walk_forward']` → skill defaults (3/1/1). Per fold: fresh `model_factory()`, fit train block, predict test block, accuracy. Persists per-fold predictions (+ optional `returns`) to `walkforward_{name}_predictions.npz` (Rule 7).

```
assert len(X)==len(y)==len(timestamps)
tr,te,st,pu,em = _resolve_sizes(...); folds = make_folds(ts, tr,te,st, pu,em)
if not folds: raise ValueError
for i, fold in enumerate(folds):
    X_tr,X_te,test_gate = fold_transform(train_idx,test_idx) or (X.iloc[...], None)
    model = model_factory(); model.fit(X_tr, y[train_idx])     # fresh per fold
    y_hat = predict_fn(model, X_te) if predict_fn else model.predict(X_te)
    score_mask = all-True & (test_gate if given)               # flat dropped upstream
    accuracies[i] = accuracy_score(y_te[score_mask], y_hat[score_mask])
    collect y_true/y_pred/fold_id/scored/(returns[test_idx])
if save: np.savez(walkforward_{name}, concatenated arrays + accuracies + windows)
return WalkForwardResult(mean/std/min/max accuracy, per_fold, npz_path)
```

### `summarize`
`(result: WalkForwardResult) -> str`

Render the headline `"X% ± y% across N folds (range a–b)"` plus a per-fold table.

```
lines = [headline mean±std range, per-fold table (window, n_train, n_test, acc)]
return "\n".join(lines)
```

### `sklearn_factory`
`(estimator_cls: type, params: dict) -> Callable[[], Any]`

Wrap an sklearn class + params into a zero-arg factory building a fresh seed-42 instance per call.

```
p = dict(params); p["random_state"] = 42
return lambda: estimator_cls(**p)
```

### `module_factory`
`(module, params: dict, tmp_path: Path) -> Callable[[], Any]`

Factory yielding a fresh `_ModuleEstimator` per fold — trains via a project model module (exact defaults + `params` overlay + seed 42), so walk-forward reproduces the production/tuned recipe.

```
return lambda: _ModuleEstimator(module, params, tmp_path)
# _ModuleEstimator.fit → module.train(X,y,params,save_path=tmp); predict → module.predict;
# predict_proba → underlying estimator's probabilities (for thresholding)
```

### `project_factories`
`(config: dict) -> dict[str, Callable[[], Any]]`

Ready factories for `logistic`/`rf`/`gbm` from `model_params(config, algo)`, seed 42.

```
return {"logistic": sklearn_factory(LogisticRegression, logistic_p or {max_iter:1000}),
        "rf": sklearn_factory(RandomForestClassifier, rf_p),
        "gbm": sklearn_factory(XGBClassifier, gbm_p)}
```

### `main`
`() -> None`

CLI: run walk-forward for one `--algo` on one `--featset`, print the summary.

```
args = parse(--algo, --featset, --train/test/step-months)
X,y,ts = _build_xy(featset)              # flat dropped, aligned
factory = project_factories(cfg)[algo]
result = walk_forward(X,y,ts,factory, config=cfg, name=f"{algo}_{featset}")
print(summarize(result))
```

## src.models.regime_hmm

Gaussian-HMM market-regime utilities shared by regime-based models.

### `build_rf`
`(params: dict) -> RandomForestClassifier`

Build an unfitted RF with `random_state=42` and `class_weight="balanced"` enforced.

```
p = dict(params); p["random_state"]=42; p["class_weight"]="balanced"
return RandomForestClassifier(**p)
```

### `fit_regime`
`(X_regime: np.ndarray) -> tuple[GaussianHMM, StandardScaler]`

Fit a 2-state full-covariance Gaussian HMM (+ train-only scaler) on the regime descriptors.

```
scaler = StandardScaler().fit(X_regime)            # train only
hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, seed=42)
hmm.fit(scaler.transform(X_regime)); return hmm, scaler
```

### `assign_regime`
`(hmm, scaler, X_regime) -> np.ndarray`

Viterbi (smoothing) decode of the state sequence. WARNING: uses the whole sequence → look-ahead; legacy `regime_binary` path only, prefer `filter_regime`.

```
return hmm.predict(scaler.transform(X_regime))     # smoothing = peeks at future
```

### `filter_regime_posterior`
`(hmm, scaler, X_regime) -> np.ndarray`

Causal forward-filtered posterior `P(state_t | descriptors_0..t)` — row `t` uses only rows ≤ `t` (descriptors themselves lag-1), so no look-ahead. Log-space forward recursion from the HMM's own params.

```
Xs = scaler.transform(X_regime)
log_b = per-state Gaussian emission logpdf; log_start=log(startprob); log_trans=log(transmat)
log_alpha[0] = log_start + log_b[0]
for t in 1..n-1: log_alpha[t] = logsumexp_i(log_alpha[t-1] + log_trans) + log_b[t]  # forward only
return exp(log_alpha - logsumexp(log_alpha, axis=1))   # normalise per row
```

### `filter_regime`
`(hmm, scaler, X_regime) -> np.ndarray`

Causal raw-state assignment = `argmax` of the filtered posterior — no-look-ahead replacement for `assign_regime`.

```
return filter_regime_posterior(hmm, scaler, X_regime).argmax(axis=1)
```

### `canonical_regime_labels`
`(regime_train, X_regime_train, vol15_col_idx) -> dict[int, int]`

Remap raw HMM states so 0 = low-vol, 1 = high-vol (by mean vol15 on train).

```
mean_vol[s] = mean(X_regime_train[regime_train==s, vol15_col_idx]) for s in {0,1}
return {0:1,1:0} if mean_vol[0] > mean_vol[1] else {0:0,1:1}
```

## src.models.regime_binary

Binary HMM-regime direction model (Experiment 5): a Gaussian HMM detects 2 regimes; a per-regime binary RF predicts up/down. No gate, no HOLD.

### `run`
`(config: dict | None = None) -> dict`

Build the flat-removed 50/50 v2 split, fit the HMM, train a per-regime binary RF (pooled fallback), predict every test bar, and persist predictions + HMM/scaler/per-regime joblibs + importances.

```
features = build_features_v2(df); split 50/50; drop flat from TRAIN only
hmm,scaler = fit_regime(X_reg_train); regime = canonical(assign_regime(...))
for r in {0,1}: dir_models[r] = build_rf(rf_params).fit(train rows in regime r)  # else pooled
for r in {0,1}: y_pred[regime_test==r] = dir_models[r].predict(...)
np.savez(exp_regime_binary_predictions, y_true, y_pred, regime, move)  # Rule 7
dump hmm/scaler/remap/dir_models + per-regime importance CSVs; return summary
```

### `load_bundle`
`(proc: Path = _PROC) -> dict`

Load the saved HMM-regime bundle (hmm, scaler, remap, per-regime dir_models) for inference.

```
for f in required exp_regime_binary_* joblibs: if missing raise FileNotFoundError
return {"hmm":..., "scaler":..., "remap":..., "dir_models": {0:..., 1:...}}
```

### `predict`
`(features: pd.DataFrame, bundle: dict) -> np.ndarray`

Decode each bar's regime (canonicalised via the saved remap) and dispatch it to that regime's direction model.

```
raw = assign_regime(bundle.hmm, bundle.scaler, X[:, regime_cols])
regime = remap[raw]
for r in {0,1}: y_pred[regime==r] = bundle.dir_models[r].predict(X[regime==r])
assert all predicted; return y_pred
```

## src.binary_suite

Binary classification suite (Experiment 4), configurable by feature builder. Flat is dropped globally; artifacts use a per-variant `{prefix}_*` name so variants never collide.

### `run`
`(config=None, algos=("logistic","rf","gbm"), build_features_fn=None, prefix="exp_noflat", display_suffix="no-flat") -> list[dict]`

Train the chosen algos on the flat-free 50/50 split, predict the whole test set, and save each model joblib, predictions `.npz` (`y_true`, `y_pred`, `move`), and importance CSV under `{prefix}_{algo}_*`.

```
X_tr,X_te,y_tr,y_te,move_te = _build_dataset(cfg, build_features_fn)   # drop_flat global
for algo in algos:
    model = _train(algo, X_tr, y_tr, params, f"{prefix}_{algo}_model.joblib")
    y_pred = _predict(algo, model, X_te)
    np.savez(f"{prefix}_{algo}_predictions.npz", y_true, y_pred, move=move_te)  # Rule 7
    _feature_importance(algo, model, cols) → CSV if available
    runs.append({algo, display, npz})
return runs
```

## src.pipeline

End-to-end driver for one production model: data → features → load-or-train → evaluate.

### `run`
`(algo, data_path="data/raw/data.csv", force_retrain=False, config=None, *, _dataset=None) -> PipelineResult`

Full pipeline for one algo. Reuses a pre-loaded `_dataset` when given (used by `--algo all`); loads the saved joblib unless `force_retrain`; writes training metadata when it actually trains.

```
X_tr,X_te,y_tr,y_te = _dataset or _build_dataset(data_path, train_size)  # drop_flat global
model, was_trained = _load_or_train(algo, X_tr, y_tr, force_retrain, params)
y_pred = _get_predictions(algo, model, X_te)
metrics = {accuracy, recall, confusion} via evaluate
if was_trained: _write_training_metadata(...)
return PipelineResult(algo, model, y_true, y_pred, metrics)
```

## src.feature_importance

### `run`
`() -> pd.DataFrame`

Train one RF (200 trees) on the 50% train split of the v2 matrix and return MDI importances ranked descending.

```
features = build_features_v2(df); X_train,_ = split(features); y = direction_labels(train)
model = RandomForestClassifier(200 trees, seed 42).fit(X_train, y)
return DataFrame(feature, importance).sort_values("importance", desc)  # 1-based rank
```

### `write_report`
`(df: pd.DataFrame) -> None`

Write the full importance ranking (Top-10 + all 49) to `docs/notes/feature_importance.md`.

```
lines = [header, Top-10 table, full 49-feature table]
_OUT_PATH.write_text("".join(lines))
```

## src.tuning

Model-selection harness that optimises no-flat **test accuracy** by selecting on a no-flat validation fold carved from the training half (the test set is touched once, at the end).

### `predict_with_threshold`
`(algo, model, X, threshold=None) -> np.ndarray`

Predict binary labels; `threshold=None` → plain `predict`, else `P(class=1) >= threshold`. Used by the GUI to apply the stored tuned threshold.

```
if threshold is None: return _predict(algo, model, X)
return (_scores(algo, model, X) >= threshold).astype(int)
```

### `build_selection_split`
`(cfg, feat_fn, val_frac=0.2) -> SelectionSplit`

Carve the flat-dropped inner-train + no-flat validation fold from the **training half only** (last `val_frac`, time-ordered) — no test-half rows, no leakage.

```
X_train, raw_train, _, _ = _load_splits(cfg, feat_fn)    # 50/50, flat dropped globally
cut = int(len(X_train) * (1 - val_frac))
inner = rows[:cut] with flat dropped; val = rows[cut:] with flat dropped
return SelectionSplit(X_inner, y_inner, move_inner, X_val, y_val, val_start)
```

### `grid_search`
`(algo, sel, grid=None) -> tuple[dict, float, list]`

Score each curated param combo by no-flat validation accuracy (fit on inner-train); return best params, best val accuracy, and all (params, acc) results.

```
for params in grid or _GRIDS[algo]:
    model = _fit(algo, sel.X_inner, sel.y_inner, params)
    acc = mean(_predict(algo, model, sel.X_val) == sel.y_val)
    track best
return best_params, best_acc, results
```

### `tune_threshold`
`(scores_val, y_val) -> float`

Sweep the decision threshold (over score quantiles) to maximise no-flat validation accuracy; return the best `>=` threshold.

```
candidates = unique(quantiles(scores_val, 2%..98%) + median)
return argmax_thr mean((scores_val >= thr) == y_val)
```

### `select_features`
`(sel, score_algo="logistic", k_grid=None) -> tuple[list[str], list]`

Rank features by L1-logistic coefficient magnitude (train-only scaling), sweep top-k, return the subset maximising no-flat validation accuracy.

```
scaler = StandardScaler().fit(sel.X_inner)       # train only
ranked = columns sorted by |L1-logistic coef| desc
for k in k_grid or {10,15,20,30,all}:
    acc = val accuracy of score_algo fit on ranked[:k]; track best subset
return best_cols, results
```

### `run_tuning`
`(cfg, algos=(...), featset="v2", use_move_weight=False, tune_thr=False, val_frac=0.2, select=False) -> list[dict]`

Per algo: grid-search → optional threshold tune → optional feature selection → retrain best on the full no-flat train half (optional |move| weighting) → evaluate once on the no-flat test slice. Persists `tuned_{featset}_{algo}_predictions.npz`, `tuned_params_{featset}.json`, and `tuning_stats_{featset}.md`.

```
sel = build_selection_split(cfg, feat_fn, val_frac)
X_tr,y_tr,move_tr, X_te,y_te,move_te, keep_te = _full_train_test(cfg, feat_fn)
if select: cols = select_features(sel); restrict all matrices to cols; save selected_*.json
for algo in algos:
    best_params = grid_search(algo, sel)[0]
    thr = tune_threshold(scores(inner-fit), sel.y_val) if tune_thr else 0.5
    model = _fit(algo, X_tr, y_tr, best_params, sample_weight=|move| if weighting)  # full no-flat train
    y_pred = (scores(model, X_te) >= thr) if tune_thr else _predict(algo, model, X_te)
    np.savez(tuned_{featset}_{algo}_predictions, y_true, y_pred, move, keep)   # Rule 7
    record no-flat TEST acc/mcc + chosen config
write tuned_params_{featset}.json + tuning_stats_{featset}.md; return results
```

### `main`
`() -> None`

CLI: run `run_tuning` for the chosen `--algos`/`--featset` with optional `--move-weight`, `--tune-threshold`, `--select`.

```
args = parse(...); run_tuning(cfg, algos, featset, move_weight, tune_thr, val_frac, select)
```

## src.run_stats

Batch statistics / leaderboards / walk-forward orchestrator over the saved prediction sets.

### `section_a`
`(cfg: dict) -> list[StatsResult]`

Load production joblibs (logistic/rf/gbm), infer on the 50% test set (no retraining), persist predictions, and return per-model stats.

```
_, X_test, _, y_test = _build_dataset(data_path, train_size)   # flat dropped globally
for algo in production models with a joblib:
    model = load(jpath); y_pred = predict(model, X_test)
    np.savez(f"{algo}_predictions.npz", y_true, y_pred)          # Rule 7
    results.append(statistics.compute(y_true, y_pred, display))
return results
```

### `section_c`
`(cfg, skip_existing=False) -> None`

Additively train the no-flat 20-feature binary suite + HMM-regime binary and write `binary_noflat_stats.md` with per-label metrics and a per-regime breakdown.

```
binary_suite.run(cfg, missing algos); regime_binary.run(cfg)
results = [statistics.compute per saved exp_noflat_* + HMM set]
per_regime_md = per-regime confusion blocks for the HMM model
write header + formatted results + per_regime_md → binary_noflat_stats.md
```

### `section_d`
`(cfg, skip_existing=False) -> None`

Additively train the 49-feature no-flat binary variant on the cached v2 matrix and write `binary_v2_stats.md`.

```
for (prefix,suffix,title) in _V2_VARIANTS:
    binary_suite.run(cfg, missing algos, build_features_fn=v2, prefix, suffix)
    results = [statistics.compute per saved {prefix}_{algo} set]
    append formatted block
write header + blocks → binary_v2_stats.md
```

### `rank_models`
`(cfg) -> list[tuple[str, str, float, float, float, float]]`

Rank saved prediction sets best-first `(stem, name, accuracy, recall, mcc, aum_pct)` on the flat-free test set (length-matched only), sorted by accuracy then MCC.

```
n_total, _, returns = _test_reference(cfg)
for each {stem}_predictions.npz (skip backtest_*):
    if len(y_pred) != n_total: skip
    res = statistics.compute(y_true, y_pred); aum = _aum_pct(y_pred, returns)
    rows.append((stem, _leaderboard_name(stem), accuracy, recall(class1), mcc, aum))
return rows sorted by (accuracy, mcc) desc
```

### `leaderboard`
`(cfg) -> None`

Write `docs/notes/leaderboard.md` (single test set): Model / Accuracy / Recall / MCC / AUM %, sorted by accuracy then MCC; list length-mismatched sets as excluded.

```
rows = rank_models(cfg); skipped = length-mismatched stems
lines = [header, excluded note, table rows]
_LEADERBOARD_PATH.write_text("".join(lines))
```

### `walkforward_curated`
`(cfg) -> None`

Rolling walk-forward for the curated core (logistic/rf/gbm × v1/v1rel/v2/v3) with **config-default** hyperparameters + an always-up baseline; persists per-fold `wf_{featset}_{algo}` sets (with returns).

```
for featset in _WF_FEATSETS:
    X,y,ts,rets = _wf_xy(cfg, featset)                 # flat dropped, aligned
    for algo in _WF_ALGOS:
        factory = module_factory(module, model_params(cfg, algo), tmp)
        walk_forward(X,y,ts, factory, config=cfg, returns=rets, name=f"wf_{featset}_{algo}")
walk_forward(v1 rows, _AlwaysUp factory, name="wf_baseline_alwaysup")
```

### `walkforward_curated_tuned`
`(cfg) -> None`

Walk-forward the **tuned (regularized)** models: for each featset with a `tuned_params_{featset}.json`, fit each fold with the validation-selected hyperparameters + stored threshold; persists `wf_tuned_{featset}_{algo}` sets.

```
for featset in _WF_FEATSETS:
    spec = json.load(tuned_params_{featset}.json) or skip
    X,y,ts,rets = _wf_xy(cfg, featset)
    for algo, entry in spec["models"]:
        params = entry["params"]; thr = entry["threshold"] if spec.tune_threshold else None
        factory = module_factory(module, params, tmp)
        walk_forward(X,y,ts, factory, config=cfg, returns=rets,
                     predict_fn=_threshold_predict_fn(thr), name=f"wf_tuned_{featset}_{algo}")
```

### `walkforward_curated_regime_orderflow`
`(cfg) -> None`

Walk-forward the base + order-flow + HMM-regime-feature combos (logistic on v1+order-flow[linear], rf/gbm on v3+order-flow[raw]), each with a per-fold causal `regime_hi_prob` appended. Persists `wf_ofhmm_{base}_{algo}` sets.

```
for (base, variant, algo) in _COMBO_RECIPES:
    X, X_reg, y, ts, rets = _combo_xy(cfg, base, variant)   # base+orderflow cols, flat-free
    ft = _hmm_fold_transform(X, X_reg, vol15_idx, "feature") # per-fold train-only HMM → regime_hi_prob
    factory = module_factory(module, model_params(cfg, algo), tmp)
    walk_forward(X, y, ts, factory, config=cfg, returns=rets,
                 fold_transform=ft, name=f"wf_ofhmm_{base}_{algo}")
```

### `leaderboard_walkforward`
`(cfg, proc=_PROC, out=_WF_LEADERBOARD_PATH) -> None`

Write `docs/notes/leaderboard-walk-forward.md` from the `walkforward_wf_*` sets: mean ± std fold accuracy, folds-won vs the always-up baseline, recall, AUM %; ranked by mean accuracy.

```
base_acc = load(walkforward_wf_baseline_alwaysup)["accuracies"]
for each walkforward_wf_* set (skip baseline):
    accs = d["accuracies"]; res = statistics.compute(y_true, y_pred)
    won = sum(accs > base_acc); aum = _aum_pct(y_pred, d["returns"])
    rows.append((name, mean(accs), std(accs), recall, won, nfolds, aum))
rows sorted by mean desc; write table → out
```

### `walkforward_results`
`(cfg, proc=_PROC, out=docs/results.md) -> None`

Per-model walk-forward report → `docs/results.md`: mean ± std accuracy, recall, confusion, Δ vs the always-up baseline, and AUM %.

```
base_mean = mean(baseline accuracies)
for each walkforward_wf_* set (skip baseline):
    accs; res = statistics.compute(y_true, y_pred)
    delta = (mean(accs) - base_mean); aum = _aum_pct(y_pred, d["returns"])
    append section (mean±std range, recall, Δ, AUM %, confusion)
write blocks → out
```

### `main`
`() -> None`

CLI dispatcher for `--sections`: `a` (production stats), `c`/`d` (binary suites), `lb` (single-test leaderboard), `wf`/`wftuned` (walk-forward default/tuned), `lbwf` (WF leaderboard + results).

```
args = parse(--sections, --skip-existing)
a → section_a + write_results; c → section_c; d → section_d
lb → leaderboard; wf → walkforward_curated; wftuned → walkforward_curated_tuned
lbwf → leaderboard_walkforward + walkforward_results
```

## src.gen_module_docs

Generate the ASCII module diagram + inventory, and check MODULES.md completeness.

### `build_inventory`
`(src: Path = _SRC) -> dict[str, list[str]]`

AST-walk `src` → dotted module name → its top-level function names (in file order).

```
for path in src.rglob("*.py") (skip __init__/__pycache__):
    funcs = [n.name for n in ast.parse(path).body if isinstance(n, FunctionDef)]
    inv[_module_name(path, src)] = funcs
return inv
```

### `build_import_graph`
`(src: Path = _SRC) -> dict[str, list[str]]`

Map each `src` module → the intra-`src` modules it imports (deduped, no self, externals ignored).

```
modules = set(build_inventory(src))
for path: walk AST; collect Import/ImportFrom targets starting "src"
graph[me] = sorted(dep for dep in deps if dep in modules and dep != me)
return graph
```

### `render_module_diagram`
`(inv, graph) -> str`

Render the ASCII import graph + a module/function inventory table as markdown.

```
lines = [title, import-graph block (module └─▶ dep per edge),
         inventory table (module | functions | count)]
return "\n".join(lines)
```

### `generate`
`(diagram_path: Path = _DIAGRAM) -> None`

(Re)write `docs/notes/module_diagram.md` from the current AST.

```
inv = build_inventory(_SRC); graph = build_import_graph(_SRC)
diagram_path.write_text(render_module_diagram(inv, graph))
```

### `check_modules_md`
`(modules_path: Path = _MODULES) -> list[str]`

Return `module.func` for every public function lacking a `### \`func\`` entry **or** a pseudocode fence in MODULES.md (private helpers exempt); drives `--check`.

```
documented = _documented(text)     # {(module, func): has_fence} from ##/### structure
for mod, funcs in build_inventory(_SRC):
    for fn in funcs where not fn.startswith("_"):
        if not documented.get((mod, fn)): missing.append(f"{mod}.{fn}")
return missing
```

### `main`
`() -> None`

CLI: default regenerates the diagram; `--check` verifies MODULES.md completeness (exit 2 if incomplete).

```
if --check:
    missing = check_modules_md(); if missing: print + sys.exit(2); else print OK
else: generate()
```
