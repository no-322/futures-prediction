# Module Reference

One entry per function per the project convention: one pseudo-code line = one function = one entry here.

## src.statistics

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute` | `(y_true, y_pred, name="", labels=None) -> StatsResult` | Compute full evaluation statistics. Works for binary and multi-class. Returns a `StatsResult` TypedDict with accuracy, macro/weighted F1, MCC, per-class precision/recall/F1/support, and confusion matrix. |
| `format_markdown` | `(result: StatsResult) -> str` | Format a `StatsResult` as a markdown section with scalar metrics table, per-class table, and confusion matrix. |
| `write_results` | `(results: list[StatsResult], path: Path) -> None` | Write a list of `StatsResult` reports to a markdown file; creates parent dirs. |
| `to_dict` | `(result: StatsResult) -> dict` | Return a JSON-serialisable plain dict from a `StatsResult`. |
| `backtest` | `(signals, bar_returns, timestamps=None, initial=1000.0, transaction_cost=0.0, periods_per_year=None, signal_encoding="binary", name="") -> BacktestResult` | Compounding long/short backtest: maps signals→positions (+1/-1/0), nets the per-bar return against the proportional per-bar `transaction_cost`, compounds equity from `initial`, and returns per-bar arrays (position/bar_return/net_return/payoff/equity/**passive_equity**) + summary scalars (final equity, total return, max drawdown, annualized Sharpe). Also computes a **passive always-long, frictionless buy-&-hold** benchmark (`passive_*` fields) on the same per-bar returns. |
| `max_drawdown` | `(equity: np.ndarray) -> float` | Largest peak-to-trough decline of an equity curve as a fraction in [0, 1]. |
| `annualized_sharpe` | `(net_returns: np.ndarray, periods_per_year: float) -> float` | `sqrt(periods_per_year) * mean/std` (ddof=1, rf=0); 0.0 on degenerate/zero-vol input. |
| `_positions_from_signals` | `(signals, encoding="binary") -> np.ndarray` | Map signals to positions: binary {1→+1, 0→-1}; three_class {0→0, 1→+1, 2→-1}. |
| `plot_equity_curve` | `(result: BacktestResult, path: Path) -> Path` | Plot strategy equity ($) vs time **with the passive buy-&-hold line** and a $initial reference line; save as PNG (lazy matplotlib import). |
| `format_backtest_markdown` | `(result: BacktestResult) -> str` | Markdown section: initial/final equity, total return, max drawdown, annualized Sharpe, transaction cost, bars/year, **passive benchmark rows + strategy−passive delta**, date range, optional equity-curve image. |
| `write_backtest_results` | `(results: list[BacktestResult], path: Path) -> None` | Write a list of `BacktestResult` sections to one markdown file. |

## src.backtest

Selection + I/O glue for backtesting saved binary models (math lives in `src.statistics`). Reconstructs the contiguous 50/50 test slice, maps a model's predictions to a trading P&L, and persists per-record results + equity-curve PNG + report.

| Function | Signature | Description |
|----------|-----------|-------------|
| `run` | `(algo: str, transaction_cost=0.0, config=None) -> list[BacktestResult]` | Backtest one registry stem or `"all"` binary models; writes `docs/notes/backtest_stats.md`. |
| `run_one` | `(algo, transaction_cost, timestamps, bar_returns) -> BacktestResult` | Backtest one model: load its `{algo}_predictions.npz`, guard length vs test bars, call `statistics.backtest`, save `backtest_{algo}_predictions.npz` + `backtest_{algo}.png`. |
| `_reconstruct_test_bars` | `(cfg: dict) -> tuple[np.ndarray, np.ndarray]` | Rebuild the 50/50 `raw_test` slice → `(timestamps, bar_returns=(Close-Open)/Open)`. |
| `_build_registry` | `() -> dict[str, str]` | Ordered map of backtestable binary prediction-set stems → display names. |

## src.walkforward

Rolling walk-forward validation — the iteration metric (see `.claude/skills/evaluation`). Fixed-width calendar windows stepped forward; fresh model per fold; reports per-fold accuracy **and** mean ± std.

| Function | Signature | Description |
|----------|-----------|-------------|
| `make_folds` | `(timestamps: pd.Series, train_months: int, test_months: int, step_months: int, purge=0, embargo=0) -> list[Fold]` | Build rolling calendar-time folds via `pd.DateOffset`: train `[t0, t0+train_months)`, test `[t0+train_months, +test_months)`, advancing `t0` by `step_months`. Asserts `timestamps` monotonic and per-fold `train_idx.max() < test_idx.min()`. Emits a fold only when both blocks are non-empty (drops partial tail). `purge` drops train rows nearest the boundary; `embargo` skips leading test rows. Returns `Fold` dataclasses (`train_idx`/`test_idx` int arrays + `train/test_start/end` Timestamps). |
| `walk_forward` | `(X, y, timestamps, model_factory, *, config=None, train_months=None, test_months=None, step_months=None, purge=None, embargo=None, keep=None, include_flat=False, drop_flat_train=False, predict_fn=None, fold_transform=None, name="wf", save=True) -> WalkForwardResult` | Run walk-forward for one sklearn-style model. Window sizes resolve explicit kwargs → `config['walk_forward']` → skill defaults (3/1/1). Per fold: fresh `model_factory()`, `fit(X_tr,y_tr)`, `predict_fn(model,X_te)` or `predict(X_te)`, accuracy. With a `keep` non-flat mask, **default scores the no-flat test slice** (`include_flat=False`) and optionally drops flat rows from each train block (`drop_flat_train`). Optional `fold_transform(full_train_idx, test_idx) -> (X_tr, X_te, test_gate)` runs **inside each fold** (before flat-drop) to inject per-fold features (e.g. a per-fold HMM regime posterior) and/or a `test_gate` AND-ed into scoring (e.g. trade high-vol bars only); per-fold `n_eligible`/`coverage` are recorded. Persists `y_true/y_pred/fold_id/kept/scored/accuracies/test_starts/test_ends` to `data/processed/walkforward_{name}_predictions.npz` (Rule 7). Returns mean/std/min/max + per-fold detail. |
| `summarize` | `(result: WalkForwardResult) -> str` | Render `"X% ± y% across N folds (range a–b)"` headline plus a per-fold table (skill reporting convention). |
| `sklearn_factory` | `(estimator_cls: type, params: dict) -> Callable[[], Any]` | Wrap an sklearn class + params into a zero-arg factory that builds a fresh instance per call with `random_state=42` injected. |
| `module_factory` | `(module, params: dict, tmp_path: Path) -> Callable[[], Any]` | Factory yielding a fresh fit/predict/predict_proba adapter (`_ModuleEstimator`) that trains via a project model module (`src.models.{logistic,rf,gbm}`) — exact module defaults + `params` overlay + seed 42, fitting to `tmp_path` each fold. Lets walk-forward reproduce a production/tuned model's exact recipe. |
| `project_factories` | `(config: dict) -> dict[str, Callable[[], Any]]` | Ready factories for `logistic/rf/gbm` from `model_params(config, algo)`, seed 42. |

## src.load

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_raw` | `(path: Path \| str) -> pd.DataFrame` | Read raw CSV and parse `"Date and Time"` as `datetime64`. |
| `validate` | `(df: pd.DataFrame) -> None` | Assert all required columns are present; log NaN count per column. Raises `ValueError` if columns are missing. |

## src.split

| Function | Signature | Description |
|----------|-----------|-------------|
| `split` | `(df: pd.DataFrame, train_size: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]` | Time-ordered split: first `train_size` fraction → train, remainder → test. Both returned with reset indices. Accepts raw df or feature matrix; timestamp validation only runs when `"Date and Time"` column is present. Raises `ValueError` if `train_size` not in (0, 1). |

## src.features

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_features` | `(df: pd.DataFrame) -> pd.DataFrame` | Build 20-dim lagged feature matrix from the full raw df (as returned by `load_raw()`). For each row t, collect [Open, Close, High, Low, VWAP] from the 4 preceding clock-minutes (t-4…t-1). Gap minutes are forward-filled on a 1-min grid. First 4 rows dropped; returns shape `(len(df)-4, 20)` with reset index. |

## src.labels

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_labels` | `(df: pd.DataFrame) -> pd.Series` | Compute binary label per row: 1 if `Close > Open`, else 0. No rows dropped — caller passes the aligned training slice (`df.iloc[4:]` of the raw data). Returns `int` Series with reset index. |
| `direction_labels` | `(raw_align: pd.DataFrame) -> pd.Series` | Alias of `build_labels` (the up/down "direction" target) used by the binary/regime suites. |
| `move_series` | `(raw_align: pd.DataFrame) -> pd.Series` | Signed intrabar move `Close − Open` per bar (float Series, reset index). |
| `flat_mask` | `(raw_align: pd.DataFrame) -> np.ndarray` | Boolean mask, True where `Close == Open` (flat bar). Used to drop flat rows from the **training** split only; computed after features are built so it never alters other rows' features. Never apply to the test split for training; applying it as an evaluation/reporting slice (the "no-flat test" stats) is fine. |

## src.models.logistic

The project's primary classifier (promoted from the former `baseline` module).

| Function | Signature | Description |
|----------|-----------|-------------|
| `train` | `(X, y, params=None, save_path=None, sample_weight=None) -> LogisticRegression` | Fit L2 logistic regression; defaults `max_iter=1000`, `random_state=42` (always enforced). Saves to `save_path` else `data/processed/logistic_model.joblib`. |
| `predict` | `(model, X) -> np.ndarray` | Class-label predictions (0/1). |
| `save` | `(model, path="data/processed/logistic_model.joblib") -> None` | Serialize fitted model to disk. |
| `load` | `(path="data/processed/logistic_model.joblib") -> LogisticRegression` | Deserialize a saved model. |

## src.baselines

Naive comparison baselines (not learned models).

| Function | Signature | Description |
|----------|-----------|-------------|
| `predict_always_up` | `(n: int) -> np.ndarray` | Predict up (1) for every one of `n` bars. |
| `predict_last_direction` | `(y_train, y_test) -> np.ndarray` | Lag-1 persistence: predict the previous bar's realised direction; first test row uses the last train label. |

## src.models.rf

| Function | Signature | Description |
|----------|-----------|-------------|
| `train` | `(X: pd.DataFrame, y: pd.Series, params: dict \| None = None, save_path: Path \| None = None, sample_weight: np.ndarray \| None = None) -> RandomForestClassifier` | Fit Random Forest; defaults: 500 trees, `sqrt` features, `min_samples_leaf=5`, `class_weight="balanced"`, `oob_score=True`, `random_state=42` (always enforced). `params` overrides defaults. `sample_weight` optionally weights rows. Saves to `save_path` if given, else `data/processed/rf_model.joblib`. |
| `predict` | `(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray` | Return class-label predictions (0 or 1) from a fitted Random Forest. |
| `save` | `(model: RandomForestClassifier, path: Path = "data/processed/rf_model.joblib") -> None` | Serialize fitted model to disk with joblib; `oob_score_` is preserved. Creates parent dirs. |
| `load` | `(path: Path = "data/processed/rf_model.joblib") -> RandomForestClassifier` | Deserialize and return a RandomForestClassifier saved by `save()`, with `oob_score_` intact. |

## src.config

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_config` | `(path: Path = "config.yaml") -> dict` | Parse `config.yaml` with `yaml.safe_load` and return the full config dict. Raises `FileNotFoundError` if the file is absent. |
| `model_params` | `(config: dict, algo: str) -> dict` | Return a copy of `config["models"][algo]`; empty dict if the algo key is missing. Used by pipeline to pass hyperparameters to each `train()` call. |

## src.pipeline

| Function | Signature | Description |
|----------|-----------|-------------|
| `run` | `(algo: str, data_path: Path = "data/raw/data.csv", force_retrain: bool = False) -> PipelineResult` | Full pipeline for one algo: load data → feature engineering → load-or-train model → evaluate. If `data/processed/<algo>_model.joblib` exists and `force_retrain=False`, loads from disk instead of retraining. Pass `_dataset` kwarg to reuse a pre-loaded split (used internally by `--algo all`). |
| `_build_dataset` | `(data_path: Path) -> tuple[DataFrame, DataFrame, Series, Series]` | Load raw CSV and return `(X_train, X_test, y_train, y_test)` via `load_raw → build_features → split → build_labels`. |
| `_load_or_train` | `(algo: str, X_train, y_train, force_retrain: bool) -> Any` | Load saved joblib if present and `force_retrain=False`; otherwise call the algo's `train()` (which auto-saves). |
| `_get_predictions` | `(algo: str, model: Any, X_test: DataFrame) -> np.ndarray` | Dispatch to the correct `predict()` function for the given algo. |

## src.models.gbm

| Function | Signature | Description |
|----------|-----------|-------------|
| `train` | `(X: pd.DataFrame, y: pd.Series, params: dict \| None = None, save_path: Path \| None = None, sample_weight: np.ndarray \| None = None) -> XGBClassifier` | Fit XGBClassifier with defaults from docstring; `random_state=42` always enforced. `params` overrides defaults. `sample_weight` optionally weights rows. Saves to `save_path` if given, else `data/processed/gbm_model.joblib`. |
| `predict` | `(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray` | Return class-label predictions (0 or 1) from a fitted XGBClassifier. |
| `save` | `(model: XGBClassifier, path: Path = "data/processed/gbm_model.joblib") -> None` | Serialize fitted model to disk with joblib; creates parent dirs. |
| `load` | `(path: Path = "data/processed/gbm_model.joblib") -> XGBClassifier` | Deserialize and return an XGBClassifier saved by `save()`. |

## src.evaluate

| Function | Signature | Description |
|----------|-----------|-------------|
| `_select` | `(y_true, y_pred, keep: np.ndarray \| None, include_flat: bool) -> tuple[np.ndarray, np.ndarray]` | Restrict predictions to the non-flat rows (`keep` mask, True = `Close != Open`) unless `include_flat` or `keep is None`. Shared no-flat gate for the metric fns. |
| `accuracy` | `(y_true, y_pred, keep=None, *, include_flat=False) -> float` | Fraction of correct predictions. **No-flat by default** when a `keep` mask is given; `include_flat=True` (or `keep=None`) scores the full set. |
| `recall` | `(y_true, y_pred, keep=None, *, include_flat=False) -> float` | Recall for class 1 (up); `zero_division=0`. No-flat by default when `keep` is given. |
| `confusion` | `(y_true, y_pred, keep=None, *, include_flat=False) -> np.ndarray` | 2×2 confusion matrix `[[TN, FP], [FN, TP]]` (`labels=[0,1]`). No-flat by default when `keep` is given. |
| `report` | `(name: str, y_true, y_pred, keep=None, *, include_flat=False) -> str` | Format accuracy, recall, and confusion matrix as a markdown section, with a one-line note recording the slice (non-flat N bars vs full set). |
| `write_results` | `(reports: list[str], path: Path, intro: str \| None = None) -> None` | Write markdown report sections to a file (default `docs/results.md`); optional `intro` overrides the default full-test header. Creates parent dirs. |

## src.models.regime_hmm

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_rf` | `(params: dict) -> RandomForestClassifier` | Build an unfitted RF with `random_state=42` and `class_weight="balanced"` enforced. |
| `fit_regime` | `(X_regime: np.ndarray) -> (GaussianHMM, StandardScaler)` | Fit a 2-state Gaussian HMM (+ train-only scaler) on the regime-descriptor features. |
| `assign_regime` | `(hmm, scaler, X_regime) -> np.ndarray` | Decode the hidden-state sequence (Viterbi **smoothing** — uses the whole sequence, so it look-aheads). Legacy `regime_binary` path only; prefer `filter_regime`. |
| `filter_regime_posterior` | `(hmm, scaler, X_regime) -> np.ndarray` | Causal forward-**filtered** posterior `P(state_t | descriptors_0..t)`, shape `(n, n_components)`. Row `t` uses only rows `≤ t` (and the descriptors are lag-1), so no look-ahead. Log-space forward recursion from the HMM's own start/trans/Gaussian-emission params. |
| `filter_regime` | `(hmm, scaler, X_regime) -> np.ndarray` | Causal raw-state assignment = `argmax` of `filter_regime_posterior`; the no-look-ahead replacement for `assign_regime`. |
| `canonical_regime_labels` | `(regime_train, X_regime_train, vol15_col_idx) -> dict[int,int]` | Remap raw HMM states so 0=low-vol, 1=high-vol (by mean vol15). |
| `REGIME_COLS` / `MIN_REGIME_ROWS` | constants | 5 regime-descriptor column names; min bars per regime to train a dedicated model. |

## src.features_v2

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_features_v2` | `(df: pd.DataFrame) -> pd.DataFrame` | Build the 49-feature v2 matrix (v1 OHLCV lags + derived indicators + target-bar time encodings). |
| `load_or_build_features_v2` | `(df: pd.DataFrame) -> pd.DataFrame` | Return the v2 matrix from the `data/processed/features_v2.parquet` cache if present, else build and cache it. |

## src.features_v3

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_features_v3` | `(df: pd.DataFrame) -> pd.DataFrame` | Build the 48-feature stationary v3 matrix: v2's 20 raw base price lags replaced by `log(value / lag1_Close)` (stationary log-ratios vs the strictly-prior close; the now-constant `lag1_Close` is dropped), all derived/time features unchanged. |
| `_transform_v3` | `(v2: pd.DataFrame) -> pd.DataFrame` | Pure row-wise transform turning a 49-col v2 matrix into the 48-col v3 matrix (log-ratio of base lags vs `lag1_Close`); fills any inf/NaN edge cells. |
| `load_or_build_features_v3` | `(df: pd.DataFrame) -> pd.DataFrame` | Return the v3 matrix from `data/processed/features_v3.parquet` if present, else build it from the cached v2 matrix and cache it. |

## src.features_v1_rel

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_features_v1_rel` | `(df: pd.DataFrame) -> pd.DataFrame` | Build the 19-feature stationary v1-relative matrix: v1's 20 raw price lags replaced by `log(value / lag1_Open)` (anchored on the strictly-prior open `Open_{t-1}`; the now-constant `lag1_Open` is dropped). The v1 analog of features_v3 (which anchors on `lag1_Close`). No look-ahead. |

## src.features_orderflow

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_features_orderflow` | `(df: pd.DataFrame, variant: str = "raw") -> pd.DataFrame` | Build the 20-feature order-flow matrix: 5 indicators (`norm_vol` = trailing-60 causal Volume z-score, `signed_vol`, `cum_td5/10/15` = trailing rolling sums of tick_delta) each lagged t-1…t-4. `variant="raw"` → `signed_vol = Volume·sign(Close−Open)` (tree models); `variant="linear"` → `signed_vol = sign(Close−Open)·Volume / trailing-mean Volume` (O(1) signed relative volume, logistic models). Only lagged columns emitted (no lag-0; `signed_vol`'s lag-0 sign equals the label). Dense-grid pattern, drop first 4 rows. |
| `_order_flow_indicators` | `(filled: pd.DataFrame, variant: str = "raw") -> dict[str, pd.Series]` | Compute the five base order-flow indicators on a dense, forward-filled 1-min frame; trailing/causal windows, leakage protection deferred to the caller's lag step. Only `signed_vol` differs between variants. |
| `load_or_build_features_orderflow` | `(df: pd.DataFrame, variant: str = "raw") -> pd.DataFrame` | Return the order-flow matrix from its variant-specific parquet cache (`features_orderflow.parquet` for `"raw"`, `features_orderflow_linear.parquet` for `"linear"`) if present, else build and cache it. |

## src.binary_suite

Binary classification suite (Experiment 4), configurable by feature set (v1/v2) and flat toggle. Trains the 4 original models and writes per-variant `{prefix}_*` artifacts (`exp_noflat`, `exp_noflat_v2`, `exp_v2`) without overwriting the originals.

| Function | Signature | Description |
|----------|-----------|-------------|
| `run` | `(config=None, algos=(...), build_features_fn=None, drop_flat=True, prefix="exp_noflat", display_suffix="no-flat") -> list[dict]` | Train the chosen algos on the 50/50 split, predict on the whole test set, and save each model joblib, predictions `.npz` (`y_true`, `y_pred`, `move`), and feature-importance CSV under `{prefix}_{algo}_*`. Returns run descriptors. |
| `_build_dataset` | `(cfg, build_features_fn=None, drop_flat=True) -> tuple[DataFrame, DataFrame, Series, Series, np.ndarray]` | Build the 50/50 split with `build_features_fn` (default v1); when `drop_flat`, remove `Close==Open` rows from training only (logs count). Returns `(X_train, X_test, y_train, y_test, move_test)`. Test untouched. |
| `_display` | `(algo: str, suffix: str) -> str` | Compose a model display name, e.g. `"Random Forest (no-flat, v2)"`. |
| `_train` | `(algo, X, y, params, save_path) -> Any` | Dispatch to the algo's `train()` with a distinct `save_path`. |
| `_predict` | `(algo, model, X) -> np.ndarray` | Dispatch to the algo's `predict()`. |
| `_feature_importance` | `(algo, model, feature_names) -> pd.DataFrame \| None` | Ranked importance table: `feature_importances_` (rf/gbm) or `abs(coef_)` (logistic); `None` otherwise. |

## src.models.regime_binary

Binary HMM-regime direction model (Experiment 5): a Gaussian HMM detects 2 regimes; a per-regime binary RF predicts up/down. No gate, no HOLD.

| Function | Signature | Description |
|----------|-----------|-------------|
| `run` | `(config: dict \| None = None) -> dict` | Build the flat-removed 50/50 split on `features_v2`, fit the HMM, train a per-regime binary RF (pooled fallback), predict every test bar, and save predictions `.npz` (`y_true`, `y_pred`, `regime`, `move`), HMM/scaler/per-regime joblibs, and per-regime importance CSVs. Returns a summary dict. |
| `_save_importance` | `(model, feature_names, path) -> None` | Write a per-regime RF MDI feature-importance CSV ranked descending. |

## src.run_stats

| Function | Signature | Description |
|----------|-----------|-------------|
| `section_c` | `(cfg: dict, skip_existing: bool = False) -> None` | Additively train the no-flat binary suite (20-feat) + HMM model and write `docs/notes/binary_noflat_stats.md` with per-label confusion-matrix metrics plus a per-regime breakdown for the HMM model. Never touches production artifacts or `all_stats.md`. |
| `section_d` | `(cfg: dict, skip_existing: bool = False) -> None` | Additively train the two 49-feature binary variants (flat-included `exp_v2_*`, no-flat `exp_noflat_v2_*`) and write `docs/notes/binary_v2_stats.md` with per-label confusion-matrix metrics. Uses the cached v2 feature matrix; never touches the 20-feature artifacts or other reports. |
| `_leaderboard_name` | `(stem, registry) -> str` | Display name for a prediction-set stem: registry hit, `exp_noflat_v1rel_{algo}` (v1-rel), `tuned_{feat}_{algo}`, else the stem. |
| `_aum_pct` | `(y_pred, returns) -> float` | % increase in AUM from the compounding long/short backtest (= `total_return`·100). |
| `_test_reference` | `(cfg) -> (int, np.ndarray, np.ndarray)` | (test length, y_true, per-bar returns) for the flat-free 50/50 test split — the single-test leaderboard reference. |
| `rank_models` | `(cfg) -> list[tuple[str,str,float,float,float,float]]` | Rank saved prediction sets best-first `(stem, name, accuracy, recall, mcc, aum_pct)` on the flat-free test set (length-matched only), sorted by accuracy then MCC. |
| `leaderboard` | `(cfg) -> None` | Write `docs/notes/leaderboard.md` (single test set): Model / Accuracy / Recall / MCC / AUM %, sorted by accuracy then MCC. |
| `_featset_builder` | `(featset) -> Callable` | Feature builder for `'v1'/'v1rel'/'v2'/'v3'/'orderflow'`. |
| `_wf_xy` | `(cfg, featset) -> (X, y, timestamps, returns)` | Flat-free walk-forward input for a feature set (features built on the full series, then flat rows dropped). |
| `walkforward_curated` | `(cfg) -> None` | Rolling walk-forward for the curated core (logistic/rf/gbm × v1/v1rel/v2/v3) + an always-up baseline; persists per-fold `wf_{featset}_{algo}` sets (with returns) for the WF leaderboard. |
| `leaderboard_walkforward` | `(cfg, proc=_PROC, out=_WF_LEADERBOARD_PATH) -> None` | Write `docs/notes/leaderboard-walk-forward.md` from `wf_*` sets: mean ± std fold accuracy, folds-won vs the always-up baseline, recall, AUM %; ranked by mean accuracy. |
| `walkforward_results` | `(cfg, proc=_PROC, out=docs/results.md) -> None` | Per-model walk-forward report → `docs/results.md`: mean ± std accuracy, recall, confusion, Δ vs always-up baseline, AUM %. |

## src.tuning

Model-selection harness that optimises **no-flat test accuracy** by selecting on a no-flat validation fold carved from the training half (the test set is touched once, at the end).

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_selection_split` | `(cfg: dict, feat_fn: Callable, val_frac: float = 0.2) -> SelectionSplit` | Carve the inner-train (flat-dropped) + no-flat validation fold from the **training half only** (last `val_frac` of train, time-ordered). No test-half rows are included (no leakage). |
| `grid_search` | `(algo: str, sel: SelectionSplit, grid: list[dict] \| None = None) -> tuple[dict, float, list]` | Score each curated param combo by no-flat validation accuracy (model fit on inner-train); return best params, best val accuracy, and all (params, acc) results. Subsamples inner-train for the RBF SVM. |
| `tune_threshold` | `(scores_val: np.ndarray, y_val: np.ndarray) -> float` | Sweep the decision threshold to maximise no-flat validation accuracy; return the best `>=` threshold. |
| `predict_with_threshold` | `(algo: str, model, X: pd.DataFrame, threshold: float \| None = None) -> np.ndarray` | Predict binary labels; `threshold=None` → plain `_predict`, else `_scores(...) >= threshold` (probability for logistic/rf/gbm). Used by the GUI to apply the stored tuned threshold. |
| `select_features` | `(sel: SelectionSplit, score_algo: str = "logistic", k_grid: list[int] \| None = None) -> tuple[list[str], list]` | Rank features by L1-logistic coefficient magnitude (train-only scaling), sweep top-k, and return the subset maximising no-flat validation accuracy. |
| `run_tuning` | `(cfg: dict, algos=…, featset="v2", use_move_weight=False, tune_thr=False, val_frac=0.2, select=False) -> list[dict]` | Per algo: grid-search → (optional) threshold tune → optional feature selection → retrain best on the full no-flat train half (optional \|move\| weighting) → evaluate once on the no-flat test slice. Persists `tuned_{featset}_{algo}_predictions.npz`, `tuned_params.json`, `selected_{featset}.json`, and `docs/notes/tuning_stats.md`. |

## src.gen_module_docs

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_inventory` | `(src=_SRC) -> dict[str, list[str]]` | AST-walk src → module → its top-level function names. |
| `build_import_graph` | `(src=_SRC) -> dict[str, list[str]]` | Intra-src import edges per module (deduped, no self). |
| `render_module_diagram` | `(inv, graph) -> str` | Render the ASCII import graph + inventory table as markdown. |
| `generate` | `(diagram_path=_DIAGRAM) -> None` | (Re)write `docs/notes/module_diagram.md`. |
| `check_modules_md` | `(modules_path=_MODULES) -> list[str]` | Return public src functions with no MODULES.md entry (private exempt); drives `--check`. |
| `main` | `() -> None` | CLI: default regenerates the diagram; `--check` verifies MODULES.md completeness (exit 2 if incomplete). |
