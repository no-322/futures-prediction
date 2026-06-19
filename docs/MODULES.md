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

## src.models.baseline

| Function | Signature | Description |
|----------|-----------|-------------|
| `train` | `(X: pd.DataFrame, y: pd.Series, params: dict \| None = None, save_path: Path \| None = None, sample_weight: np.ndarray \| None = None) -> LogisticRegression` | Fit logistic regression; defaults `max_iter=1000`, `random_state=42` (always enforced). `params` dict from `config.model_params()` overrides defaults. `sample_weight` optionally weights rows (e.g. \|Close−Open\|). Saves to `save_path` if given, else `data/processed/baseline_model.joblib`. |
| `predict` | `(model: LogisticRegression, X: pd.DataFrame) -> np.ndarray` | Return class-label predictions (0 or 1) from a fitted logistic regression. |
| `predict_always_up` | `(n: int) -> np.ndarray` | Baseline: return an array of `n` ones (always predict up). |
| `predict_last_direction` | `(y_train: pd.Series, y_test: pd.Series) -> np.ndarray` | Baseline: for each test row, predict the direction of the previous bar; first row uses last training label. |
| `save` | `(model: LogisticRegression, path: Path = "data/processed/baseline_model.joblib") -> None` | Serialize fitted model to disk with joblib; creates parent dirs. |
| `load` | `(path: Path = "data/processed/baseline_model.joblib") -> LogisticRegression` | Deserialize and return a LogisticRegression saved by `save()`. |

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

## src.models.svm

| Function | Signature | Description |
|----------|-----------|-------------|
| `train` | `(X: pd.DataFrame, y: pd.Series, params: dict \| None = None, save_path: Path \| None = None, sample_weight: np.ndarray \| None = None) -> SVMModel` | Fit `StandardScaler` on `X_train` only, then fit `SVC` with defaults from docstring; `random_state=42` always enforced. `params` overrides defaults. `sample_weight` optionally weights rows. Saves to `save_path` if given, else `data/processed/svm_model.joblib`. |
| `predict` | `(model: SVMModel, X: pd.DataFrame) -> np.ndarray` | Apply the training-fit scaler to `X`, then return class-label predictions (0 or 1) from the fitted SVC. |
| `save` | `(model: SVMModel, path: Path = "data/processed/svm_model.joblib") -> None` | Serialize the `SVMModel` (scaler + clf) to disk with joblib; creates parent dirs. |
| `load` | `(path: Path = "data/processed/svm_model.joblib") -> SVMModel` | Deserialize and return an `SVMModel` saved by `save()`. |

## src.evaluate

| Function | Signature | Description |
|----------|-----------|-------------|
| `accuracy` | `(y_true: np.ndarray, y_pred: np.ndarray) -> float` | Fraction of correct predictions on the test set. |
| `recall` | `(y_true: np.ndarray, y_pred: np.ndarray) -> float` | Recall for class 1 (up direction); `zero_division=0`. |
| `confusion` | `(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray` | 2×2 confusion matrix `[[TN, FP], [FN, TP]]`. |
| `report` | `(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> str` | Format accuracy, recall, and confusion matrix for one model as a markdown section. |
| `write_results` | `(reports: list[str], path: Path) -> None` | Write list of markdown report sections to `docs/results.md` (or any path). Creates parent dirs. |

## src.models.regime_hmm

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_rf` | `(params: dict) -> RandomForestClassifier` | Build an unfitted RF with `random_state=42` and `class_weight="balanced"` enforced. |
| `fit_regime` | `(X_regime: np.ndarray) -> (GaussianHMM, StandardScaler)` | Fit a 2-state Gaussian HMM (+ train-only scaler) on the regime-descriptor features. |
| `assign_regime` | `(hmm, scaler, X_regime) -> np.ndarray` | Decode the hidden-state sequence (Viterbi) for new descriptor rows. |
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

## src.binary_suite

Binary classification suite (Experiment 4), configurable by feature set (v1/v2) and flat toggle. Trains the 4 original models and writes per-variant `{prefix}_*` artifacts (`exp_noflat`, `exp_noflat_v2`, `exp_v2`) without overwriting the originals.

| Function | Signature | Description |
|----------|-----------|-------------|
| `run` | `(config=None, algos=(...), build_features_fn=None, drop_flat=True, prefix="exp_noflat", display_suffix="no-flat") -> list[dict]` | Train the chosen algos on the 50/50 split, predict on the whole test set, and save each model joblib, predictions `.npz` (`y_true`, `y_pred`, `move`), and feature-importance CSV under `{prefix}_{algo}_*`. Returns run descriptors. |
| `_build_dataset` | `(cfg, build_features_fn=None, drop_flat=True) -> tuple[DataFrame, DataFrame, Series, Series, np.ndarray]` | Build the 50/50 split with `build_features_fn` (default v1); when `drop_flat`, remove `Close==Open` rows from training only (logs count). Returns `(X_train, X_test, y_train, y_test, move_test)`. Test untouched. |
| `_display` | `(algo: str, suffix: str) -> str` | Compose a model display name, e.g. `"Random Forest (no-flat, v2)"`. |
| `_train` | `(algo, X, y, params, save_path) -> Any` | Dispatch to the algo's `train()` with a distinct `save_path`. |
| `_predict` | `(algo, model, X) -> np.ndarray` | Dispatch to the algo's `predict()`. |
| `_feature_importance` | `(algo, model, feature_names) -> pd.DataFrame \| None` | Ranked importance table: `feature_importances_` (rf/gbm) or `abs(coef_)` (baseline); `None` for RBF SVM. |

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
| `test_flat_mask` | `(cfg: dict) -> np.ndarray` | Reconstruct the 50/50 test slice (`df.iloc[4:]` → split) and return the boolean "keep" mask (`True` where `Close != Open`), aligned 1-to-1 with every binary model's saved test predictions. |
| `section_noflat_test` | `(cfg: dict) -> None` | No-flat-test evaluation slice: read every existing `{stem}_predictions.npz` (no retraining), drop flat (`Open == Close`) test rows via `test_flat_mask`, recompute stats, and write three sibling reports (`all_stats_noflat_test.md`, `binary_noflat_stats_noflat_test.md`, `binary_v2_stats_noflat_test.md`). Length-mismatched sets (3-class / two-stage) are skipped. |

## src.tuning

Model-selection harness that optimises **no-flat test accuracy** by selecting on a no-flat validation fold carved from the training half (the test set is touched once, at the end).

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_selection_split` | `(cfg: dict, feat_fn: Callable, val_frac: float = 0.2) -> SelectionSplit` | Carve the inner-train (flat-dropped) + no-flat validation fold from the **training half only** (last `val_frac` of train, time-ordered). No test-half rows are included (no leakage). |
| `grid_search` | `(algo: str, sel: SelectionSplit, grid: list[dict] \| None = None) -> tuple[dict, float, list]` | Score each curated param combo by no-flat validation accuracy (model fit on inner-train); return best params, best val accuracy, and all (params, acc) results. Subsamples inner-train for the RBF SVM. |
| `tune_threshold` | `(scores_val: np.ndarray, y_val: np.ndarray) -> float` | Sweep the decision threshold to maximise no-flat validation accuracy; return the best `>=` threshold. |
| `predict_with_threshold` | `(algo: str, model, X: pd.DataFrame, threshold: float \| None = None) -> np.ndarray` | Predict binary labels; `threshold=None` → plain `_predict`, else `_scores(...) >= threshold` (probability for baseline/rf/gbm, SVM margin for svm). Used by the GUI to apply the stored tuned threshold. |
| `select_features` | `(sel: SelectionSplit, score_algo: str = "baseline", k_grid: list[int] \| None = None) -> tuple[list[str], list]` | Rank features by L1-logistic coefficient magnitude (train-only scaling), sweep top-k, and return the subset maximising no-flat validation accuracy. |
| `run_tuning` | `(cfg: dict, algos=…, featset="v2", use_move_weight=False, tune_thr=False, val_frac=0.2, select=False) -> list[dict]` | Per algo: grid-search → (optional) threshold tune → optional feature selection → retrain best on the full no-flat train half (optional \|move\| weighting) → evaluate once on the no-flat test slice. Persists `tuned_{featset}_{algo}_predictions.npz`, `tuned_params.json`, `selected_{featset}.json`, and `docs/notes/tuning_stats.md`. |
