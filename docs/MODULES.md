# Module Reference

One entry per function per the project convention: one pseudo-code line = one function = one entry here.

## src.statistics

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute` | `(y_true, y_pred, name="", labels=None) -> StatsResult` | Compute full evaluation statistics. Works for binary and multi-class. Returns a `StatsResult` TypedDict with accuracy, macro/weighted F1, MCC, per-class precision/recall/F1/support, and confusion matrix. |
| `format_markdown` | `(result: StatsResult) -> str` | Format a `StatsResult` as a markdown section with scalar metrics table, per-class table, and confusion matrix. |
| `write_results` | `(results: list[StatsResult], path: Path) -> None` | Write a list of `StatsResult` reports to a markdown file; creates parent dirs. |
| `to_dict` | `(result: StatsResult) -> dict` | Return a JSON-serialisable plain dict from a `StatsResult`. |

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

## src.models.baseline

| Function | Signature | Description |
|----------|-----------|-------------|
| `train` | `(X: pd.DataFrame, y: pd.Series, params: dict \| None = None) -> LogisticRegression` | Fit logistic regression; defaults `max_iter=1000`, `random_state=42` (always enforced). `params` dict from `config.model_params()` overrides defaults. Auto-saves to `data/processed/baseline_model.joblib`. |
| `predict` | `(model: LogisticRegression, X: pd.DataFrame) -> np.ndarray` | Return class-label predictions (0 or 1) from a fitted logistic regression. |
| `predict_always_up` | `(n: int) -> np.ndarray` | Baseline: return an array of `n` ones (always predict up). |
| `predict_last_direction` | `(y_train: pd.Series, y_test: pd.Series) -> np.ndarray` | Baseline: for each test row, predict the direction of the previous bar; first row uses last training label. |
| `save` | `(model: LogisticRegression, path: Path = "data/processed/baseline_model.joblib") -> None` | Serialize fitted model to disk with joblib; creates parent dirs. |
| `load` | `(path: Path = "data/processed/baseline_model.joblib") -> LogisticRegression` | Deserialize and return a LogisticRegression saved by `save()`. |

## src.models.rf

| Function | Signature | Description |
|----------|-----------|-------------|
| `train` | `(X: pd.DataFrame, y: pd.Series, params: dict \| None = None) -> RandomForestClassifier` | Fit Random Forest; defaults: 500 trees, `sqrt` features, `min_samples_leaf=5`, `class_weight="balanced"`, `oob_score=True`, `random_state=42` (always enforced). `params` overrides defaults. Auto-saves to `data/processed/rf_model.joblib`. |
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
| `train` | `(X: pd.DataFrame, y: pd.Series, params: dict \| None = None) -> XGBClassifier` | Fit XGBClassifier with defaults from docstring; `random_state=42` always enforced. `params` overrides defaults. Auto-saves to `data/processed/gbm_model.joblib`. |
| `predict` | `(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray` | Return class-label predictions (0 or 1) from a fitted XGBClassifier. |
| `save` | `(model: XGBClassifier, path: Path = "data/processed/gbm_model.joblib") -> None` | Serialize fitted model to disk with joblib; creates parent dirs. |
| `load` | `(path: Path = "data/processed/gbm_model.joblib") -> XGBClassifier` | Deserialize and return an XGBClassifier saved by `save()`. |

## src.models.svm

| Function | Signature | Description |
|----------|-----------|-------------|
| `train` | `(X: pd.DataFrame, y: pd.Series, params: dict \| None = None) -> SVMModel` | Fit `StandardScaler` on `X_train` only, then fit `SVC` with defaults from docstring; `random_state=42` always enforced. `params` overrides defaults. Auto-saves to `data/processed/svm_model.joblib`. |
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
