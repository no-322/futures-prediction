# Module Reference

One entry per function per the project convention: one pseudo-code line = one function = one entry here.

## src.load

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_raw` | `(path: Path \| str) -> pd.DataFrame` | Read raw CSV and parse `"Date and Time"` as `datetime64`. |
| `validate` | `(df: pd.DataFrame) -> None` | Assert all required columns are present; log NaN count per column. Raises `ValueError` if columns are missing. |

## src.split

| Function | Signature | Description |
|----------|-----------|-------------|
| `split` | `(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]` | Time-ordered 50/50 split: first half → train, second half → test. Both returned with reset indices. Accepts raw df or feature matrix; timestamp validation only runs when `"Date and Time"` column is present. |

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
| `train` | `(X: pd.DataFrame, y: pd.Series) -> LogisticRegression` | Fit logistic regression on training features and labels; `random_state=42`, `max_iter=1000`. |
| `predict` | `(model: LogisticRegression, X: pd.DataFrame) -> np.ndarray` | Return class-label predictions (0 or 1) from a fitted logistic regression. |
| `predict_always_up` | `(n: int) -> np.ndarray` | Baseline: return an array of `n` ones (always predict up). |
| `predict_last_direction` | `(y_train: pd.Series, y_test: pd.Series) -> np.ndarray` | Baseline: for each test row, predict the direction of the previous bar; first row uses last training label. |

## src.models.rf

| Function | Signature | Description |
|----------|-----------|-------------|
| `train` | `(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier` | Fit Random Forest (500 trees, `sqrt` features, `min_samples_leaf=5`, `class_weight="balanced"`, `oob_score=True`, `random_state=42`). OOB accuracy available as `model.oob_score_`. |
| `predict` | `(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray` | Return class-label predictions (0 or 1) from a fitted Random Forest. |

## src.evaluate

| Function | Signature | Description |
|----------|-----------|-------------|
| `accuracy` | `(y_true: np.ndarray, y_pred: np.ndarray) -> float` | Fraction of correct predictions on the test set. |
| `recall` | `(y_true: np.ndarray, y_pred: np.ndarray) -> float` | Recall for class 1 (up direction); `zero_division=0`. |
| `confusion` | `(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray` | 2×2 confusion matrix `[[TN, FP], [FN, TP]]`. |
| `report` | `(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> str` | Format accuracy, recall, and confusion matrix for one model as a markdown section. |
| `write_results` | `(reports: list[str], path: Path) -> None` | Write list of markdown report sections to `docs/results.md` (or any path). Creates parent dirs. |
