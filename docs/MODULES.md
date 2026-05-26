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
| `split` | `(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]` | Time-ordered 50/50 split: first half → train, second half → test. Both returned with reset indices. Raises `ValueError` if timestamps are not monotonically increasing. |

## src.features

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_features` | `(df: pd.DataFrame) -> pd.DataFrame` | Build 20-dim lagged feature matrix. For each row t, collect [Open, Close, High, Low, VWAP] from the 4 preceding clock-minutes (t-4…t-1). Gap minutes are forward-filled on a 1-min grid. First 4 rows dropped; returns shape `(len(df)-4, 20)` with reset index. |

## src.labels

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_labels` | `(df: pd.DataFrame) -> pd.Series` | Compute binary label per row: 1 if `Close > Open`, else 0. First 4 rows dropped to align with `build_features` output. Returns `int` Series with reset index. |
