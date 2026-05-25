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
