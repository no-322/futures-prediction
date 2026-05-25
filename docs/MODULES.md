# Module Reference

One entry per function per the project convention: one pseudo-code line = one function = one entry here.

## src.load

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_raw` | `(path: Path \| str) -> pd.DataFrame` | Read raw CSV and parse `"Date and Time"` as `datetime64`. |
| `validate` | `(df: pd.DataFrame) -> None` | Assert all required columns are present; log NaN count per column. Raises `ValueError` if columns are missing. |
