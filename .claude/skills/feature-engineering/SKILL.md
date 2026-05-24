---
name: feature-engineering
description: >
  Apply when editing or reviewing src/features.py, tests/test_features.py,
  or any code that constructs, modifies, or tests feature vectors for the
  futures prediction model. Also apply when a prompt mentions lags, OHLCV,
  VWAP, rolling windows, or feature matrix shape.
---

# Feature Engineering Skill

## When to apply

Activate this skill when:
- Editing or creating `src/features.py` or `tests/test_features.py`
- Any prompt mentions lags, OHLCV, VWAP, rolling windows, or feature matrix shape
- Reviewing code that builds or transforms input features for the model
- Debugging unexpected model inputs or shape mismatches

## 20-dim construction rule

For each target row `t`, build a 20-dim feature vector using only the four minutes strictly before `t`.

| Position | Lag | Features (in order) |
|----------|-----|---------------------|
| 0–4      | t-4 | Open, Close, High, Low, VWAP |
| 5–9      | t-3 | Open, Close, High, Low, VWAP |
| 10–14    | t-2 | Open, Close, High, Low, VWAP |
| 15–19    | t-1 | Open, Close, High, Low, VWAP |

- **Lags**: t-4, t-3, t-2, t-1 — exactly four minutes, strictly before t
- **Features per lag**: Open, Close, High, Low, VWAP — exactly five values
- **Column order**: all five features for the oldest lag first, then moving forward to t-1
- **Final shape**: `(n_rows, 20)` — assert this before returning

## No-leakage rule

Minute `t` must **never** appear in the feature vector for row `t`. The allowed range is `[t-4, t-1]` inclusive.

### pandas pitfalls

**`rolling()` without shift includes the current row:**
```python
# WRONG — df.rolling(4) includes row t in the window for row t
df['feature'] = df['close'].rolling(4).mean()

# CORRECT — shift(1) pushes the window one step back before computing
df['feature'] = df['close'].shift(1).rolling(4).mean()
```

**`shift()` direction:**
- `df.shift(1)` moves values *down* (lags by 1 step) — correct direction
- `df.shift(-1)` moves values *up* (looks ahead by 1 step) — never use this for features

**Constructing lags explicitly:**
```python
# Safe explicit approach
for lag in [4, 3, 2, 1]:
    for col in ['Open', 'Close', 'High', 'Low', 'VWAP']:
        features[f'{col}_lag{lag}'] = df[col].shift(lag)
```

### "t-1 ≈ t" is not leakage

If minute t-1 values are nearly identical to minute t values (low-volatility periods, illiquid instruments), that is fine. **Temporal ordering defines leakage, not value distance.** The rule is about which timestamp the data originates from, not how much prices moved between bars.

### NaN rows from lagging

Shifting by 4 makes the first 4 rows all-NaN. These must be dropped — but **log the count first**, never drop silently:

```python
n_before = len(X)
X = X.dropna()
print(f"Dropped {n_before - len(X)} NaN rows from lagging")
```

After dropping, reset the index so label alignment works correctly:
```python
X = X.reset_index(drop=True)
y = y.loc[X.index].reset_index(drop=True)  # or align by timestamp
```

### Verification check

After building the raw (pre-drop) feature matrix, confirm no leakage by inspection:
```python
assert X.iloc[:4].isna().all().all(), "First 4 rows should be all-NaN before drop"
assert X.shape[1] == 20, f"Expected 20 features, got {X.shape[1]}"
```

## Verification checklist

Before declaring feature work done, all of these must pass:

- [ ] `pytest tests/test_features.py` — all tests green
- [ ] `X.shape[1] == 20` — exact column count asserted in code
- [ ] First 4 rows are NaN before drop; drop count logged to stdout
- [ ] `docs/MODULES.md` updated: every new or changed function gets an entry (signature, Args, Returns, one-line description)
- [ ] No row in the training split contains values sourced from that row's own timestamp

## Common mistakes

| Mistake | Why it's wrong | Fix |
|---------|---------------|-----|
| `df.rolling(4).mean()` without `.shift(1)` | Includes row t in its own window | Add `.shift(1)` before `.rolling()` |
| `df.shift(-1)` for a lag | Looks one step into the future | Use `.shift(1)` for 1-step lag, `.shift(n)` for n-step |
| Fitting scaler on full dataset | Leaks test-set statistics into training | Fit scaler on train split only, transform test separately |
| Sorting by price or volume instead of timestamp | Breaks time ordering | Always sort by the timestamp column before splitting |
| Silent `dropna()` | Hides how many rows were lost | Log `n_before - n_after` before every drop |
| Not resetting index after drop | Misaligned joins when merging features with labels | `reset_index(drop=True)` on both X and y after filtering |
| Including `close(t)` in features for row `t` | Direct leakage of current bar | Lags must start at t-1; `close(t)` is part of the label, not features |
