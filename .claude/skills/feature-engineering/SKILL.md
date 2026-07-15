---
name: feature-engineering
description: >
  Apply when editing or reviewing any feature pipeline (src/features.py,
  src/features_v1_rel.py, src/features_v2.py, src/features_v3.py,
  src/features_orderflow.py) or its test, or any code that constructs, modifies,
  or tests feature columns for the futures prediction model. Also apply when a
  prompt mentions lags, OHLCV, VWAP, rolling windows, order-flow, stationarity,
  or feature matrix shape.
---

# Feature Engineering Skill

## When to apply

Activate this skill when:
- Editing or creating any `src/features*.py` pipeline or its `tests/test_features*.py`
- Any prompt mentions lags, OHLCV, VWAP, rolling windows, order-flow, stationarity, or
  feature matrix shape
- Reviewing code that builds or transforms input features for the model
- Debugging unexpected model inputs or shape mismatches

## Feature pipelines, not one fixed vector

The project runs **several parallel feature pipelines** — do not assume a single 20-dim
vector. **`docs/FEATURES.md` is the source of truth**: it registers every pipeline
(v1, v1-rel, v2, v3, order-flow, HMM-regime) with its column list and per-feature
pseudocode. New feature work **adds a pipeline or a variant** under the contract below and
registers it in `FEATURES.md` (and `docs/MODULES.md`, with a pseudocode block).

## The feature-pipeline contract

Every pipeline must satisfy all of these:

1. **Dense 1-minute grid.** Build on a continuous minute grid (reindex + forward-fill
   gaps) so `shift(k)` is a true clock-minute offset, not a row offset. Gaps (overnight,
   weekends, contract rolls) must not be treated as adjacent minutes.
2. **Causal only (k ≥ 1).** Every emitted column is a `shift(k)` with `k ≥ 1`. **No lag-0 /
   current bar, ever.** A feature for minute `t` depends only on data ≤ `t-1`. This is the
   whole ball game: e.g. `signed_vol`'s lag-0 sign equals the label, so only its lagged
   copies may be emitted.
3. **Warm-up dropped.** Shifting/rolling makes the leading rows NaN — drop them (**log the
   count**), fill any residual warm-up NaN, and `reset_index(drop=True)` so X stays
   row-aligned with labels and timestamps.
4. **Registered + documented.** Add the pipeline to `docs/FEATURES.md` and give every new
   function a `docs/MODULES.md` entry with signature + description + **pseudocode**.

Assert the pipeline's own column count before returning (the number is pipeline-specific —
20 for v1, 19 for v1-rel, 49 for v2, 48 for v3, 20 for order-flow), not a universal 20.

## No-leakage rule

Minute `t` must **never** appear in the feature columns for row `t`. Allowed range is
`[t-k, t-1]` for the pipeline's lags.

### pandas pitfalls

**`rolling()` without shift includes the current row:**
```python
# WRONG — df.rolling(4) includes row t in the window for row t
df['feature'] = df['close'].rolling(4).mean()

# CORRECT — compute the indicator, then lag by >=1 before it becomes a feature column
ind = df['close'].rolling(60).mean()      # trailing/causal is fine on the grid
feat = ind.shift(1)                        # the emitted feature is lagged
```
(A trailing rolling window that includes the current bar is acceptable **only** because
the emitted feature is then `shift(k≥1)` — leakage protection comes from the lag, not the
window.)

**`shift()` direction:**
- `df.shift(1)` moves values *down* (lags by 1 step) — correct direction
- `df.shift(-1)` moves values *up* (looks ahead) — never use this for features

**Constructing lags explicitly:**
```python
for lag in [4, 3, 2, 1]:                    # all k >= 1
    for name in indicators:
        features[f'lag{lag}_{name}'] = indicators[name].shift(lag)
```

### "t-1 ≈ t" is not leakage

If minute t-1 values are nearly identical to minute t values (low-volatility periods),
that is fine. **Temporal ordering defines leakage, not value distance** — the rule is about
which timestamp the data originates from, not how much prices moved.

### NaN rows from lagging / warm-up

Shifting by k makes the first k rows NaN; rolling windows extend the warm-up further.
Drop/fill them — but **log the count first**, never silently:

```python
n_before = len(X)
# drop the fixed warm-up rows, then fill any residual rolling-window NaN
X = X.iloc[k:].reset_index(drop=True)
print(f"Dropped {n_before - len(X)} warm-up rows")
```

## Verification checklist

Before declaring feature work done:

- [ ] `pytest tests/test_features*.py` for the pipeline you touched — green
- [ ] The pipeline's own column count is asserted in code (pipeline-specific)
- [ ] Warm-up rows dropped; drop/fill count logged to stdout
- [ ] A **perturbation / no-look-ahead** check: perturbing bar `t`'s own OHLC/volume/tick
      data does not change the feature row whose target is `t` (features depend only on ≤ t-1)
- [ ] `docs/FEATURES.md` + `docs/MODULES.md` updated (signature, description, **pseudocode**)
- [ ] No row contains values sourced from that row's own timestamp

## Common mistakes

| Mistake | Why it's wrong | Fix |
|---------|---------------|-----|
| Emitting a lag-0 / current-bar column | Direct leakage of bar `t` (e.g. `signed_vol` sign = label) | Emit only `shift(k≥1)` copies |
| `df.rolling(w).mean()` used directly as a feature | Includes row t in its own window | Lag the indicator by `k≥1` before emitting |
| `df.shift(-1)` for a lag | Looks one step into the future | Use `.shift(k)` with `k≥1` |
| Fitting scaler on full dataset | Leaks test-set statistics into training | Fit scaler on the train split/fold only |
| Row offsets instead of timestamp/grid | Gaps break the "consecutive minutes" assumption | Build on the dense 1-min grid |
| Silent `dropna()` | Hides how many rows were lost | Log `n_before - n_after` before every drop |
| Not resetting index after drop | Misaligned joins with labels/timestamps | `reset_index(drop=True)` after filtering |
