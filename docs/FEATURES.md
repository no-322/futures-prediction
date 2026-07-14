# Feature Reference — v1 / v2 / v3 / order-flow / HMM

Quick lookup for which feature set contains what, and the pseudocode logic for each
engineered feature. Source of truth: `src/features.py`, `src/features_v2.py`,
`src/features_v3.py`, `src/features_orderflow.py`, `src/models/regime_hmm.py`.

## Quick-reference table

| Set | Module | # cols | Composition | Used by (leaderboard) |
|-----|--------|--------|-------------|-----------------------|
| **v1** | `features.py` | 20 | 5 OHLCV price columns × 4 lags | Logistic Regression (base, no-flat, tuned v1) |
| **v2** | `features_v2.py` | 49 | 20 v1 base lags + 26 derived-indicator lags (13 × lags {1,4}) + 3 target-bar time cols | GBM tuned v2; also the source of the HMM regime inputs |
| **v3** | `features_v3.py` | 48 | v2, but the 20 raw price lags → stationary log-ratios vs `lag1_Close` (drop constant `lag1_Close`) | GBM tuned v3, RF tuned v3 (top of leaderboard) |
| **v1-rel** | `features_v1_rel.py` | 19 | v1, but the 20 raw price lags → log-ratios vs `lag1_Open` (drop constant `lag1_Open`) | Logistic Regression (no-flat, v1-rel) — degenerate, see below |
| **order-flow** | `features_orderflow.py` | 20 | 5 volume/tick indicators × 4 lags (`raw` and scale-stable `linear` variants) | Concatenated onto tree models (order-flow experiments) |
| **HMM regime** | `regime_hmm.py` | 5 inputs → 1 signal | 2-state Gaussian HMM over 5 v2 columns → causal high-vol posterior | Added as a feature (`regime_hi_prob`) or a trading gate |

## Shared construction rules (all sets)

- **Dense 1-minute grid:** reindex to a continuous minute grid
  (`date_range(freq="1min")`) and **forward-fill** gaps (overnight/weekend/roll), so
  `shift(k)` means *k clock-minutes back*, not *k rows back*.
- **No look-ahead:** every feature for target bar `t` is a `shift(k)` with **k ≥ 1**
  (uses only bars ≤ t-1). No feature ever reads bar `t` itself.
- **Warm-up drop:** the first 4 rows (max base lag) are dropped; index reset 0-based.
  Later warm-up NaNs (rolling windows) are forward/zero-filled and logged.
- Column naming: `lag{k}_{name}` for the value of `{name}` at bar `t-k`.

```
# common skeleton (pseudocode)
grid   = date_range(first_ts, last_ts, freq="1min")
filled = raw.set_index(time).reindex(grid).ffill()   # gap-safe
indicators = compute_on(filled)                      # per-set, see below
for k in [4,3,2,1]:                                  # lags, k>=1 → causal
    for name in indicators:
        col[f"lag{k}_{name}"] = indicators[name].shift(k)
features = project_back_to_original_timestamps(col)
features = features.iloc[4:].reset_index(drop=True)  # drop warm-up
```

---

## v1 — raw OHLCV lags (20)

Columns = `{Open, Close, High, Low, VWAP}` × lags `{4,3,2,1}`, order
`[lag4_Open … lag4_VWAP, lag3_…, … lag1_VWAP]`.

```
for k in [4,3,2,1]:
    for col in [Open, Close, High, Low, VWAP]:
        lag{k}_{col} = filled[col].shift(k)          # price at bar t-k
```

Note: these are **raw price levels** → non-stationary across contracts (the reason v3
exists).

---

## v2 — v1 + 13 derived indicators (×lags {1,4}) + 3 time cols (49)

All 13 indicators computed on the dense grid, then lagged at **1 and 4** (26 cols).
`close`, `open`, `high`, `low`, `vwap`, `Up/Down Ticks`, `Tick Count` are grid columns.

```
vwap_dev   = (Close - VWAP) / VWAP                       # % distance from VWAP
bar_range  = High - Low                                  # bar height
body_ratio = |Close - Open| / (High - Low + 1e-8)        # candle body vs range, in [0,1]
tick_delta = (UpTicks - DownTicks) / TickCount           # signed tick imbalance, in [-1,1]
ret        = Close / Close.shift(1) - 1                  # 1-bar simple return
log_ret    = log(Close / Close.shift(1))                 # 1-bar log return
rsi5       = RSI(Close, 5)                                # Wilder RSI, 5-bar
rsi15      = RSI(Close, 15)                               # Wilder RSI, 15-bar
vol5       = log_ret.rolling(5).std()                    # realized vol, 5-bar
vol15      = log_ret.rolling(15).std()                   # realized vol, 15-bar
macd_line, macd_signal, macd_hist = MACD(Close, 6, 13, 4)

# lag each of the 13 above at k in {1,4}  → 26 columns
```

Indicator helpers:

```
def RSI(close, w):                    # Wilder's smoothing (ewm com=w-1)
    delta = close.diff()
    gain  = ewm(max(delta, 0),  com=w-1)     # avg gain
    loss  = ewm(max(-delta, 0), com=w-1)     # avg loss
    rs    = gain / loss
    return 100 - 100 / (1 + rs)

def MACD(close, fast=6, slow=13, signal=4):  # minute-adapted (≈ 12/26/9 scaled)
    line   = EMA(close, fast) - EMA(close, slow)
    sig    = EMA(line, signal)
    return line, sig, line - sig             # macd_line, macd_signal, macd_hist
```

Target-bar **time features** (no lag — clock time of bar `t` is known before it opens,
so not look-ahead):

```
minute_of_day = hour*60 + minute
tod_sin = sin(2π · minute_of_day / 1440)                 # cyclic time-of-day
tod_cos = cos(2π · minute_of_day / 1440)
session_min = minutes since session start                # new session when gap ≥ 12h
```

Total = 20 (base) + 26 (13 indicators × 2 lags) + 3 (time) = **49**.

---

## v3 — v2 made stationary (48)

Only the **20 raw price lags** change: each becomes a **log-ratio vs the most recent
close** `lag1_Close` (= Close_{t-1}). The 26 indicators + 3 time cols are already
stationary and unchanged. `lag1_Close` → `log(1)=0` (constant) → dropped.

```
ref = v2["lag1_Close"]                                   # Close_{t-1}
for col in the 20 base price lags:                       # lag{k}_{Open/Close/High/Low/VWAP}
    v3[col] = log( v2[col] / ref )                       # stationary, contract-scale invariant
drop "lag1_Close"                                        # now identically 0
# guard inf/NaN at edges → ffill → fillna(0)
```

Effect: values hover near 0, preserve momentum (e.g. `log(Close_{t-4}/Close_{t-1})`) and
intrabar shape, and transfer across the 13 contracts. Count = 49 − 1 = **48**.

---

## v1-rel — v1 anchored to the last open (19)

Same idea as v3, applied to v1: each of the 20 raw price lags → log-ratio vs the most
recent open `lag1_Open` (= Open_{t-1}); the now-constant `lag1_Open` is dropped ⇒ 19.

```
ref = v1["lag1_Open"]                                    # Open_{t-1}
for col in the 20 base price lags:
    v1rel[col] = log( v1[col] / ref )                    # stationary, level-free
drop "lag1_Open"                                         # now identically 0
```

Caveat (empirical): the resulting features are tiny (std ≈ 2e-4), and an **unscaled**
logistic regression (C=1.0 L2) collapses to the intercept and predicts "up" for every
bar → no-flat test accuracy **0.5058** (≈ the majority-class rate, MCC 0.0000), *below*
raw v1's 0.5472. Stationary log-ratios help scale-invariant **tree** models (v3) but
need standardization to help a **linear** model.

## order-flow — volume & tick-flow lags (20)

5 indicators on the dense grid, lagged at `{4,3,2,1}` — **only lagged columns emitted**
(never lag-0: `signed_vol`'s lag-0 sign equals the label). `_VOL_WINDOW = 60`.

```
tick_delta = (UpTicks - DownTicks) / TickCount           # same as v2

norm_vol   = (Volume - Volume.rolling(60).mean())        # trailing 60-bar z-score
             / Volume.rolling(60).std()                  #   of volume (causal)

signed_vol = Volume * sign(Close - Open)                 # variant="raw"  (tree models)
           # sign(Close-Open) * Volume / Volume.rolling(60).mean()   # variant="linear"
           #   → signed *relative* volume, O(1) scale (logistic models)

cum_td5    = tick_delta.rolling(5).sum()                 # cumulative tick imbalance
cum_td10   = tick_delta.rolling(10).sum()                #   over trailing 5 / 10 / 15 bars
cum_td15   = tick_delta.rolling(15).sum()

# lag each of the 5 at k in [4,3,2,1] → 20 columns; no lag-0
```

Two variants differ **only** in `signed_vol`: `raw` (full volume magnitude, helps trees)
vs `linear` (divided by trailing-mean volume → O(1) scale, so it doesn't dominate an
unscaled logistic fit).

---

## HMM regime — causal high-vol detector (5 inputs → 1 signal)

Not a standalone matrix: a 2-state Gaussian HMM fit on 5 **v2** columns, used to label
each bar high-vol / low-vol without look-ahead.

```
REGIME_COLS = [lag1_vol15, lag1_macd_hist, lag1_rsi15, lag1_tick_delta, lag1_return]
```

Logic (per walk-forward fold / train split — fit on TRAIN only):

```
scaler, hmm = fit( StandardScaler + GaussianHMM(n_states=2), X_regime[train] )

# canonicalize: state with higher mean lag1_vol15 on train = "high-vol" (state 1)
hi_state = argmax_s mean(lag1_vol15 | train state == s)

# CAUSAL assignment = forward FILTERING (not Viterbi smoothing):
#   posterior_t = P(state_t | descriptors_0..t), depends only on bars <= t
#   (descriptors are lag1 → effectively data <= t-1)
posterior = hmm_forward_filter(scaler.transform(X_regime))     # log-space alpha recursion
regime_hi_prob = posterior[:, hi_state]                        # P(high-vol) in [0,1]
```

Two uses:
- **Feature:** append `regime_hi_prob` as one extra column to the base feature set.
- **Gate:** trade/score only bars where `regime_hi_prob >= 0.5` (high-vol), report
  coverage.

Key point: uses **filtering (forward algorithm)**, not `hmm.predict` (Viterbi
smoothing), because Viterbi decodes each bar using the *whole* sequence → look-ahead.

---

## Cheat-sheet: where each engineered signal lives

| Feature/logic | v1 | v2 | v3 | order-flow | HMM input |
|---|:--:|:--:|:--:|:--:|:--:|
| Raw OHLCV price lags | ✅ | ✅ | log-ratio | | |
| vwap_dev, bar_range, body_ratio | | ✅ | ✅ | | |
| return / log_return | | ✅ | ✅ | | ✅ (return) |
| RSI 5/15, MACD line/signal/hist | | ✅ | ✅ | | ✅ (rsi15, macd_hist) |
| vol5 / vol15 (realized vol) | | ✅ | ✅ | | ✅ (vol15) |
| tick_delta | | ✅ | ✅ | ✅ | ✅ |
| tod_sin/cos, session_min | | ✅ | ✅ | | |
| norm_vol (vol z-score) | | | | ✅ | |
| signed_vol (raw / linear) | | | | ✅ | |
| cum_td 5/10/15 | | | | ✅ | |
| regime_hi_prob (HMM posterior) | derived signal, appended to any base set | | | | |
