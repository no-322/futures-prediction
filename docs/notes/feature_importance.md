# Feature Importance — v2 Features (49 features, RF 200 trees)

Model: RandomForestClassifier, MDI (mean decrease in impurity)
Training set: 50% time-ordered split, binary direction labels

## Top 10

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `lag4_macd_hist` | 0.03773 |
| 2 | `lag1_macd_hist` | 0.03660 |
| 3 | `lag4_vol15` | 0.03646 |
| 4 | `lag1_vol15` | 0.03638 |
| 5 | `lag1_macd_line` | 0.03489 |
| 6 | `lag1_vwap_dev` | 0.03486 |
| 7 | `lag1_rsi5` | 0.03484 |
| 8 | `lag4_rsi15` | 0.03484 |
| 9 | `lag1_rsi15` | 0.03481 |
| 10 | `lag4_rsi5` | 0.03474 |

## Full ranking (49 features)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `lag4_macd_hist` | 0.03773 |
| 2 | `lag1_macd_hist` | 0.03660 |
| 3 | `lag4_vol15` | 0.03646 |
| 4 | `lag1_vol15` | 0.03638 |
| 5 | `lag1_macd_line` | 0.03489 |
| 6 | `lag1_vwap_dev` | 0.03486 |
| 7 | `lag1_rsi5` | 0.03484 |
| 8 | `lag4_rsi15` | 0.03484 |
| 9 | `lag1_rsi15` | 0.03481 |
| 10 | `lag4_rsi5` | 0.03474 |
| 11 | `lag4_macd_signal` | 0.03462 |
| 12 | `lag1_macd_signal` | 0.03460 |
| 13 | `lag4_macd_line` | 0.03428 |
| 14 | `lag4_vwap_dev` | 0.03367 |
| 15 | `session_min` | 0.03295 |
| 16 | `tod_cos` | 0.03245 |
| 17 | `lag4_vol5` | 0.03225 |
| 18 | `lag1_vol5` | 0.03214 |
| 19 | `tod_sin` | 0.03140 |
| 20 | `lag1_tick_delta` | 0.02230 |
| 21 | `lag4_tick_delta` | 0.02229 |
| 22 | `lag4_VWAP` | 0.01481 |
| 23 | `lag1_VWAP` | 0.01476 |
| 24 | `lag3_VWAP` | 0.01469 |
| 25 | `lag2_VWAP` | 0.01464 |
| 26 | `lag4_log_return` | 0.01416 |
| 27 | `lag4_return` | 0.01401 |
| 28 | `lag1_log_return` | 0.01362 |
| 29 | `lag1_return` | 0.01356 |
| 30 | `lag4_Open` | 0.01039 |
| 31 | `lag2_Open` | 0.00992 |
| 32 | `lag3_Open` | 0.00988 |
| 33 | `lag3_Close` | 0.00985 |
| 34 | `lag1_Open` | 0.00976 |
| 35 | `lag2_Close` | 0.00975 |
| 36 | `lag1_Close` | 0.00966 |
| 37 | `lag1_High` | 0.00963 |
| 38 | `lag4_Low` | 0.00956 |
| 39 | `lag2_High` | 0.00955 |
| 40 | `lag3_High` | 0.00955 |
| 41 | `lag4_High` | 0.00953 |
| 42 | `lag2_Low` | 0.00949 |
| 43 | `lag1_Low` | 0.00948 |
| 44 | `lag3_Low` | 0.00947 |
| 45 | `lag4_Close` | 0.00944 |
| 46 | `lag4_body_ratio` | 0.00843 |
| 47 | `lag1_bar_range` | 0.00812 |
| 48 | `lag1_body_ratio` | 0.00794 |
| 49 | `lag4_bar_range` | 0.00726 |
