# D1 — Univariate forecasting

*Section:* D — Time Series

**Description:** Predict future values of a single series from its own history.

**Real-life examples:**
- Demand forecast
- Server load
- Energy consumption

## Benchmarks
- **M4** — https://github.com/Mcompetitions/M4-methods
  - metrics: smape, mase

## Models
### NaiveForecaster — baseline / tier1_core

**Roles:** temporal_dynamics
**Status:** ✓ cadw, ✓ agent

**Purpose:** Last-value / seasonal / drift baseline; every method must beat it.

**Use cases:**
- M4 naive2 baseline

**Config:** `{'strategy': 'str', 'period': 'int'}`

**CADW:** `cadw.ts.models.NaiveForecaster`

**Benchmarks:**
- **[literature]** M4 → smape=14.35
  - algorithm: `naive2 forecaster` (docs)
  - number: https://github.com/Mcompetitions/M4-methods
  - code/paper: https://otexts.com/fpp3/
  - note: naive2 M4 benchmark [legacy v0.9.44, migrated v0.9.50]

### ETS — baseline / tier1_core

**Roles:** temporal_dynamics, probabilistic
**Status:** desc

**Purpose:** Exponential smoothing with explicit trend+seasonality components.

**Use cases:**
- Inventory replenishment

**Config:** `{'trend': 'str', 'seasonal': 'str'}`

**Benchmarks:**
- **[literature]** M4 → smape=13.52
  - algorithm: `statsmodels ExponentialSmoothing` (docs)
  - number: https://github.com/Mcompetitions/M4-methods
  - code/paper: https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
  - note: [legacy v0.9.44, migrated v0.9.50]

### SARIMA — sota / tier1_core

**Roles:** temporal_dynamics
**Status:** ✓ cadw, ✓ agent

**Purpose:** Seasonal ARIMA via Hannan-Rissanen CSS; explicit AR/MA orders.

**Use cases:**
- Traffic patterns with weekly seasonality

**Config:** `{'order': 'tuple', 'seasonal_order': 'tuple'}`

**CADW:** `cadw.ts.models.SARIMAForecaster`

**Benchmarks:**
- **[literature]** M4 → smape=12.84
  - algorithm: `statsmodels SARIMAX` (docs)
  - number: https://github.com/Mcompetitions/M4-methods
  - code/paper: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
  - note: [legacy v0.9.44, migrated v0.9.50]

### GBDTForecaster — sota / tier1_core

**Roles:** nonlinear_interactions, temporal_dynamics
**Status:** ✓ cadw, ✓ agent

**Purpose:** GBDT on lag features; SOTA on heterogeneous M4/M5-type series.

**Use cases:**
- Retail SKU forecast

**Config:** `{'lags': 'tuple', 'mode': 'str'}`

**CADW:** `cadw.ts.models.GBDTForecaster`

**Benchmarks:**
- **[literature]** M4 → smape=12.3
  - algorithm: `sklearn.ensemble.HistGradientBoostingRegressor` (docs)
  - number: https://github.com/Mcompetitions/M4-methods
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
  - note: [legacy v0.9.44, migrated v0.9.50]

### Chronos-Bolt-Tiny — sota / tier3_export

**Roles:** temporal_dynamics, representation
**Status:** desc, export

**Purpose:** Chronos-Bolt-Tiny (Amazon, 2024): 9M-parameter pretrained T5 encoder-decoder for time-series forecasting; patch-based input, direct multi-step, quantile-native output (produces full predictive distribution). 250x faster and 20x more memory-efficient than original Chronos. Notable NEGATIVE scaling result: on electricity-price data Tiny (8.65M) beats Bolt-Mini (21M), Bolt-Small (48M), Bolt-Base (205M) - smaller is better here.

**Use cases:**
- CPU-only zero-shot forecasting baseline
- Quantile forecast for inventory

**Config:** `{'model_id': 'str', 'context_length': 'int', 'prediction_length': 'int'}`

**Benchmarks:**
- **[literature]** fev_bench → mase=null (n=19)
  - algorithm: `Chronos-Bolt-Tiny` (code)
  - number: https://github.com/autogluon/fev
  - code/paper: https://huggingface.co/amazon/chronos-bolt-tiny
  - note: no_public_benchmark: fev-bench leaderboard renders client-side; Tiny MASE from HF model card plots is ~1.40 (geomean, 19 fev datasets); absolute value depends on the fev snapshot. IEEE electricity: MAE=0.0393, RMSE=0.317, MAPE=0.0242 (best among Chronos-Bolt variants).

### Chronos-Bolt-Mini — sota / tier3_export

**Roles:** temporal_dynamics, representation
**Status:** desc, export

**Purpose:** Chronos-Bolt-Mini (Amazon, 2024): 21M params. Larger than Tiny; on most fev-bench datasets has comparable or slightly better MASE. IEEE electricity showed Tiny beats Mini (MAE 0.0393 vs 0.0394). Apache 2.0, CPU-feasible.

**Use cases:**
- Slightly larger CPU-friendly zero-shot baseline

**Config:** `{'model_id': 'str'}`

**Benchmarks:**
- **[literature]** fev_bench → mase=null (n=19)
  - algorithm: `Chronos-Bolt-Mini` (code)
  - number: https://github.com/autogluon/fev
  - code/paper: https://huggingface.co/amazon/chronos-bolt-mini
  - note: no_public_benchmark: fev-bench aggregate-only; Mini is comparable to Tiny (roughly MASE 1.39-1.40).

### Chronos-Bolt-Small — sota / tier3_export

**Roles:** temporal_dynamics, representation
**Status:** desc, export

**Purpose:** Chronos-Bolt-Small (Amazon, 2024): 48M params. Underperforms Tiny on electricity-price (MAE 0.0530 vs 0.0393); kept as horizontal reference showing "larger does not help".

**Use cases:**
- Upper-bound reference for Bolt scaling

**Config:** `{'model_id': 'str'}`

**Benchmarks:**
- **[literature]** fev_bench → mase=null (n=19)
  - algorithm: `Chronos-Bolt-Small` (code)
  - number: https://github.com/autogluon/fev
  - code/paper: https://huggingface.co/amazon/chronos-bolt-small
  - note: no_public_benchmark: fev-bench aggregate-only. IEEE electricity showed Small is WORSE than Tiny (MAE 0.0530 vs 0.0393) - negative result on model-size scaling.

### Chronos-Bolt-Base — sota / tier3_export

**Roles:** temporal_dynamics, representation
**Status:** desc, export

**Purpose:** Chronos-Bolt-Base (Amazon, 2024): 205M params. Top of Bolt family on fev-bench (MASE 1.393); but performs WORSE than Tiny on electricity. Illustrates that rank depends on benchmark - no universal "best Bolt".

**Use cases:**
- High-parameter CPU reference

**Config:** `{'model_id': 'str'}`

**Benchmarks:**
- **[literature]** fev_bench → mase=1.393 (n=19)
  - algorithm: `Chronos-Bolt-Base` (code)
  - number: https://github.com/autogluon/fev
  - code/paper: https://huggingface.co/amazon/chronos-bolt-base
  - note: fev-bench MASE 1.393 (geomean over 19 datasets); 5th of 25 in fev leaderboard at release; beats Tiny here BUT worse than Tiny on electricity (MAE 0.0477 vs 0.0393).

### Chronos-2-small — sota / tier3_export

**Roles:** temporal_dynamics, representation, nonlinear_interactions
**Status:** desc, export

**Purpose:** Chronos-2-small (Amazon, Nov 2025): 28M params. Second-gen Chronos: multivariable native support, covariates (past+future, real+categorical), cross-learning. Context 8192 (4x Bolt). Only foundation model reference for D2 multi-forecast and D1 exog.

**Use cases:**
- Multivariate zero-shot forecast
- Forecast with exogenous covariates

**Config:** `{'model_id': 'str', 'context_length': 'int'}`

**Benchmarks:**
- **[literature]** GIFT_Eval → mase=null
  - algorithm: `Chronos-2-small` (code)
  - number: https://github.com/SalesforceAIResearch/gift-eval
  - code/paper: https://huggingface.co/amazon/chronos-2-small
  - note: no_public_benchmark: GIFT-Eval dynamic leaderboard; no single quoted MASE available. Chronos-2 120M is the flagship; small 28M is CPU-feasible variant. DO NOT compare Chronos-2 to Chronos-Bolt - different architecture + capabilities.

### AutoGluon-TS — sota / tier3_export

**Roles:** nonlinear_interactions, temporal_dynamics, calibration
**Status:** desc, export

**Purpose:** AutoGluon-TimeSeries (AWS Labs, 2023+): stack ensemble over statistical + GBDT + (optional) DL; learned weights via repeated k-fold bagging + holdout MASE. The winning approach family for M5 (WRMSE 0.52). Reference for simple-weighted / FFORMA-style ensembles in D1.

**Use cases:**
- M5 winner architecture (stack over LightGBM + stat)
- AutoML baseline for retail forecasting

**Config:** `{'presets': 'str', 'time_limit': 'int'}`

**Benchmarks:**
- **[literature]** M5 → wrmse=0.52 (n=1)
  - algorithm: `AutoGluon-TimeSeries (stack)` (docs)
  - number: https://www.kaggle.com/competitions/m5-forecasting-accuracy
  - code/paper: https://auto.gluon.ai/stable/tutorials/timeseries/index.html
  - note: Approx WRMSE for M5 winning pipeline (LightGBM + AutoGluon transforms; the exact 0.52 is the M5 top solution which used AutoGluon-style stacking). AutoGluon-TS not itself the winner submission but the winning approach family; treat as literature reference with standard M5 caveat.

### Timer-S1 — sota / tier3_export

**Roles:** temporal_dynamics, representation
**Status:** desc, export

**Purpose:** Timer-S1 (ByteDance, 2026): 1B+ param decoder-only foundation model for time series; top-1 of GIFT-Eval at snapshot (MASE 0.693, CRPS 0.485). GPU-only; true SOTA horizon for foundation forecasters.

**Use cases:**
- Absolute SOTA zero-shot reference

**Config:** `{'model_id': 'str'}`

**Benchmarks:**
- **[literature]** GIFT_Eval → mase=0.693
  - algorithm: `Timer-S1` (code)
  - number: https://github.com/SalesforceAIResearch/gift-eval
  - code/paper: https://github.com/thuml/Timer
  - note: GIFT-Eval snapshot 2026: MASE 0.693, CRPS 0.485 - best of pretrained models at release. 1B+ params; GPU required; not CPU-feasible for CADW. SOTA horizon only.

### t0-beta — sota / tier3_export

**Roles:** temporal_dynamics, representation
**Status:** desc, export

**Purpose:** t0-beta (2026): foundation model second on GIFT-Eval (MASE 0.687, CRPS 0.474) - marginally beats Timer-S1 on MASE/CRPS. Included for completeness; only one of {Timer-S1, t0-beta} needed in ensemble_recommendation horizon.

**Use cases:**
- Alternative SOTA zero-shot reference

**Config:** `{'model_id': 'str'}`

**Benchmarks:**
- **[literature]** GIFT_Eval → mase=0.687
  - algorithm: `t0-beta` (code)
  - number: https://github.com/SalesforceAIResearch/gift-eval
  - code/paper: https://github.com/SalesforceAIResearch/gift-eval
  - note: GIFT-Eval snapshot 2026: MASE 0.687, CRPS 0.474. Marginally better than Timer-S1 on MASE; same GPU-only constraint.

## Metrics
- **mase** (anchor, lower)
  - Scale-free error; comparable across series.
  - *special:* MASE<1 means beating naive; only compare within same seasonal_period.
- **smape** (diagnostic, lower)
  - Symmetric percentage error; M4/M5 competition standard.
  - *special:* Punishes over-forecast more than under-forecast; not comparable with MASE.

## Ensemble rules
- **[temporal_dynamics, nonlinear_interactions]** → SARIMA + GBDTForecaster (max_per_role=2)
  - SARIMA captures linear temporal dynamics; GBDT captures nonlinear interactions + seasonal patterns. max_per_role=2 because both expose temporal_dynamics. Weight forecasts by inverse MASE on holdout.
