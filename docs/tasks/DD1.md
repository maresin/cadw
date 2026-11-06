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
**Status:** ✓ cadw, ✓ agent

**Purpose:** Holt-Winters exponential smoothing (additive-error, additive/multiplicative seasonality, damped trend). Classical decomposition init + bounded Nelder-Mead grid search over (alpha, beta, gamma, phi); analytic Gaussian prediction intervals. Reference: Hyndman et al. 2002.

**Use cases:**
- Inventory replenishment
- Retail demand forecast

**Config:** `{'seasonal_period': 'int', 'seasonal': 'str', 'trend': 'bool', 'damped': 'bool'}`

**CADW:** `cadw.ts.ets.ETSForecaster`

**Benchmarks:**
- **[literature]** M4 → smape=13.52
  - algorithm: `statsmodels ExponentialSmoothing` (docs)
  - number: https://github.com/Mcompetitions/M4-methods
  - code/paper: https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
  - note: [legacy v0.9.44, migrated v0.9.50]
- **[cadw_measured]** cadw_internal → fit_time_s=17.153 (n=3)
  - algorithm: `cadw.ts.ets.ETSForecaster` (code)
  - number: .pingis/benches/ets.json
  - code/paper: cadw/ts/ets.py
  - note: CADW-native: fit_predict_s=0.18/1.74/17.15 @n=500/5k/50k (seasonal_period=12; scripts/bench_ets.py, .pingis/benches/ets.json). Residual sigma=0.61/0.54/0.52. Peak RSS 4.9/0.1/1.4 MB. Hardware-specific; NOT comparable to external LB. Range 1000x across n - O(n * grid_size) fit; no cross-n comparable fit_time.

### AutoETS — sota / tier1_core

**Roles:** temporal_dynamics, probabilistic
**Status:** ✓ cadw, ✓ agent

**Purpose:** AutoETS: AICc-selected ETS. Evaluates up to 9 configurations (trend N/A/Ad x season N/A/M), skips invalid combos (n<2s for seasonal; y<=0 for multiplicative), picks minimum AICc = n*ln(SSE/n) + 2k + 2k(k+1)/(n-k-1) with k = sigma+alpha+[beta]+ [gamma]+[phi]. Falls back to NaiveForecaster(last) if no config fits. Wraps ETSForecaster; same prediction interval logic.

**Use cases:**
- Automatic retail demand forecast (no manual model selection)
- Batch forecasting where ETS configuration varies per series

**Config:** `{'seasonal_period': 'int', 'warn_slow_fit': 'bool', 'slow_fit_threshold_s': 'float'}`

**CADW:** `cadw.ts.auto_ets.AutoETSForecaster`

**Benchmarks:**
- **[cadw_measured]** cadw_internal → mase=0.4998 (n=1)
  - algorithm: `cadw.ts.auto_ets.AutoETSForecaster` (code)
  - number: .pingis/benches/auto_ets.json
  - code/paper: cadw/ts/auto_ets.py
  - note: CADW-native: MASE=0.4998 on synthetic_seasonal_linear (n=500, seasonal_period=12); 9 configs evaluated, selected ETS(A,A,N) (trend dominates seasonal in this DGP); fit_time 44.1s. WARNING: 44s fit time is slow for n=500 — grid search is O(9 configs * 600 evals * n) Python loop. Backlog item "AutoETS grid search optimization" tracks vectorization / coordinate-descent fix.

### Theta — baseline / tier1_core

**Roles:** temporal_dynamics, probabilistic
**Status:** ✓ cadw, ✓ agent

**Purpose:** Theta method (STM variant): STL decomposition -> OLS theta_0 (trend) + SES theta_2 (curvature) -> 0.5/0.5 combine -> reseasonalize. Capability flags: handles_trend=True, handles_seasonality=True. KNOWN LIMITATION: STM underperforms on trend-dominated series (MASE 1.4 vs AutoETS 0.5 on synthetic trend=0.3,season_amp=3) — SES on theta_2 lags behind trend by (1-alpha)/alpha*slope per step; STL fixes decomposition leak but not SES lag. This is the STM baseline; OTM variant (weight optimisation) in backlog. Use for M3-like series where seasonality dominates trend.

**Use cases:**
- M3-competition-style seasonal demand
- Fast baseline for univariate forecast

**Config:** `{'seasonal_period': 'int', 'n_iter': 'int'}`

**CADW:** `cadw.ts.theta.ThetaForecaster`

**Benchmarks:**
- **[cadw_measured]** cadw_internal → fit_time_s=0.981 (n=3)
  - algorithm: `cadw.ts.theta.ThetaForecaster` (code)
  - number: .pingis/benches/theta.json
  - code/paper: cadw/ts/theta.py
  - note: CADW-native: fit_predict_s=0.012/0.103/0.981 @n=1k/10k/100k (seasonal_period=12; scripts/bench_theta.py; .pingis/benches/theta.json). alpha=0.65 stable across n; sigma 0.677/0.674/0.680. Peak RSS 4.9/0.6/6.1 MB. Hardware-specific; NOT comparable to external LB. ~40x faster than AutoETS on same DGP (single alpha grid vs 600-eval grid).

### AutoTheta — sota / tier1_core

**Roles:** temporal_dynamics, probabilistic
**Status:** ✓ cadw, ✓ agent

**Purpose:** AutoTheta: CV-selected STM vs OTM. Evaluates 4 candidates (STM/OTM x additive/multiplicative); holdout MASE selection; refits winner on full series. STM = fixed w2=0.5; OTM = weight-optimized (w2, alpha). Additive via STL (y=T+S+R); multiplicative via log-transform + additive STL. Skips multiplicative when y<=0. Never worse than best-of-{STM,OTM} on validation by construction. SOTA-class classical forecaster.

**Use cases:**
- Automatic forecasting without manual STM/OTM choice
- Batch demand forecast with mixed trend/seasonality

**Config:** `{'seasonal_period': 'int', 'val_steps': 'int', 'n_iter': 'int'}`

**CADW:** `cadw.ts.auto_theta.AutoThetaForecaster`

**Benchmarks:**
- **[cadw_measured]** cadw_internal → mase=0.2093 (n=2)
  - algorithm: `cadw.ts.auto_theta.AutoThetaForecaster` (code)
  - number: .pingis/benches/auto_theta.json
  - code/paper: cadw/ts/auto_theta.py
  - note: CADW-native: M3-like MASE=0.209 (selected OTM_additive; beats AutoETS 0.276 by 24%, AutoTheta 1.4s vs AutoETS 38.6s); trend-dominated MASE=0.296 (selected OTM; beats STM 0.652 2.2x). HONEST LIMITATION: on M3-like, CV-holdout selected OTM but STM generalized slightly better (0.201 vs 0.209); holdout-selection risk documented. K-fold CV is backlog.
- **[literature]** fev_bench → mase=1.381 (n=19)
  - algorithm: `statsforecast AutoTheta` (code)
  - number: https://github.com/autogluon/fev
  - code/paper: https://github.com/Nixtla/statsforecast
  - note: fev-bench (anofox-forecast head-to-head, 2025): AutoTheta MASE 1.381 (Nixtla fev auto_theta 1.362) — SOTA-class classical, 1.4% behind Nixtla, 5.9% ahead of AutoETS (1.525). Tier3 export reference value; CADW implementation is a practical OTM/STM hybrid, not exact Nixtla port.

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

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** GBDT on lag features; SOTA on heterogeneous M4/M5-type series. v0.9.76c: role temporal_dynamics removed — GBDTForecaster captures nonlinear_interactions in lag-feature space; temporal structure is implicit in lag features, not a native temporal_dynamics model (SARIMA/AutoETS are the temporal_dynamics representatives). Enables 3-role ensemble in D1.

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

### EnsembleForecaster — ensemble / tier1_core

**Roles:** temporal_dynamics, nonlinear_interactions, probabilistic
**Status:** ✓ cadw

**Purpose:** Weighted ensemble of heterogeneous forecasters (ETS/Theta/GBDT/Naive), per-member pipelines. Different inductive biases -> decorrelated errors. v0.9.82 (verified). Weight learning: inverse-MASE on rolling holdout.

**Use cases:**
- Robust forecasts across regimes
- Automated model combination

**Config:** `{'members': 'list[ForecasterBase]', 'weighting': 'uniform | inverse_mase | optimized'}`

**CADW:** `cadw.ts.EnsembleForecaster`

**Benchmarks:**
- **[cadw_measured]** cadw_internal → mase=null (n=3)
  - algorithm: `cadw.ts.ensemble.EnsembleForecaster` (code)
  - number: .pingis/benches/ensemble.json
  - code/paper: cadw/ts/ensemble.py
  - note: CADW-native ensemble; see .pingis/benches/ensemble.json (v0.9.82).

### OTMThetaForecaster — sota / tier1_core

**Roles:** temporal_dynamics
**Status:** ✓ cadw

**Purpose:** OTM: optimized theta method (theta=2 fixed, (w2, alpha) optimized by MLE). v0.9.78 (verified). Better than STM on trend-dominated series.

**Use cases:**
- Trend-dominated forecast
- M3/M4-style univariate tasks

**Config:** `{'seasonal_period': 'int'}`

**CADW:** `cadw.ts.OTMThetaForecaster`

**Benchmarks:**
- **[literature]** M3 → mase=null
  - algorithm: `OTM (Optimized Theta Method)` (paper)
  - number: https://forecasters.org/resources/time-series-data/m3-competition/
  - code/paper: https://arxiv.org/abs/1505.03004
  - note: M3 winner family (Makridakis 2000); OTM weights optimized per series.

### TemporalHierarchy — sota / tier1_core

**Roles:** temporal_dynamics, calibration
**Status:** ✓ cadw

**Purpose:** Temporal Hierarchical Forecasting: aggregate base series to K temporal levels, fit forecaster per level, reconcile (bottom-up / top-down / optimal-OLS / MinT-WLS). v0.9.89 (verified).

**Use cases:**
- Multi-level demand forecasting (day/week/month)
- Hierarchy-consistent forecasts

**Config:** `{'levels': 'list[int]', 'reconciliation': 'bu | td | optimal_ols | optimal_wls'}`

**CADW:** `cadw.ts.TemporalHierarchy`

**Benchmarks:**
- **[literature]** M5 → mase=null
  - algorithm: `MinT-WLS temporal reconciliation` (paper)
  - number: https://www.kaggle.com/competitions/m5-forecasting-accuracy
  - code/paper: https://robjhyndman.com/papers/mint.pdf
  - note: MinT (Wickramasuriya 2019); temporal aggregation + reconciliation.

### TransformedTargetForecaster — baseline / tier1_core

**Roles:** temporal_dynamics, representation
**Status:** ✓ cadw

**Purpose:** Pipeline: transformers (in order) then forecaster. At fit: y -> t1 -> ... -> tN -> base.fit; at predict: base.predict -> tN.inverse -> ... . v0.9.79a (verified). Canonical composition wrapper for AutoForecastPipeline.

**Use cases:**
- Detrend + forecast + inverse
- STL-deseasonalize + forecast

**Config:** `{'transformers': 'list', 'forecaster': 'ForecasterBase'}`

**CADW:** `cadw.ts.TransformedTargetForecaster`

**Benchmarks:**
- **[cadw_measured]** cadw_internal → mase=null
  - algorithm: `cadw.ts.transforms.TransformedTargetForecaster` (code)
  - number: .pingis/benches/transforms.json
  - code/paper: cadw/ts/transforms.py
  - note: CADW-native pipeline; see .pingis/benches/transforms.json (v0.9.79).

### backtest_forecaster — baseline / tier2_wrapper

**Roles:** calibration
**Status:** desc, export

**Purpose:** Utility, not a model. Expanding/sliding-window CV for forecasters (MAE/RMSE/MAPE/sMAPE/MASE, optional pinball). Clones forecaster per fold. Canonical reference for forecast-quality evaluation.

**Use cases:**
- Rolling-origin CV for forecast comparison
- Hyperparameter selection for forecasters

**Config:** `{'horizon': 'int', 'n_folds': 'int', 'strategy': 'expanding | sliding', 'metrics': 'list[str]'}`

**CADW:** `cadw.ts.backtest_forecaster`

**Benchmarks:**
- **[cadw_measured]** cadw_internal → mase=null
  - algorithm: `cadw.ts.backtest.backtest_forecaster` (code)
  - number: cadw/ts/backtest.py
  - code/paper: cadw/ts/backtest.py
  - note: Utility (not a model); provides CV-based MASE for other forecasters.

### Chronos-Bolt-Tiny — sota / tier3_export

**Roles:** temporal_dynamics, representation
**Status:** export

**Purpose:** Chronos-Bolt-Tiny (Amazon, 2024): 9M-parameter pretrained T5 encoder-decoder for time-series forecasting; patch-based input, direct multi-step, quantile-native output (produces full predictive distribution). 250x faster and 20x more memory-efficient than original Chronos. Notable NEGATIVE scaling result: on electricity-price data Tiny (8.65M) beats Bolt-Mini (21M), Bolt-Small (48M), Bolt-Base (205M) - smaller is better here. Requires torch; inference-only (fit not in CADW).

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
**Status:** export

**Purpose:** Chronos-Bolt-Mini (Amazon, 2024): 21M params. Larger than Tiny; on most fev-bench datasets has comparable or slightly better MASE. IEEE electricity showed Tiny beats Mini (MAE 0.0393 vs 0.0394). Apache 2.0, CPU-feasible. Requires torch; inference-only (fit not in CADW).

**Use cases:**
- Slightly larger CPU-friendly zero-shot baseline

**Config:** `{'model_id': 'str'}`

**Benchmarks:**
- **[literature]** fev_bench → mase=null (n=19)
  - algorithm: `Chronos-Bolt-Mini` (code)
  - number: https://github.com/autogluon/fev
  - code/paper: https://huggingface.co/amazon/chronos-bolt-mini
  - note: no_public_benchmark: fev-bench aggregate-only; Mini is comparable to Tiny (roughly MASE 1.39-1.40).

### Chronos-2-small — sota / tier3_export

**Roles:** temporal_dynamics, representation, nonlinear_interactions
**Status:** export

**Purpose:** Chronos-2-small (Amazon, Nov 2025): 28M params. Second-gen Chronos: multivariable native support, covariates (past+future, real+categorical), cross-learning. Context 8192 (4x Bolt). Only foundation model reference for D2 multi-forecast and D1 exog. Requires torch; inference-only (fit not in CADW).

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

## Metrics
- **mase** (anchor, lower)
  - Scale-free error; comparable across series.
  - *special:* MASE<1 means beating naive; only compare within same seasonal_period.
- **smape** (diagnostic, lower)
  - Symmetric percentage error; M4/M5 competition standard.
  - *special:* Punishes over-forecast more than under-forecast; not comparable with MASE.
- **mape** (diagnostic, lower)
  - Mean absolute percentage error (%).
  - *special:* NaN when any actual is zero; not comparable across scales.
- **rmse** (diagnostic, lower)
  - Root mean squared error in y units.
  - *special:* Sensitive to outliers; prefer mase for cross-series comparison.
- **mae** (diagnostic, lower)
  - Mean absolute error in y units.
  - *special:* Same units as target; not comparable across different series.
- **wape** (diagnostic, lower)
  - Weighted absolute percentage error (sum|y-f|/sum|y| * 100).
  - *special:* Robust to zeros in actuals (unlike MAPE, which NaNs); NaN when sum|y|=0.
- **mqlike** (diagnostic, lower)
  - Mean QLIKE (variance-robust volatility loss).
  - *special:* Requires y>0, f>0; NaN otherwise. Standard for volatility forecast.
- **msis** (diagnostic, lower)
  - Mean Scaled Interval Score (Gneiting & Raftery 2007) — width plus penalty for misses at level 1-misis_alpha.
  - *special:* Requires forecaster.prediction_interval; skipped if unsupported. Not comparable to point metrics; measures calibration + width.
- **pinball** (diagnostic, lower)
  - Pinball loss at quantile q (a.k.a. quantile loss).
  - *special:* Requires forecaster quantile support; report per-q (pinball_0.1, pinball_0.5, pinball_0.9 separately, not summed).
- **wql** (diagnostic, lower)
  - Weighted Quantile Loss (weighted mean pinball across quantile levels).
  - *special:* Requires forecaster quantile support and `quantiles` parameter. Primary metric for probabilistic forecast in fev-bench (GIFT-Eval-style).
- **sql** (diagnostic, lower)
  - Scaled Quantile Loss (WQL / naive_MAE) — scale-free WQL for cross-series comparison.
  - *special:* Requires forecaster quantile support; NaN when naive_MAE=0.
- **crps** (diagnostic, lower)
  - Continuous Ranked Probability Score (approximate from quantile losses; CRPS ~ 2 * mean pinball).
  - *special:* Requires forecaster quantile support; approximate, not exact integral.

## Ensemble rules
- **[temporal_dynamics, probabilistic, nonlinear_interactions]** → SARIMA + AutoTheta + GBDTForecaster (max_per_role=2)
  - 3 complementary paradigms: SARIMA (temporal_dynamics, seasonal ARIMA via Hannan-Rissanen CSS), AutoTheta (probabilistic, CV-selected STM/OTM Theta on STL — SOTA-class classical), GBDTForecaster (nonlinear_interactions, tree-based on lag features). v0.9.78c: AutoTheta replaces AutoETS in the ensemble (both expose temporal_dynamics+probabilistic; AutoTheta has better fev MASE 1.381 vs AutoETS 1.525 and is 5-30x faster). AutoETS remains as standalone card. max_per_role=2 because SARIMA and AutoTheta both expose temporal_dynamics. Average z-scored point forecasts weighted by inverse holdout MASE.
