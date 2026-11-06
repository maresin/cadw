# D2 — Multivariate forecasting

*Section:* D — Time Series

**Description:** Predict multiple correlated series jointly.

**Real-life examples:**
- Store-level sales with promotions
- Sensor network

## Benchmarks
- **M5** — https://www.kaggle.com/competitions/m5-forecasting-accuracy
  - metrics: wrmse, smape

## Models
### VAR — baseline / tier1_core

**Roles:** temporal_dynamics
**Status:** ✓ cadw, ✓ agent

**Purpose:** Vector autoregression; explicit cross-series coefficients.

**Use cases:**
- Macroeconomic indicator forecast

**Config:** `{'p': 'int'}`

**CADW:** `cadw.ts.var.VARModel`

**Benchmarks:**
- **[literature]** M5 → smape=14.2
  - algorithm: `statsmodels VAR` (docs)
  - number: https://www.kaggle.com/competitions/m5-forecasting-accuracy
  - code/paper: https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VAR.html
  - note: [legacy v0.9.44, migrated v0.9.50]
- **[cadw_reproduced]** cadw_internal → rmse=5.83672798857639e-18 (n=1)
  - algorithm: `cadw.ts.models.fit_var` (code)
  - reproduces: `statsmodels VAR`
  - number: .pingis/benches/reproduction/var.json
  - code/paper: cadw/ts/models.py
  - note: M2 parity on synthetic_m4_like_multivariate DGP: RMSE(cadw_forecast, statsmodels_forecast)=0.0 (5.8e-18 round-off); verdict=consistent

### GBDT_multiseries — sota / tier1_core

**Roles:** nonlinear_interactions, temporal_dynamics
**Status:** ✓ cadw, ✓ agent

**Purpose:** GBDT on lag features per series (global model).

**Use cases:**
- M5 store-item forecast

**Config:** `{'lags': 'tuple', 'group_col': 'str'}`

**CADW:** `cadw.ts.models.GBDTForecaster`

**Benchmarks:**
- **[literature]** M5 → smape=12.8
  - algorithm: `sklearn.ensemble.HistGradientBoostingRegressor` (docs)
  - number: https://www.kaggle.com/competitions/m5-forecasting-accuracy
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
  - note: [legacy v0.9.44, migrated v0.9.50]

### BVAR — sota / tier1_core

**Roles:** temporal_dynamics
**Status:** ✓ cadw, ✓ agent

**Purpose:** Bayesian VAR with Minnesota (random-walk) prior; Normal-Inverted Wishart conjugate posterior. Shrinks coefficients toward random walk, stabilizes forecasts on short series and large panels (Litterman 1986; Banbura-Giannone-Reichlin 2010). Streaming via sufficient statistics XTX/XTY/YTY Reduce.

**Use cases:**
- FRED-MD macroeconomic panel forecasting
- Central bank short-horizon projections

**Config:** `{'p': 'int', 'lambda1': 'float', 'lambda2': 'float', 'mu': 'float'}`

**CADW:** `cadw.ts.bvar.BVARModel`

**Benchmarks:** —

### VECM — sota / tier1_core

**Roles:** temporal_dynamics
**Status:** ✓ cadw, ✓ agent

**Purpose:** Vector Error Correction Model via Johansen MLE; recovers cointegrating relations β and adjustment speeds α for series sharing a stochastic trend. Streaming cross-covariance accumulation + generalized eigenproblem on k×k matrices (Johansen 1988/1991).

**Use cases:**
- Yield curve modeling (level/slope/curvature cointegrated)
- Pairs trading / spread forecasting
- Macro panel with cointegrated series

**Config:** `{'k_ar_diff': 'int', 'coint_rank': 'int', 'det_order': 'int'}`

**CADW:** `cadw.ts.vecm.VECMModel`

**Benchmarks:** —

### GaussianProcessForecaster — sota / tier2_wrapper

**Roles:** probabilistic
**Status:** desc, export

**Purpose:** GP-based multivariate forecaster with kernel-induced credible bands.

**Use cases:**
- Small multivariate series with intervals
- Sensor network forecast

**Config:** `{'kernel': 'string', 'n_restarts': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → mase=null
  - algorithm: `GaussianProcessForecaster` (docs)
  - number: .pingis/benches/
  - code/paper: https://scikit-learn.org/stable/modules/gaussian_process.html
  - note: no_public_benchmark: GP per-dataset

## Metrics
- **wrmse** (anchor, lower)
  - Hierarchical weighted error (M5 official).
  - *special:* Only defined when the target hierarchy is declared; not comparable with flat sMAPE.
- **smape** (diagnostic, lower)
  - Flat-scale accuracy.
  - *special:* Ignores hierarchy; use WRMSSE as the decision metric.

## Ensemble rules
- **[temporal_dynamics]** → VAR + BVAR + VECM + GBDT_multiseries (max_per_role=4)
  - Complementary inductive biases — linear (VAR) + Bayesian shrinkage (BVAR) + cointegration (VECM) + nonlinear lag (GBDT_multiseries). Use VAR/GBDT on stationary panels; BVAR on short samples / large d; VECM when series share a stochastic trend.
