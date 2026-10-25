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
