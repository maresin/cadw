# D2 — Multivariate forecasting

*Section:* D — Time Series

**Description:** Predict multiple correlated series jointly.

**Real-life examples:**
- Store-level sales with promotions
- Sensor network

## Benchmarks
- **M5** — https://www.kaggle.com/competitions/m5-forecasting-accuracy
  - metrics: wrmssе, smape

## Models
### VAR — baseline / tier1_core

**Roles:** temporal_dynamics
**Status:** ✓ cadw, ✓ agent

**Purpose:** Vector autoregression; explicit cross-series coefficients.

**Use cases:**
- Macroeconomic indicator forecast

**Config:** `{'p': 'int'}`

**CADW:** `cadw.ts.models.VAR`

**Benchmarks:**
- **[literature]** M5 → smape=14.2
  - algorithm: `VAR` (description)
  - number: https://www.kaggle.com/competitions/m5-forecasting-accuracy
  - note: [migrated v0.9.47b: algorithm/algorithm_source_url unverified]

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
  - algorithm: `GBDT_multiseries` (description)
  - number: https://www.kaggle.com/competitions/m5-forecasting-accuracy
  - note: [migrated v0.9.47b: algorithm/algorithm_source_url unverified]

## Metrics
- **wrmssе** (anchor, lower)
  - Hierarchical weighted error (M5 official).
  - *special:* Only defined when the target hierarchy is declared; not comparable with flat sMAPE.
- **smape** (diagnostic, lower)
  - Flat-scale accuracy.
  - *special:* Ignores hierarchy; use WRMSSE as the decision metric.
