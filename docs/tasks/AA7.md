# A7 — Quantile / interval regression

*Section:* A — Classification & Regression

**Description:** Predict conditional quantiles (e.g. P10/P50/P90 of y|x).

**Real-life examples:**
- Delivery-time ETA
- Inventory safety stock
- Risk bands

## Benchmarks
- **AMLB** — https://openml.github.io/automlbenchmark/
  - metrics: rmse

## Models
### LinearQuantReg — baseline / tier1_core

**Roles:** linear_signal
**Status:** desc

**Purpose:** Linear quantile regression (pinball loss); fast baseline.

**Use cases:**
- Salary quantile regression

**Config:** `{'quantiles': 'list', 'alpha': 'float'}`

**Benchmarks:**
- **[literature]** AMLB → rmse=0.5
  - algorithm: `sklearn.linear_model.QuantileRegressor` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html
  - note: pinball loss, not rmse; use for ranking only [legacy v0.9.44, migrated v0.9.50]

### GBDT_pinball — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** desc

**Purpose:** GBDT with pinball loss; SOTA for nonlinear quantile regression.

**Use cases:**
- ETA with confidence intervals

**Config:** `{'quantiles': 'list', 'n_estimators': 'int'}`

**Benchmarks:**
- **[literature]** AMLB → rmse=0.46
  - algorithm: `sklearn.ensemble.GradientBoostingRegressor` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
  - note: [legacy v0.9.44, migrated v0.9.50]

## Metrics
- **pinball_loss** (anchor, lower)
  - Official quantile regression loss.
  - *special:* Sum over quantiles = 'interval score'; report per-quantile, not aggregated.
- **coverage_90** (pareto_axis, target)
  - Fraction of true y within [P05, P95]; target 0.90.
  - *special:* Diagnostic for calibration of quantile intervals, not a score.
