# A3 — Regression

*Section:* A — Classification & Regression

**Description:** Predict y in R.

**Real-life examples:**
- Price prediction
- Demand forecast
- ETA estimation

## Benchmarks
- **AMLB** — https://openml.github.io/automlbenchmark/
  - metrics: rmse

## Models
### Ridge — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw, ✓ agent

**Purpose:** L2-regularised linear regression; closed-form solution, O(d^2) state.

**Use cases:**
- Housing price baseline

**Config:** `{'alpha': 'float'}`

**CADW:** `cadw.ml.linear.fit_linear`

**Benchmark:** AMLB → rmse=0.42 (normalised RMSE median across AMLB regression tasks)

### Lasso — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw, ✓ agent

**Purpose:** L1-regularised linear regression; produces sparse coefficients.

**Use cases:**
- Gene expression regression

**Config:** `{'alpha': 'float', 'max_iter': 'int'}`

**CADW:** `cadw.ml.linear.fit_lasso`

**Benchmark:** AMLB → rmse=0.43

### GBDT_L2 — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** Gradient-boosted trees with squared loss; general-purpose SOTA.

**Use cases:**
- Kaggle tabular regression winner baseline

**Config:** `{'n_estimators': 'int', 'max_depth': 'int', 'learning_rate': 'float'}`

**CADW:** `cadw.ml.gbdt.fit_gbdt_regressor`

**Benchmark:** AMLB → rmse=0.36

## Metrics
- **rmse** (anchor, lower)
  - Standard regression loss; interpretable in y units.
  - *special:* Sensitive to outliers; prefer mae when target has heavy tails.
- **mae** (pareto_axis, lower)
  - Median-scaled loss; robust to outliers.
  - *special:* Not linearly related to rmse; do not combine into one score.
- **r2** (diagnostic, higher)
  - Fraction of variance explained.
  - *special:* Can be negative on test data; only meaningful relative to a baseline.

## Ensemble rules
- **[linear_signal, nonlinear_interactions]** → Ridge + GBDT_L2 (max_per_role=2)
  - Linear extrapolation + tree interpolation; average predictions.
