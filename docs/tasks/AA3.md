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
**Status:** ✓ cadw

**Purpose:** L2-regularised linear regression; closed-form solution, O(d^2) state.

**Use cases:**
- Housing price baseline

**Config:** `{'alpha': 'float'}`

**CADW:** `cadw.ml.linear.fit_linear`

**Benchmarks:**
- **[literature]** AMLB → rmse=0.42
  - algorithm: `sklearn.linear_model.Ridge` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
  - note: normalised RMSE median across AMLB regression tasks [legacy v0.9.44, migrated v0.9.50]

### Lasso — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw

**Purpose:** L1-regularised linear regression; produces sparse coefficients.

**Use cases:**
- Gene expression regression

**Config:** `{'alpha': 'float', 'max_iter': 'int'}`

**CADW:** `cadw.ml.linear.fit_lasso`

**Benchmarks:**
- **[literature]** AMLB → rmse=0.43
  - algorithm: `sklearn.linear_model.Lasso` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
  - note: [legacy v0.9.44, migrated v0.9.50]

### ElasticNet — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw

**Purpose:** L1+L2 regularised linear regression; l1_ratio=1.0 -> Lasso, 0.0 -> Ridge.

**Use cases:**
- High-dimensional regression with correlated features
- Gene expression with grouped predictors

**Config:** `{'alpha': 'float', 'l1_ratio': 'float', 'max_iter': 'int'}`

**CADW:** `cadw.ml.linear.fit_elasticnet`

**Benchmarks:**
- **[literature]** AMLB → rmse=0.43
  - algorithm: `sklearn.linear_model.ElasticNet` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
  - note: [legacy v0.9.44, migrated v0.9.50; delegation to fit_linear]

### GBDT_L2 — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** Gradient-boosted trees with squared loss; general-purpose SOTA.

**Use cases:**
- Kaggle tabular regression winner baseline

**Config:** `{'n_estimators': 'int', 'max_depth': 'int', 'learning_rate': 'float'}`

**CADW:** `cadw.ml.gbdt.fit_gbdt_regressor`

**Benchmarks:**
- **[literature]** AMLB → rmse=0.36
  - algorithm: `sklearn.ensemble.GradientBoostingRegressor` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
  - note: [legacy v0.9.44, migrated v0.9.50]
- **[cadw_reproduced]** cadw_internal → rmse=1.0970246900316962 (n=1)
  - algorithm: `cadw.ml.gbdt.fit_gbdt_regressor` (code)
  - reproduces: `sklearn.ensemble.GradientBoostingRegressor`
  - number: .pingis/benches/reproduction/gbdt_regressor.json
  - code/paper: cadw/ml/gbdt.py
  - note: M2 parity on synthetic_amlb_regression_like DGP; ref(sklearn.GradientBoostingRegressor)=1.0498, verdict=consistent

### GaussianProcessRegressor — sota / tier2_wrapper

**Roles:** probabilistic
**Status:** desc, export

**Purpose:** GP regression with kernel prior; credible intervals. Top-1 on small-n tabular regression.

**Use cases:**
- Small-n regression with uncertainty
- Bayesian optimization surrogate

**Config:** `{'kernel': 'string', 'alpha': 'float'}`

**Benchmarks:**
- **[literature]** cadw_internal → rmse=null
  - algorithm: `GaussianProcessRegressor` (docs)
  - number: .pingis/benches/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
  - note: no_public_benchmark: GP per-dataset; not on AMLB tabular regression LB

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
