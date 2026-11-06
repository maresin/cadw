# H1 — Treatment effect (ATE / CATE)

*Section:* H — Causal Inference & Uplift

**Description:** Estimate average and heterogeneous treatment effect from observational or RCT data.

**Real-life examples:**
- A/B test incrementality
- Drug efficacy
- Marketing campaign lift

## Benchmarks
- **RCT_IHDP** — https://www.fredjo.com/
  - metrics: ate_error, pehe

## Models
### T_learner — baseline / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** Two outcome models, one per arm; CATE = mu1 - mu0 (Kunzel et al. 2019).

**Use cases:**
- A/B test CATE estimation

**Config:** `{'n_estimators': 'int', 'propensity_alpha': 'float'}`

**CADW:** `cadw.ml.meta_learners.fit_meta_learner`

**Benchmarks:**
- **[literature]** RCT_IHDP → ate_error=0.1
  - algorithm: `T-learner (Kunzel et al.)` (paper)
  - number: https://www.fredjo.com/
  - code/paper: https://arxiv.org/abs/1706.03461
  - note: [legacy v0.9.44, migrated v0.9.50]

### S_learner — baseline / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** Single model with treatment as a feature; CATE = f(x,1) - f(x,0).

**Use cases:**
- Marketing campaign CATE

**Config:** `{'n_estimators': 'int', 'propensity_alpha': 'float'}`

**CADW:** `cadw.ml.meta_learners.fit_meta_learner`

**Benchmarks:**
- **[literature]** RCT_IHDP → ate_error=0.12
  - algorithm: `S-learner (Kunzel et al.)` (paper)
  - number: https://www.fredjo.com/
  - code/paper: https://arxiv.org/abs/1706.03461
  - note: [legacy v0.9.44, migrated v0.9.50]

### DR_learner — sota / tier3_export

**Roles:** nonlinear_interactions, probabilistic
**Status:** desc, export

**Purpose:** Doubly-robust (AIPW) meta-learner; uses propensity + outcome model.

**Use cases:**
- Observational causal inference

**Benchmarks:**
- **[literature]** RCT_IHDP → ate_error=0.06
  - algorithm: `DR-learner (EconML)` (docs)
  - number: https://www.fredjo.com/
  - code/paper: https://econml.azurewebsites.net/spec/estimation/dr.html
  - note: [legacy v0.9.44, migrated v0.9.50]

### CausalForest — sota / tier2_wrapper

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Generalized Random Forest (Athey 2019): honest-split local CATE estimation.

**Use cases:**
- Heterogeneous treatment effects
- Subgroup policy targeting

**Config:** `{'n_trees': 'int', 'honesty': 'bool'}`

**Benchmarks:**
- **[literature]** cadw_internal → cate_rmse=null
  - algorithm: `CausalForest` (paper)
  - number: .pingis/benches/
  - code/paper: https://arxiv.org/abs/1510.04342
  - note: no_public_benchmark: CausalForest per-dataset

## Metrics
- **ate_error** (anchor, lower)
  - Absolute error of the estimated average treatment effect.
  - *special:* ATE error alone hides heterogeneity quality; report PEHE alongside.
- **pehe** (pareto_axis, lower)
  - Precision in Estimation of Heterogeneous Effect; PEHE(x) RMSE.
  - *special:* PEHE is sensitive to noise level; not comparable across DGPs.

## Ensemble rules
- **[nonlinear_interactions]** → T_learner + S_learner (max_per_role=2)
  - Average CATE; T-learner wins when arms have distinct dynamics, S-learner when sparse.
