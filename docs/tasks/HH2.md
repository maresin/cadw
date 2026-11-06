# H2 — Uplift modeling (Qini)

*Section:* H — Causal Inference & Uplift

**Description:** Rank customers by incremental gain from a treatment.

**Real-life examples:**
- Targeted promotions (avoid 'sure things')
- Retention campaigns

## Benchmarks
- **RCT_IHDP** — https://www.fredjo.com/
  - metrics: ate_error, pehe

## Models
### T_learner_uplift — baseline / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** Two-model uplift (see _fit_uplift_two_model); reference baseline.

**Use cases:**
- Promotions targeting

**Config:** `{'n_estimators': 'int'}`

**CADW:** `cadw.ml.uplift._fit_uplift_two_model`

**Benchmarks:**
- **[literature]** RCT_IHDP → pehe=0.15
  - algorithm: `T-learner (Kunzel et al.)` (paper)
  - number: https://www.fredjo.com/
  - code/paper: https://arxiv.org/abs/1706.03461
  - note: [legacy v0.9.44, migrated v0.9.50]

### CausalTree — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** ED-divergence splits + honest leaves (Rzepakowski-Jaroszewicz 2012, Wager-Athey 2018).

**Use cases:**
- Uplift campaigns

**Config:** `{'n_estimators': 'int', 'max_depth': 'int', 'max_samples': 'int'}`

**CADW:** `cadw.ml.uplift.fit_uplift`

**Benchmarks:**
- **[literature]** RCT_IHDP → pehe=0.12
  - algorithm: `Causal tree (Athey & Imbens)` (paper)
  - number: https://www.fredjo.com/
  - code/paper: https://arxiv.org/abs/1504.01132
  - note: [legacy v0.9.44, migrated v0.9.50]

### CausalForest — sota / tier2_wrapper

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Generalized Random Forest (Athey 2019): local CATE estimation via honest trees; used for uplift targeting.

**Use cases:**
- Uplift targeting with local CATE
- Subgroup policy for uplift

**Config:** `{'n_trees': 'int', 'honesty': 'bool'}`

**Benchmarks:**
- **[literature]** cadw_internal → qini_coefficient=null
  - algorithm: `CausalForest` (paper)
  - number: .pingis/benches/
  - code/paper: https://arxiv.org/abs/1510.04342
  - note: no_public_benchmark: causal forest uplift per-dataset

### DR_learner_uplift — sota / tier2_wrapper

**Roles:** probabilistic
**Status:** desc, export

**Purpose:** Doubly-robust meta-learner for uplift; propensity + outcome models give semiparametric efficiency.

**Use cases:**
- Semiparametric uplift estimation
- Robust Qini vs misspecification

**Config:** `{'n_estimators': 'int', 'propensity': 'logistic'}`

**Benchmarks:**
- **[literature]** cadw_internal → qini_coefficient=null
  - algorithm: `DR_learner_uplift` (paper)
  - number: .pingis/benches/
  - code/paper: https://arxiv.org/abs/1712.05898
  - note: no_public_benchmark: DR-learner per-dataset

## Metrics
- **qini** (anchor, higher)
  - Area between Qini curve and random-targeting baseline.
  - *special:* Noisy at n_test<2000; sample std ~sqrt(qini*(1-qini)/n); report SE.
- **auuc** (pareto_axis, higher)
  - Area Under Uplift Curve; alternative to Qini.
  - *special:* Monotone transform of Qini under uniform weighting; use one, not both.

## Ensemble rules
- **[nonlinear_interactions]** → T_learner_uplift + CausalTree (max_per_role=2)
  - Causal tree wins on high-dim noise / deep interactions / large n. On simple DGPs parity. See d3_3_uplift.md.
