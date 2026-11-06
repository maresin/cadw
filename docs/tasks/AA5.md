# A5 — Ordinal classification

*Section:* A — Classification & Regression

**Description:** Predict ranked classes (e.g. ratings 1-5).

**Real-life examples:**
- Star ratings
- Pain severity 1-10
- Satisfaction surveys

## Benchmarks
- **AMLB** — https://openml.github.io/automlbenchmark/
  - metrics: accuracy

## Models
### OrdinalLogit — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw, ✓ agent

**Purpose:** Cumulative-link logistic; respects class order.

**Use cases:**
- Satisfaction survey prediction

**Config:** `{'alpha': 'float'}`

**CADW:** `cadw.ml.ordinal.fit_ordinal_logit`

**Benchmarks:**
- **[literature]** AMLB → accuracy=0.55
  - algorithm: `mord.LogisticAT` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://pythonhosted.org/mord/
  - note: exact-match accuracy [legacy v0.9.44, migrated v0.9.50]

### GBDT_threshold — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** desc

**Purpose:** GBDT on cumulative logits; SOTA when relations are nonlinear.

**Use cases:**
- Risk bands in finance

**Config:** `{'n_estimators': 'int', 'max_depth': 'int'}`

**Benchmarks:**
- **[literature]** AMLB → accuracy=0.6
  - algorithm: `sklearn.ensemble.GradientBoostingClassifier` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
  - note: [legacy v0.9.44, migrated v0.9.50]

### OrderedProbitBayesian — sota / tier2_wrapper

**Roles:** probabilistic
**Status:** desc, export

**Purpose:** Bayesian ordered probit: latent Gaussian + cutpoints; full posterior over thresholds.

**Use cases:**
- Ordinal survey responses
- Severity grading

**Config:** `{'n_draws': 'int', 'prior_scale': 'float'}`

**Benchmarks:**
- **[literature]** cadw_internal → mae_ordinal=null
  - algorithm: `OrderedProbitBayesian` (paper)
  - number: .pingis/benches/
  - code/paper: https://arxiv.org/abs/2104.05494
  - note: no_public_benchmark: ordered probit per-dataset

## Metrics
- **mae_ordinal** (anchor, lower)
  - Distance-aware accuracy; respects ordered classes.
  - *special:* Preferred to accuracy; a 1→2 error is less costly than 1→5.
- **accuracy** (diagnostic, higher)
  - Exact-match accuracy.
  - *special:* Ignores distance; misleads when ordinal distance matters.

## Ensemble rules
- **[linear_signal, nonlinear_interactions]** → OrdinalLogit (max_per_role=1)
  - Single model — GBDT-threshold is described_only, wait for verification.
