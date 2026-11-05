# T1 — Bayesian optimization

*Section:* T — Optimization

**Description:** Model the objective with a surrogate (GP / TPE) and select next point by acquisition. Sample-efficient for expensive black-box.

**Real-life examples:**
- Hyperparameter search
- Simulation-based design
- Neural-architecture search

## Benchmarks
- **HPOBench** — https://github.com/automl/HPOBench
  - metrics: best_found_value, n_evals_to_target

## Models
### GP-EI — baseline / tier2_wrapper

**Roles:** probabilistic
**Status:** desc, export

**Purpose:** Mockus 1975: Gaussian-process surrogate + Expected Improvement acquisition. Sample-efficient for low-dim spaces; O(n^3) GP fit limits n_evals.

**Use cases:**
- Continuous HPO
- Expensive simulation
- Low-dim design

**Config:** `{'n_initial': 'int', 'n_iter': 'int', 'kernel': 'string'}`

**Benchmarks:**
- **[literature]** HPOBench → best_found_value=null
  - algorithm: `skopt.gp_minimize` (docs)
  - number: https://github.com/automl/HPOBench
  - code/paper: https://scikit-optimize.github.io/
  - note: HPOBench suites; GP-EI is canonical baseline.

### TPE — sota / tier2_wrapper

**Roles:** probabilistic, local_structure
**Status:** desc, export

**Purpose:** Bergstra et al. 2011: Tree-structured Parzen Estimator. Density-ratio surrogate; handles categorical / conditional spaces better than GP. Optuna default.

**Use cases:**
- Mixed-type HPO
- Conditional search spaces
- Optuna production

**Config:** `{'n_initial': 'int', 'n_iter': 'int', 'gamma': 'float'}`

**Benchmarks:**
- **[literature]** HPOBench → best_found_value=null
  - algorithm: `optuna.samplers.TPESampler` (docs)
  - number: https://github.com/automl/HPOBench
  - code/paper: https://optuna.readthedocs.io/
  - note: Bergstra 2011; Optuna implementation.

### SMAC — sota / tier2_wrapper

**Roles:** probabilistic
**Status:** desc, export

**Purpose:** Hutter et al. 2011: Sequential Model-based Algorithm Configuration. Random forest surrogate; robust for categorical/conditional spaces.

**Use cases:**
- Algorithm configuration
- Mixed-type HPO

**Config:** `{'n_initial': 'int', 'n_iter': 'int', 'intensification': 'float'}`

**Benchmarks:**
- **[literature]** HPOBench → best_found_value=null
  - algorithm: `smac.SMAC4HPO` (docs)
  - number: https://github.com/automl/HPOBench
  - code/paper: https://automl.github.io/SMAC3/
  - note: Hutter 2011; SMAC3 implementation.

## Metrics
- **best_found_value** (anchor, higher)
  - Best objective value found at fixed eval budget.
  - *special:* Direction depends on optimization sign; report as-is with sign convention.
- **n_evals_to_target** (diagnostic, lower)
  - Number of evals to reach target value; sample-efficiency measure.
  - *special:* Requires predefined target; not always available.
