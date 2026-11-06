# T2 — Derivative-free optimization

*Section:* T — Optimization

**Description:** Non-model-based or evolutionary; useful when the objective is noisy / non-stationary / high-dim.

**Real-life examples:**
- Hyperparameter search
- Noisy simulation optimization
- High-dim black-box

## Benchmarks
- **HPOBench** — https://github.com/automl/HPOBench
  - metrics: best_found_value, n_evals_to_target

## Models
### NelderMead — baseline / tier2_wrapper

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Nelder & Mead 1965: simplex reflection/expansion/contraction. No derivatives; local; fast in low-dim but poor in high-dim.

**Use cases:**
- Low-dim smooth optimization
- Cheap objective

**Config:** `{'xatol': 'float', 'fatol': 'float', 'maxiter': 'int'}`

**Benchmarks:**
- **[literature]** HPOBench → best_found_value=null
  - algorithm: `scipy.optimize.minimize(method="Nelder-Mead")` (docs)
  - number: https://github.com/automl/HPOBench
  - code/paper: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
  - note: scipy reference implementation.

### CMA-ES — sota / tier2_wrapper

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Hansen & Ostermeier 2001: Covariance Matrix Adaptation Evolution Strategy. Learns variable correlations; competitive with BO in mid-dim. Population-based.

**Use cases:**
- Mid-dim continuous optimization
- Noisy objective
- Non-convex

**Config:** `{'popsize': 'int', 'sigma0': 'float', 'maxiter': 'int'}`

**Benchmarks:**
- **[literature]** HPOBench → best_found_value=null
  - algorithm: `cma.CMAEvolutionStrategy` (code)
  - number: https://github.com/automl/HPOBench
  - code/paper: https://github.com/CMA-ES/pycma
  - note: Hansen 2001; pycma reference.

### BayesianOpt_GP — sota / tier2_wrapper

**Roles:** probabilistic
**Status:** desc, export

**Purpose:** GP surrogate + EI acquisition; sample-efficient HPO alternative to simplex/evolution methods.

**Use cases:**
- Sample-efficient HPO
- Expensive simulation optimization

**Config:** `{'n_initial': 'int', 'n_iter': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → best_found_value=null
  - algorithm: `BayesianOpt_GP` (paper)
  - number: .pingis/benches/
  - code/paper: https://arxiv.org/abs/1012.2599
  - note: no_public_benchmark: HPOBench suites (no single LB)

### SMAC — sota / tier2_wrapper

**Roles:** nonlinear_interactions
**Status:** desc, export

**Purpose:** Random-forest surrogate + intensification; nonlinear HPO surrogate for categorical/conditional.

**Use cases:**
- Algorithm configuration
- Categorical search space

**Config:** `{'n_initial': 'int', 'intensification': 'float'}`

**Benchmarks:**
- **[literature]** cadw_internal → best_found_value=null
  - algorithm: `SMAC` (docs)
  - number: .pingis/benches/
  - code/paper: https://automl.github.io/SMAC3/
  - note: no_public_benchmark: HPOBench suites

## Metrics
- **best_found_value** (anchor, higher)
  - Best objective value at fixed eval budget.
  - *special:* Direction depends on optimization sign.
