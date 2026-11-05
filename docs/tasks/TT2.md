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

## Metrics
- **best_found_value** (anchor, higher)
  - Best objective value at fixed eval budget.
  - *special:* Direction depends on optimization sign.
