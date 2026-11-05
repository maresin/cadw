# W2 — Markov Chain Monte Carlo

*Section:* W — Probabilistic inference

**Description:** Sample from exact posterior via Markov chains. Asymptotically unbiased; slower than VI.

**Real-life examples:**
- Gold-standard posteriors
- Small-data Bayesian inference
- Model comparison

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: ess, rhat

## Models
### MetropolisHastings — baseline / tier2_wrapper

**Roles:** probabilistic
**Status:** desc, export

**Purpose:** Metropolis et al. 1953: random-walk MH. Propose + accept/reject. Simple; tuning-sensitive; slow mixing in high dim.

**Use cases:**
- Simple posterior sampling
- Baseline MCMC

**Config:** `{'n_samples': 'int', 'proposal_std': 'float', 'burn_in': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → ess=null
  - algorithm: `Metropolis-Hastings` (paper)
  - number: https://arxiv.org/abs/1701.02434
  - code/paper: https://arxiv.org/abs/1701.02434
  - note: Metropolis 1953.

### NUTS — sota / tier2_wrapper

**Roles:** probabilistic
**Status:** desc, export

**Purpose:** Hoffman & Gelman 2014: No-U-Turn Sampler (adaptive HMC). Auto-tunes trajectory length; efficient in mid-dim. Stan/PyMC default.

**Use cases:**
- Gold-standard Bayesian inference
- Stan/PyMC production

**Config:** `{'n_samples': 'int', 'n_warmup': 'int', 'target_accept': 'float'}`

**Benchmarks:**
- **[literature]** cadw_internal → ess=null
  - algorithm: `NUTS` (paper)
  - number: https://arxiv.org/abs/1111.4246
  - code/paper: https://arxiv.org/abs/1111.4246
  - note: Hoffman & Gelman 2014; Stan/PyMC default.

## Metrics
- **ess** (anchor, higher)
  - Effective Sample Size: independent-sample equivalent; higher = better mixing.
  - *special:* Must be >= 100 * n_chains for reliable inference.
- **rhat** (diagnostic, target)
  - Potential Scale Reduction Factor across chains; ~1.0 = converged.
  - *special:* Target 1.0; values > 1.01 signal non-convergence.
