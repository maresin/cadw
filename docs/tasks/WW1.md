# W1 — Variational inference

*Section:* W — Probabilistic inference

**Description:** Approximate posterior q(theta) by optimizing ELBO. Fast, scalable; biased vs MCMC.

**Real-life examples:**
- Bayesian neural nets
- Large-data posteriors
- Latent variable models

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: elbo, runtime_s

## Models
### MeanFieldVI — baseline / tier2_wrapper

**Roles:** probabilistic
**Status:** desc, export

**Purpose:** Blei et al. 2017: mean-field VI. Factorized q = prod q_i(theta_i). Fast, closed-form coordinate ascent for conjugate models. Underestimates posterior variance.

**Use cases:**
- LDA topic models
- Fast Bayesian approximation
- Conjugate priors

**Config:** `{'max_iter': 'int', 'tol': 'float'}`

**Benchmarks:**
- **[literature]** cadw_internal → elbo=null
  - algorithm: `mean-field VI` (paper)
  - number: https://arxiv.org/abs/1601.00670
  - code/paper: https://arxiv.org/abs/1601.00670
  - note: Blei et al. 2017 tutorial; per-model ELBO only.

### BlackBoxVI — sota / tier2_wrapper

**Roles:** probabilistic, representation
**Status:** desc, export

**Purpose:** Ranganath et al. 2014: black-box VI. Score-function / reparameterization gradients for arbitrary models. Used in VAEs, Bayesian NN.

**Use cases:**
- VAE training
- Bayesian NN
- General model VI

**Config:** `{'n_samples': 'int', 'learning_rate': 'float', 'max_iter': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → elbo=null
  - algorithm: `black-box VI` (paper)
  - number: https://arxiv.org/abs/1401.0118
  - code/paper: https://arxiv.org/abs/1401.0118
  - note: Ranganath et al. 2014.

### SVI — sota / tier2_wrapper

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Stochastic Variational Inference (Hoffman 2013): minibatch VI with local steps; scalable to large data.

**Use cases:**
- Large-data Bayesian fit
- Streaming VI

**Config:** `{'n_steps': 'int', 'learning_rate': 'float'}`

**Benchmarks:**
- **[literature]** cadw_internal → elbo=null
  - algorithm: `SVI` (paper)
  - number: .pingis/benches/
  - code/paper: https://arxiv.org/abs/1303.2225
  - note: no_public_benchmark: SVI convergence per-model

## Metrics
- **elbo** (anchor, higher)
  - Evidence Lower BOund: VI objective; higher = better fit.
  - *special:* Only comparable within the same model; different models have different ELBO scales.
- **runtime_s** (diagnostic, lower)
  - Wall-clock fit time.
  - *special:* Hardware-specific.
