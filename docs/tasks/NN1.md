# N1 — Unsupervised density

*Section:* N — Density estimation

**Description:** Estimate p(x) for tabular data; score new samples and detect low-density regions.

**Real-life examples:**
- Novelty / anomaly detection via low-likelihood
- Generative sampling for simulation
- Cohort modeling / mixture discovery

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: fit_time_s, peak_rss_mb, mean_log_likelihood, n_clusters

## Models
### GaussianMixture — baseline / tier1_core

**Roles:** probabilistic
**Status:** ✓ cadw

**Purpose:** EM-fitted mixture of Gaussians. Inductive bias: data are drawn from a fixed number of Gaussian components — strong when clusters are approximately ellipsoidal and K is known; the canonical parametric density baseline.

**Use cases:**
- Gaussian-cluster customer segmentation
- Anomaly detection via low likelihood
- Synthetic sampling for simulation

**Config:** `{'n_components': 'int', 'covariance_type': 'string', 'random_state': 'int', 'max_iter': 'int'}`

**CADW:** `cadw.ml.gmm.fit_gmm`

**Benchmarks:**
- **[cadw_measured]** cadw_internal → mean_log_likelihood=-7.095 ± 0.0005 (n=2)
  - algorithm: `cadw.ml.gmm` (code)
  - number: .pingis/benches/gmm.json
  - code/paper: cadw/ml/gmm.py
  - note: CADW-native: mean_ll=-7.095 @n=100k and -7.096 @n=1M (scripts/bench_gmm.py run 2026-10-24, .pingis/benches/gmm.json); std=half-range; NOT comparable to external LB.
- **[cadw_measured]** cadw_internal → fit_time_s=3.258 ± 1.444 (n=2)
  - algorithm: `cadw.ml.gmm` (code)
  - number: .pingis/benches/gmm.json
  - code/paper: cadw/ml/gmm.py
  - note: CADW-native: fit_s=0.37 @n=100k and 3.258 @n=1M (scripts/bench_gmm.py, .pingis/benches/gmm.json); std=half-range; hardware-specific.

### KernelDensity — baseline / tier2_wrapper

**Roles:** local_structure, probabilistic
**Status:** desc

**Purpose:** Non-parametric kernel density estimator; the density is a sum of kernel bumps at every training point. Inductive bias: smooth local structure with bandwidth-controlled smoothing — no parametric form, but O(n) scoring and fragile in high dimensions.

**Use cases:**
- 1-2D density visualization
- Anomaly scoring in low-d sensor streams
- Smooth residual distributions

**Config:** `{'kernel': 'string', 'bandwidth': 'float'}`

**Benchmarks:**
- **[literature]** cadw_internal → mean_log_likelihood=null
  - algorithm: `sklearn.neighbors.KernelDensity` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/density.html
  - note: no_public_benchmark: AMLB/OpenML measure supervised metrics only; no density log-likelihood leaderboard

### BIRCH — baseline / tier1_core

**Roles:** local_structure, representation
**Status:** ✓ cadw, ✓ agent

**Purpose:** Balanced iterative reducing and clustering using hierarchies (Zhang 1996). Inductive bias: cluster-feature tree with radius threshold; density is approximated by per-leaf counts. Not a true density model but a scalable cluster-based density proxy.

**Use cases:**
- Streaming cluster discovery
- Large-catalog bucketing
- Approximate density for n>>memory

**Config:** `{'threshold': 'float', 'branching_factor': 'int', 'n_clusters': 'int'}`

**CADW:** `cadw.ml.birch.fit_birch`

**Benchmarks:**
- **[cadw_measured]** cadw_internal → fit_time_s=3.388 ± 1.551 (n=3)
  - algorithm: `cadw.ml.birch` (code)
  - number: .pingis/benches/birch.json
  - code/paper: cadw/ml/birch.py
  - note: CADW-native: fit_s=0.286/1.336/3.388 @n=5k/20k/50k (scripts/bench_birch.py, .pingis/benches/birch.json); std=half-range; hardware-specific.

### HDBSCAN — sota / tier2_wrapper

**Roles:** local_structure
**Status:** desc

**Purpose:** Hierarchical density-based clustering (Campello 2013). Inductive bias: clusters are contiguous high-density regions at variable density — automatically discovers K, robust to noise. Strong on irregular densities but no explicit p(x).

**Use cases:**
- Open-set clustering / noise detection
- Arbitrary-shape density modes
- Geospatial hot-spot detection

**Config:** `{'min_cluster_size': 'int', 'min_samples': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → mean_log_likelihood=null
  - algorithm: `hdbscan.HDBSCAN` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
  - note: no_public_benchmark: HDBSCAN does not output log-likelihood; no unified density leaderboard

### NormalizingFlow — sota / tier3_export

**Roles:** representation, probabilistic
**Status:** export

**Purpose:** Invertible neural network (RealNVP/MAF/Glow) that maps data to a latent Gaussian via a chain of bijections. Inductive bias: arbitrary smooth density with exact likelihood — SOTA in high dimensions but torch-only, reached only via to_batches(format='torch').

**Use cases:**
- High-d generative modeling
- Likelihood-based anomaly detection
- Tabular density for CATE posterior

**Config:** `{'flow_type': 'string', 'n_layers': 'int', 'hidden_sizes': 'list'}`

**Benchmarks:**
- **[literature]** cadw_internal → mean_log_likelihood=null
  - algorithm: `Normalizing flows (Papamakarios et al. 2021)` (paper)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://arxiv.org/abs/1912.02762
  - note: no_public_benchmark: flows reported per-dataset in papers; no unified public density LB on the AMLB task set

## Metrics
- **mean_log_likelihood** (anchor, higher)
  - Mean held-out log-likelihood; the canonical density-quality metric.
  - *special:* Scale-dependent: not comparable across dataset normalizations. Report median across runs when n_datasets>=2.
- **fit_time_s** (diagnostic, lower)
  - Wall-clock fit time.
  - *special:* Hardware-specific; only comparable within the same bench run.
- **peak_rss_mb** (diagnostic, lower)
  - Peak resident memory.
  - *special:* Includes Python + numpy overhead; not a pure model cost.

## Ensemble rules
- **[probabilistic, local_structure]** → GaussianMixture + BIRCH (max_per_role=2)
  - GMM gives parametric densities; BIRCH gives scalable cluster-based density proxy. Complementary when n spans 10k-1M; use GMM when K is known and parametric log-likelihood is needed.
