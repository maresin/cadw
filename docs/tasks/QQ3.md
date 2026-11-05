# Q3 — Manifold learning

*Section:* Q — Dimensionality reduction

**Description:** Non-linear dim reduction that preserves local neighborhoods or topology.

**Real-life examples:**
- Visualization (k=2)
- Non-linear feature extraction
- Clustering preprocessing

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: trustworthiness, fit_time_s

## Models
### t-SNE — baseline / tier2_wrapper

**Roles:** representation, local_structure
**Status:** desc, export

**Purpose:** van der Maaten & Hinton 2008: t-distributed Stochastic Neighbor Embedding. Preserves local neighborhoods via KL divergence between similarity distributions. Strong at 2D visualization.

**Use cases:**
- 2D visualization
- Local-structure discovery
- Cluster inspection

**Config:** `{'n_components': 'int', 'perplexity': 'float', 'max_iter': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → trustworthiness=null
  - algorithm: `sklearn.manifold.TSNE` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
  - note: no_public_benchmark: dim-reduction not on AMLB.

### UMAP — sota / tier2_wrapper

**Roles:** representation, local_structure
**Status:** desc, export

**Purpose:** McInnes et al. 2018: Uniform Manifold Approximation. Fuzzy topological representation; faster than t-SNE and preserves both local and some global structure.

**Use cases:**
- Fast visualization (n > 10^5)
- Local + global structure
- Preprocessing for clustering

**Config:** `{'n_components': 'int', 'n_neighbors': 'int', 'min_dist': 'float'}`

**Benchmarks:**
- **[literature]** cadw_internal → trustworthiness=null
  - algorithm: `umap.UMAP` (paper)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://arxiv.org/abs/1802.03426
  - note: McInnes et al. 2018.

## Metrics
- **trustworthiness** (anchor, higher)
  - Agreement between neighbor ranks in high-D and low-D; local-structure preservation.
  - *special:* n_neighbors parameter sensitive; report across k.
- **fit_time_s** (diagnostic, lower)
  - Wall-clock fit time.
  - *special:* Hardware-specific; t-SNE vs UMAP speed differs ~10x.
