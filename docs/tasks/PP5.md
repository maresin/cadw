# P5 — Biclustering

*Section:* P — Clustering

**Description:** Cluster rows AND columns simultaneously.

**Real-life examples:**
- Gene expression (samples × genes)
- Recommender co-clustering
- Text co-clustering

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: mean_squared_residue, fit_time_s

## Models
### SpectralCoclustering — baseline / tier2_wrapper

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Dhillon 2001: spectral co-clustering on checkerboard patterns. SVD on normalized matrix; rows and cols cluster together.

**Use cases:**
- Gene expression submatrices
- Users × items co-clusters

**Config:** `{'n_clusters': 'int', 'svd_method': 'string'}`

**Benchmarks:**
- **[literature]** cadw_internal → mean_squared_residue=null
  - algorithm: `sklearn.cluster.SpectralCoclustering` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralCoclustering.html
  - note: no_public_benchmark: no unified biclustering LB.

### SpectralBiclustering — sota / tier2_wrapper

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Kluger 2003: generalized spectral biclustering. Handles checkerboard + non-checkerboard patterns; rows and cols cluster independently after SVD.

**Use cases:**
- Gene expression submatrices
- General biclustering

**Config:** `{'n_clusters': 'tuple', 'method': 'string', 'svd_method': 'string'}`

**Benchmarks:**
- **[literature]** cadw_internal → mean_squared_residue=null
  - algorithm: `sklearn.cluster.SpectralBiclustering` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralBiclustering.html
  - note: no_public_benchmark: no unified biclustering LB.

## Metrics
- **mean_squared_residue** (anchor, lower)
  - Coherence of bicluster submatrices (lower = tighter).
  - *special:* Depends on data scale; report per-dataset.
- **fit_time_s** (diagnostic, lower)
  - Wall-clock fit time.
  - *special:* Hardware-specific.
