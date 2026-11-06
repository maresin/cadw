# B2 — Density-based clustering

*Section:* B — Clustering

**Description:** Arbitrary-shaped clusters as dense regions; noise points labelled -1; k not needed.

**Real-life examples:**
- Unknown-k segmentation
- Anomaly via noise label
- Spatial clustering

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: adjusted_rand_score, noise_ratio, fit_time_s

## Models
### DBSCAN — baseline / tier2_wrapper

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Ester et al. 1996: clusters as dense regions via eps-neighborhood + min_pts. Flat; single global eps struggles with varying density.

**Use cases:**
- Spatial clustering
- Anomaly (noise points)
- No k required

**Config:** `{'eps': 'float', 'min_samples': 'int', 'metric': 'string'}`

**Benchmarks:**
- **[literature]** cadw_internal → adjusted_rand_score=null
  - algorithm: `sklearn.cluster.DBSCAN` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
  - note: no_public_benchmark: no unified clustering LB.

### HDBSCAN — sota / tier2_wrapper

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Campello et al. 2013: hierarchical DBSCAN. Mutual-reachability MST + condensed cluster tree. No global eps; handles varying density.

**Use cases:**
- Unknown-k segmentation
- Anomaly via noise label
- Spatial clustering with varying density

**Config:** `{'min_cluster_size': 'int', 'min_samples': 'int', 'metric': 'string'}`

**Benchmarks:**
- **[literature]** cadw_internal → adjusted_rand_score=null
  - algorithm: `hdbscan.HDBSCAN` (code)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://github.com/scikit-learn-contrib/hdbscan
  - note: no_public_benchmark: Campello 2013 per-dataset numbers. Reference impl in .pingis/impl/N/HDBSCAN.txt.

### GaussianMixtureClustering — sota / tier2_wrapper

**Roles:** probabilistic
**Status:** desc, export

**Purpose:** Density view via GMM; contrasts with DBSCAN/HDBSCAN (no explicit p(x)).

**Use cases:**
- Density-mode discovery
- Soft density cluster

**Config:** `{'n_components': 'int', 'covariance_type': 'string'}`

**Benchmarks:**
- **[literature]** cadw_internal → adjusted_rand_score=null
  - algorithm: `GaussianMixtureClustering` (docs)
  - number: .pingis/benches/
  - code/paper: https://scikit-learn.org/stable/modules/mixture.html
  - note: no_public_benchmark: GMM clustering per-dataset

### SpectralClustering — sota / tier2_wrapper

**Roles:** representation
**Status:** desc, export

**Purpose:** Graph-based density clustering: mode-seeking via spectral eigenvectors of similarity graph.

**Use cases:**
- Arbitrary-shape density cluster
- Manifold density

**Config:** `{'n_clusters': 'int', 'affinity': 'string'}`

**Benchmarks:**
- **[literature]** cadw_internal → adjusted_rand_score=null
  - algorithm: `SpectralClustering` (paper)
  - number: .pingis/benches/
  - code/paper: https://papers.nips.cc/paper/2001/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html
  - note: no_public_benchmark: spectral clustering per-dataset

## Metrics
- **adjusted_rand_score** (anchor, higher)
  - Pairwise agreement with ground truth, chance-corrected.
  - *special:* Requires labels; use silhouette for unsupervised mode.
- **noise_ratio** (diagnostic, target)
  - Fraction of points labelled -1.
  - *special:* Domain-dependent; high noise may signal wrong eps.
