# P3 — Hierarchical clustering

*Section:* P — Clustering

**Description:** Tree of nested clusters (dendrogram).

**Real-life examples:**
- Taxonomy induction
- Dendrogram visualization
- Cluster count exploration

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: cophenetic_correlation, fit_time_s

## Models
### AgglomerativeClustering — baseline / tier1_core

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Bottom-up hierarchical: merge closest pairs by linkage (ward/complete/average/single). O(n^2) memory naive.

**Use cases:**
- Dendrogram (cluster hierarchy)
- Taxonomy discovery
- Small n (< 10^4)

**Config:** `{'n_clusters': 'int', 'linkage': 'string', 'metric': 'string'}`

**Benchmarks:**
- **[literature]** cadw_internal → cophenetic_correlation=null
  - algorithm: `sklearn.cluster.AgglomerativeClustering` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
  - note: no_public_benchmark: no unified clustering LB.

### BIRCH — sota / tier1_core

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Zhang et al. 1996: Clustering Feature (CF) tree. Bounded memory; O(n) single-pass. Leaf CFs = microclusters. Scalable streaming hierarchical.

**Use cases:**
- Streaming clustering
- Large n bounded memory
- Pre-clustering for k-means

**Config:** `{'threshold': 'float', 'branching_factor': 'int', 'n_clusters': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → cophenetic_correlation=null
  - algorithm: `sklearn.cluster.Birch` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html
  - note: no_public_benchmark: CADW-native measurement in .pingis/benches/birch.json (fit_time_s).

## Metrics
- **cophenetic_correlation** (anchor, higher)
  - Preservation of pairwise distances in dendrogram.
  - *special:* Requires dendrogram; for flat methods (k-means) use silhouette.
- **fit_time_s** (diagnostic, lower)
  - Wall-clock fit time.
  - *special:* Hardware-specific.
