# B1 — Centroid-based clustering

*Section:* B — Clustering

**Description:** Partition points into k clusters by centroid proximity.

**Real-life examples:**
- Customer segmentation (k known)
- Vector quantization
- Anomaly via distance-to-centroid

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: silhouette, inertia, fit_time_s

## Models
### KMeans — baseline / tier1_core

**Roles:** local_structure
**Status:** ✓ cadw

**Purpose:** Lloyd k-means with k-means++ init. Spherical equal-variance clusters; centroid objective. Canonical baseline.

**Use cases:**
- Customer segmentation (k known)
- Vector quantization
- Anomaly via distance to nearest centroid

**Config:** `{'k': 'int', 'n_init': 'int', 'max_iter': 'int', 'tol': 'float', 'seed': 'int'}`

**CADW:** `cadw.ml.kmeans.fit_kmeans`

**Benchmarks:**
- **[literature]** cadw_internal → inertia=null
  - algorithm: `sklearn.cluster.KMeans` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
  - note: no_public_benchmark: KMeans on AMLB is pre-processing; no unified clustering LB. Measure v0.9.124.

### MiniBatchKMeans — baseline / tier1_core

**Roles:** local_structure
**Status:** ✓ cadw

**Purpose:** Sculley 2010 streaming k-means: per-centroid updates with eta = 1/count. Reservoir init + streaming sweeps.

**Use cases:**
- Large-scale segmentation (n > 10^6)
- Online clustering with drift

**Config:** `{'n_clusters': 'int', 'batch_size': 'int', 'max_iter': 'int', 'tol': 'float', 'seed': 'int'}`

**CADW:** `cadw.ml.minibatch.fit_minibatch_kmeans`

**Benchmarks:**
- **[literature]** cadw_internal → inertia=null
  - algorithm: `sklearn.cluster.MiniBatchKMeans` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
  - note: no_public_benchmark: same as KMeans; measure v0.9.124.

### BisectingKMeans — sota / tier1_core

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Top-down hierarchical k-means: recursively bisect each cluster via 2-means. Better for non-convex shapes than Lloyd.

**Use cases:**
- Taxonomy induction
- Document clustering
- Non-spherical clusters

**Config:** `{'n_clusters': 'int', 'bisecting_strategy': 'string'}`

**Benchmarks:**
- **[literature]** cadw_internal → inertia=null
  - algorithm: `sklearn.cluster.BisectingKMeans` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html
  - note: no_public_benchmark: not on AMLB.

## Metrics
- **silhouette** (anchor, higher)
  - Unsupervised cluster separation; primary diagnostic without labels.
  - *special:* O(n^2) naive; sample-based for large n. Not comparable across k.
- **inertia** (diagnostic, lower)
  - Within-cluster sum of squared distances (KMeans objective).
  - *special:* Decreases with k; only comparable at same k.
