# P4 — Shape-based clustering (time series)

*Section:* P — Clustering

**Description:** Cluster series by shape, invariant to amplitude/offset.

**Real-life examples:**
- TS motif discovery
- Equipment operating mode clustering
- Patient trajectory phenotyping

## Benchmarks
- **UCR_Classification** — https://www.timeseriesclassification.com/
  - metrics: adjusted_rand_score

## Models
### EuclideanKMeansZ — baseline / tier1_core

**Roles:** local_structure, temporal_dynamics
**Status:** desc, export

**Purpose:** K-means on z-normalized series with Euclidean distance. Shape-aware baseline; ignores temporal shifts (unlike KShape).

**Use cases:**
- TS clustering (baseline)
- Shape-aware segmentation

**Config:** `{'n_clusters': 'int', 'max_iter': 'int'}`

**Benchmarks:**
- **[literature]** UCR_Classification → adjusted_rand_score=null
  - algorithm: `k-means on z-norm (naive baseline)` (paper)
  - number: https://www.timeseriesclassification.com/
  - code/paper: https://arxiv.org/abs/1506.07608
  - note: Baseline referenced in Paparrizos & Gravano 2015; naive Euclidean k-means on z-normalized series.

### KShape — sota / tier1_core

**Roles:** local_structure, temporal_dynamics
**Status:** desc, export

**Purpose:** Paparrizos & Gravano 2015: k-shape. Shape-based distance via cross-correlation; centroids as shape-averaged series. Invariant to amplitude and offset.

**Use cases:**
- TS motif clustering
- Equipment mode discovery
- Patient trajectory grouping

**Config:** `{'n_clusters': 'int', 'max_iter': 'int', 'tol': 'float'}`

**Benchmarks:**
- **[literature]** UCR_Classification → adjusted_rand_score=null
  - algorithm: `k-Shape (Paparrizos & Gravano 2015)` (paper)
  - number: https://www.timeseriesclassification.com/
  - code/paper: https://arxiv.org/abs/1506.07608
  - note: Paper reports ARI per UCR dataset; no unified LB. CADW TS-anomaly implementation in D_time_series (fit_kshape); clustering-task view here.

## Metrics
- **adjusted_rand_score** (anchor, higher)
  - Pairwise agreement with ground truth, chance-corrected.
  - *special:* Requires labels (UCR benchmark); use silhouette for unsupervised.
