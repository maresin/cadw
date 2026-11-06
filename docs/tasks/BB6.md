# B6 — Ensemble clustering

*Section:* B — Clustering

**Description:** Consensus of multiple clustering runs (different algorithms/seeds/k).

**Real-life examples:**
- Robust segmentation
- Flag ambiguous points (disagreement)

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: adjusted_rand_score, agreement_fraction

## Models
### KMeansOnly — baseline / tier1_core

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Single-run KMeans baseline: what consensus must beat. No cross-algorithm agreement.

**Use cases:**
- Simple segmentation baseline

**Config:** `{'k': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → adjusted_rand_score=null
  - algorithm: `sklearn.cluster.KMeans` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
  - note: baseline for consensus comparison.

### KMeansHDBSCAN_consensus — sota / tier1_core

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Consensus of centroid (KMeans) and density (HDBSCAN) partitions. Different biases -> robust agreement core; disagreement signals ambiguous points.

**Use cases:**
- Robust segmentation
- Flag ambiguous points

**Config:** `{'kmeans_k': 'int', 'hdbscan_min_cluster_size': 'int', 'agreement_method': 'string'}`

**Benchmarks:**
- **[literature]** cadw_internal → adjusted_rand_score=null
  - algorithm: `consensus (KMeans + HDBSCAN) -- Strehl & Ghosh 2002 framework` (paper)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://www.sciencedirect.com/science/article/abs/pii/S0031320301001751
  - note: no_public_benchmark: custom ensemble; ARI vs each base partition. Strehl & Ghosh 2002 consensus-clustering framework.

## Metrics
- **adjusted_rand_score** (anchor, higher)
  - Agreement with ground truth (synthetic) or each base partition.
  - *special:* Requires labels or a reference partition.
- **agreement_fraction** (diagnostic, target)
  - Fraction of points where all runs agree.
  - *special:* Higher agreement = more confident core; disagreement = ambiguous.
