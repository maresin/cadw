# R1 — Frequent itemset mining

*Section:* R — Association rule mining

**Description:** Discover itemsets appearing together above a frequency threshold. Basis for association rules (A -> B).

**Real-life examples:**
- Market basket analysis
- Cross-sell recommendations
- Web clickstream co-occurrence

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: mean_lift, runtime_s

## Models
### Apriori — baseline / tier2_wrapper

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Agrawal & Srikant 1994: level-wise candidate generation with downward-closure pruning. Simple, but repeated full scans; slow when max-itemset is large.

**Use cases:**
- Market basket analysis
- Cross-sell rules
- Small dense transactions

**Config:** `{'min_support': 'float', 'min_confidence': 'float', 'max_len': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → mean_lift=null
  - algorithm: `mlxtend.frequent_patterns.apriori` (docs)
  - number: https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - code/paper: https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
  - note: no_public_benchmark: no unified ARM LB. mlxtend reference.

### Eclat — baseline / tier2_wrapper

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Zaki 2000: depth-first vertical tid-set intersection. Equivalent to Apriori output; fewer scans when transactions sparse. DFS helps memory in some regimes.

**Use cases:**
- Sparse transactions
- Vertical data layout
- Frequent itemset mining

**Config:** `{'min_support': 'float', 'max_len': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → mean_lift=null
  - algorithm: `pyclustering.cluster.eclat` (code)
  - number: https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - code/paper: https://github.com/annoviko/pyclustering
  - note: no_public_benchmark: no unified ARM LB.

### FP-growth — sota / tier2_wrapper

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Han et al. 2000: FP-tree construction + pattern growth. Two DB scans total; usually 1-2 orders faster than Apriori on dense data. Canonical SOTA for frequent itemsets.

**Use cases:**
- Dense transactions
- Large itemsets
- Production ARM

**Config:** `{'min_support': 'float', 'max_len': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → mean_lift=null
  - algorithm: `mlxtend.frequent_patterns.fpgrowth` (docs)
  - number: https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - code/paper: https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/
  - note: no_public_benchmark: Han et al. 2000 numbers per-dataset.

## Metrics
- **mean_lift** (anchor, higher)
  - Average lift of discovered rules; primary association quality.
  - *special:* Requires min_support threshold; report across 3 thresholds.
- **runtime_s** (diagnostic, lower)
  - Wall-clock fit time.
  - *special:* Hardware-specific; strongly dataset-dependent.
