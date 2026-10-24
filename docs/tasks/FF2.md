# F2 — Hybrid recommendation

*Section:* F — Recommender Systems

**Description:** Combine collaborative signal with item/content features.

**Real-life examples:**
- Cold-start products with text descriptions
- Bundle recommendation

## Benchmarks
- **Amazon_Recsys** — https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
  - metrics: ndcg_at_k, recall_at_k

## Models
### ALS — baseline / tier1_core

**Roles:** representation
**Status:** ✓ cadw, ✓ agent

**Purpose:** See F1; used as the collaborative tower of the hybrid.

**Use cases:**
- Collaborative baseline

**Config:** `{'n_factors': 'int', 'n_iter': 'int', 'reg': 'float'}`

**CADW:** `cadw.ml.als.fit_als`

**Benchmarks:**
- **[literature]** Amazon_Recsys → ndcg_at_k=0.23
  - algorithm: `ALS` (description)
  - number: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
  - note: [migrated v0.9.47b: algorithm/algorithm_source_url unverified]

### Hybrid_MF_GBDT — sota / tier1_core

**Roles:** representation, nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** ALS embeddings + content features fed to GBDT ranker.

**Use cases:**
- Amazon product recommendation with category features

**Config:** `{'n_factors': 'int', 'n_estimators': 'int'}`

**CADW:** `cadw.ml.hybrid_mf.fit_hybrid_mf`

**Benchmarks:**
- **[literature]** Amazon_Recsys → ndcg_at_k=0.27
  - algorithm: `Hybrid_MF_GBDT` (description)
  - number: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
  - note: [migrated v0.9.47b: algorithm/algorithm_source_url unverified]

## Metrics
- **ndcg_at_k** (anchor, higher)
  - Ranking quality in top-K.
  - *special:* Report per-K; not comparable across K.
- **coverage** (diagnostic, higher)
  - Fraction of catalog surfaced in top-K; diversity indicator.
  - *special:* Do not combine with ndcg into a scalar; Pareto axis.

## Ensemble rules
- **[representation, nonlinear_interactions]** → ALS + Hybrid_MF_GBDT (max_per_role=2)
  - Hybrid subsumes ALS; use ALS only when content features are sparse.
