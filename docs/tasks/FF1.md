# F1 — Collaborative filtering

*Section:* F — Recommender Systems

**Description:** Predict user-item preference from interaction matrix alone.

**Real-life examples:**
- Movie recommendation
- Product recommendation
- Song/playlist

## Benchmarks
- **Amazon_Recsys** — https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
  - metrics: recall_at_k, ndcg_at_k

## Models
### ItemKNN — baseline / tier3_export

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Item-item cosine similarity; classic non-personalized baseline.

**Use cases:**
- E-commerce 'you might also like'

**Config:** `{'k': 'int', 'metric': 'str'}`

**Benchmark:** Amazon_Recsys → ndcg_at_k=0.18

### ALS — sota / tier1_core

**Roles:** representation
**Status:** ✓ cadw, ✓ agent

**Purpose:** Alternating Least Squares on implicit feedback; scalable to millions.

**Use cases:**
- Spotify-like playlist recommendation

**Config:** `{'n_factors': 'int', 'n_iter': 'int', 'reg': 'float'}`

**CADW:** `cadw.ml.als.fit_als`

**Benchmark:** Amazon_Recsys → ndcg_at_k=0.23

## Metrics
- **ndcg_at_k** (anchor, higher)
  - Position-weighted ranking quality in top-K.
  - *special:* Compare only within the same K and the same negative sampling.
- **recall_at_k** (pareto_axis, higher)
  - Hit rate in top-K, unweighted.
  - *special:* Higher when K grows; not comparable across K.

## Ensemble rules
- **[representation]** → ALS (max_per_role=1)
  - Single model — ItemKNN is tier3_export (reference only).
