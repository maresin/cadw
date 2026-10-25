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

**Benchmarks:**
- **[literature]** Amazon_Recsys → ndcg_at_k=0.18
  - algorithm: `sklearn.neighbors.NearestNeighbors` (docs)
  - number: https://web.archive.org/web/20240101000000/https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
  - note: [legacy v0.9.44, migrated v0.9.50] canonical cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ currently refuses connection; using wayback 2024-01-01 snapshot

### ALS — sota / tier1_core

**Roles:** representation
**Status:** ✓ cadw, ✓ agent

**Purpose:** Alternating Least Squares on implicit feedback; scalable to millions.

**Use cases:**
- Spotify-like playlist recommendation

**Config:** `{'n_factors': 'int', 'n_iter': 'int', 'reg': 'float'}`

**CADW:** `cadw.ml.als.fit_als`

**Benchmarks:**
- **[literature]** Amazon_Recsys → ndcg_at_k=0.23
  - algorithm: `implicit.als.AlternatingLeastSquares` (code)
  - number: https://web.archive.org/web/20240101000000/https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
  - code/paper: https://github.com/benfred/implicit
  - note: [legacy v0.9.44, migrated v0.9.50] canonical cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ currently refuses connection; using wayback 2024-01-01 snapshot
- **[cadw_reproduced]** Amazon_Recsys → ndcg_at_k=0.3827250969586643 (n=1)
  - algorithm: `cadw.ml.als.fit_als` (code)
  - reproduces: `implicit.als.AlternatingLeastSquares`
  - number: .pingis/benches/reproduction/als.json
  - code/paper: cadw/ml/als.py
  - note: M2 parity on synthetic_recsys_like DGP; cadw=0.38273, ref(implicit)=0.36734, verdict=consistent; NDCG@10 on 20% holdout, user-item matrix ~10% density

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
