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
  - algorithm: `implicit.als.AlternatingLeastSquares` (code)
  - number: https://web.archive.org/web/20240101000000/https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
  - code/paper: https://github.com/benfred/implicit
  - note: [legacy v0.9.44, migrated v0.9.50] canonical cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ currently refuses connection; using wayback 2024-01-01 snapshot
- **[cadw_reproduced]** cadw_internal → ndcg_at_k=0.3827250969586643 (n=1)
  - algorithm: `cadw.ml.als.fit_als` (code)
  - reproduces: `implicit.als.AlternatingLeastSquares`
  - number: .pingis/benches/reproduction/als.json
  - code/paper: cadw/ml/als.py
  - note: M2 parity on synthetic_recsys_like DGP; cadw=0.38273, ref(implicit)=0.36734, verdict=consistent; NDCG@10 on 20% holdout, user-item matrix ~10% density

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
  - algorithm: `sklearn.ensemble.HistGradientBoostingRegressor` (docs)
  - number: https://web.archive.org/web/20240101000000/https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
  - note: [legacy v0.9.44, migrated v0.9.50] canonical cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ currently refuses connection; using wayback 2024-01-01 snapshot

### SVDpp — sota / tier2_wrapper

**Roles:** local_structure
**Status:** desc, export

**Purpose:** SVD++ (Koren 2008): MF + implicit-neighborhood signals; local structure of user-item graph.

**Use cases:**
- Implicit-feedback recsys
- Neighborhood-augmented MF

**Config:** `{'n_factors': 'int', 'n_epochs': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → ndcg_at_k=null
  - algorithm: `SVDpp` (paper)
  - number: .pingis/benches/
  - code/paper: https://arxiv.org/abs/0804.0861
  - note: no_public_benchmark: SVD++ per-dataset

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
