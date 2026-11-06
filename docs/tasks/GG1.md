# G1 — Ranking (LTR)

*Section:* G — Learning to Rank

**Description:** Learn a scoring function that ranks items per query.

**Real-life examples:**
- Search ranking
- Recommendation re-rank
- Ad ranking

## Benchmarks
- **LETOR** — https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/
  - metrics: ndcg_at_k, map_at_k

## Models
### PairwiseLogistic — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ agent, desc

**Purpose:** Pairwise logistic on (item_a, item_b) preference pairs; simplest LTR baseline.

**Use cases:**
- Simple search re-rank

**Config:** `{'alpha': 'float', 'max_iter': 'int'}`

**Benchmarks:**
- **[literature]** LETOR → ndcg_at_k=0.42
  - algorithm: `sklearn.linear_model.LogisticRegression` (docs)
  - number: https://web.archive.org/web/20240101000000/https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  - note: [legacy v0.9.44, migrated v0.9.50] canonical microsoft.com LETOR page returns HTTP 403; using wayback 2024-01-01 snapshot

### LambdaMART — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** GBDT with lambda gradients (Burges 2010); SOTA on LTR benchmarks.

**Use cases:**
- Bing search ranking
- Kaggle LTR competitions

**Config:** `{'n_estimators': 'int', 'max_depth': 'int', 'learning_rate': 'float'}`

**CADW:** `cadw.ml.lambdamart.fit_lambdamart`

**Benchmarks:**
- **[literature]** LETOR → ndcg_at_k=0.51
  - algorithm: `LightGBM lambdarank` (code)
  - number: https://web.archive.org/web/20240101000000/https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/
  - code/paper: https://github.com/microsoft/LightGBM/tree/master/examples/lambdarank
  - note: [legacy v0.9.44, migrated v0.9.50] canonical microsoft.com LETOR page returns HTTP 403; using wayback 2024-01-01 snapshot

### ListNet — sota / tier2_wrapper

**Roles:** probabilistic
**Status:** desc, export

**Purpose:** ListNet (Cao 2007): Plackett-Luce probability of permutation; listwise probabilistic LTR.

**Use cases:**
- Whole-page ranking
- Query-document list

**Config:** `{'n_epochs': 'int', 'lr': 'float'}`

**Benchmarks:**
- **[literature]** cadw_internal → ndcg_at_10=null
  - algorithm: `ListNet` (paper)
  - number: .pingis/benches/
  - code/paper: https://arxiv.org/abs/2002.07416
  - note: no_public_benchmark: ListNet per-dataset

## Metrics
- **ndcg_at_k** (anchor, higher)
  - Position-weighted ranking quality in top-K.
  - *special:* Report per-K; not comparable across K or across different relevance labels.
- **map_at_k** (pareto_axis, higher)
  - Mean Average Precision in top-K.
  - *special:* Binary-relevance variant of NDCG; do not combine into a scalar.

## Ensemble rules
- **[linear_signal, nonlinear_interactions]** → PairwiseLogistic + LambdaMART (max_per_role=1)
  - PairwiseLogistic is described_only; ensemble is a spec for Phase F/H.
