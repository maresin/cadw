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

**Benchmark:** LETOR → ndcg_at_k=0.42

### LambdaMART — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** GBDT with lambda gradients (Burges 2010); SOTA on LTR benchmarks.

**Use cases:**
- Bing search ranking
- Kaggle LTR competitions

**Config:** `{'n_estimators': 'int', 'max_depth': 'int', 'learning_rate': 'float'}`

**CADW:** `cadw.ml.lambdamart.fit_lambdamart`

**Benchmark:** LETOR → ndcg_at_k=0.51

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
