# Q2 — Matrix factorization

*Section:* Q — Dimensionality reduction

**Description:** Factor X ≈ W·H with non-negativity (NMF), implicit-feedback (ALS), or general constraints.

**Real-life examples:**
- Topic modeling (docs × words)
- Recommender factorization (users × items)
- Parts-based representation

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: reconstruction_error, fit_time_s

## Models
### NMF — baseline / tier2_wrapper

**Roles:** representation, probabilistic
**Status:** desc, export

**Purpose:** Lee & Seung 1999: non-negative matrix factorization X ≈ W·H with W,H ≥ 0. Parts-based representation; interpretable components. Canonical CADW: .pingis/impl/J/nmf.py (reference impl), J_nlp.J3.NMF card.

**Use cases:**
- Topic modeling (docs × words)
- Parts-based image decomposition
- Interpretable dim reduction

**Config:** `{'n_components': 'int', 'max_iter': 'int', 'init': 'random|nndsvd'}`

**Benchmarks:**
- **[literature]** cadw_internal → reconstruction_error=null
  - algorithm: `sklearn.decomposition.NMF` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
  - note: no_public_benchmark: dim-reduction not on AMLB. Reference impl .pingis/impl/J/nmf.py; card J_nlp.J3.NMF.

### ALS — sota / tier1_core

**Roles:** representation, local_structure
**Status:** desc, export

**Purpose:** Hu et al. 2008: Alternating Least Squares for implicit-feedback matrices. Solves weighted least squares per user/item iteratively. Canonical CADW: cadw.ml.als.fit_als (verified in F_recsys.ALS).

**Use cases:**
- Implicit-feedback recommenders
- Large sparse matrix factorization
- Low-rank approximation

**Config:** `{'n_factors': 'int', 'regularization': 'float', 'iterations': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → reconstruction_error=null
  - algorithm: `implicit.als.AlternatingLeastSquares` (paper)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://arxiv.org/abs/1205.2618
  - note: Hu et al. 2008. CADW impl: fit_als (F_recsys).

## Metrics
- **reconstruction_error** (anchor, lower)
  - ||X - W·H||_F; quality of low-rank approximation.
  - *special:* Normalize by ||X||_F.
- **fit_time_s** (diagnostic, lower)
  - Wall-clock fit time.
  - *special:* Hardware-specific.
