# C1 — Linear projection

*Section:* C — Dimensionality reduction

**Description:** Project data onto k linear directions that maximize variance (PCA) or a related objective. Interpretable, invertible.

**Real-life examples:**
- Preprocessing for downstream models
- Noise reduction / denoising
- Visualization (k=2)

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: reconstruction_error, explained_variance_ratio, fit_time_s

## Models
### PCA — baseline / tier1_core

**Roles:** representation
**Status:** ✓ cadw

**Purpose:** Pearson 1901: orthogonal linear projection maximizing variance. Eigendecomposition of covariance; exact for full SVD, randomized for large d. Canonical baseline.

**Use cases:**
- Feature preprocessing
- Denoising
- 2D visualization

**Config:** `{'n_components': 'int', 'method': 'auto|exact|randomized'}`

**CADW:** `cadw.ml.pca.fit_pca`

**Benchmarks:**
- **[literature]** cadw_internal → reconstruction_error=null
  - algorithm: `sklearn.decomposition.PCA` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
  - note: no_public_benchmark: dim-reduction not on AMLB; measure v0.9.124.

### TruncatedSVD — baseline / tier2_wrapper

**Roles:** representation
**Status:** desc, export

**Purpose:** SVD without centering (LSA-style). Unlike PCA, does not require mean removal; suited for sparse inputs.

**Use cases:**
- Sparse text embeddings
- LSA preprocessing
- No mean removal needed

**Config:** `{'n_components': 'int', 'algorithm': 'randomized|arpack'}`

**Benchmarks:**
- **[literature]** cadw_internal → reconstruction_error=null
  - algorithm: `sklearn.decomposition.TruncatedSVD` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
  - note: no_public_benchmark: dim-reduction not on AMLB.

### randomized_svd — sota / tier1_core

**Roles:** representation
**Status:** desc, export

**Purpose:** Halko et al. 2011: randomized SVD for large d. Approximate; O(d·k·p) vs O(d²·k) exact. Canonical CADW implementation: cadw.ml.randsvd.fit_randomized_svd (verified in J_nlp.J2.LSA_TFIDF_SVD).

**Use cases:**
- Large-d PCA
- Approximate SVD for sparse / low-rank
- Truncated LSA

**Config:** `{'n_components': 'int', 'n_oversamples': 'int', 'random_state': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → reconstruction_error=null
  - algorithm: `sklearn.utils.extmath.randomized_svd` (paper)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://arxiv.org/abs/0909.4061
  - note: Halko et al. 2011. CADW impl: fit_randomized_svd (J_nlp J2).

### KernelPCA — sota / tier2_wrapper

**Roles:** nonlinear_interactions
**Status:** desc, export

**Purpose:** Kernel PCA (Scholkopf 1998): nonlinear projection via kernel-induced feature map.

**Use cases:**
- Nonlinear feature compression
- Denoising with kernel ridge

**Config:** `{'n_components': 'int', 'kernel': 'string'}`

**Benchmarks:**
- **[literature]** cadw_internal → reconstruction_error=null
  - algorithm: `KernelPCA` (docs)
  - number: .pingis/benches/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
  - note: no_public_benchmark: kernel PCA per-dataset

### SparsePCA — sota / tier2_wrapper

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Sparse PCA (Zou 2006): elastic-net-regularized components; local (sparse) loading structure.

**Use cases:**
- Interpretable feature projection
- Sparse loadings for report

**Config:** `{'n_components': 'int', 'alpha': 'float'}`

**Benchmarks:**
- **[literature]** cadw_internal → reconstruction_error=null
  - algorithm: `SparsePCA` (docs)
  - number: .pingis/benches/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html
  - note: no_public_benchmark: sparse PCA per-dataset

## Metrics
- **reconstruction_error** (anchor, lower)
  - Frobenius norm of (X - X_reconstructed); lower = better preservation.
  - *special:* Scale-dependent; normalize by ||X||_F for cross-dataset comparability.
- **explained_variance_ratio** (diagnostic, higher)
  - Fraction of total variance in top-k components.
  - *special:* Only meaningful for PCA (centered); TruncatedSVD lacks center.
