# O2 — Data cleaning / imputation

*Section:* O — Data Science outside ML (EDA / cleaning / feature engineering / validation)

**Description:** Fill missing values without leaking information across the split; preserve distributional shape.

**Real-life examples:**
- Sensor dropouts in telemetry
- Survey non-response
- Medical record gaps

## Benchmarks
- **sklearn_reference** — https://scikit-learn.org/stable/modules/impute.html
  - metrics: rmse, accuracy

## Models
### SimpleImputer — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw, ✓ agent

**Purpose:** Replace missing values with a column statistic (mean/median/mode) computed by Reduce. Inductive bias: missing-at-random per column; destroys inter-column correlation.

**Use cases:**
- Filling numeric gaps quickly
- Pre-model baseline
- Survey mode fill

**Config:** `{'strategy': 'string'}`

**CADW:** `cadw.dataframe.fillna`

**Benchmarks:**
- **[literature]** sklearn_reference → rmse=null
  - algorithm: `sklearn.impute.SimpleImputer` (docs)
  - number: https://scikit-learn.org/stable/modules/impute.html
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
  - note: no_public_benchmark: SimpleImputer is a reference primitive; no published per-dataset RMSE table exists.

### KNNImputer — baseline / tier1_core

**Roles:** local_structure
**Status:** desc

**Purpose:** Impute from k nearest neighbours in feature space. Inductive bias: local similarity predicts missing values; O(n²) naive, requires blocking.

**Use cases:**
- Filling correlated sensor gaps
- Recommendation-style fill
- Small-n survey imputation

**Config:** `{'n_neighbors': 'int'}`

**Benchmarks:**
- **[literature]** sklearn_reference → rmse=null
  - algorithm: `sklearn.impute.KNNImputer` (docs)
  - number: https://scikit-learn.org/stable/modules/impute.html
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
  - note: no_public_benchmark: KNNImputer docs example numbers are illustrative, not a benchmark ranking.

### IterativeImputer — sota / tier2_wrapper

**Roles:** linear_signal, representation
**Status:** desc

**Purpose:** Iteratively regress each feature on the others (MICE / EM-like). Inductive bias: linear cross-feature structure; captures correlations SimpleImputer loses.

**Use cases:**
- Clinical multi-feature gaps
- Survey with correlated items
- Sensor fusion with redundancy

**Config:** `{'max_iter': 'int', 'estimator': 'string'}`

**Benchmarks:**
- **[literature]** sklearn_reference → rmse=null
  - algorithm: `sklearn.impute.IterativeImputer` (docs)
  - number: https://scikit-learn.org/stable/modules/impute.html
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
  - note: no_public_benchmark: IterativeImputer ships no per-dataset RMSE table; it is the reference MICE implementation.

## Metrics
- **rmse** (anchor, lower)
  - RMSE of imputed vs held-out true values.
  - *special:* Only comparable within the same missingness mask and split; MCAR vs MAR matters.
- **accuracy** (diagnostic, higher)
  - Downstream classification accuracy after imputation.
  - *special:* Downstream-dependent; report the model used.

## Ensemble rules
- **[linear_signal, local_structure]** → SimpleImputer (max_per_role=2)
  - Use SimpleImputer as the fast fallback; escalate to KNNImputer only for small-n correlated gaps.
