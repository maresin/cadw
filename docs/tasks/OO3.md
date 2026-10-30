# O3 — Feature engineering / encoding

*Section:* O — Data Science outside ML (EDA / cleaning / feature engineering / validation)

**Description:** Transform raw columns into model-ready features: categorical encoders, binning, polynomial expansions.

**Real-life examples:**
- Categorical columns to one-hot
- Continuous variables to bins
- Interaction features for linear models

## Benchmarks
- **OpenML_Benchmarks** — https://www.openml.org/search?type=benchmark
  - metrics: accuracy

## Models
### OneHotEncoder — baseline / tier1_core

**Roles:** representation
**Status:** —

**Purpose:** Expand a categorical column to one binary indicator per category. Inductive bias: categories are independent; high cardinality explodes dimensionality.

**Use cases:**
- Small-cardinality categoricals
- Linear model preprocessing
- Tree-friendly encodings

**Config:** `{'handle_unknown': 'string'}`

**Benchmarks:**
- **[literature]** OpenML_Benchmarks → accuracy=null
  - algorithm: `sklearn.preprocessing.OneHotEncoder` (docs)
  - number: https://www.openml.org/search?type=benchmark
  - code/paper: https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
  - note: no_public_benchmark: encoding is a preprocessing step; OpenML leaderboards rank full pipelines, not encoders alone.

### KBinsDiscretizer — baseline / tier1_core

**Roles:** representation
**Status:** desc

**Purpose:** Bin a continuous column into k quantile/uniform bins. Inductive bias: piecewise-constant effect of the feature; can help non-linear models and regularize.

**Use cases:**
- Age buckets
- Income deciles
- Monotone encoding for linear models

**Config:** `{'n_bins': 'int', 'strategy': 'string'}`

**Benchmarks:**
- **[literature]** OpenML_Benchmarks → accuracy=null
  - algorithm: `sklearn.preprocessing.KBinsDiscretizer` (docs)
  - number: https://www.openml.org/search?type=benchmark
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html
  - note: no_public_benchmark: binning has no standalone accuracy leaderboard.

### PolynomialFeatures — sota / tier1_core

**Roles:** nonlinear_interactions, representation
**Status:** desc

**Purpose:** Expand features to degree-k polynomial terms, adding cross-products. Inductive bias: interaction structure is multiplicative; explodes d^k without regularization.

**Use cases:**
- Linear model with interactions
- Physics formula recovery
- Feature crosses for CTLs

**Config:** `{'degree': 'int', 'interaction_only': 'bool'}`

**Benchmarks:**
- **[literature]** OpenML_Benchmarks → accuracy=null
  - algorithm: `sklearn.preprocessing.PolynomialFeatures` (docs)
  - number: https://www.openml.org/search?type=benchmark
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
  - note: no_public_benchmark: polynomial expansion is a preprocessing transform; no standalone leaderboard.

### EncodeCategorical — baseline / tier1_core

**Roles:** representation
**Status:** ✓ cadw, ✓ agent

**Purpose:** Map each unique category to a dense int32 id, sorted for determinism. Pass 1: pc.value_counts per batch, union in a Python set (O(K) RAM). Pass 2: pc.index_in streaming over batches (O(batch) per batch). Inductive bias: none — pure representation; reverse_maps is O(K) by contract.

**Use cases:**
- Categorical to tree/linear feature
- Deterministic id assignment across runs
- Pre-fit encoding for downstream models

**Config:** `{'columns': 'str or Sequence[str]'}`

**CADW:** `cadw.dataframe.encode_categorical`

**Benchmarks:** —

## Metrics
- **accuracy** (anchor, higher)
  - Downstream accuracy of the model trained on the transformed features.
  - *special:* Depends on the model and the split; report both.
