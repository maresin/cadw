# A4 — Multilabel classification

*Section:* A — Classification & Regression

**Description:** Predict multiple binary labels per row.

**Real-life examples:**
- Document tagging
- Medical diagnosis (multiple conditions)

## Benchmarks
- **AMLB** — https://openml.github.io/automlbenchmark/
  - metrics: accuracy

## Models
### OneVsRest_Logistic — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw, ✓ agent

**Purpose:** Independent per-label logits; ignores label correlations.

**Use cases:**
- Tag suggestion

**Config:** `{'alpha': 'float'}`

**CADW:** `cadw.ml.chains.fit_chains`

**Benchmarks:**
- **[literature]** AMLB → accuracy=0.62
  - algorithm: `sklearn.multiclass.OneVsRestClassifier` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
  - note: subset accuracy [legacy v0.9.44, migrated v0.9.50]

### ClassifierChains — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** Sequential per-label classifiers; each sees earlier labels.

**Use cases:**
- Medical coding (ICD)

**Config:** `{'n_estimators': 'int', 'order': 'list'}`

**CADW:** `cadw.ml.chains.fit_chains`

**Benchmarks:**
- **[literature]** AMLB → accuracy=0.66
  - algorithm: `sklearn.multioutput.ClassifierChain` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html
  - note: [legacy v0.9.44, migrated v0.9.50]

### ML-kNN — baseline / tier2_wrapper

**Roles:** probabilistic
**Status:** desc, export

**Purpose:** Multi-label kNN with Bayesian prior (Zhang & Zhou 2007). Label posterior from neighbour counts.

**Use cases:**
- Multi-label text tagging
- Gene function prediction

**Config:** `{'n_neighbors': 'int', 'smoothing': 'float'}`

**Benchmarks:**
- **[literature]** cadw_internal → hamming_loss=null
  - algorithm: `ML-kNN` (docs)
  - number: .pingis/benches/
  - code/paper: https://scikit.ml/api/skmultilearn.adapt.mlknn.html
  - note: no_public_benchmark: ML-kNN per-dataset; no unified multilabel LB

## Metrics
- **subset_accuracy** (anchor, higher)
  - Exact-match accuracy per row.
  - *special:* Strict — one wrong label fails the row; use hamming_loss when labels are numerous.
- **hamming_loss** (pareto_axis, lower)
  - Fraction of incorrect labels per row.
  - *special:* Handles many labels gracefully; not comparable to subset_accuracy.

## Ensemble rules
- **[linear_signal, nonlinear_interactions]** → OneVsRest_Logistic + ClassifierChains (max_per_role=1)
  - OvR as baseline, chains as SOTA; use chains when label correlation is strong.
