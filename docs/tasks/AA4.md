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

**CADW:** `cadw.ml.chains.fit_classifier_chains`

**Benchmarks:**
- **[literature]** AMLB → accuracy=0.62
  - algorithm: `OneVsRest_Logistic` (description)
  - number: https://openml.github.io/automlbenchmark/
  - note: subset accuracy [migrated v0.9.47b: algorithm/algorithm_source_url unverified]

### ClassifierChains — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** Sequential per-label classifiers; each sees earlier labels.

**Use cases:**
- Medical coding (ICD)

**Config:** `{'n_estimators': 'int', 'order': 'list'}`

**CADW:** `cadw.ml.chains.fit_classifier_chains`

**Benchmarks:**
- **[literature]** AMLB → accuracy=0.66
  - algorithm: `ClassifierChains` (description)
  - number: https://openml.github.io/automlbenchmark/
  - note: [migrated v0.9.47b: algorithm/algorithm_source_url unverified]

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
