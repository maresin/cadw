# C4 — Feature selection

*Section:* C — Dimensionality reduction

**Description:** Select subset of features by importance, eliminating noise / redundant signals.

**Real-life examples:**
- Model interpretability (few features)
- Remove noisy features
- Speed up downstream training

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: downstream_accuracy, n_selected, fit_time_s

## Models
### PermutationImportance — baseline / tier1_core

**Roles:** calibration
**Status:** desc, export

**Purpose:** Breiman 2001: feature importance by permuting each feature and measuring score drop. Model-agnostic; honest (unlike in-sample impurity). Canonical CADW: cadw.ml.importance.permutation_importance (verified in O_eda.O8).

**Use cases:**
- Feature ranking post-hoc
- Model-agnostic importance
- Sanity check for built-in importance

**Config:** `{'n_repeats': 'int', 'random_state': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → downstream_accuracy=null
  - algorithm: `sklearn.inspection.permutation_importance` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
  - note: Breiman 2001. CADW impl: permutation_importance (O_eda O8, verified).

### Boruta — sota / tier2_wrapper

**Roles:** calibration, nonlinear_interactions
**Status:** desc, export

**Purpose:** Kursa & Rudnicki 2010: all-relevant feature selection. Shadow features + iterative random-forest comparison; keeps all features statistically more important than random.

**Use cases:**
- Biomarker discovery (all-relevant)
- Nonlinear feature importance
- Small-nomics data

**Config:** `{'n_estimators': 'int', 'max_iter': 'int', 'alpha': 'float'}`

**Benchmarks:**
- **[literature]** cadw_internal → downstream_accuracy=null
  - algorithm: `boruta.BorutaPy` (paper)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://www.jstatsoft.org/article/view/v036i11
  - note: Kursa & Rudnicki 2010.

## Metrics
- **downstream_accuracy** (anchor, higher)
  - Accuracy/R2 of model trained on selected features only; ultimate test of feature-selection quality.
  - *special:* Depends on downstream model; report across 2-3 model families.
- **n_selected** (diagnostic, lower)
  - Number of retained features.
  - *special:* Tradeoff: fewer features = faster but may lose signal.
