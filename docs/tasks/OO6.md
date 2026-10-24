# O6 — Sampling / splitting / resampling

*Section:* O — Data Science outside ML (EDA / cleaning / feature engineering / validation)

**Description:** Split, stratify, reservoir-sample, or resample class imbalance.

**Real-life examples:**
- Train/test split with class balance
- Reservoir sampling from a stream
- SMOTE on rare events

## Benchmarks
- **sklearn_reference** — https://scikit-learn.org/stable/modules/
  - metrics: accuracy, auc

## Models
### StratifiedSplit — baseline / tier1_core

**Roles:** probabilistic
**Status:** ✓ agent, desc

**Purpose:** Split preserving class proportions in each fold. Inductive bias: class is the stratifier; correct when the target is the source of imbalance.

**Use cases:**
- Cross-val on imbalanced target
- Holdout with matching priors
- Grouped CV by class

**Config:** `{'test_size': 'float', 'random_state': 'int'}`

**Benchmarks:**
- **[literature]** sklearn_reference → accuracy=null
  - algorithm: `sklearn.model_selection.StratifiedKFold` (docs)
  - number: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
  - code/paper: https://scikit-learn.org/stable/modules/cross_validation.html
  - note: no_public_benchmark: splitter is not a model; accuracy depends entirely on the downstream estimator.

### ReservoirSample — baseline / tier1_core

**Roles:** probabilistic
**Status:** ✓ agent, desc

**Purpose:** Uniform sample of k items from a stream in one pass, O(k) state. Inductive bias: uniform inclusion; unbiased but ignores strata.

**Use cases:**
- Sample from a log stream
- Streaming EDA
- Online quality monitoring

**Config:** `{'k': 'int'}`

**Benchmarks:**
- **[literature]** sklearn_reference → accuracy=null
  - algorithm: `Vitter reservoir sampling (Algorithm R)` (description)
  - number: https://en.wikipedia.org/wiki/Reservoir_sampling
  - code/paper: https://en.wikipedia.org/wiki/Reservoir_sampling
  - note: no_public_benchmark: sampling scheme has no accuracy benchmark; correctness is proven analytically.

### SMOTE — sota / tier2_wrapper

**Roles:** nonlinear_interactions, probabilistic
**Status:** desc, export

**Purpose:** Synthesize minority-class points along kNN segments. Inductive bias: minority manifold is locally linear; effective on low-dim dense features, harmful on high-dim sparse.

**Use cases:**
- Rare-event classification
- Fraud oversampling
- Medical rare disease

**Config:** `{'k_neighbors': 'int', 'sampling_strategy': 'string'}`

**Benchmarks:**
- **[literature]** sklearn_reference → auc=null
  - algorithm: `imblearn.over_sampling.SMOTE` (paper)
  - number: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
  - code/paper: https://arxiv.org/abs/1106.1813
  - note: no_public_benchmark: imbalanced-learn docs page has usage examples but no cross-dataset AUC leaderboard for SMOTE.

## Metrics
- **auc** (anchor, higher)
  - ROC AUC of the downstream model on the minority class.
  - *special:* Sampling-scheme-dependent; report the downstream model and splitter.
- **accuracy** (diagnostic, higher)
  - Downstream accuracy (misleading under imbalance).
  - *special:* Report only alongside AUC; never alone.

## Ensemble rules
- **[probabilistic, nonlinear_interactions]** → StratifiedSplit (max_per_role=2)
  - Use StratifiedSplit for correct folds; apply SMOTE only inside the training fold (never on the test fold).
