# O7 — Hyperparameter tuning / AutoML

*Section:* O — Data Science outside ML (EDA / cleaning / feature engineering / validation)

**Description:** Search a hyperparameter space under a CV budget; AutoML wraps the whole pipeline.

**Real-life examples:**
- Tune GBDT depth/lr on tabular
- AutoML baseline for a new task
- Budgeted search for a Kaggle-like contest

## Benchmarks
- **OpenML_Benchmarks** — https://www.openml.org/search?type=benchmark
  - metrics: accuracy

## Models
### GridSearch — baseline / tier1_core

**Roles:** linear_signal
**Status:** desc

**Purpose:** Exhaustive grid over a discrete parameter space with CV scoring. Inductive bias: none — it is a search strategy; guarantees finding the best grid point but scales exponentially. Deferred per .pingis/impl/REPORT.md §4 (orchestration wrapper; no new memory model).

**Use cases:**
- Small discrete grids
- Reproducible baselines
- Grid for 2-3 params

**Config:** `{'param_grid': 'dict'}`

**Benchmarks:**
- **[literature]** OpenML_Benchmarks → accuracy=null
  - algorithm: `sklearn.model_selection.GridSearchCV` (docs)
  - number: https://scikit-learn.org/stable/modules/grid_search.html
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
  - note: no_public_benchmark: search strategy is not a model; accuracy is downstream.

### RandomizedSearch — baseline / tier1_core

**Roles:** probabilistic
**Status:** desc

**Purpose:** Sample N random parameter configurations from declared distributions. Inductive bias: random coverage of the space, best when only a few hyperparameters matter (Bergstra & Bengio 2012). Deferred per .pingis/impl/REPORT.md §4 (orchestration wrapper; no new memory model).

**Use cases:**
- Large continuous spaces
- Budget-limited tuning
- Neural net hparam search

**Config:** `{'n_iter': 'int', 'distributions': 'dict'}`

**Benchmarks:**
- **[literature]** OpenML_Benchmarks → accuracy=null
  - algorithm: `sklearn.model_selection.RandomizedSearchCV` (paper)
  - number: https://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization
  - code/paper: https://arxiv.org/abs/1206.2944
  - note: no_public_benchmark: original Bergstra & Bengio paper reports per-experiment gains, not a cross-dataset accuracy table.

### OptunaTPE — sota / tier2_wrapper

**Roles:** probabilistic, representation
**Status:** desc, export

**Purpose:** Sequential model-based search; Tree-structured Parzen Estimator models p(x|y) and proposes the next promising trial. Inductive bias: smooth response surface; sample-efficient but needs a budget.

**Use cases:**
- Production AutoML
- Budgeted tuning
- Multi-objective tuning

**Config:** `{'n_trials': 'int', 'sampler': 'string'}`

**Benchmarks:**
- **[literature]** OpenML_Benchmarks → accuracy=null
  - algorithm: `optuna.samplers.TPESampler` (docs)
  - number: https://optuna.readthedocs.io/en/stable/
  - code/paper: https://github.com/optuna/optuna
  - note: no_public_benchmark: Optuna docs do not publish a cross-dataset leaderboard; TPE is the standard reference for BO-based search.

## Metrics
- **accuracy** (anchor, higher)
  - Downstream CV accuracy of the best configuration found.
  - *special:* Budget-dependent; report n_trials and CV folds.
