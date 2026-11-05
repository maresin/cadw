# U1 — Full AutoML (pipeline search + ensemble)

*Section:* U — AutoML

**Description:** Search over full pipelines: preprocessing, model family, hyperparameters, ensemble. Produces a fitted pipeline given a task.

**Real-life examples:**
- Rapid prototyping on new tabular tasks
- Non-expert ML deployment
- Baseline reference for custom models

## Benchmarks
- **AMLB** — https://openml.github.io/automlbenchmark/
  - metrics: mean_rank, mean_elapsed_time

## Models
### Auto-sklearn — baseline / tier3_export

**Roles:** nonlinear_interactions, linear_signal
**Status:** desc, export

**Purpose:** Feurer et al. 2015: Bayesian HPO over sklearn pipelines + meta-learning + ensemble. Historically first strong AutoML; AutoGluon now dominates benchmarks.

**Use cases:**
- Baseline AutoML
- Small-data prototyping

**Config:** `{'time_left_for_this_task': 'int', 'per_run_time_limit': 'int'}`

**Benchmarks:**
- **[literature]** AMLB → mean_rank=null
  - algorithm: `auto-sklearn 2.0` (paper)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://arxiv.org/abs/2007.04074
  - note: AMLB 2025: not top-3 on modern benchmarks.

### H2OAutoML — baseline / tier3_export

**Roles:** nonlinear_interactions, linear_signal
**Status:** desc, export

**Purpose:** LeDell & Poirier 2020: stacked ensemble over GBM/DRF/XGBoost/GLM. Fast; well-tuned defaults; competitive on mid-size tabular.

**Use cases:**
- Production AutoML
- Medium tabular tasks

**Config:** `{'max_models': 'int', 'max_runtime_secs': 'int'}`

**Benchmarks:**
- **[literature]** AMLB → mean_rank=null
  - algorithm: `H2O AutoML` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
  - note: AMLB 2025: top-5 on many tasks.

### AutoGluon_1_2 — sota / tier3_export

**Roles:** nonlinear_interactions, linear_signal, representation, probabilistic
**Status:** desc, export

**Purpose:** Erickson et al. 2020: multi-layer stacking + bagging + repeated k-fold. AutoML Benchmark 2025 SOTA: Pareto-efficient quality/speed, >10k samples/sec inference. Reference impl: AutoGluon TabularPredictor.

**Use cases:**
- SOTA tabular AutoML
- High-quality baseline for custom models

**Config:** `{'time_limit': 'int', 'presets': 'string'}`

**Benchmarks:**
- **[literature]** AMLB → mean_rank=null
  - algorithm: `AutoGluon 1.2` (paper)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://arxiv.org/abs/2003.06505
  - note: AMLB 2025: statistically significantly better than other AutoML (paper).

## Metrics
- **mean_rank** (anchor, lower)
  - Mean rank across AMLB tasks; primary AutoML leaderboard metric.
  - *special:* Number of tasks matters; report with n_datasets. Not comparable to per-dataset accuracy.
- **mean_elapsed_time** (diagnostic, lower)
  - Wall-clock per task (fit + predict).
  - *special:* Hardware-specific; time budget matters for rank comparability.
