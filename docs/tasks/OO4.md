# O4 — Feature selection

*Section:* O — Data Science outside ML (EDA / cleaning / feature engineering / validation)

**Description:** Rank and drop features by relevance, redundancy or stability before modeling.

**Real-life examples:**
- High-dimensional genomics / wide tables
- Reducing training cost for GBDT
- Removing leakage / constant columns

## Benchmarks
- **OpenML_Benchmarks** — https://www.openml.org/search?type=benchmark
  - metrics: accuracy
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: fit_time_s, peak_rss_mb

## Models
### MutualInfoSelect — baseline / tier1_core

**Roles:** linear_signal, probabilistic
**Status:** ✓ cadw, ✓ agent

**Purpose:** Rank features by mutual information to the target (binned for continuous). Inductive bias: any monotone or non-monotone dependence is detected; no assumption of linearity.

**Use cases:**
- Genomic feature triage
- Text n-gram pruning
- Wide-table reduction

**Config:** `{'k': 'int', 'n_bins': 'int'}`

**CADW:** `cadw.ml.feature_select.fit_feature_select`

**Benchmarks:**
- **[cadw_measured]** cadw_internal → fit_time_s=3.962 (n=2)
  - algorithm: `cadw.ml.feature_select.fit_feature_select (mutual_info)` (code)
  - number: .pingis/benches/feature_select.json
  - code/paper: cadw/ml/feature_select.py
  - note: CADW-native: mutual_info fit_s=1.508 @n=10k, 3.962 @n=50k, d=20 (feature_select.json). NOT comparable to external.
- **[cadw_measured]** cadw_internal → peak_rss_mb=13.7 (n=2)
  - algorithm: `cadw.ml.feature_select.fit_feature_select (mutual_info)` (code)
  - number: .pingis/benches/feature_select.json
  - code/paper: cadw/ml/feature_select.py
  - note: CADW-native: mutual_info peak_mb=85.0 @n=10k, 13.7 @n=50k (feature_select.json; higher at small n due to fixed overheads). NOT comparable to external.
- **[literature]** OpenML_Benchmarks → accuracy=null
  - algorithm: `sklearn.feature_selection.mutual_info_classif` (docs)
  - number: https://scikit-learn.org/stable/modules/feature_selection.html
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
  - note: no_public_benchmark: sklearn docs describe mutual_info_classif but publish no cross-dataset accuracy leaderboard for the selector.

### VarianceThreshold — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw, ✓ agent

**Purpose:** Drop features whose variance is below a threshold. Inductive bias: low variance means low information; catches constants and near-constants cheaply.

**Use cases:**
- Drop constant columns
- Cheap pre-filter
- Remove near-zero-variance sensors

**Config:** `{'threshold': 'float'}`

**CADW:** `cadw.ml.feature_select.fit_feature_select`

**Benchmarks:**
- **[cadw_measured]** cadw_internal → fit_time_s=0.032 (n=2)
  - algorithm: `cadw.ml.feature_select.fit_feature_select (variance)` (code)
  - number: .pingis/benches/feature_select.json
  - code/paper: cadw/ml/feature_select.py
  - note: CADW-native: variance fit_s=0.003 @n=10k, 0.032 @n=50k (feature_select.json). NOT comparable to external.

### BorutaSelect — sota / tier2_wrapper

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** Shadow-feature wrapper: repeated GBDT fits compare real vs shuffled shadow features; selects only those that beat all shadows. Inductive bias: relevance detectable by tree ensembles; stable, expensive.

**Use cases:**
- Tabular feature triage
- Genomic relevance
- Stable selection for production

**Config:** `{'n_iter': 'int', 'max_depth': 'int'}`

**CADW:** `cadw.ml.feature_select.fit_feature_select`

**Benchmarks:**
- **[cadw_measured]** cadw_internal → fit_time_s=8.355 (n=3)
  - algorithm: `cadw.ml.feature_select.fit_feature_select (boruta)` (code)
  - number: .pingis/benches/feature_select.json
  - code/paper: cadw/ml/feature_select.py
  - note: CADW-native: boruta fit_s=4.642 @n=2k, 5.394 @n=5k, 8.355 @n=10k; n_selected=2 of d=20 (feature_select.json). NOT comparable to external.

## Metrics
- **accuracy** (anchor, higher)
  - Downstream model accuracy after feature selection.
  - *special:* Model-dependent; report the downstream model.
- **fit_time_s** (diagnostic, lower)
  - Selection wall-clock.
- **peak_rss_mb** (diagnostic, lower)
  - Selection peak memory.

## Ensemble rules
- **[linear_signal, nonlinear_interactions]** → VarianceThreshold + MutualInfoSelect (max_per_role=2)
  - Always start with VarianceThreshold (near-free); layer MutualInfo on survivors.
