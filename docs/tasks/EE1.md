# E1 — Unsupervised anomaly detection

*Section:* E — Anomaly Detection (tabular)

**Description:** Detect rare rows with no labelled anomalies available.

**Real-life examples:**
- Fraud detection (labels delayed)
- Network intrusion
- Sensor fault detection

## Benchmarks
- **ODDS** — https://odds.cs.stonybrook.edu/
  - metrics: roc_auc, pr_auc
- **ADBench** — https://github.com/Minqi824/ADBench
  - metrics: roc_auc, pr_auc

## Models
### zscore — baseline / tier1_core

**Roles:** local_structure
**Status:** ✓ agent, desc

**Purpose:** Univariate z-score per column; row score = max |z|; simplest baseline.

**Use cases:**
- Server metric alarms

**Config:** `{'threshold': 'float'}`

**Benchmark:** ODDS → roc_auc=0.72

### IForest — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** Random-split isolation ensemble; O(n) fit; strong on high-dim tabular.

**Use cases:**
- Network intrusion
- Kaggle fraud datasets

**Config:** `{'n_estimators': 'int', 'subsample_size': 'int', 'contamination': 'float'}`

**CADW:** `cadw.ml.iforest.fit_isolation_forest`

**Benchmark:** ODDS → roc_auc=0.87

### LOF — baseline / tier3_export

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Local Outlier Factor via sklearn/PyOD; strong on clustered data.

**Use cases:**
- Astronomy outlier catalogs

**Config:** `{'n_neighbors': 'int'}`

**Benchmark:** ODDS → roc_auc=0.78

## Metrics
- **roc_auc** (anchor, higher)
  - Rank-based discrimination across all thresholds.
  - *special:* Over-optimistic when contamination < 1%; report pr_auc alongside.
- **pr_auc** (pareto_axis, higher)
  - Discrimination under strong imbalance.
  - *special:* Prefer to roc_auc when contamination < 5%; not comparable across datasets with different prevalence.

## Ensemble rules
- **[local_structure, nonlinear_interactions]** → zscore + IForest (max_per_role=1)
  - Average normalized anomaly scores; zscore for univariate extremes, IForest for global structure.
