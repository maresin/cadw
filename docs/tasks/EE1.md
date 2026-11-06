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

**Benchmarks:**
- **[literature]** ODDS → roc_auc=0.72
  - algorithm: `scipy.stats.zscore` (docs)
  - number: http://web.archive.org/web/20240913183309/https://odds.cs.stonybrook.edu/
  - code/paper: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html
  - note: canonical odds.cs.stonybrook.edu now returns HTTP 403; using wayback snapshot 2024-09-13 [legacy v0.9.44, migrated v0.9.50]

### IForest — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw

**Purpose:** Random-split isolation ensemble; O(n) fit; strong on high-dim tabular.

**Use cases:**
- Network intrusion
- Kaggle fraud datasets

**Config:** `{'n_estimators': 'int', 'subsample_size': 'int', 'contamination': 'float'}`

**CADW:** `cadw.ml.iforest.fit_isolation_forest`

**Benchmarks:**
- **[literature]** ODDS → roc_auc=0.87
  - algorithm: `sklearn.ensemble.IsolationForest` (docs)
  - number: http://web.archive.org/web/20240913183309/https://odds.cs.stonybrook.edu/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
  - note: canonical odds.cs.stonybrook.edu now returns HTTP 403; using wayback snapshot 2024-09-13 [legacy v0.9.44, migrated v0.9.50]
- **[cadw_reproduced]** cadw_internal → roc_auc=0.9987789473684211 (n=1)
  - algorithm: `cadw.ml.iforest.fit_isolation_forest` (code)
  - reproduces: `sklearn.ensemble.IsolationForest`
  - number: .pingis/benches/reproduction/iforest.json
  - code/paper: cadw/ml/iforest.py
  - note: M2 parity on synthetic_odds_like DGP; cadw=0.9988, ref(sklearn.IsolationForest)=0.9992, verdict=consistent

### LOF — baseline / tier3_export

**Roles:** local_structure
**Status:** desc, export

**Purpose:** Local Outlier Factor via sklearn/PyOD; strong on clustered data.

**Use cases:**
- Astronomy outlier catalogs

**Config:** `{'n_neighbors': 'int'}`

**Benchmarks:**
- **[literature]** ODDS → roc_auc=0.78
  - algorithm: `sklearn.neighbors.LocalOutlierFactor` (docs)
  - number: http://web.archive.org/web/20240913183309/https://odds.cs.stonybrook.edu/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
  - note: canonical odds.cs.stonybrook.edu now returns HTTP 403; using wayback snapshot 2024-09-13 [legacy v0.9.44, migrated v0.9.50]

### GMMAnomaly — sota / tier2_wrapper

**Roles:** probabilistic
**Status:** desc, export

**Purpose:** Anomaly via low-likelihood threshold on a fitted Gaussian mixture.

**Use cases:**
- Multimodal anomaly detection
- Cohort outlier detection

**Config:** `{'n_components': 'int', 'threshold_q': 'float'}`

**Benchmarks:**
- **[literature]** cadw_internal → roc_auc=null
  - algorithm: `GMMAnomaly` (docs)
  - number: .pingis/benches/
  - code/paper: https://scikit-learn.org/stable/modules/mixture.html
  - note: no_public_benchmark: GMM anomaly per-dataset

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
