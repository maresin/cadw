# E2 — Supervised anomaly detection

*Section:* E — Anomaly Detection (tabular)

**Description:** Detect rare rows with labels available (binary anomaly classes).

**Real-life examples:**
- Fraud with confirmed cases
- Manufacturing defect detection

## Benchmarks
- **AMLB** — https://openml.github.io/automlbenchmark/
  - metrics: roc_auc, pr_auc

## Models
### Logistic_balanced — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw, ✓ agent

**Purpose:** LogReg with class_weight='balanced'; simplest cost-sensitive baseline.

**Use cases:**
- Fraud baseline

**Config:** `{'class_weight': 'str', 'alpha': 'float'}`

**CADW:** `cadw.ml.logistic.fit_logistic`

**Benchmarks:**
- **[literature]** AMLB → pr_auc=0.42
  - algorithm: `sklearn.linear_model.LogisticRegression` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  - note: [legacy v0.9.44, migrated v0.9.50]

### GBDT_scale_pos_weight — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** Cost-sensitive GBDT; weight positives by neg/pos (XGBoost semantics).

**Use cases:**
- Fraud detection
- Defect detection

**Config:** `{'scale_pos_weight': 'float', 'n_estimators': 'int'}`

**CADW:** `cadw.ml.gbdt.fit_gbdt_classifier`

**Benchmarks:**
- **[literature]** AMLB → pr_auc=0.51
  - algorithm: `XGBoost scale_pos_weight` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://xgboost.readthedocs.io/en/stable/parameter.html
  - note: [legacy v0.9.44, migrated v0.9.50]

## Metrics
- **pr_auc** (anchor, higher)
  - Discrimination of the rare class under strong imbalance.
  - *special:* Use only pr_auc for ranking; roc_auc is overly optimistic.
- **recall_at_1pct** (pareto_axis, higher)
  - Business-aligned recall at 1% false-positive budget.
  - *special:* Prevalence-dependent; not comparable across datasets.

## Ensemble rules
- **[linear_signal, nonlinear_interactions]** → Logistic_balanced + GBDT_scale_pos_weight (max_per_role=1)
  - Average probabilities; calibrate on holdout before deployment.
