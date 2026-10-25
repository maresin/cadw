# A6 — Imbalanced classification

*Section:* A — Classification & Regression

**Description:** Binary/multiclass with strong class imbalance (positive_rate<5%).

**Real-life examples:**
- Fraud detection
- Rare disease diagnosis
- Defect detection

## Benchmarks
- **AMLB** — https://openml.github.io/automlbenchmark/
  - metrics: roc_auc, pr_auc

## Models
### Logistic_balanced — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw, ✓ agent

**Purpose:** LogReg with class_weight='balanced'; simplest cost-sensitive baseline.

**Use cases:**
- Imbalanced credit scoring

**Config:** `{'class_weight': 'str', 'alpha': 'float'}`

**CADW:** `cadw.ml.logistic.fit_logistic`

**Benchmarks:**
- **[literature]** AMLB → pr_auc=0.42
  - algorithm: `sklearn.linear_model.LogisticRegression` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  - note: AMLB imbalance subset [legacy v0.9.44, migrated v0.9.50]

### GBDT_scale_pos_weight — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** Cost-sensitive GBDT; weight positives by neg/pos ratio (XGBoost semantics).

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
  - Discrimination of the rare class; prefers models that rank positives high.
  - *special:* Much lower than roc_auc when imbalance is strong; use only pr_auc comparisons for this subtype.
- **roc_auc** (diagnostic, higher)
  - Discrimination of rare class (legacy comparison).
  - *special:* Overly optimistic under heavy imbalance; do not use as anchor.
- **recall_at_1pct** (pareto_axis, higher)
  - Business-aligned recall at 1% false-positive budget.
  - *special:* Not comparable across datasets with different prevalence.

## Ensemble rules
- **[linear_signal, nonlinear_interactions]** → Logistic_balanced + GBDT_scale_pos_weight (max_per_role=2)
  - Weighted probability average; apply calibration before deployment.
