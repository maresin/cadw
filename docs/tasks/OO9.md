# O9 — Fairness / bias

*Section:* O — Data Science outside ML (EDA / cleaning / feature engineering / validation)

**Description:** Measure group-level disparity in model outcomes or errors; audit before deployment.

**Real-life examples:**
- Credit/lending decision audit
- Hiring model parity check
- Healthcare outcome equity

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: dp_diff, eo_diff, fit_time_s
- **sklearn_reference** — https://fairlearn.org/main/user_guide/fairness_in_machine_learning.html
  - metrics: dp_diff, eo_diff

## Models
### FairnessMetrics — baseline / tier1_core

**Roles:** calibration
**Status:** ✓ cadw, ✓ agent

**Purpose:** Group-wise TPR/FPR and positive-rate disparities via Reduce. Inductive bias: sensitive groups are the audit unit; metrics are statistical, no model.

**Use cases:**
- Loan-approval audit
- Hiring pipeline check
- Medical model equity

**Config:** `{'group_col': 'string', 'y_true_col': 'string', 'y_pred_col': 'string'}`

**CADW:** `cadw.ml.fairness.fairness_metrics`

**Benchmarks:**
- **[cadw_measured]** cadw_internal → dp_diff=0.0096 (n=2)
  - algorithm: `cadw.ml.fairness.fairness_metrics` (code)
  - number: .pingis/benches/fairness.json
  - code/paper: cadw/ml/fairness.py
  - note: CADW-native: dp=0.0096 @n=100k (8 groups), 0.0052 @n=1M (fairness.json). NOT comparable to external.
- **[cadw_measured]** cadw_internal → eo_diff=0.0166 (n=2)
  - algorithm: `cadw.ml.fairness.fairness_metrics` (code)
  - number: .pingis/benches/fairness.json
  - code/paper: cadw/ml/fairness.py
  - note: CADW-native: eo=0.0166 @n=100k, 0.0085 @n=1M (fairness.json). NOT comparable to external.
- **[cadw_measured]** cadw_internal → fit_time_s=0.098 (n=2)
  - algorithm: `cadw.ml.fairness.fairness_metrics` (code)
  - number: .pingis/benches/fairness.json
  - code/paper: cadw/ml/fairness.py
  - note: CADW-native: fairness metrics time_s=0.029 @n=100k, 0.098 @n=1M (fairness.json). NOT comparable to external.

### FairlearnParity — sota / tier2_wrapper

**Roles:** calibration
**Status:** desc, export

**Purpose:** Reductions to parity constraints on top of any estimator. Inductive bias: enforce group-parity by constrained optimization; trades accuracy for parity.

**Use cases:**
- Constrained logistic regression
- Fair ranking
- Regulatory compliance

**Config:** `{'constraint': 'string', 'eps': 'float'}`

**Benchmarks:**
- **[literature]** sklearn_reference → dp_diff=null
  - algorithm: `fairlearn.reductions.ExponentiatedGradient` (docs)
  - number: https://fairlearn.org/main/user_guide/fairness_in_machine_learning.html
  - code/paper: https://github.com/fairlearn/fairlearn
  - note: no_public_benchmark: Fairlearn docs define parity but publish no cross-dataset ranking; tradeoff accuracy-vs-parity is problem-specific.

## Metrics
- **dp_diff** (anchor, lower)
  - Difference in positive-prediction rate across groups (demographic parity).
  - *special:* Reporting-dependent; always report per-group counts.
- **eo_diff** (diagnostic, lower)
  - Difference in true-positive rate across groups (equalized odds).
  - *special:* Requires ground-truth labels in the audit set.
- **fit_time_s** (diagnostic, lower)
  - Wall-clock for the metrics pass.

## Ensemble rules
- **[calibration]** → FairnessMetrics (max_per_role=2)
  - Audit with FairnessMetrics; if disparity is unacceptable, wrap with FairlearnParity (accepting the accuracy cost).
