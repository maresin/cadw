# O8 — Explainability (XAI)

*Section:* O — Data Science outside ML (EDA / cleaning / feature engineering / validation)

**Description:** Attribute a fitted model's prediction to its input features for debugging, trust, and compliance.

**Real-life examples:**
- Feature importance for stakeholders
- Debugging model regressions
- Regulatory attribution

## Benchmarks
- **sklearn_reference** — https://scikit-learn.org/stable/modules/permutation_importance.html
  - metrics: auc, fidelity

## Models
### PermutationImportance — baseline / tier1_core

**Roles:** calibration
**Status:** ✓ cadw, ✓ agent

**Purpose:** Shuffle one feature at a time and measure the drop in a chosen metric. Inductive bias: model-agnostic, captures any dependence; noisy under correlated features.

**Use cases:**
- Model debug on tabular
- Feature triage
- Model card attributions

**Config:** `{'n_repeats': 'int', 'scoring': 'string'}`

**CADW:** `cadw.ml.importance.permutation_importance`

**Benchmarks:**
- **[literature]** sklearn_reference → auc=null
  - algorithm: `sklearn.inspection.permutation_importance` (docs)
  - number: https://scikit-learn.org/stable/modules/permutation_importance.html
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
  - note: no_public_benchmark + cadw_measured_without_bench: sklearn docs page has usage but no cross-dataset fidelity leaderboard; CADW implements permutation_importance (cadw.ml.importance) but no .pingis/benches/importance.json exists — reproduce before ranking.

### SHAPTreeExplainer — sota / tier3_export

**Roles:** calibration, representation
**Status:** desc, export

**Purpose:** Shapley-value attributions with the tree-exact algorithm for GBDT. Inductive bias: additive feature attribution; exact for trees, O(T·L) per row.

**Use cases:**
- Explaining GBDT predictions
- Regulatory attribution
- Model debugging

**Config:** `{'max_evals': 'int', 'model_output': 'string'}`

**Benchmarks:**
- **[literature]** sklearn_reference → auc=null
  - algorithm: `shap.TreeExplainer` (paper)
  - number: https://arxiv.org/abs/1705.07874
  - code/paper: https://github.com/shap/shap
  - note: no_public_benchmark: SHAP paper (Lundberg & Lee 2017) is a theoretical/methodological paper; no cross-dataset accuracy leaderboard.

## Metrics
- **auc** (anchor, higher)
  - Downstream model AUC (XAI is evaluated indirectly via model sanity).
  - *special:* XAI has no ground-truth metric; report fidelity against a held-out baseline where possible.
- **fidelity** (diagnostic, higher)
  - Agreement between attribution and a held-out black-box (for LIME-style local surrogates).

## Ensemble rules
- **[calibration]** → PermutationImportance (max_per_role=2)
  - Permutation is model-agnostic and cheap; use SHAPTreeExplainer for exact per-row attributions on GBDT.
