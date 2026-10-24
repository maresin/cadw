# A2 — Multiclass classification

*Section:* A — Classification & Regression

**Description:** Predict P(y=k|x) for K>=3 classes.

**Real-life examples:**
- Handwritten digit recognition
- Product category tagging

## Benchmarks
- **AMLB** — https://openml.github.io/automlbenchmark/
  - metrics: accuracy, logloss

## Models
### LogisticRegression_multinomial — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw, ✓ agent

**Purpose:** Multinomial log-odds; strong when classes are roughly linearly separable.

**Use cases:**
- Iris-style small-d tasks

**Config:** `{'alpha': 'float', 'max_iter': 'int'}`

**CADW:** `cadw.ml.logistic.fit_logistic`

**Benchmarks:**
- **[literature]** AMLB → accuracy=0.74
  - algorithm: `LogisticRegression_multinomial` (description)
  - number: https://openml.github.io/automlbenchmark/
  - note: [migrated v0.9.47b: algorithm/algorithm_source_url unverified]

### GBDT_multiclass — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** One-vs-all softmax GBDT; SOTA on tabular multiclass.

**Use cases:**
- Product categorization

**Config:** `{'n_estimators': 'int', 'max_depth': 'int'}`

**CADW:** `cadw.ml.gbdt.fit_gbdt_classifier`

**Benchmarks:**
- **[literature]** AMLB → accuracy=0.81
  - algorithm: `GBDT_multiclass` (description)
  - number: https://openml.github.io/automlbenchmark/
  - note: [migrated v0.9.47b: algorithm/algorithm_source_url unverified]

### GaussianNB — baseline / tier1_core

**Roles:** probabilistic
**Status:** ✓ cadw, ✓ agent

**Purpose:** Gaussian class-conditionals; extremely fast.

**Use cases:**
- Baseline in medical tabular data

**Config:** `{'var_smoothing': 'float'}`

**CADW:** `cadw.ml.naive_bayes.fit_gaussian_nb`

**Benchmarks:**
- **[literature]** AMLB → accuracy=0.68
  - algorithm: `GaussianNB` (description)
  - number: https://openml.github.io/automlbenchmark/
  - note: [migrated v0.9.47b: algorithm/algorithm_source_url unverified]

## Metrics
- **accuracy** (anchor, higher)
  - Overall correct decisions when classes are balanced.
  - *special:* Use balanced_accuracy or macro-F1 when classes are imbalanced.
- **logloss** (diagnostic, lower)
  - Calibration-aware multiclass loss.
  - *special:* Punishes confident wrong calls; not comparable to accuracy.
- **macro_f1** (pareto_axis, higher)
  - Per-class balance, robust to class imbalance.
  - *special:* Degrades when one class is very rare and the model ignores it.

## Ensemble rules
- **[linear_signal, nonlinear_interactions]** → LogisticRegression_multinomial + GBDT_multiclass (max_per_role=2)
  - Linear + trees; average softmax probabilities.
