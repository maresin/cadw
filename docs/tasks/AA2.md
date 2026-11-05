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
**Status:** ✓ cadw

**Purpose:** Multinomial log-odds; strong when classes are roughly linearly separable.

**Use cases:**
- Iris-style small-d tasks

**Config:** `{'alpha': 'float', 'max_iter': 'int'}`

**CADW:** `cadw.ml.logistic.fit_logistic`

**Benchmarks:**
- **[literature]** AMLB → accuracy=0.74
  - algorithm: `sklearn.linear_model.LogisticRegression` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  - note: [legacy v0.9.44, migrated v0.9.50]

### GBDT_multiclass — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** One-vs-all softmax GBDT; SOTA on tabular multiclass.

**Use cases:**
- Product categorization

**Config:** `{'n_estimators': 'int', 'max_depth': 'int'}`

**CADW:** `cadw.ml.gbdt.fit_gbdt_multiclass`

**Benchmarks:**
- **[literature]** AMLB → accuracy=0.81
  - algorithm: `sklearn.ensemble.GradientBoostingClassifier` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
  - note: [legacy v0.9.44, migrated v0.9.50]

### GaussianNB — baseline / tier1_core

**Roles:** probabilistic
**Status:** ✓ cadw

**Purpose:** Gaussian class-conditionals; extremely fast.

**Use cases:**
- Baseline in medical tabular data

**Config:** `{'var_smoothing': 'float'}`

**CADW:** `cadw.ml.naive_bayes.fit_gaussian_nb`

**Benchmarks:**
- **[literature]** AMLB → accuracy=0.68
  - algorithm: `sklearn.naive_bayes.GaussianNB` (docs)
  - number: https://openml.github.io/automlbenchmark/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
  - note: [legacy v0.9.44, migrated v0.9.50]

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
