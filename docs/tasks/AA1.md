# A1 — Binary classification

*Section:* A — Classification & Regression

**Description:** Predict P(y=1|x) with y in {0,1}.

**Real-life examples:**
- Credit scoring (default yes/no)
- Churn prediction
- Fraud detection (rare positive)
- CTR prediction

## Benchmarks
- **AMLB** — https://openml.github.io/automlbenchmark/
  - metrics: roc_auc, accuracy, logloss

## Models
### LogisticRegression — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw, ✓ agent

**Purpose:** Additive log-odds; fast on high-d, low-n; interpretable coefficients.

**Use cases:**
- Credit scoring
- Medical risk scores

**Config:** `{'alpha': 'float', 'max_iter': 'int', 'l1_ratio': 'float'}`

**CADW:** `cadw.ml.logistic.fit_logistic`

**Benchmark:** AMLB → roc_auc=0.78 (median across AMLB binary tasks; ranking only)

### GBDT — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** ✓ cadw, ✓ agent

**Purpose:** Gradient-boosted trees; SOTA on most tabular Kaggle tasks.

**Use cases:**
- Kaggle credit scoring
- Ad click prediction

**Config:** `{'n_estimators': 'int', 'max_depth': 'int', 'learning_rate': 'float'}`

**CADW:** `cadw.ml.gbdt.fit_gbdt_classifier`

**Benchmark:** AMLB → roc_auc=0.82 (median across AMLB binary tasks)

### BernoulliNB — baseline / tier1_core

**Roles:** probabilistic
**Status:** ✓ cadw, ✓ agent

**Purpose:** Naive Bayes for binary features; fast baseline.

**Use cases:**
- Spam filter
- Text classification

**Config:** `{'alpha': 'float'}`

**CADW:** `cadw.ml.naive_bayes.fit_bernoulli_nb`

**Benchmark:** AMLB → roc_auc=0.72

### TabPFN — sota / tier3_export

**Roles:** nonlinear_interactions
**Status:** desc, export

**Purpose:** Prior-fitted transformer for small tabular datasets; SOTA on AMLB small-N.

**Use cases:**
- Medical tabular prediction (n<2000)

**Config:** `{'n_estimators': 'int'}`

**Benchmark:** AMLB → roc_auc=0.85 (Top of AMLB small-N subset; export via to_batches(format='torch'))

## Metrics
- **roc_auc** (anchor, higher)
  - Threshold-invariant discrimination.
  - *special:* Insensitive to class imbalance; prefer pr_auc when positive_rate<0.05.
- **pr_auc** (pareto_axis, higher)
  - Discrimination under class imbalance.
  - *special:* Only use as anchor when positive_rate<0.05; not comparable to roc_auc numerically.
- **brier** (pareto_axis, lower)
  - Calibration + discrimination mixed; safest for probability use.
  - *special:* Decompose to (reliability, resolution, uncertainty) to see where loss comes from.
- **logloss** (diagnostic, lower)
  - Strict calibration-aware loss.
  - *special:* Very sensitive to confident wrong predictions; do not sum with brier.
- **accuracy** (diagnostic, higher)
  - Business-default decision quality.
  - *special:* Uses a fixed threshold 0.5; only meaningful when costs are symmetric.

## Ensemble rules
- **[linear_signal, nonlinear_interactions]** → LogisticRegression + GBDT (max_per_role=2)
  - Complementary inductive biases; average predicted probabilities.
- **[linear_signal, probabilistic]** → LogisticRegression + BernoulliNB (max_per_role=2)
  - Only when features are binary; Naive Bayes adds fast likelihood.
