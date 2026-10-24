# A5 — Ordinal classification

*Section:* A — Classification & Regression

**Description:** Predict ranked classes (e.g. ratings 1-5).

**Real-life examples:**
- Star ratings
- Pain severity 1-10
- Satisfaction surveys

## Benchmarks
- **AMLB** — https://openml.github.io/automlbenchmark/
  - metrics: accuracy

## Models
### OrdinalLogit — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw, ✓ agent

**Purpose:** Cumulative-link logistic; respects class order.

**Use cases:**
- Satisfaction survey prediction

**Config:** `{'alpha': 'float'}`

**CADW:** `cadw.ml.ordinal.fit_ordinal_logit`

**Benchmark:** AMLB → accuracy=0.55 (exact-match accuracy)

### GBDT_threshold — sota / tier1_core

**Roles:** nonlinear_interactions
**Status:** desc

**Purpose:** GBDT on cumulative logits; SOTA when relations are nonlinear.

**Use cases:**
- Risk bands in finance

**Config:** `{'n_estimators': 'int', 'max_depth': 'int'}`

**Benchmark:** AMLB → accuracy=0.6

## Metrics
- **mae_ordinal** (anchor, lower)
  - Distance-aware accuracy; respects ordered classes.
  - *special:* Preferred to accuracy; a 1→2 error is less costly than 1→5.
- **accuracy** (diagnostic, higher)
  - Exact-match accuracy.
  - *special:* Ignores distance; misleads when ordinal distance matters.

## Ensemble rules
- **[linear_signal, nonlinear_interactions]** → OrdinalLogit (max_per_role=1)
  - Single model — GBDT-threshold is described_only, wait for verification.
