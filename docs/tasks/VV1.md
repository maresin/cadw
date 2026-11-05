# V1 — Calibration

*Section:* V — Uncertainty quantification

**Description:** Post-hoc adjustment of predicted probabilities so that P(y=1|p_hat) = p_hat. Improves reliability of classifiers / soft scores.

**Real-life examples:**
- Risk-critical decision systems
- Post-hoc classifier reliability
- Ensemble probability fusion

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: ece, brier_score

## Models
### PlattScaling — baseline / tier1_core

**Roles:** calibration
**Status:** desc, export

**Purpose:** Platt 1999: logistic sigmoid fit on scores. Simple, 2 parameters; preserves ranking; standard calibration baseline. CADW impl exists (cadw.ml.calibration.fit_calibration, Platt default).

**Use cases:**
- Calibrate SVM/logistic scores
- Rank-preserving calibration

**Config:** `{'method': 'platt'}`

**Benchmarks:**
- **[cadw_measured]** cadw_internal → ece=null
  - algorithm: `cadw.ml.calibration (platt)` (code)
  - number: .pingis/benches/calibration.json
  - code/paper: cadw/ml/calibration.py
  - note: CADW impl: fit_calibration(method="platt"); reference also sklearn.calibration.CalibratedClassifierCV.

### TemperatureScaling — baseline / tier2_wrapper

**Roles:** calibration
**Status:** desc, export

**Purpose:** Guo et al. 2017: single temperature parameter on softmax logits. Standard for neural nets; preserves accuracy; 1 parameter.

**Use cases:**
- Neural softmax calibration
- Confidence reliability

**Config:** `{'temperature': 'float'}`

**Benchmarks:**
- **[literature]** cadw_internal → ece=null
  - algorithm: `temperature scaling` (paper)
  - number: https://arxiv.org/abs/1706.04599
  - code/paper: https://arxiv.org/abs/1706.04599
  - note: Guo et al. 2017 per-dataset ECE.

### IsotonicCalibration — sota / tier1_core

**Roles:** calibration
**Status:** desc, export

**Purpose:** Zadrozny & Elkan 2002: non-parametric monotonic piecewise-constant map. More flexible than Platt; overfits on small calibration data. CADW impl exists (fit_calibration(method='isotonic')).

**Use cases:**
- Non-linear calibration map
- Large calibration set

**Config:** `{'method': 'isotonic'}`

**Benchmarks:**
- **[cadw_measured]** cadw_internal → ece=null
  - algorithm: `cadw.ml.calibration (isotonic)` (code)
  - number: .pingis/benches/calibration.json
  - code/paper: cadw/ml/calibration.py
  - note: CADW impl: fit_calibration(method="isotonic").

## Metrics
- **ece** (anchor, lower)
  - Expected Calibration Error: |p_hat - empirical|, binned. Lower = better calibrated.
  - *special:* Bin-count sensitive; report with 15 or 20 bins. Not comparable across different bin counts.
- **brier_score** (diagnostic, lower)
  - Mean squared error of probabilities (proper scoring rule).
  - *special:* Combines calibration + sharpness; not pure calibration metric.
