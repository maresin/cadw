# V2 — Conformal prediction

*Section:* V — Uncertainty quantification

**Description:** Distribution-free prediction intervals with finite-sample coverage guarantee (under exchangeability).

**Real-life examples:**
- Risk-critical intervals
- Model-agnostic uncertainty
- Guaranteed coverage

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: empirical_coverage, mean_interval_width

## Models
### SplitConformal — baseline / tier2_wrapper

**Roles:** calibration
**Status:** desc, export

**Purpose:** Papadopoulos 2002 / Vovk 2005: split conformal. Hold-out calibration set; interval = prediction +/- quantile of calibration residuals. 1/n coverage guarantee.

**Use cases:**
- Interval around point predictions
- Model-agnostic uncertainty

**Config:** `{'coverage': 'float', 'calibration_fraction': 'float'}`

**Benchmarks:**
- **[literature]** cadw_internal → empirical_coverage=null
  - algorithm: `split conformal` (paper)
  - number: https://arxiv.org/abs/2107.07511
  - code/paper: https://arxiv.org/abs/2107.07511
  - note: Angelopoulos & Bates 2021 tutorial; standard method.

### CVplus — sota / tier2_wrapper

**Roles:** calibration
**Status:** desc, export

**Purpose:** Barber et al. 2021: cross-conformal (CV+) prediction. Uses k-fold residuals; finite-sample coverage without split loss; tighter intervals than split conformal.

**Use cases:**
- Tighter intervals (no holdout loss)
- Small-data conformal

**Config:** `{'coverage': 'float', 'n_folds': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → empirical_coverage=null
  - algorithm: `CV+ conformal` (paper)
  - number: https://arxiv.org/abs/2107.07511
  - code/paper: https://arxiv.org/abs/2107.07511
  - note: Barber et al. 2021; CV+ method.

## Metrics
- **empirical_coverage** (anchor, target)
  - Observed fraction of true y in prediction interval; should match target coverage (e.g. 0.95).
  - *special:* Target = nominal coverage; deviation matters.
- **mean_interval_width** (diagnostic, lower)
  - Average interval width; tighter is better at same coverage.
  - *special:* Trades off with coverage; report jointly.
