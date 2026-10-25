# D4 — TS classification

*Section:* D — Time Series

**Description:** Predict a categorical label per series.

**Real-life examples:**
- Activity recognition
- ECG arrhythmia classes

## Benchmarks
- **UCR_Classification** — https://www.timeseriesclassification.com/
  - metrics: accuracy

## Models
### IntervalFeatures_GBDT — baseline / tier1_core

**Roles:** nonlinear_interactions, temporal_dynamics
**Status:** ✓ cadw, ✓ agent

**Purpose:** tsfresh-subset interval features + GBDT.

**Use cases:**
- Sensor activity recognition

**Config:** `{'windows': 'tuple', 'stats': 'tuple'}`

**CADW:** `cadw.ts.stats.rolling_stats`

**Benchmarks:**
- **[literature]** UCR_Classification → accuracy=0.82
  - algorithm: `sklearn.ensemble.HistGradientBoostingRegressor` (docs)
  - number: https://www.timeseriesclassification.com/
  - code/paper: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
  - note: [legacy v0.9.44, migrated v0.9.50]

### MiniRocket — sota / tier1_core

**Roles:** representation
**Status:** ✓ cadw, ✓ agent

**Purpose:** Random convolutional kernels; near-SOTA with 1/1000 of the compute.

**Use cases:**
- UCR activity recognition

**Config:** `{'num_kernels': 'int'}`

**CADW:** `cadw.ml.minirocket.fit_minirocket`

**Benchmarks:**
- **[literature]** UCR_Classification → accuracy=0.89
  - algorithm: `MiniRocket` (paper)
  - number: https://www.timeseriesclassification.com/
  - code/paper: https://arxiv.org/abs/2103.01827
  - note: [legacy v0.9.44, migrated v0.9.50]

### HIVE-COTE — sota / tier3_export

**Roles:** representation, temporal_dynamics, nonlinear_interactions
**Status:** desc, export

**Purpose:** Ensemble of 5+ TS classifiers; top of UCR leaderboards.

**Use cases:**
- UCR leaderboard

**Benchmarks:**
- **[literature]** UCR_Classification → accuracy=0.93
  - algorithm: `HIVE-COTE 2.0` (paper)
  - number: https://www.timeseriesclassification.com/
  - code/paper: https://arxiv.org/abs/2010.09155
  - note: [legacy v0.9.44, migrated v0.9.50]

## Metrics
- **accuracy** (anchor, higher)
  - Mean accuracy over resamples (UCR official).
  - *special:* UCR uses 30 resamples; single-split accuracy is not comparable.
- **mean_rank** (pareto_axis, lower)
  - Average rank across UCR datasets.
  - *special:* Rank on many datasets; not comparable to accuracy on one.

## Ensemble rules
- **[representation, nonlinear_interactions]** → MiniRocket + IntervalFeatures_GBDT (max_per_role=1)
  - Stack features from both and fit a linear head; +1-2% accuracy.
