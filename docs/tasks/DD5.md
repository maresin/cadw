# D5 — TS change-point detection (alias of D3.2)

*Section:* D — Time Series

**Description:** Same as D3.2 — kept for taxonomy completeness. Use D3.2.

**Real-life examples:**
- See D3.2

## Benchmarks
- **UCR_Anomaly** — https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
  - metrics: auprc

## Models
### BinarySegmentation — baseline / tier1_core

**Roles:** temporal_dynamics
**Status:** desc

**Purpose:** Greedy top-down recursive splitting on L2 cost; simplest change-point baseline.

**Use cases:**
- Simple regime detection

**Config:** `{'n_bkps': 'int'}`

**Benchmarks:**
- **[literature]** UCR_Anomaly → auprc=0.38
  - algorithm: `ruptures Binseg` (docs)
  - number: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
  - code/paper: https://centre-borelli.github.io/ruptures-docs/user-guide/detection/binseg/
  - note: [legacy v0.9.44, migrated v0.9.50]

### PELT — sota / tier1_core

**Roles:** temporal_dynamics
**Status:** ✓ cadw, ✓ agent

**Purpose:** See D3.2.

**Use cases:**
- See D3.2

**Config:** `{'penalty': 'float'}`

**CADW:** `cadw.ml.pelt.fit_pelt`

**Benchmarks:**
- **[literature]** UCR_Anomaly → auprc=0.51
  - algorithm: `ruptures Pelt` (docs)
  - number: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
  - code/paper: https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/
  - note: [legacy v0.9.44, migrated v0.9.50]

## Metrics
- **auprc** (anchor, higher)
  - See D3.2.
  - *special:* See D3.2.

## Ensemble rules
- **[temporal_dynamics]** → PELT (max_per_role=1)
  - Use D3.2 for the full taxonomy; this is a stub.
