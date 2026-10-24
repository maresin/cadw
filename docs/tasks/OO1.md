# O1 — EDA / descriptive statistics

*Section:* O — Data Science outside ML (EDA / cleaning / feature engineering / validation)

**Description:** Summarize a dataset: per-column statistics, distributions, frequencies — aggregate-first, no row materialization.

**Real-life examples:**
- Quick profile of a new table
- Report-ready summary statistics
- Distribution sanity check before modeling

## Benchmarks
- **sklearn_reference** — https://scikit-learn.org/stable/modules/
  - metrics: describe, hist, value_counts

## Models
### Describe — baseline / tier1_core

**Roles:** linear_signal
**Status:** ✓ cadw, ✓ agent

**Purpose:** Per-column count/mean/std/min/max/quantiles computed by a Reduce over chunks. Inductive bias: none — pure descriptive summary; no model.

**Use cases:**
- New-table profiling
- Feature sanity before fit
- Report tables

**Config:** `{'quantiles': 'list'}`

**CADW:** `cadw.dataframe.describe`

**Benchmarks:**
- **[cadw_measured]** cadw_internal → fit_time_s=9.2618 (n=2)
  - algorithm: `cadw.CadwDataFrame.describe` (code)
  - number: .pingis/benches/eda.json
  - code/paper: cadw/dataframe/_aggregations.py
  - note: CADW-native: describe fit_time_s=0.4961 @n=10k, 9.2618 @n=100k (d=11, single-chunk; eda.json). NOT comparable to external.
- **[cadw_measured]** cadw_internal → peak_rss_mb=7.13 (n=2)
  - algorithm: `cadw.CadwDataFrame.describe` (code)
  - number: .pingis/benches/eda.json
  - code/paper: cadw/dataframe/_aggregations.py
  - note: CADW-native: peak_mb=3.46 @n=10k, 7.13 @n=100k (eda.json). NOT comparable to external.

### Hist — baseline / tier1_core

**Roles:** representation
**Status:** ✓ cadw, ✓ agent

**Purpose:** Binned per-column histogram via a Shuffle+Reduce over chunk-wise partial counts. Inductive bias: fixed-width bins; no model.

**Use cases:**
- Distribution plots
- Skew detection
- Report figures

**Config:** `{'bins': 'int'}`

**CADW:** `cadw.plot_accessor.hist`

**Benchmarks:**
- **[literature]** sklearn_reference → accuracy=null
  - algorithm: `sklearn.preprocessing.KBinsDiscretizer` (docs)
  - number: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html
  - code/paper: https://scikit-learn.org/stable/modules/preprocessing.html
  - note: no_public_benchmark: binning is a preprocessing primitive; no accuracy leaderboard applies.

### ValueCounts — baseline / tier1_core

**Roles:** probabilistic
**Status:** ✓ cadw, ✓ agent

**Purpose:** Per-category frequency via groupby+Reduce. Inductive bias: none — counts only.

**Use cases:**
- Cardinality check
- Top-K category report
- Class imbalance look

**Config:** `{'normalize': 'bool'}`

**CADW:** `cadw.dataframe.value_counts`

**Benchmarks:**
- **[literature]** sklearn_reference → accuracy=null
  - algorithm: `pandas.Series.value_counts` (docs)
  - number: https://scikit-learn.org/stable/modules/feature_selection.html
  - code/paper: https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html
  - note: no_public_benchmark: frequency counting has no accuracy benchmark. CADW multi-chunk value_counts hits BUG-1 (see .pingis/audits/v0.9.48_bugs.md).

### YDataProfiling — sota / tier2_wrapper

**Roles:** representation
**Status:** desc, export

**Purpose:** Full dataset profile (missingness, cardinality, correlations, distributions) in one pass; external library, not shipped.

**Use cases:**
- Automatic EDA report
- Data-quality hand-off
- Notebook profiling

**Config:** `{'minimal': 'bool'}`

**Benchmarks:**
- **[literature]** sklearn_reference → accuracy=null
  - algorithm: `ydata-profiling.ProfileReport` (docs)
  - number: https://scikit-learn.org/stable/
  - code/paper: https://github.com/ydataai/ydata-profiling
  - note: no_public_benchmark: no EDA-profiling leaderboard exists. Official docs URL https://ydata-profiling.ydata.ai/docs/master/ returned status=404 (fetch_failed); the algorithm source is the GitHub repo.

## Metrics
- **fit_time_s** (anchor, lower)
  - Wall-clock for a full describe over the frame.
  - *special:* Chunk-size and d dependent; report n and d.
- **peak_rss_mb** (diagnostic, lower)
  - Peak resident memory during describe.
  - *special:* tracemalloc-based; single-chunk numbers only (see BUG-1).

## Ensemble rules
- **[representation]** → Describe + Hist (max_per_role=3)
  - Layer the external profiler on top of CADW primitives for a richer report while keeping compute in the frame.
