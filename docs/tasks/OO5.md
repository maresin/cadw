# O5 — Validation / data quality

*Section:* O — Data Science outside ML (EDA / cleaning / feature engineering / validation)

**Description:** Assert schema and statistical contracts on incoming batches before fitting or scoring.

**Real-life examples:**
- Pipeline entry validation
- Schema drift detection
- Data contract enforcement

## Benchmarks
- **sklearn_reference** — https://scikit-learn.org/stable/modules/
  - metrics: validation_rate

## Models
### PanderaSchema — baseline / tier2_wrapper

**Roles:** calibration
**Status:** desc

**Purpose:** Declarative dataframe schema (dtypes, ranges, unique, custom checks). Inductive bias: a fixed contract; no learning.

**Use cases:**
- Parquet-load schema guard
- Statistical range asserts
- Pandas-style data contracts

**Config:** `{'strict': 'bool'}`

**Benchmarks:**
- **[literature]** sklearn_reference → validation_rate=null
  - algorithm: `pandera.DataFrameSchema` (docs)
  - number: https://pandera.readthedocs.io/en/stable/
  - code/paper: https://github.com/unionai-oss/pandera
  - note: no_public_benchmark: schema validators have no accuracy benchmark; correctness is binary by construction.

### GreatExpectationsSuite — sota / tier3_export

**Roles:** calibration
**Status:** desc, export

**Purpose:** Suite of declarative expectations over data (types, ranges, distributions) with data-docs generation. Inductive bias: fixed contract suite, no learning.

**Use cases:**
- Batch pipeline contracts
- Data-doc generation
- Cross-team data SLAs

**Config:** `{'run_name': 'string'}`

**Benchmarks:**
- **[literature]** sklearn_reference → validation_rate=null
  - algorithm: `great_expectations.Validator` (docs)
  - number: https://docs.greatexpectations.io/docs/
  - code/paper: https://github.com/great-expectations/great_expectations
  - note: no_public_benchmark: validator libraries publish no benchmark ranking; value is correctness of the contract.

## Metrics
- **validation_rate** (anchor, higher)
  - Fraction of batches that pass all assertions.
  - *special:* Contract-specific; report the contract version.
