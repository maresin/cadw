# R2 — Sequential pattern mining

*Section:* R — Association rule mining

**Description:** Frequent subsequences in event sequences (order matters). Patterns like (A -> B -> C) over time-ordered transactions.

**Real-life examples:**
- User journey patterns
- Gene sequence motifs
- Log sequence mining

## Benchmarks
- **cadw_internal** — https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - metrics: sequence_coverage, runtime_s

## Models
### GSP — baseline / tier2_wrapper

**Roles:** local_structure, temporal_dynamics
**Status:** desc, export

**Purpose:** Srikant & Agrawal 1996: Generalized Sequential Patterns — Apriori-style for sequences. Multiple DB scans; simple but slow.

**Use cases:**
- User journey patterns
- Log sequence discovery

**Config:** `{'min_support': 'float', 'max_gap': 'int', 'max_len': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → sequence_coverage=null
  - algorithm: `SPMF GSP` (code)
  - number: https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - code/paper: https://www.philippe-fournier-viger.com/spmf/
  - note: no_public_benchmark: SPMF library reference (Fournier-Viger).

### PrefixSpan — sota / tier2_wrapper

**Roles:** local_structure, temporal_dynamics
**Status:** desc, export

**Purpose:** Pei et al. 2001: prefix-projected pattern growth. No candidate generation; depth-first on projected DBs. Usually much faster than GSP on large sequences.

**Use cases:**
- Large event logs
- User journey patterns
- Gene sequence motifs

**Config:** `{'min_support': 'float', 'max_len': 'int'}`

**Benchmarks:**
- **[literature]** cadw_internal → sequence_coverage=null
  - algorithm: `SPMF PrefixSpan` (code)
  - number: https://github.com/maresin/Cadw_framework/tree/main/.pingis/benches
  - code/paper: https://www.philippe-fournier-viger.com/spmf/
  - note: no_public_benchmark: SPMF reference. Pei et al. 2001.

## Metrics
- **sequence_coverage** (anchor, higher)
  - Fraction of input sequences containing at least one discovered pattern.
  - *special:* Depends on min_support; report across 3 thresholds.
- **runtime_s** (diagnostic, lower)
  - Wall-clock fit time.
  - *special:* Hardware-specific.
