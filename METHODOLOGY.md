# CADW — Research & ML-tools methodology

Version: 0.9.47 (2026-10-24)
Bridge between chat sessions. This file lives in the repo root (not in
`docs/`) so that `scripts/collect_project.py --mode=tiny` includes it
fully in the snapshot — `docs/**` is STUB-only in tiny mode.

## 1. Discovery — where to look

### SOTA leaderboards (primary)
- **Papers with Code** (paperswithcode.com/sota) — filter by task and by
  modality (tabular / time-series / anomaly / recsys).
- **Kaggle competitions** — what wins, with a date. Older than 3 years:
  check if a simpler model has since overtaken it.
- **UCR/UEA** (timeseriesclassification.com), **ODDS**, **ADBench**,
  **M4/M5**, **LETOR**, **Amazon_Recsys** — accepted benchmarks. Numbers
  are not cross-comparable (different split / prevalence / label).

### Reviews (secondary)
- Annual Review of Statistics and Its Application (2018-2025).
- JMLR / NeurIPS / ICML / ICLR proceedings — prefer a survey over a
  single paper.
- arXiv: search with `survey` and a date filter.

### Ecosystem check (triage)
- sklearn / PyOD / sktime / implicit / lifelines / statsmodels docs —
  what is production-ready, what is marked experimental.
- `awesome-ml`, `awesome-time-series`, `awesome-recsys` — breadth only.

## 2. Triage — what to include

| Tier | Inclusion rule |
|---|---|
| **tier1_core** | Implementable on CADW primitives (Reduce / Shuffle / Sort); < 500 LOC; no torch/GPU. |
| **tier2_wrapper** | Thin wrapper < 50 LOC over a light pure-Python library (< 5 MB). |
| **tier3_export** | SOTA but requires torch/GPU/heavy deps — reachable only via `to_batches`. |

**Reject** if: proprietary; no reference implementation; numbers not
verifiable from a public benchmark; no usable `roles` slot.

**Roles are mandatory** (`roles` in YAML, enum in `docs/ml_tools/enums.py`):
`linear_signal`, `nonlinear_interactions`, `local_structure`,
`probabilistic`, `temporal_dynamics`, `representation`, `calibration`.

## 3. Benchmark selection — multi-benchmark per subtype

A **benchmark** defines a canonical dataset set and a protocol for
aggregation (typically mean ± std across N datasets). A **leaderboard**
ranks models by the aggregate. `task_type` names the benchmark's task
family subset (AMLB covers binary / multiclass / regression; the
`n_datasets` count differs per task).

For each subtype, collect **every benchmark** with published results in
papers from 2022+:

| Task family | Typical benchmarks | Count |
|---|---|---|
| Binary / multiclass / regression | AMLB, OpenML_CC18, Kaggle (1-2 recent) | 2-4 |
| Unsupervised anomaly | ODDS, ADBench, PyOD benchmark | 2-3 |
| Supervised anomaly | AMLB | 1 |
| TS forecast | M4, M5, M3, tourism | 2-4 |
| TS classification | UCR, UEA | 2 |
| TS anomaly | UCR_Anomaly, TSB-AD | 1-2 |
| Recsys | Amazon_Recsys, MovieLens-1M, Yelp | 2-3 |
| Uplift / causal | RCT_IHDP, ACIC, Lalonde | 2-3 |
| LTR | LETOR, MSLR-WEB30K | 1-2 |
| Survival | CoxPH_Survival, SUPPORT2, METABRIC | 2-3 |
| Density | sklearn density benchmark, UCI (1-2) | 1-2 |
| Bandits | MovieLens for bandits, synthetic Yahoo R6 | 1-2 |
| NLP | SST-2, AG News, 20 Newsgroups, Reuters | 2-4 |

**Minimum:** 1 benchmark (niche subtype). **Mainstream:** 2-3.

**Metrics per benchmark:** anchor + 1-2 diagnostics. Include **every**
metric the public leaderboard publishes. If leaderboard gives AUC and
F1 — both go in.

**Per-dataset stories** (highest/lowest value) go in `note` as prose,
not as a structured field. Benchmarks define the canonical dataset
set; enumerating it per card is duplication.

**Special case `cadw_internal`:** for models whose only verified number
is CADW's own measurement (memory, runtime, synthetic DGP). Registered
in `benchmarks.yaml` with `warning: "CADW-native measurement, NOT
comparable to external leaderboards"`.

## 4. Verification — two sources per number + 3 attribution tiers

Every `BenchmarkResult` records **two URLs**:

| Field | What | Example |
|---|---|---|
| `benchmark_url` | WHERE the number comes from | leaderboard page, or `.pingis/benches/<name>.json` |
| `algorithm_source_url` | WHERE the algorithm is documented | paper DOI, arXiv, GitHub, official docs |

These URLs are often on **different domains** and are **never the same
URL** for a literature entry. `algorithm_source_kind` ∈
{`paper`, `code`, `docs`, `pseudocode`, `description`} labels what kind
of source it is.

### Attribution tiers (3)

| `attribution` | Meaning | When set |
|---|---|---|
| `literature` | Number from external publication | PINGIS fetched page, quoted number |
| `cadw_reproduced` | CADW matched the original within tolerance | M2 `reproduce_benchmark.py` verdict=consistent |
| `cadw_measured` | CADW measured, no external reference (memory/runtime/synthetic DGP) | `.pingis/benches/<name>.json` |

**Rule:** before a `cadw_reproduced` exists, source = `literature` (with
`algorithm` naming the external implementation and `algorithm_source_url`
pointing to its paper/code). When CADW matches it, **add** a second
`cadw_reproduced` entry — do not replace the literature one. Both stay,
so the matrix shows "CADW at parity with LightGBM on AMLB".

`reproduces` field is required iff `attribution == "cadw_reproduced"`.
It names the **literature algorithm** (must match an `algorithm` in a
sibling literature entry for the same `benchmark` + `metric`).

### Discrepancy

If literature value differs from reference reproduction: prefer the
reference, document in `note: "lit=0.53, ref=0.48 (different split)"`.
Never overwrite the literature value with the reproduction value — they
are separate rows.

### Enforced by

- `tests/test_ml_tools_schema.py` — attribution-specific rules
  (literature ≠ cadw.*, reproduces required iff cadw_reproduced,
  note required when value set, std requires n_datasets >= 2).
- `scripts/audit_url_log.py` — every literature entry needs a
  `url_log.jsonl` entry with `status=200` for BOTH `benchmark_url` and
  `algorithm_source_url`; cadw_reproduced needs a `.pingis` JSON with
  `verdict=consistent`; cadw_measured needs both paths to exist.

## 5. Registration — how to add a model

1. YAML card in `docs/ml_tools/<Section>_*.yaml` per `schema.py`.
2. `pytest -q tests/test_ml_tools_schema.py` — validator (roles enum,
   anchor, 1 baseline + 1 sota, non-verified NOT in ensemble).
3. `uv run python scripts/render_matrix.py` — regenerate MATRIX.md +
   `docs/tasks/*.md`.
4. If `verified_in_cadw: true` — a reproducer in
   `scripts/reproduce_benchmark.py` + `.pingis/benches/reproduction/<name>.json`.
5. CHANGELOG entry.

## 6. Debugging divergence

When a reproducer reports `divergent` — do **not** start tuning
parameters. Run this ladder in order:

1. **One-step on tiny synthetic** (5 users x 4 items, `n_iter=1`).
   Compare against the manual formula computed with `_solve_user` /
   `_gram` / equivalent. Reveals: is the formula correct? Is one
   update equivalent?
2. **Multi-step (2-3 iter), same tiny synthetic.** Reveals: order of
   updates? Cache invalidation?
3. **Init equivalence.** If reference uses a different
   `seed -> init` mapping, one-step comparisons are meaningless until
   X0/Y0 are aligned.
4. **Parameter sweep** (init, alpha, reg) — only if 1-3 pass.
5. **DGP coverage check.** `assert len(unique(train_items)) == n_items`
   and `len(unique(train_users)) == n_users`. Sparse DGP with `top-k`
   selection silently loses cold items — a bug in the sample, not the
   model.
6. **Metric comparison** (NDCG / RMSE) — final gate.

**Lesson v0.9.46 ALS:** 6 iterations spent on steps 4/5 first; the bug
was found on step 1 (item-sweep sorted by `["i"]` = user_pos instead of
`["u"]` = item_pos). Always start with step 1.

## 7. PINGIS vs human

| Role | Does |
|---|---|
| **PINGIS** (research) | Discovery, triage, benchmark selection, literature verification, YAML cards. |
| **Human** (reviewer) | Approve additions, spot-check 10% cards, run reproduction gates. |
| **Auto-tests** | Schema validator + reproduction consistency. |

PINGIS **does not implement** models without a separate TASK. Discovery
produces cards with `described_only: true` or `not_implemented_in_cadw: true`,
no code.

## 8. Sources of truth

- `docs/ml_tools/enums.py` — the 7 roles, 3 tiers, 3 combination values.
- `docs/ml_tools/benchmarks.yaml` — the 12+ registered benchmarks with
  mandatory `warning`.
- `docs/ml_tools/schema.py` — dataclass contract + `validate_matrix()`.
- `docs/ml_tools/README.md` — full debugging ladder (this file is a
  shorter bridge for chat sessions).
- `.pingis/tasks/` — PINGIS assignments.
