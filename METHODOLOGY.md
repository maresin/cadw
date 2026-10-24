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

## 3. Benchmark selection — one primary + 1-2 secondary

| Task | Primary | Secondary |
|---|---|---|
| Binary / multiclass / regression | AMLB | Kaggle (different splits!) |
| Unsupervised anomaly | ODDS | ADBench |
| Supervised anomaly | AMLB | — |
| TS forecast | M4 | M5 (hierarchical only) |
| TS classification | UCR | UEA |
| TS anomaly | UCR_Anomaly | TSB-AD |
| Recsys | Amazon_Recsys | MovieLens |
| Uplift / causal | RCT_IHDP | — |
| LTR | LETOR | — |
| Survival | CoxPH_Survival | — |

**Rule:** each benchmark in `benchmarks.yaml` MUST have a `warning`
field stating the comparability caveat. Cross-benchmark numbers are
never comparable. A new benchmark is added only if it is used by
>= 3 papers from 2022+ and has a public split.

## 4. Verification — three levels

| Level | What | Where |
|---|---|---|
| **literature** | number from paper / leaderboard | `benchmark_result.source + url + value` |
| **reference** | run in isolated env (`requirements-bench.txt`) | `.pingis/benches/reproduction/<name>.json` |
| **discrepancy** | if literature != reference — prefer reference, document | `notes: "lit=0.53, ref=0.48 (different split)"` |

Never substitute one number for another without an explicit `notes`.
`tests/test_ml_tools_schema.py` enforces that `source` is in the
registry.

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
