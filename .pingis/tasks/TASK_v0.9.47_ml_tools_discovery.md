# TASK v0.9.47 — ML-tools matrix: remaining sections (I, J, K, L, M, N, O)

**Branch:** pingis/dev-audit
**Baseline:** HEAD = v0.9.46-fix2, tests 2966 passed, 2 skipped.
**Mode:** discovery — cards only (`described_only` or
`not_implemented_in_cadw`). **No `cadw/ml/*.py` code.**
**Patch:** v0.9.47 docs-only. One commit.

## Goal

Fill 7 remaining matrix sections. Priority: already-implemented-in-CADW
models first; SOTA additions only as `tier3_export`.

| Priority | Section | Subtypes | Already in CADW |
|---|---|---|---|
| 1 | **I** survival | I1 time-to-event | D.3.1 CoxPH |
| 2 | **N** density | N1 unsupervised density | D.6.8 GMM |
| 3 | **J** NLP | J1 text classification, J2 embeddings, J3 topic | D.6.9 TF-IDF |
| 4 | **M** bandits | M1 contextual bandit | D.6.7 LinUCB |
| 5 | **O** EDA / cleaning | O1 EDA, O2 cleaning, O3 FE, O4 selection, O5 validation, O6 sampling, O8 XAI, O9 fairness, O10 drift, O11 A/B | describe, hist, fillna, feature_select, perm_importance, fairness (D.1.4), drift (D.6.2) |
| 6 | **L** graph | L1 node embeddings (summary only) | — (reference only) |
| 7 | **K** vision | K1 summary only | — (torch-only, out-of-scope) |

## Per-section recipe

1. **Subtype** — per `docs/roadmap_research/part2_taxonomy.md`.
2. **Baseline** (>= 1) — well-known method + reference in CADW if exists.
   `tier=baseline`, `tier_level=tier1_core` (or `tier3_export` if no native).
3. **SOTA** (>= 1) — top-3 from primary benchmark (METHODOLOGY.md §3).
   torch/GPU-requiring → `tier3_export` + `not_implemented_in_cadw: true`.
4. **Benchmarks** — from `benchmarks.yaml` registry; new benchmark only
   if >= 3 papers from 2022+ and public split.
5. **Metrics** — exactly **one anchor**, 1-2 diagnostic/pareto_axis.
   Anchor from primary benchmark (e.g. C-index for survival, NMI for LDA).
6. **Roles** — from `docs/ml_tools/enums.py` (7 total).
7. **purpose** (2-3 sentences) + **real_life_examples** (>= 1).
8. **benchmark_result** — `source`/`url`/`metric`/`value` from literature.
   `notes` if lit != reference.

## Hard gates

- `pytest -q tests/test_ml_tools_schema.py` — all parametrized section
  tests pass (13 sections total after this patch: A/D/E/F/G/H + I/J/K/L/M/N/O)
- `pytest -q tests/test_render_matrix.py` — 15+ passed
- `uv run python scripts/render_matrix.py` — regenerate MATRIX.md +
  `docs/tasks/*.md`
- Each subtype: >= 1 baseline + >= 1 sota; anchor has `special`;
  non-verified models not in ensemble
- Each `benchmark_result.source` in `benchmarks.yaml`
- Each new benchmark has `warning`

## Do NOT

- **Do not implement** models in `cadw/ml/*.py` — cards only.
- **Do not add** new dependencies to `pyproject.toml`.
- **Do not update** `docs/ML_TASKS_TAXONOMY.md` (deprecated).
- **Do not touch** A/D/E/F/G/H YAML — accepted.

## Report

`.pingis/audits/v0.9.47_discovery_report.md`:

| Section | Subtypes | Cards | New benchmarks | Sources |
|---|---|---|---|---|
| I | 1 | 5 | CoxPH_Survival (existing) | Springer 2005, lifelines docs |
| N | 1 | 4 | — | Blei et al. 2003, sklearn GMM |
| J | 3 | 8 | — | sklearn TfidfVectorizer docs, gensim |
| ... |

Every `benchmark_result.url` must be a real URL. Every `tier3_export`
must have a reason (`needs torch` / `needs GPU` / `needs proprietary data`).

## Success metric

- 13 sections total (A-O, minus unused), ~50 subtypes, ~140 cards
- 100% `benchmark_result.source` in registry
- 100% `real_life_examples` non-empty
- 0 `tier3_export` without `not_implemented_in_cadw: true`
- `pytest tests/test_ml_tools_schema.py tests/test_render_matrix.py` green

## Not in scope

- M2.2 (LOF/OCSVM) — deferred to Phase H native.
- Phase F (Light NNs) — separate task.
- Reproduction for new models — separate task (needs reference).
