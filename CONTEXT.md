# CADW — Context Handoff

**Version:** 0.9.32 (2026-10-23)
**Goal:** out-of-core dataframe на Apache Arrow для данных, не
влезающих в RAM, на слабых машинах.
**Constraint:** ≤1 batch в RAM, всё тяжёлое через 3 примитива:
Reduce / HashShuffle / Sort.

## Phase Status

| Phase | Status | Notes |
|---|---|---|
| A — Stabilization | ✅ completed | 11 items, A.6 → G |
| B — Infrastructure | ✅ completed | 8/8 milestones (B.1 split, B.2 decisions, B.3 tests, B.5 scripts, B.6 cleanup) |
| C — Pandas P0 | ✅ completed | 20/20 features (isin, between, idxmax, … , dropna, fillna) |
| D — Algorithms P1 | 🚀 in progress | D.1.1 IForest, D.1.3 cost-sensitive GBDT done; D.3.1 CoxPH deferred |
| E — Recipes | planned | — |
| F — Light NNs | planned | — |
| G — Tech debt | planned | A.6 deferred here |
| H — P2 / long-tail | planned | — |

## Where to look

### Decisions & rules
- `docs/decisions/00_index.md` — TOC + reading guide (начни здесь)
- `docs/decisions/01_intro_principles.md` — philosophy, glossary
- `docs/decisions/02_primitives.md` — Reduce, Shuffle, Sort, why 3
- `docs/decisions/10_9_memoryguard*.md` — OOM safety, formulas
- `docs/decisions/14-15a-d_*.md` — §13.4 lessons (60+ Phase B lessons)
- `DECISIONS.md` — 64-line index (ссылки на всё выше)

### Code (v0.9.30)
- `cadw/core/memory_guard.py` (857) — planning, formulas
- `cadw/core/sampling.py`, `models*.py` — memory model ops
- `cadw/ops/` (15 modules) — primitives + groupby/join/dedup/rank/quantile/resample/pivot/topk/sort/shuffle/histogram/metrics
- `cadw/ml/` (6 modules) — linear, logistic, kmeans, pca, gbdt, naive_bayes
- `cadw/ts/` (6 modules) — features, stats, models, backtest, thf
- **`cadw/dataframe/` (18 modules, B.1 split from 7872-LOC monolith)**:
  - `_core.py` (~530 LOC) — class + `__init__`, `__getitem__`, props, `info`
  - `_helpers.py` — 5 module-level helpers (build_out_schema, detect_mode, normalize_scalar, infer_arrow_type_from_value, apply_filter_with_positions)
  - 12 mixins: `_exports`, `_iterators`, `_misc`, `_indexing`, `_aggregations`, `_transformers_fill`, `_transformers_reshape`, `_rolling`, `_cumulative`, `_per_group`, `_resample_reshape`, `_join`, `_ml_bindings`, `_columns`
  - `_column_proxy.py` (~320) — `_ColumnProxy` class
  - `_groupby.py` (~220) — `_GroupBy` class
- `cadw/dataset/` — `Dataset`, `CadwIndex`, `MaterializedIndex`, `RowFilter`
- `cadw/io/` — batch reader/writer
- `cadw/cv.py`, `cadw/pipeline.py`, `cadw/scalers.py` — cross-cut

### Tests
- `tests/` — ~2500 fast tests (~145s single, ~74s parallel)
- Selective: `pytest -m ts` (103), `-m ml` (406), `-m ops` (414), `-m viz` (52), `-m pipeline` (105)
- Parallel: `pytest -n auto -m "not slow"` → 1.9× speedup
- Slow: `pytest -m slow` (4 tests, E2E + OOM)

## How to work

```bash
# Fast suite
pytest -q -m "not slow"           # ~145s
pytest -m "ts and not slow"       # ~5s
pytest -n auto -m "not slow"      # ~74s

# Find §13.4 lesson
grep -rn "§13.4/274" docs/decisions/

# Context snapshot
uv run python scripts/collect_project.py --mode=tiny --output /tmp/tiny.md  # 141 KB manifest
# (--phase B and --doc docs/decisions/ BUGGED post-B — see known gaps)

# History
git log --oneline -20
```

## Known gaps (v0.9.30)

- ⚠️ **README.md stale** (says v0.8.8, pre-B.1) — not authoritative
- ⚠️ **Phase C roadmap items show 0/7 done** — should be 20/20 (accounting drift, not code)
- ⚠️ **Phase B items have stale `file` paths** — B.1 says `cadw/dataframe.py`, real: `cadw/dataframe/*.py`
- ⚠️ **`collect_project --phase/--doc` broken** — parser not updated for `docs/decisions/` + split files
- ✅ **`tail()` added v0.9.26y2** — was missing pre-B

## Immediate next

v0.9.32 closed: F9 gbdt RAM gap (batch_memory_factor); D.1.3 scale_pos_weight + tests.
Next: D.3.1 CoxPH (deferred), then D.1.2 Matrix Profile / D.1.4 fairness.
FOR_PINGIS: AUDIT_v0931.md (findings), algorithm_notes.md (D.1.3 sources + D.3.1 deferral).

## Documentation navigation (v0.9.33)

- **`docs/ML_TASKS_TAXONOMY.md`** — master navigator: 54 ML task
  types → 3-tier algorithms (baseline / best / ensemble) + CADW
  status. **Start here** when picking a model.
- **`docs/BUG_BACKLOG.md`** — bug backlog (Phase I), 8 items, exact
  patches.
- **`docs/ALGORITHM_BACKLOG.md`** — pending Phase D, 25 items, with
  universal test/bench requirements.
- **`docs/ROADMAP.md`** — consolidated phase roadmap (A-I).
- **`docs/roadmap_research/`** — PINGIS research (part1/part2/
  consolidated roadmap).
- **`.pingis/tasks/`** — PINGIS work assignments (TASK_v0934 bugs,
  TASK_v0935 algorithms).
- **`.pingis/audits/`** — PINGIS audit reports.
- **`.pingis/algorithm_notes/`** — per-algorithm design notes.
- **`.pingis/benches/`** — raw bench numbers.

## Current state (v0.9.33)

- Bugs backlog: 8 items → TASK_v0934 (bug pass, no deferral).
- Algorithm backlog: 25 items → TASK_v0935 (algorithm pass, no
  deferral).
- PINGIS workspace: `.pingis/` — all artifacts centralised.

## Roadmap update (v0.9.43)

**Phases A-D:** ✅ closed (Phase D = 29/29 algorithms, 25 notes
+ 25 benches).

| Phase | Status | DoD |
|---|---|---|
| **G** tech debt | ✅ closed v0.9.43 (A.6, .hypothesis, bench-deps isolation) | — |
| **M1** matrix schema | 🔜 v0.9.44 (blocker for F/H) | pydantic schema + 5 sections (A, D, E, F, H) + validator test |
| **F** Light NNs | after M1 | model + tests + bench + **card in matrix** |
| **H** Long-tail | after M1 | model + tests + bench + **card in matrix** |
| **E** Recipes | after F/H | reduced to views over the matrix |
| **M2** Reproduction | parallel | isolated requirements-bench.txt + scripts/reproduce_benchmark.py |
| **M3** Fill remaining | parallel with F/H | 10 more sections (B, C, G, I, J, K, L, M, N, O) |

**DoD change for F/H:** a model is `done` only when:

1. `cadw/ml/<name>.py` implemented + ≥8 tests + bench
2. Card in `docs/ml_tools/<section>.yaml` with:
   `roles`, `purpose`, `real_life_examples`, `benchmark_result`,
   `implementation_status: {verified_in_cadw: true, verified_by_agent: true}`
3. `tests/test_ml_tools_schema.py` passes

**Deprecated:** `docs/ML_TASKS_TAXONOMY.md` (top-of-file notice) —
replaced by matrix after M1 populates 5 sections.

**Bench-deps isolation (v0.9.43):**

- `requirements-bench.txt` — pyod, ruptures, lifelines, implicit,
  statsmodels, sklearn. NOT in `pyproject.toml`.
- AST guard `tests/test_no_bench_deps_in_core.py` prevents `cadw/`
  from importing these. Reference work lives only in `scripts/` and
  `tests/`.

