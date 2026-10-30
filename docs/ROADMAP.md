# CADW Roadmap

Last updated: 2026-10-30 (after v0.9.99).

## Current: v0.9.99

**Closed:** streaming-output migration (v0.9.92), wide-d quartet
(v0.9.93), unified narrow (v0.9.94), unified broad (v0.9.95),
G11 transitive-migration gate (v0.9.96),
YAML bounded_state_dims + df.estimate_memory (v0.9.97),
tfidf.transform_sparse(output=) (v0.9.98),
metrics guards verification + encode_categorical error (v0.9.99).

All 13 producers return CadwDataFrame regardless of output=;
output= selects storage strategy only. Pipeline works with all
migrated models out-of-the-box.

| Producer class | Status |
|---|---|
| ops (groupby/dedup/topk) | ✅ CadwDataFrame |
| ML predict/transform (iforest, chains, ordinal, gmm, tfidf, randsvd, coxph) | ✅ CadwDataFrame |

**Legacy (v0.9.92):** streaming-output migration.
- All ops + ML predict return `CadwDataFrame` via facade
  (or `AutoResult` / `AutoArrayResult` direct, `output="auto"`).
- No hidden OOM in migrated functions.
- Test count: 4308 passed, 8 skipped (fast; includes G1-G11;
  G11 adds 142 transitive-migration parameters).

**Known non-OOC** (per STREAMING §6.1) — documented, not regressions:
- `binseg`, `stray`: intrinsic O(n)/O(n²), refuse-at-guard.
- `linear`/`logistic`/`coxph`/`pca`: O(d²) state, wide-d not handled.
- `encode_categorical`, `metrics`-classification: K guard, no spill.
- TS single-series fits: `tier1_core_exception`.
- sparse `tfidf.transform`: needs CSR SpillBuffer.

## Next 3 sessions

### v0.9.100 — version bump + audit pass 3

**Goal:** close the long-standing `__version__ = "0.9.88"` while
tags exist through v0.9.99; audit source conformance.

**Deliverables:**
- `cadw/__init__.py::__version__` → current tag (`"0.9.100"`).
- `pyproject.toml::version` → `"0.9.100"`.
- `test_version_consistency.py` tightened: on master, `git tag
  --points-at HEAD == vX.Y.Z` requires `pyproject.toml::version ==
  X.Y.Z == cadw.__version__`. (Currently only checks
  pyproject==cadw; tag mismatch silent.)
- Audit pass 3 (STREAMING §6.1 backlog): grep `check_state` count
  per module, verify no unguarded n-sized allocs in v0.9.90-99
  new files, source conformance report.

**Tests:** `test_version_consistency.py` + audit script output.

**Breaking:** none (metadata).

### v0.9.101 — lda::_build_vocab K-spill

**Goal:** OOC LDA vocab build for K > state_limit (no API break).

**Deliverables:**
- 2-pass HashShuffle on tokens:
  1. Shuffle docs by token hash → P partitions, per-partition
     token counts.
  2. K-way merge top-`vocab_size` by `(-count, token)`.
- Bit-identical to current in-memory when K ≤ state_limit.
- `MemoryGuard` guards on `state_limit` for per-partition dicts.

**Tests:** K=10^6 on 10^5 docs without OOM; determinism;
tie-breaking matches current impl.

**Breaking:** none.

### v0.9.102 — lazy reverse_maps (breaking, opt-in)

**Goal:** `encode_categorical(columns, reverse_maps="disk")` —
SpillMap on disk instead of `list[unique_values]`.

**Deliverables:**
- New `SpillMap` type: `.get(k)`, `.index(v)` (disk scan),
  `.len()`, `.iter()`. NOT a `list` — breaking API.
- `encode_categorical(columns, *, reverse_maps="memory"|"disk")`,
  default `"memory"` (backward compat, current behavior).
- CHANGELOG banner: BREAKING only when user opts `"disk"`.
- Underlying state becomes O(batch); K-axis moves to disk.

**Tests:** memory==disk at K=10^5 (bit-identical); K=10^7 OOC
fit; `.index` semantics; SpillMap cleanup on GC.

**Breaking:** yes (opt-in only).


## Do not re-solve

Per STREAMING §6.1 — won't-fix with rationale:

- **eif bit-identical chunked** (S6b-full): 40 GB I/O at n_est=100,
  n_w=10^6, d=50. Zero functional gain (tests pass at recall 0.5).
- **kmeans_ad bit-identical** (S6c-full): 3.2 GB I/O at k=8,
  n_w=10^6. Reservoir is deterministic; tests pass.
- **autoencoder beyond n_w = state_limit/16** (S6d-full):
  2-3 sessions, no urgent use case.
- **sparse tfidf.transform**: variable-width rows, needs CSR
  SpillBuffer design (separate).
- **binseg / stray**: intrinsic O(n) / O(n²). `refuse_at_guard`
  is correct behaviour.
- **TS single-series fits** (STL, VAR, VECM, ETS, Theta):
  `tier1_core_exception`, n ≤ 10^6 typical.

- **CoxPH concordance streaming** (v0.9.97 audit): current
  implementation is O(n) chunked (~8 MB/chunk bool matrix),
  NOT O(n²) as v0.9.96 brief §8 claimed. True streaming
  (external sort + Fenwick inversion) gives O(n log n) I/O
  for a statistic unstable at n > 10⁶ anyway. Won't-fix until
  a real use case appears.

If a future session considers one of these, **read STREAMING §6.1
first** — cost calculations and rationale are there.

## Backlog (not scheduled)

- `df.estimate_memory(op, params)` — pre-flight RAM estimate
  (per-axis). Would let users see `fits: false, reason, alternatives`
  before calling.
- YAML `bounded_state_dims` for all 84 verified cards.
- Audit pass 3 (after v0.9.95).
- `sgd_streaming` mini-batch optimization (currently single-sample).

## Session discipline

- Before commit: `pytest -q <affected files>` (focused, ~20s).
- Before tag: `pytest -q -m "not slow"` (full fast, ~10min).
- Before release: `pytest -q` (all, includes slow).
- See `AI_CONSTITUTION.md` §4 for any new `fit_*` / `predict_*`.
