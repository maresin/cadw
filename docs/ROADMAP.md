# CADW Roadmap

Last updated: 2026-10-30 (after v0.9.97).

## Current: v0.9.97

**Closed:** streaming-output migration (v0.9.92), wide-d quartet
(v0.9.93), unified narrow (v0.9.94), unified broad (v0.9.95),
G11 transitive-migration gate (v0.9.96),
YAML bounded_state_dims + df.estimate_memory (v0.9.97).

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

### v0.9.98 — CSR SpillBuffer + K-spill (sparse outputs)

**Goal:** unblock two remaining K-axis / sparse-output cases.

**Deliverables:**
- CSR SpillBuffer primitive: variable-width per-row payload,
  chunked Arrow IPC writes/reads (STREAMING §6.1 backlog).
- `tfidf.transform` (sparse mode) via CSR SpillBuffer.
- K-spill for `encode_categorical` (vocab > state_limit) via
  HashShuffle.
- K-spill for `metrics.classification` (K > state_limit).
- Optionally `lda::_build_vocab`, `tfidf::fit_tfidf` (vocab via
  HashShuffle).

**Tests:** `K = 10^7` categories OOC; `K = 10^6` classes OOC;
sparse tfidf nnz preserved.

**Breaking:** none (opt-in via `output="disk"` / `sparse=True`).

### v0.9.99 — D4 ROCKET variants + TS forecast extension

**Goal:** expand algorithm coverage in two families.

**Deliverables:**
- D4: `MultiRocket`, `Hydra`, `BOSS`, `Catch22` forecasters.
- TS forecast: `AutoARIMA`, `Croston`.
- YAML cards for all new models with §7 mandatory fields.
- G11 auto-guards any new producer with `output=`.

**Tests:** numerical parity vs paper reference; YAML §7 complete.

**Breaking:** none.

### v0.9.100 — version bump + release audit

**Goal:** fix the long-standing `__version__ = "0.9.88"` while tags
exist through v0.9.9x.

**Deliverables:**
- `cadw/__init__.py::__version__` and `pyproject.toml::version`
  synchronized with current tag.
- `test_version_consistency.py` verifies `tag == pyproject ==
  cadw.__version__` on master.
- Audit pass 3 (source conformance, STREAMING backlog).
- `sgd_streaming` mini-batch optimization.

**Breaking:** none (metadata only).

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
