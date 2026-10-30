# CADW Roadmap

Last updated: 2026-10-30 (after v0.9.98).

## Current: v0.9.98

**Closed:** streaming-output migration (v0.9.92), wide-d quartet
(v0.9.93), unified narrow (v0.9.94), unified broad (v0.9.95),
G11 transitive-migration gate (v0.9.96),
YAML bounded_state_dims + df.estimate_memory (v0.9.97),
tfidf.transform_sparse(output=) (v0.9.98).

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

### v0.9.99 — encode_categorical K-spill (HashShuffle)

**Goal:** encode categorical columns with vocab > state_limit
without O(K) RAM.

**Deliverables:**
- 3-pass HashShuffle encoder:
  1. hash vocab to P partitions, count per partition;
  2. assign global id ranges per partition;
  3. re-hash & encode rows with partition-local id offset.
- `CadwDataFrame.encode_categorical(columns, *, spill_mode="auto")`.
- `reverse_maps` reconstructed via partition-local id+name pairs.

**Tests:**
- K = 10^7 unique categories encoded without OOM.
- Bit-identical to current in-memory encoder for K <= 10^5.
- Determinism: same seed → same ids across two runs.

**Breaking:** none (opt-in via `spill_mode="disk"`).

### v0.9.100 — metrics.classification K-guard probe + fix

**Goal:** understand & fix `metrics.classification` for K > state_limit.
Probe v0.9.98 showed **zero** `check_state` calls in `cadw/ops/metrics.py`
— potential silent OOM at K=10^6 (confusion matrix K² cells).

**Deliverables:**
- Probe: which metric fns grow with K, which with K².
- Add `check_state` guards at the right places (confusion matrix,
  per-class arrays, ROC-AUC class loop).
- If K² is intrinsic (macro-metrics confusion): document
  `n_scalability: refuse_at_guard` in YAML card.
- If K linear: wrap per-class loop in SpillBuffer.

**Tests:**
- K = 10^5: all metrics work, no OOM.
- K = 10^6: either OOC, or explicit `MemoryLimitExceeded` with
  actionable message.

**Breaking:** none (guards fail-loud instead of silent OOM).

### v0.9.101 — lda::_build_vocab K-spill

**Goal:** same 3-pass HashShuffle pattern as v0.9.99, for LDA vocab.

**Deliverables:**
- `_build_vocab` via HashShuffle (partition by token hash, count
  per partition, k-way merge top-vocab_size).
- `tfidf::fit_tfidf` may reuse same primitive (check if same shape).

**Tests:**
- vocab_size = 10^6 on 10^5 docs without OOM.
- Deterministic top-vocab_size (same tie-breaking as current:
  `(-count, token)`).

**Breaking:** none.

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
