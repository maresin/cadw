# CADW Refactor Plan — v0.9.87

**Started:** 2026-10-27
**Source:** audit d07072d + fix1a-c (sub_pca/eif/kshape/s2g/dwt_mlead/
tspq/autoencoder/kmeans_ad/BOCPD)
**Governing doc:** docs/ml_tools/METHODOLOGY_ALGORITHM_SELECTION.md

## Phase 0 — Foundation

- [ ] **0a.** Write METHODOLOGY_ALGORITHM_SELECTION.md (this phase's
      deliverable)
- [ ] **0b.** Implement `cadw/core/spill_array.py` (SpillArray
      primitive) + tests
- [ ] **0c.** Implement 4 gate tests with initial whitelist:
      - [ ] G1 `tests/test_primitive_reuse.py`
      - [ ] G2 `tests/test_guard_before_alloc.py`
      - [ ] G3 `tests/test_no_dense_windows.py`
      - [ ] G4 `tests/test_ml_tools_schema.py` §7 field additions

## Phase 1 — Stop the bleeding (reclassified 2026-10-27)

**Important finding on P1/P2 scope:** sequential-per-class GBDT
training and lambdamart's codes=(n,d) uint8 both make the "narrow
SpillArray fix" insufficient for full out-of-core. Real fixes are
architectural rewrites (2-3 sessions each). Plan below separates
narrow vs full scope.

- [ ] **P4a.** sub_pca: strided view + chunked Welford cov + chunked
      residual scoring. Peak O(chunk·w + w² + k·w), output bit-identical
      within tolerance. **← NEXT** Removes G3 entry sub_pca::_sliding_windows.
- [ ] **P4b.** dwt_mlead: chunked mahalanobis (per level). Removes G3
      entry dwt_mlead::_sliding_windows.
- [ ] **P4c.** s2g: stream conv/quantize; verify _local_convolution
      chunkable. Removes 2 G3 entries (fit_s2g, _score_queries).
- [ ] **P4d.** patch_memory_ad: **requires SpillDir** (training loop
      re-reads X_norm (n_w,d) K times; strided alone insufficient).
      Removes G3 entry patch_memory_ad::fit_patch_memory_ad.
- [ ] **P3.** kshape → 2-pass chunked centroid assignment (special
      case: strided alone insufficient; needs algorithm change).
      Removes G3 entry kshape::_sliding_windows.
- [x] **P2a.** Lambdamart — pre-allocated X/y (eliminate vstack
      transient 2x peak). DONE 2026-10-27, commit f95... (P2a).
      HONEST SCOPE: X still O(n*d) in RAM (whitelisted in G1 via
      plan gate-whitelist). codes still O(n*d) uint8. y O(n).
- [ ] **P2b.** Lambdamart — X + codes (n, d) → memmap; chunked
      _build_tree histogram computation (fancy indexing
      codes[idx, j] must be replaced with per-chunk bincount
      aggregation). REMOVES G1 whitelist for lambdamart.
- [ ] **P2c.** Lambdamart — scores/grad/hess per-iteration spilling.
- [ ] **P1a.** GBDT multiclass — parallel-per-class rewrite
      (XGBoost-style: K trees simultaneously per batch, grad/hess
      used for all K histograms at once). 2-3 sessions. Bit-identical
      output vs current sequential required.
- [ ] **P1b.** GBDT multiclass — pred + cur_leaf_c to SpillDir
      (after P1a). Removes G1 entry.

## Phase 2 — Audit + registration

- [ ] **P5.** Pass 2 audit — manual classification of 47 ML + 15 TS
      models: state_class, state_type, bounded_state, verified vs
      source.

## Phase 3 — Registration

- [ ] **P6.** YAML cards for all models with §7 fields.
      Whitelist removal proportional to progress.
- [ ] **P7.** Remove all whitelist entries when refactor done;
      gates become pure regression guards.

## Whitelist (violations to refactor)

Each entry = a real code smell. Remove on fix commit.

| File | Function | Gate | Backlog item |
|---|---|---|---|
| cadw/ml/gbdt.py | fit_gbdt_multiclass | G1, G4 | P1 |
| cadw/ml/lambdamart.py | fit_lambdamart | G1 | P2 |
| cadw/ml/kshape.py | fit_kshape | G3 | P3 |
| *(others populated by 0c initial scan)* | | | |

## Progress log

- 2026-10-27: Phase 0a started (this doc). fix1a-c complete
  (3546 passed). GBDT multiclass fix1d deferred — SpillArray needed.

## Not in scope

- v0.9.87 fix1d (GBDT multiclass bandaid per-class buffers) —
  superseded by P1 with SpillArray.
- D2, D4 — blocked until Phase 0 complete (methodology first).
- DU-1 — starts after Phase 0 + at least P1/P2 (so the section
  inherits a working SpillArray).

## Gate whitelist (machine-readable)

Entries below are ACTIVE violations that gates G1-G4 permit until
refactored. Each entry references a phase (P1-P7) in the plan above.

Removal of an entry = its corresponding refactor commit landed.
When the block is empty, refactor is complete; gates become pure
regression guards.

<!-- gate-whitelist
G1: cadw/ml/gbdt.py::_fit_boosting_multiclass
G1: cadw/ml/lambdamart.py::fit_lambdamart
G4: A.*
G4: D1.*
G4: D2.*
G4: D3.1.*
G4: D3.2.*
G4: D3.3.*
G4: D4.*
G4: D5.*
G4: E.*
G4: F.*
G4: G.*
G4: H.*
G4: I.*
G4: J.*
G4: K.*
G4: L.*
G4: M.*
G4: N.*
G4: O.*
-->


## P1 reclassification (2026-10-27)

**Original P1**: GBDT multiclass → SpillDir.

**Finding**: sequential-per-class training (current architecture) makes
SpillArray insufficient. True O(batch·K) requires parallel-per-class
(XGBoost-style): K trees trained simultaneously per batch, grad/hess
computed once and used for all K histograms. This is an architectural
rewrite, not a wrapper.

**Cost comparison** (n=1e9, K=10, num_levels=3):
  - Current: O(n·K) RAM = 328 GB, 0 disk I/O
  - Spill-only: O(batch·K) RAM but ~60 TB I/O per round × 100 rounds
  - Parallel rewrite: O(batch·K) RAM, ~n·K bytes I/O per round (feasible)

**Decision**: Split P1 into:
  - P1a (deferred, separate session): parallel-per-class rewrite.
    Scope: ~500 LOC, 2-3 sessions, tests rewritten for
    bit-identical-output vs current sequential.
  - P1b (deferred): after P1a, switch pred + cur_leaf_c to SpillDir.

**Priority order updated**:
  P2 (lambdamart vstack → SpillDir)        ← next session
  P3 (kshape 2-pass)                        ← P1a alternative (smaller)
  P4 (windowed strided)                     ← medium
  P1a (parallel-per-class GBDT rewrite)     ← dedicated session
  P1b (GBDT pred/leaf spill)                ← after P1a
  P5 (Pass 2 audit)
  P6 (YAML, flip G4)
  P7 (whitelist empty)

**Whitelist update**: G1 entry for gbdt.py::_fit_boosting_multiclass
STAYS until P1b. P2 will remove its OWN entry from plan §whitelist
(lambdamart is tracked in plan text, not G1 whitelist).

## P4 audit correction (2026-10-27)

FINAL_REPORT §7 said "merge patch_sub_pca pattern into all 9
windowed modules" — implies single-template. Source review shows:

  - sub_pca: 2-pass reduce (cov) + chunked scoring — fits template
  - dwt_mlead: per-level chunked mahalanobis — simpler than template
  - s2g: conv/quantize streaming — depends on _local_convolution
  - patch_memory_ad: training loop re-reads X_norm K times -> SpillDir
    needed, NOT a strided-view fix

So P4 is 4-5 sessions, not 1. Audit's "45.7->4.9 MB" applies only to
sub_pca prototype. Each sub-phase is production-quality complete fix
of ONE module, not partial patch of many.

Sub_pca _sliding_windows, kshape _sliding_windows etc are all
whitelisted in G3 until each P4 sub-phase lands.

## P2b-narrow status (2026-10-28)

P2b narrow done: X in fit_lambdamart now np.memmap (SpillDir-backed),
freed after _bin_features. Peak during training drops from
n*(9d + 40) to n*(d + 48). Codes, y, scores, grad, hess still in RAM
(P2b-full territory).

P2b-full DEFERRED (2-3 sessions, architecture change):
  - X + codes (n, d) -> memmap
  - _build_tree rewrite: level-synchronous BFS with node_id (n,)
    int32 memmap; per-level chunked bincount aggregation;
    codes[idx, j] fancy indexing removed
  - _predict_tree chunked by rows
  - scores/grad/hess memmap (fancy indexing on scores[idx] remains)
  - Estimated 4n peak RAM (idx int32 + chunk buffers) vs current
    58n after P2b-narrow; full OOC vs half-fix.

Whitelist G1 for lambdamart STAYS (y = np.empty(n_total),
codes = np.empty((n, d), uint8) still RAM). Full removal requires
P2b-full.

## P5 (Pass 2 audit) — DONE 2026-10-28

Audit output:
  - .pingis/audits/state_class_pass2_ml.md (48 files, 417 fns)
  - .pingis/audits/state_class_pass2_ops_ts.md (28 files, 301 fns)
  - scripts/audit_pass2_ml.py --dirs/--out flags for future re-scan

Triage:

### Real new issues (P5a-fix candidates)

- [ ] **P5a-fix-gmm**: gmm.py::_responsibilities — np.empty((n, k))
      per EM iteration. Same pattern as GBDT multiclass (O(n·k) RAM,
      2-3 alloc per iter). Estimated fix: 1-2 sessions (chunked
      E-step + M-step via Reduce-like accumulator).
- [ ] **P5a-fix-coxph**: coxph.py::fit_coxph — vstack@L189 builds
      risk matrix (n, d). P2a-class fix: pre-allocate (n, d). 0.5 session.
- [ ] **P5a-fix-lda**: lda.py::fit_lda — vstack@L257 builds
      (n, vocab) matrix. P2a-class fix: pre-allocate or memmap.
      0.5 session (vocab can be large — memmap preferred).
- [ ] **P5a-review-als**: als.py::fit_als — concat@L154. Check if
      events (O(n)) or per-user factors (O(n_users), bounded).
- [ ] **P5a-review-feature_select**: feature_select.py::_collect —
      vstack@L51. Check if (n, d) fit-path or (n_models, d) bounded.

### Acceptable by design (document in P6 YAML)

- TS models with n-scaling (STL, SES, SARIMA design matrix):
  bounded_state: false, n_scalability: linear,
  tier1_core_exception: single-series TS, n ≤ 1e6 typical.
  Modules: _decomposition.stl_decompose, ets._ses_fit,
  theta._ses_fit, otm._ses, models._ar_fit, _hannan_rissanen,
  _arma_forecast (in-sample).

- Predict-path n-alloc: n = batch size (by design, per-batch output).
  ~30 functions across predict/transform/_iter_pred — all OK.

- Output-not-state: sub_pca._map_max/_map_mean, kshape._map_scores_*,
  s2g._aggregate_to_timestamps, dwt_mlead._reverse_window_boolean —
  (n,) scores returned, not persistent state.

- Bounded fit state: coxph._breslow (n_events), pelt.pelt (DP n+1),
  stray._kth_nn_distances (documented n<=20k),
  _ar_fit (n·p, p small).

### False positives (tool limitation, not issues)

- shuffle._vectorized_hash_numeric, _string_hash_column — n = batch
- models._predict_in_sample x2 — n = output
- ensemble.prediction_interval, thf.predict, var._companion — bounded
- s2g._local_convolution — n_w = chunk size after P4c streaming

### Updated phase plan

P5 phase closed. Remaining refactor by priority:

1. **C = P1a-b (GBDT multiclass parallel rewrite)** — HIGH ROI,
   biggest MEDIUM issue, same pattern as gmm._responsibilities
   (so P1a-b fixes both). Estimated 2-3 sessions.
2. **P2b-full (lambdamart codes/y/scores)** — MEDIUM ROI,
   2-3 sessions. May defer if lambdamart not used at huge n.
3. **P5a-fix-coxph + lda + gmm** — MEDIUM, 1-2 sessions total
   (all P2a-class narrow fixes).
4. **P5a-review-als + feature_select** — review first, fix if real.
5. **P6 (YAML state fields)** — 1-2 sessions, after refactors settle.
6. **P7 (flip G4, whitelist empty)** — 0.5 session.

P5a-fix-gmm overlaps with P1a-b (same pattern O(n·k) per-iteration).
Consolidate: do P1a-b first, then P5a-fix-gmm as separate narrow
task OR skip if GBDT-parallel infrastructure applies.
