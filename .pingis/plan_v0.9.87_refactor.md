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
- [ ] **P2a.** Lambdamart — replace Xs list + vstack with SpillDir
      (narrow: fixes transient peak, codes remains O(n·d)).
- [ ] **P2b.** Lambdamart — codes=(n,d) uint8 → SpillDir + streaming
      tree building (full out-of-core).
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
G3: cadw/ml/s2g.py::_score_queries
G3: cadw/ml/s2g.py::fit_s2g
G3: cadw/ml/kshape.py::_sliding_windows
G3: cadw/ml/patch_memory_ad.py::fit_patch_memory_ad
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
