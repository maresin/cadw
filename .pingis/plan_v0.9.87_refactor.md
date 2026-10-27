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

## Phase 1 — Stop the bleeding

- [ ] **P1.** GBDT multiclass → SpillArray for pred; on-the-fly
      grad/hess (drop grad_all/hess_all materialization entirely)
- [ ] **P2.** Lambdamart → SpillArray for X stack
- [ ] **P3.** kshape → 2-pass chunked centroid assignment
      (windowed family — audit prototype exists)

## Phase 2 — Eliminate remaining waste

- [ ] **P4.** 9 windowed models: strided_view + chunked (audit
      prototype: 45.7 → 4.9 MB, 5.4× faster). Note: fix1a-c already
      fixed guard order and est_bytes; P4 is the strided-view
      optimization (separate concern from order).
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
