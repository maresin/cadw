# CADW Roadmap

Last updated: 2026-11-02 (after v0.9.108).

## Current: v0.9.109

Last updated: 2026-11-02. Framework + refactor milestones done (v0.9.105-108).
Next: fill pending cards via EXPERIMENT_PROCEDURE (probe: 124 entries, ~37 unique fns in A/B; alias-heavy).

**Closed:** methodology docs (v0.9.102), ram_formula mechanism (v0.9.103),
R1-R4 enforcement, MatrixProfile x2 verified, bounded_by_domain field,
21 draft reasons. v0.9.104-fixes: R2 syntax check, evaluate_formula
exception wrap, MatrixProfile stale field.

**Status baseline:**

| Status | Count |
|---|---|
| verified | 2 (MatrixProfile x2) |
| draft | 30 (21 + 9 from v0.9.105: linear, lasso, logistic x4, NB x3) |
| bounded_by_domain | 0 |
| pending (no status) | 64 |

**Known gaps:**

- ~~DEC-012: `_stack_numeric` AP2~~ — **closed v0.9.105**. Formulas
  derived (see `EXPERIMENT_PROCEDURE.md §7`). Remaining blocker for
  `verified` is DEC-023 (framework), not DEC-012.
- **DEC-022** (v0.9.105): `batch_memory_factor` scalar insufficient
  for large d (logistic d=1000: guard 4.0 → 32 MB, actual 56 MB).
- **DEC-023** (v0.9.105): framework chunking gap.
  `_iter_filtered_batches` does not honor `batch_size`; production
  path uses `batch=n`. OOC broken for all 9 cards.
- DEC-013: `fit_zms` ×3 - O(n) output needs `SpillArray`. Deferred.
- DEC-014: `fit_pelt` ×2 - `_collect_series` container. Deferred.
- `fit_ab_test` - tiny O(1) state, STABILITY §4 special case.

## Closed milestones (v0.9.105-v0.9.108)

- **v0.9.105** — 9 cards formula-calibrated (draft). EXPERIMENT_PROCEDURE.md.
- **v0.9.106** — framework chunking + state-aware guard (DEC-022/023 closed).
- **v0.9.107** — 9 cards `verified` (tag). 11 verified total (2+9).
- **v0.9.108** — refactor `_stack_numeric` → `_common.py` (23 dedup sites).

## Next 3 sessions

### v0.9.109 — Batch 1: fill 5 pending cards (per EXPERIMENT_PROCEDURE)

**Candidates (simple group A/B, batch + state, no second knob):**
- `fit_coxph` — ReduceOp, state O(d²), like linear/logistic
- `fit_pca` — ReduceOp + CovarianceOp, state O(d²)
- `fit_iforest` — sample-based, bounded n_estimators × subsample
- `fit_lda` — dict state, has `test_lda_bounded_memory` (existing)
- `fit_kmeans` — ReduceOp + k·d state, k user knob (already tested)

Per card: probe → formula from code → matrix 3×3 → `status: draft`
(not verified yet — framework gaps check per card). If OOC blocked
by known gap (SpillArray for zms, _collect_series for pelt,
GAP for lda if dict O(n)), card stays draft with DEC-030+ reason.

**Exit criterion (PROCEDURE §5):** all 5 cards closed (draft with
reason OR verified). Not "one more iteration" loop.

### v0.9.110 — Batch 2: 5-7 more pending cards

Similar. Prefer group B (batch + state) before group C (needs P/k/w).

### v0.9.111 — Batch 3: remaining pending + milestone tag

Tag `v0.9.111` if 15+ verified total (currently 11).

## Backlog (not scheduled)

- 53+ pending cards: batches ~15/session v0.9.108-v0.9.111
- `solve_batch(op, budget, **params)` inverse solver
- `fit_ab_test` §4 special case (tiny O(1) state)
- AP2 sweep: grep `column_stack|vstack|concatenate` in cadw/ml/
- MatrixProfile default S=4096, w=32 auto-inject

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

- ~~DEC-012 AP2 in _stack_numeric~~ — closed v0.9.105; not "won't fix",
  actually resolved (formula + code fix; framework blocker separate).
- **CoxPH concordance streaming** (v0.9.97 audit): current
  implementation is O(n) chunked (~8 MB/chunk bool matrix),
  NOT O(n²) as v0.9.96 brief §8 claimed. True streaming
  (external sort + Fenwick inversion) gives O(n log n) I/O
  for a statistic unstable at n > 10⁶ anyway. Won't-fix until
  a real use case appears.

If a future session considers one of these, **read STREAMING §6.1
first** — cost calculations and rationale are there.

## Session discipline

- Before commit: `pytest -q <affected files>` (focused, ~20s).
- Before tag: `pytest -q -m "not slow"` (full fast, ~10min).
- Before release: `pytest -q` (all, includes slow).
- See `AI_CONSTITUTION.md` §4 for any new `fit_*` / `predict_*`.
