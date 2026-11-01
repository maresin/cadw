# CADW Roadmap

Last updated: 2026-10-30 (after v0.9.101).

## Current: v0.9.104

Last updated: 2026-11-01 (after v0.9.104-fixes).

**Closed:** methodology docs (v0.9.102), ram_formula mechanism (v0.9.103),
R1-R4 enforcement, MatrixProfile x2 verified, bounded_by_domain field,
21 draft reasons. v0.9.104-fixes: R2 syntax check, evaluate_formula
exception wrap, MatrixProfile stale field.

**Status baseline:**

| Status | Count |
|---|---|
| verified | 2 (MatrixProfile x2) |
| draft | 21 |
| tier1_core_exception | 0 (replaced by bounded_by_domain) |
| pending (no status) | 64 |

**Known gaps** (DEC-012..DEC-014):

- `_stack_numeric` AP2 - blocks 9 verified (linear, logistic x4, lasso,
  NB x3) [DEC-012]
- `fit_zms` x3 - O(n) output needs SpillArray [DEC-013]
- `fit_pelt` x2 - `_collect_series` container [DEC-014]
- `fit_ab_test` - tiny O(1) state, STABILITY §4 special case

## Next 3 sessions

### v0.9.105 - `_stack_numeric` refactor (DEC-012)

Unblock 9 verified. Pre-allocated `np.empty((batch, d))` + per-column
slice assign in `cadw/ml/linear.py`, `cadw/ml/naive_bayes.py`.
`extra_factor=3.0` default in `reduce.check_batch`. Formula + 9-cell
matrix per card -> 9 verified. DEC-012 closed.

### v0.9.106 - `SpillArray` for `fit_zms` (DEC-013)

Unblock 3 verified. Probe `cadw/core/spill.py` for SpillArray. Wrap
`decision_scores` (n,) + `labels` (n,) in SpillArray. Formula
(chunk x w x 8) + matrix + OOC check. DEC-013 closed.

### v0.9.107 - `fit_pelt` `_collect_series` refactor (DEC-014)

Unblock 2 verified. Bounded per-batch buffering. Formula from code.
pelt x2 -> verified. DEC-014 closed.

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
