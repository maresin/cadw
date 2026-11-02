# CADW Roadmap

Last updated: 2026-11-02 (after v0.9.105).

## Current: v0.9.105

Last updated: 2026-11-02. Framework chunking gap exposed (DEC-023).

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

## Next 3 sessions

### v0.9.106 — Framework: chunking + state-aware guard

Closes DEC-022 + DEC-023. Required before ANY card can be `verified`.

**Part 1 — Chunking wiring (DEC-023):**
- Propagate `batch_size` through `CadwDataFrame._iter_filtered_batches`
  → `Dataset.iter_batches` (currently kwarg exists in `Dataset.iter_batches`
  but not propagated)
- `MemoryGuard.get_batch_rows` integration as default batch_size
- User override: `fit_*(df, ..., batch_size=N)` (public API)

**Part 2 — State-aware guard (DEC-022):**
- `get_batch_rows(state_bytes=...)` — subtract state from usable budget
- `check_batch` + `check_state` coordination (reserved counter)
- `reduce()` uses `plan(op, op_context={"state_bytes": ...})`

**Verify:** OOC check passes (peak flat with n). Formulas stay
`status: draft` — no optimizations applied yet, coef unchanged.
Tests green.

### v0.9.107 — Optimizations + flip 9 → verified

After v0.9.106 framework fix (chunking + guard works, but coef still
3×/4× without optimization), apply deferred optimizations, then flip
all 9 cards to `verified` in one cycle.

**Part 3 — Optimizations (deferred from v0.9.105):**
- O2b: `del X` in `LogisticGradOp.update` (peak 3× → 2× batch; logistic
  coef should drop from a=3.05 to a≈2.0)
- O4: `np.einsum('ij,ij->j', Xc, Xc)` in Gaussian update
- Refactor `_stack_numeric` (5 copies) → `cadw/ml/_common.py`
  (single source of truth per AI_CONSTITUTION §5.1)

**Then flip 9 → verified:**
- Re-measure `test_*_formula_accuracy` for all 6 ops (production-path,
  no patch) with post-optimization coefs
- Update `ram_formula` in 9 cards (measured coefs, not pre-opt)
- All 9 cards `draft` → `verified` (STABILITY §6 criteria met)
- DEC-012, DEC-022, DEC-023 formally closed
- Tag `v0.9.107`

### v0.9.108 — Back to `SpillArray` for `fit_zms` (DEC-013)

Unblock 3 cards. Probe `cadw/core/spill.py` for SpillArray. Wrap
`decision_scores` (n,) + `labels` (n,) in SpillArray. Formula + matrix
+ OOC. DEC-013 closed.

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
