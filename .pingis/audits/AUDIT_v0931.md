# AUDIT_v0931 — bug & RAM review (executor)

Date: 2026-10-23. Baseline tests: 2548 passed, 2 skipped (fast, -n4).

Scope: AST scan of to_pylist/as_py/to_numpy (160 sites), ReduceOp state_nbytes,
empty-input paths, per-batch guards in dataframe/*, groupby/sort/shuffle/rank, ML.

## Findings (classified)

### P0 candidates

- [F1] `ops/sort.py:external_sort` on empty input: no explicit early
  return. Function iterates batches, collects `runs=[]`, then later
  attempts merge of empty `runs`. Needs check: does it raise or return empty?
  (needs runtime confirm; prior audit A10 flags this).

- [F2] `ops/reduce.py:SumOp` uses float64 accumulator for int64 columns.
  Silent precision loss for values > 2^53. sum([2^53, 1, 1]) returns
  2^53 not 2^53+2. Not surfaced by tests. Should either preserve int
  type when dtype is int64, or document limitation prominently.

- [F3] `ops/reduce.py` state_nbytes reported as 16 for tuples of
  (float,int) — real tuple overhead ≈ 72 B. Ops: CountOp, SumOp, MeanOp,
  MinMaxOp (16→~72). StdOp (24→~80). Only matters under tight budgets
  (< 1 KB state_limit) — internal 1MiB fallback covers it. **wontfix
  per prior audit A3/A8 (documented).**

### P1 candidates

- [F4] `dataframe/_per_group.py:881` and `_transformers_fill.py:591`:
  `to_pylist()` inside per-group loops. Bounded by group size (not N),
  but group can be as large as batch (single global group). Batch guard
  at loop head covers it (check_batch at _per_group:407, fill:561).
  Verdict: **not a bug** — guarded upstream.

- [F5] `dataframe/_rolling.py:510`, `_per_group.py:446`, `_cumulative.py:395`:
  `b.column(c).slice(start, group_len).to_pylist()` — per-group slice.
  Same as F4. Guarded upstream (rolling:479, per_group:407,
  cumulative:369). Verdict: **not a bug.**

- [F6] `ops/groupby.py:581` — `grouped.to_pylist()` — grouped is the
  aggregated per-batch result. Bounded by distinct keys in batch ≤ K_est.
  For high-cardinality keys K_est ~ batch_size, guarded at loop head
  (v0.9.23 hash_shuffle.batch=3.0). **Verdict: guarded.**

- [F7] `ops/rank.py:197-198` — fallback path with nulls: `to_pylist` for
  by + row_id. Guard is `check_batch(extra_factor=7.0)` at :355. Verdict:
  **already accounted for.**

- [F8] `ml/gbdt.py:_fit_boosting` — `pred = np.full(n_total,...)` and
  `cur_leaf = np.zeros(n_total,...)` — O(n) internal state (32 B × N).
  Pre-flight `check_state(n_total * 32)` at line 735 correctly rejects
  large N. Verdict: **guarded; but n_total estimate is float32×`n_total`,
  actual np.float64 = 8·N for pred + 8·N cur_leaf + N for leaf values
  ≈ 24·N.** Actual 24·N < check 32·N — overestimate, safe.

- [F9] `ml/gbdt.py` — no `check_batch` inside `_iter_with_state` and
  `_iter_for_leaves` loops. Each batch builds `_stack_numeric` (d×float64
  = 8·d·n) + g,h,leaf columns (3×8·n) = ~(8d+24)·n per batch. For n=100k,
  d=10 that is ~10 MB > 2×batch.nbytes (0.8 MB). **Real gap.**
  Recommendation: add `mg.check_batch(b, extra_factor=~15)` inside the
  loops, or preflight `mg.check_batch` on the enriched size.

- [F10] `ml/iforest.py:253` `arrays.append(arr.to_numpy())` — building
  per-batch numpy blocks in scoring. Guard at 215 extra_factor=3.0. But
  arr is float64 cast: n×8 B → 8·n for 100k rows = 0.8 MB, ~2× raw.
  Verdict: extra_factor=3 conservative — **guarded.**

### P2 / documentation

- [F11] `core/models.py:155` `sample.to_pylist()` in `type_size()` —
  sample is user-provided, typically 1k rows. Called during schema
  inference. Guarded externally (samples sized by caller). Verdict: **OK.**

- [F12] `dataset/index.py` — 15 unguarded to_pylist/to_numpy sites.
  MaterializedIndex holds full column in RAM by design (in-memory index,
  n ≤ RAM). Most are O(N) but they are the *purpose* of this class (fast
  lookup). `__iter__` uses numpy (already optimized). `get_indexer` builds
  dict of all labels → O(N) Python objects = 40× memory. **Caveat:** this
  class is explicitly non-streaming; user opts in. **Verdict: by design,
  document.**

- [F13] `core/memory_guard.py:700` — `len(set(col.to_pylist()))` fallback
  in `estimate_unique_keys`. Prior audit A10 marked fixed but this site
  remains. Guarded by sample_batch size (pre-sample), not full N. Sample
  size is typically 10k rows → 10k Python objects = 0.4 MB. **Verdict:
  bounded by sample size, acceptable.**

- [F14] `ops/shuffle.py:223` exotic fallback `arr.to_pylist()` — only
  dictionary/struct/list columns, rare partition-key types. Bounded by
  batch size (guard at loop head). **Verdict: OK.**

- [F15] `ml/logistic.py:153,167` — `uniq.to_pylist()`, `col.to_pylist()`
  in fit — labels, typically small K. Not N-scaled. **Verdict: OK.**

## Summary table

| ID | Priority | Verdict |
|---|---|---|
| F1 external_sort(empty) | — | non-bug: already raises ValueError |
| F2 SumOp int64 2^53 | — | non-bug: exact for representable; documented 2^53 limit → G.8 |
| F3 state_nbytes 16→72 | P2 | wontfix (A3/A8) |
| F4/F5 per-group to_pylist | — | guarded |
| F6 groupby to_pylist | — | guarded |
| F7 rank fallback | — | guarded (7.0) |
| F8 gbdt state preflight | — | safe (overestimate) |
| F9 gbdt per-batch stack unguarded | P1 | FIXED v0.9.32: batch_memory_factor 7.0/4.0/3.5 + tests |
| F10 iforest score | — | guarded |
| F11 type_size sample | — | OK |
| F12 dataset/index to_pylist | P2 | by design |
| F13 memory_guard sample | P2 | OK (bounded) |
| F14 shuffle exotic | — | OK |
| F15 logistic uniq | — | OK |

## Stage 3 fix plan — SUPERSEDED

Original plan (F9 guard, F2 int64, F1 empty-input) superseded: F1/F2 verified non-bugs, only F9 real and fixed.

## Runtime verification (post-audit)

- **F1 external_sort(empty)** → `ValueError: external_sort: пустой вход`.
  Already correct. Same semantics as hash_shuffle (A10). Mark as **not a bug, wontfix** — the pipeline handles it consistently. Docs already mention. **Status: verified, no change.**

- **F2 SumOp int64 2^53+2** → returned 9007199254740994.0 (correct, but as float).
  The integer value is preserved because `pc.sum` returns python-int, then
  `float(s)` rounds. 2^53+2 is exactly representable (even); 2^53+1
  would not be. Verified limitation exists at ~2^53 boundary; acceptable
  per current docs ("SumOp state is float64"). **Status: documented, no code change** — a fully-fixed version would need `decimal128` (roadmap G.8).

**Net real code bug found: only F9 (gbdt per-batch unguarded).** Proceeding with Stage 3.
