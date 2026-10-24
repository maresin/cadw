# D.2.2 — Implicit-feedback ALS

**Version:** v0.9.36 · **Module:** `cadw/ml/als.py` · **Mixin:** `df.fit_als(...)`
**Complexity:** M · **State:** O((U+I)·k)

## Sources

- Hu, Koren, Volinsky, *Collaborative Filtering for Implicit Feedback
  Datasets*, ICDM 2008 — the confidence weighting `c = 1 + alpha*w` and the
  alternating least-squares fixed point.
- Takacs, Pilaszy, Nemeth, Tikk, *Applications of the Conjugate Gradient
  Method for Implicit Feedback Collaborative Filtering*, KDD 2011.
- `implicit.cpu.als.AlternatingLeastSquares` (BSD) — API reference
  (`factors`, `alpha`, `regularization`, `iterations`).

## Algorithm

Per user u, with item factors Y and confidence c_ui:

```
A_u = Y^T Y + sum_i (c_ui - 1) y_i y_i^T + reg*I
b_u = sum_i c_ui y_i
x_u = A_u^-1 b_u
```

and symmetrically for items. One sweep = one pass over users + one over items.

## Out-of-core composition

The rating matrix is never materialized. Each sweep:

1. `external_sort` on the user column → a user-sorted run on disk.
2. Stream the run grouping by user (`_iter_user_groups`), accumulating the
   per-user Gram correction and RHS, then one k×k solve.
3. Repeat with `external_sort` on the item column.

`HashShuffle` is unnecessary (the sort gives the grouping); the `Reduce`
primitive cannot express the per-group k×k solve, so the sorted-stream
pattern is used instead. This is stated rather than presented as a Reduce.

## RAM model

| Component | Bytes |
|---|---|
| User factors X | `U·k·8` |
| Item factors Y | `I·k·8` |
| Per-group temp | O(k²) |
| Sort spill | on disk (external_sort) |

`check_batch(b, extra_factor=2.0)` on every batch, `check_state((U+I)*k*8)`
before allocating the factors. Guard test: `(200+200)*256*8 = 819 200 B`
raises `MemoryLimitExceeded` under a 1 MB budget.

## Instrumentation (pattern for later groups)

`log_fit("als", iter_i, iter_n, loss, state_bytes)` is emitted **once per
sweep**, and each sweep is itself multi-batch (every batch goes through
`check_batch(..., extra_factor=2.0)`). The loss is the observed part of the
Hu-Koren-Volinsky objective
`sum (1+alpha*w)(1 - x_u·y_i)^2 + reg*(||X||^2+||Y||^2)`; the unobserved mass
is constant in the factors. `test_objective_decreases` asserts it falls
monotonically across 8 sweeps.

## Benchmark (measured)

`scripts/bench_als.py` → `.pingis/benches/als.json`, k=16, 3 sweeps,
20 interactions/user, 100k-row batches:

| users × items × per_user | interactions | fit | peak RSS | state |
|---|---|---|---|---|
| 2 000 × 800 × 20 | 40 000 | 6.55 s | 19.4 MB | 358 400 B |
| 10 000 × 4 000 × 20 | 200 000 | 31.29 s | 64.0 MB | 1 792 000 B |

State matches `(U+I)·k·8` exactly at both sizes. A 1M-interaction run was
attempted and exceeded the 300 s bench budget (the per-row Python grouping
in `_iter_user_groups` dominates); it is **not** reported as a number.

## Tests (10)

smoke/shapes/state · objective decreases · recommend hits observed items ·
closed-form user update reference (explicit k×k system) · determinism ·
multi-batch invariance (chunk 1 vs 100 000) · state independent of
interaction count · guard · errors (missing column, bad k, empty) · source
contract.

## Edge cases

- Empty input → `ValueError`.
- `n_factors`/`n_iter` < 1 → `ValueError`.
- Items or users with no interactions simply keep their seed factors.
- `weight_col=None` means every interaction has weight 1.
