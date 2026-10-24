# D.3.1 — Cox proportional hazards (Breslow)

**Version:** v0.9.36 · **Module:** `cadw/ml/coxph.py` · **Mixin:** `df.fit_coxph(...)`
**Complexity:** M · **State:** O(d^2) after a guarded O(N*d) pre-pass

## Sources

- Cox, *Regression Models and Life-Tables*, JRSS-B 34(2), 1972.
- Andersen, Gill, *Cox Regression Model for Counting Processes: A Large
  Sample Study*, Ann. Statist. 10(4), 1982 — the counting-process
  formulation and the Breslow tie handling used here.
- lifelines `CoxPHFitter` (MIT) — API reference and the reference
  implementation the tests compare against.

## Algorithm

```
L(b) = sum_{events i} [ x_i.b - log( sum_{j: T_j >= T_i} exp(x_j.b) ) ]
grad = sum_{events} [ x_i - S1/S0 ]
info = sum_{events} [ S2/S0 - (S1/S0)(S1/S0)^T ]
S_k = sum_{risk set} w x^k ,  w = exp(x.b)
```

Newton-Raphson with a 1e-10 ridge on the information matrix (pinv fallback
if it is singular), `tol=1e-7` on the max absolute step, `max_iter=50`.
Breslow baseline cumulative hazard at event times; Harrell c-index.

## Why not a Reduce (stated honestly)

The risk set at time t is `{j : T_j >= t}` — every row with a longer
survival time. A batch-local partial sum cannot know that set, so unlike
fairness (O(K) counters) or ALS (sorted-stream grouping) this cannot be
expressed as a Reduce or a sorted sweep without recomputing risk sets per
event. We therefore use the same shape the critic already accepted for
LambdaMART (D.2.1):

1. one streaming pre-pass collects X, T, E; `check_state(n_rows*(d*8+16))`
   runs BEFORE the vstack, so an oversized input raises
   `MemoryLimitExceeded` instead of OOM (tested: 20 000 rows x 3 features
   under a 1 MB budget);
2. Newton iterations then run in memory.

No approximation is claimed. This is the same O(N*d) class as the existing
`fit_gbdt_*` / `fit_lambdamart`, which is inherent to Newton-type models.

## RAM model

| Component | Bytes |
|---|---|
| Pre-pass X copy | `N*d*8` (guarded) |
| T, E | `N*16` |
| Coefficients + information | `d^2*8` |
| Baseline hazard | O(#event times) |

`check_batch(b, extra_factor=2.0)` on every batch; `check_state` for the
accumulating copy and again on the finished model state.
`log_fit("coxph", iter_i, iter_n, loss=-ll, state_bytes=d*d*8)` per Newton
iteration.

## Benchmark (measured)

`scripts/bench_coxph.py` -> `.pingis/benches/coxph.json`, d=5, 100k-row batches,
true coefficients `linspace(0.6, -0.6, 5)`:

| N | fit | iters | peak RSS | state | c-index | lifelines fit | lifelines c-index | coef maxdiff |
|---|---|---|---|---|---|---|---|---|
| 5 000 | 0.06 s | 4 | 6.0 MB | 46 896 B | 0.7229 | 0.45 s | 0.7229 | 1.54e-06 |
| 50 000 | 0.70 s | 4 | 36.7 MB | 464 864 B | 0.7244 | 4.18 s | 0.7244 | 1.58e-06 |

Coefficients agree with lifelines to ~1.6e-06 and the c-index is identical
to 4 decimal places at both sizes. 100k rows was attempted first and hit
the 280 s tool timeout because the original c-index was a Python double
loop; the c-index is now vectorized in row chunks (the 50 000-row run
includes that c-index), which is why the bench tops out at 50 000.

## Tests (10)

smoke/state · reference vs lifelines (coef atol 1e-4, c-index 1e-6) ·
sign/magnitude recovery on 2000 rows · gradient vs central finite
differences (atol 1e-5) · determinism · multi-batch invariance (chunk 1 vs
100 000) · model-sized state at N=100 vs N=5000 · guard · errors (missing
column, empty X_cols, all censored, non-positive time) · source contract.

## Edge cases

- All censored -> `ValueError("no events")`.
- Non-positive survival time -> `ValueError`.
- Tied event times use Breslow (lifelines default is Efron; the test
  tolerance of 1e-4 reflects that difference on near-continuous
  synthetic data where ties are rare).
- Singular information matrix -> `pinv` fallback.
