# Section I — algorithm specs (survival)

Common conventions for all specs in this folder:
- input: `batch_iter_factory: Callable[[], Iterator[pa.RecordBatch]]` (multi-pass safe, no buffering)
- memory: `MemoryGuard(max_memory_mb=budget, safety=0.7, state_safety=0.5)`
- no `pa.Table`, no `read_all()`, no per-row python loops over N
- columns: `duration` (float64 > 0), `event` (bool/int8), features `X` (float64)

## I1.KaplanMeier
- state_class: O(u) where u = #unique event times on disk (spilled); O(1) in RAM
- state_type: cross_row (Sort + Reduce)
- bounded_state: true — RAM state is a running (n_at_risk, n_events, S) triple + fixed-size write buffer
- n_scalability: O(N log N) time (external sort), O(1) RAM; u may be O(N) but lives on disk
- primitive_reuse: `external_sort(key_columns=['t'])` -> `read_sorted` -> `ReduceOp` streaming step function; `SpillDir` + `array_schema/array_to_record_batch/spill_array_writer` for the step function when u > state_limit/row_size
- passes: 2 (sort pass, then scan sorted run once)
- estimate_state: 3 float64 + buffer, ~64 B; `estimate_state_nbytes()` returns 64
- edge cases: no events -> S(t)=1 everywhere (warn); ties in t handled by grouping on equal t (cumulative d_i/n_i per unique t, not per row)
- output: `KaplanMeierModel(times: np.ndarray, survival: np.ndarray, n_at_risk, n_events, median_survival)`; `.predict(t)` via `np.searchsorted` on the spilled times (loaded only if u*24 <= state_limit, else streamed chunk-by-chunk)
- deviation from lifelines: lifelines keeps all u in RAM; we spill and only expose median + chunked survival curve. Improvement: log-log CI computed from Greenwood variance accumulated in the same Reduce pass.

## I1.WeibullAFT
- state_class: O(d^2 + d) sufficient stats (XTWX, XTWy, sum w, sum y) — bounded by `check_state(8*(d*d+2d))`
- state_type: aggregation (Reduce over sufficient stats) + iterative (Newton/Raphson on (beta, log sigma))
- bounded_state: true — state independent of N; d must satisfy `8*d*d <= state_limit_bytes` (else raise `MemoryLimitExceeded` before allocating)
- n_scalability: O(N*d^2) time per Newton iteration (dense XTWX update), O(1) RAM w.r.t. N; ~5-10 iterations
- primitive_reuse: `ReduceOp` subclass `XTWXOp` (vectorized `X.T @ X`, `X.T @ y` per batch), `MemoryGuard.estimate_state_nbytes` + `check_state`, pattern copied from `cadw/ml/coxph.py`
- passes: P passes over data, P = #Newton iterations (data re-streamed each pass; no materialization)
- edge cases: censoring handled by log-likelihood gradient term (not a simple weighted regression — use full Weibull log-lik with censoring); sigma -> 0 guarded; X rank deficiency detected via `np.linalg.cond(XTWX)` -> ridge fallback lambda=1e-6*mean(diag)
- output: `WeibullAFTModel(beta, log_sigma, scale, baseline_cumhazard, feature_names)`; `.predict_median(X)`; `to_dict()`
- deviation from lifelines: lifelines uses full-batch L-BFGS; we use Newton with analytic Hessian on sufficient stats (converges in ~6 iters for well-conditioned data). Improvement: optional ridge for numerical stability.
