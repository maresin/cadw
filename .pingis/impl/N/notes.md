# Section N — algorithm specs (unsupervised density)

## N1.KernelDensity
- state_class: O(K*d) centers (K fixed by user, d feature dim), bounded by state_limit; if K*d*8 > state_limit -> spill to disk and stream centers chunk-by-chunk
- state_type: partitioned (Reservoir-sample centers) + reduce (log-sum-exp per query batch)
- bounded_state: true
- n_scalability: fit = 1 streaming pass (Reservoir O(N)); query = O(N_query * K * d) time, RAM = query_batch*d*8 + chunk_centers*d*8
- primitive_reuse: `np.random.default_rng` reservoir loop (per-batch vectorized), `MemoryGuard.check_state(8*K*d)`, `SpillDir` + `spill_array_writer/reader` when K exceeds state limit, `ReduceOp` `LogSumExpOp` for per-batch log-likelihood
- passes: 2 (reservoir sample pass; per-query-batch scan pass)
- algorithm:
  fit: reservoir sample K rows from stream (Algorithm R, vectorized per batch — draw `n_new` indices, compare with reservoir slot indices using geometric skip for large batches)
  score(x) = logsumexp_i(-||x-c_i||^2 / (2h^2)) - log(K) - d*log(h*sqrt(2pi))
  If K spilled: read centers in `center_chunk` rows, accumulate logsumexp via `np.logaddexp`
- estimate_state: `8*K*d + 8*K + 64`
- edge cases: empty input -> raise; K > N -> warn, use N centers; h <= 0 -> raise
- output: `KernelDensityModel(centers_, bandwidth_, metric_)`; `.score_samples(batch_iter)` yields batch of floats; `.score(X)` mean
- deviation from sklearn: sklearn keeps full training set (N*d) in RAM and does kNN per query; we keep only K centers (K << N recommended, e.g. 4096) -> O(1) RAM. Improvement: Silverman's rule-of-thumb for h if not given (`h = 1.06 * std * N^(-1/5)`).

## SKIP rationale
- N1.HDBSCAN: mutual-reachability MST requires O(n log n) memory or heavy out-of-core graph — not feasible under 2.8 GB without a dedicated graph engine
- N1.NormalizingFlow: torch -> out of scope
- N1.GaussianMixture / BIRCH: already in cadw.ml; no new algorithm
