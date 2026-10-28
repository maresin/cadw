# Section M — algorithm specs (contextual / multi-armed bandits)

Input: replay stream of `RecordBatch` with columns `arm:int64`, `reward:float64` (Yahoo R6A / Open Bandit
triplets `(arm, propensity, reward)` supported via optional `propensity:float64`).
All three impls share a single `ReduceOp` for per-arm sufficient stats.

Common:
- input: `batch_iter_factory` (multi-pass safe); online mode is single-pass
- memory: `MemoryGuard(max_memory_mb=budget)`; `check_state(8*A)` before `init()`
- no `pa.Table`, no per-row loops

## Shared reducer `ArmStatsOp(ReduceOp)`
- state: `sums: np.ndarray(A)`, `counts: np.ndarray(A)`, optional `successes/failures` for TS
- `update`: `np.add.at(sums, arm_ids, rewards)`; `np.add.at(counts, arm_ids, 1)` — vectorized, no python loop
- `state_nbytes = 8*A + 8*A + 64`; `estimate_state_nbytes = 8*A*2 + 64`
- `batch_memory_factor = 1.0` (numpy gather is cheap); no extra buffer

## M1.EpsilonGreedy
- state_class: O(A) Q[a], N[a]
- state_type: aggregation (single pass)
- bounded_state: true
- n_scalability: O(N) time, O(A) RAM (A = #arms, small)
- primitive_reuse: `ArmStatsOp` + `reduce`
- algorithm: `Q[a] = sums[a]/max(counts[a],1)`; online: incremental `Q[a] += (r - Q[a])/N[a]`. `eps` schedule: `eps_t = eps0 / sqrt(1 + t/decay)` (common improvement over fixed eps)
- output: `EpsilonGreedyPolicy(q_, counts_, eps_, rng_)`; `.select(context=None)` -> argmax with eps-greedy; `.update(arm, reward)` for online
- deviation: bandit simulation reuses Numpy RNG so smoke tests are reproducible (sklearn/simglucose compare, not exact match)

## M1.UCB1
- state_class: O(A)
- state_type: aggregation
- bounded_state: true
- n_scalability: O(N) time, O(A) RAM
- primitive_reuse: `ArmStatsOp` + `reduce`
- algorithm: UCB index `Q[a] + c * sqrt(2*ln(N) / max(N[a],1))`; `c=1.0` default, `c=2.0` for conservative exploration
- output: `UCB1Policy(q_, counts_, c_)`; `.select()` -> argmax; `.update(arm, reward)` for online

## M1.ThompsonSampling
- state_class: O(A) Beta(a, b) posterior per arm
- state_type: aggregation
- bounded_state: true
- n_scalability: O(N) time, O(A) RAM
- primitive_reuse: `ArmStatsOp` extended with `successes/failures` arrays; reward in {0,1} (Bernoulli) or thresholded from continuous (optional)
- algorithm: per arm `a_a = 1 + successes[a]`, `b_a = 1 + failures[a]`; sample `theta_a ~ Beta(a_a, b_a)`; select `argmax`
- output: `ThompsonSamplingPolicy(alpha_, beta_, rng_)`; `.select()` -> argmax of samples; `.update(arm, reward)` for online

## SKIP rationale
- M1.NeuralUCB: torch -> out of scope
- M1.LinUCB: already in cadw.ml.linucb; no new algorithm needed
