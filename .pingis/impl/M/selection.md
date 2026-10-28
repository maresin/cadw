# Section M - Contextual bandits

Benchmarks: Yahoo_R6A https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=1 ; Open_Bandit_Dataset https://github.com/st-tech/zr-obp ; Mushroom https://archive.ics.uci.edu/dataset/73/mushroom
Anchor: ctr

## IMPLEMENT

### M1.EpsilonGreedy [baseline]
- algorithm_source_url: https://en.wikipedia.org/wiki/Multi-armed_bandit
- ref: Sutton & Barto, RL: An Introduction, Ch. 2
- benchmark_url: https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=1
- metric: ctr
- state_class: O(A) rewards/counts; bounded_state: true
- state_type: aggregation (online incremental)
- primitive_reuse: Reduce (sum rewards, count pulls)
- algorithm: online incremental Q[a] += (r - Q[a])/N[a]; with prob eps explore uniformly, else argmax. Multi-pass not needed; single pass over stream.

### M1.UCB1 [baseline]
- algorithm_source_url: https://arxiv.org/abs/0803.3798
- ref: Auer et al. 2002, Finite-time Analysis of the Multiarmed Bandit Problem
- benchmark_url: https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=1
- metric: ctr
- state_class: O(A); bounded_state: true
- state_type: aggregation
- primitive_reuse: Reduce (per-arm reward sum + pull count)
- algorithm: pick arm maximizing Q[a] + sqrt(2 ln N / N[a]); O(A) state, online update.

### M1.ThompsonSampling [sota, tier2 -> out-of-core tier1]
- algorithm_source_url: https://en.wikipedia.org/wiki/Thompson_sampling (fetched 200)
- ref: Chapelle & Li 2011, An Empirical Evaluation of Thompson Sampling
- benchmark_url: https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=1
- metric: ctr
- state_class: O(A) Beta(a,b) posterior; bounded_state: true
- state_type: aggregation
- primitive_reuse: Reduce (per-arm a,b updates)
- algorithm: maintain Beta(1+successes, 1+failures) per arm; sample theta_a ~ Beta; pick argmax. Online, O(A).

## SKIP (tier3_export)
- M1.NeuralUCB - torch. Not implementable.
- M1.LinUCB - verified_in_cadw; no new algorithm.
