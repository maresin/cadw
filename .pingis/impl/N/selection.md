# Section N - Unsupervised density

Benchmarks: ODDS https://odds.cs.stonybrook.edu/ ; sklearn density https://scikit-learn.org/stable/modules/density.html
Anchor metric: mean_log_likelihood

## IMPLEMENT

### N1.KernelDensity [baseline]
- algorithm_source_url: https://scikit-learn.org/stable/modules/density.html (fetched 200)
- ref: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
- benchmark_url: https://scikit-learn.org/stable/modules/density.html
- metric: mean_log_likelihood
- state_class: O(K*d) K centers, bounded by state_limit; bounded_state: true
- state_type: partitioned (center subsample) + reduce (log-sum-exp per batch)
- primitive_reuse: Reservoir sampling over batches, Reduce (logsumexp), SpillDir if K>state_limit
- algorithm: two-pass. (1) reservoir-sample K centers O(K). (2) per query batch compute log p(x) = logsumexp_i(-||x-c_i||^2/(2h^2)). If K > state_limit, chunk centers from disk.

## SKIP (tier3 or heavy)
- N1.HDBSCAN - needs mutual-reachability MST O(n log n); described_only, not out-of-core under 2.8GB
- N1.NormalizingFlow - torch tier3_export
- N1.GaussianMixture / BIRCH - verified_in_cadw; no new algorithm
