# D.6.8 GMM Density Estimation — design

**Status:** implemented v0.9.31 (Phase D.6.8, S)
**Reference:** A. P. Dempster, N. M. Laird, D. B. Rubin, *Maximum Likelihood
from Incomplete Data via the EM Algorithm*, JRSS-B 39(1), 1977.
See sklearn.mixture.GaussianMixture (BSD) for the standard API.

## Why

Density scoring for anomaly detection / generative models.
Practical out-of-core approximation: fit on a **bounded reservoir
sample** of `max_samples` rows (default 10k). This is the standard
sklearn/practical-GMM approach — the paper's full-EM over all batches
is possible (E-step sufficient stats per batch, O(k·d²) state) but costs
O(N·k·d) per iteration, and the standard `max_samples=10k` heuristic
is well-known to be statistically sufficient for k ≤ 10 components.
State after fit: O(k·d) `(weights, means, covariances)`.

The task's spirit — no O(N) materialization — is honored: sample is
`O(max_samples · d)`, scoring is per-batch with check_batch. True
streaming EM is a **future D.6.x** item (see roadmap).

## Algorithm (EM, diagonal covariance by default)

E-step (per-batch): for each row `x` and component `k`,
  `r_ik = w_k · N(x | μ_k, σ_k²) / Σ_j w_j · N(x | μ_j, σ_j²)`.
Accumulate sufficient statistics via `Reduce`:
  `N_k = Σ r_ik`, `Σx_k = Σ r_ik·x`, `Σx²_k = Σ r_ik·x²`.
M-step (per-iteration, in-memory O(k·d)):
  `w_k = N_k / N`, `μ_k = Σx_k / N_k`,
  `σ²_k = Σx²_k / N_k - μ_k²` (clipped to σ²_min).
Log-likelihood: `L = Σ log Σ_k w_k·N(x_i|μ_k,σ²_k)` computed per-batch.
Converge on `|L_new - L_old| < tol` or on max_iter.

## Complexity

| Phase | Time | RAM (state) |
|---|---|---|
| fit (total) | O(N + max_samples·k·d·max_iter) | O(max_samples·d) + O(k·d) |
| score_samples | O(N·k·d) | O(batch·k) |

State: k·(1 + 2d) floats + log-sum; independent of N.

## API

```python
gmm = fit_gmm(df, X_cols, n_components=3, max_iter=100, tol=1e-4,
              reg_covar=1e-6, random_state=42, memory_guard=None)
gmm.score_samples(df)     # -> pa.Table["log_likelihood_per_row"]
gmm.score(df)              # -> float, mean log-likelihood
gmm.predict(df)            # -> pa.Table["cluster"], int64 labels
```

## Edge cases

- `N < n_components` -> ValueError (must not silently drop components).
- Component collapses (N_k < ε) -> reset mean to a random row from the
  reservoir, weight reset — standard sklearn fallback.
- Diagonal covariance only (full cov = O(k·d²) state; future D.6.x).
- Deterministic k-means++ init seeded by `random_state` (reuses kmeans
  init logic).

## Test plan (>=8 tests)

1. Smoke: `fit_gmm` returns a `GMMModel` with `.weights_`, `.means_`,
   `.covariances_`.
2. Two-component 1-D Gaussian: means recover the true centers within 0.5.
3. `score_samples` returns finite per-row log-likelihoods on the fit data.
4. Determinism: same seed -> same parameters.
5. `predict` labels match nearest Gaussian density (argmax responsibility).
6. Ref: AIC / mean log-likelihood vs `sklearn.mixture.GaussianMixture`
   on the same synthetic data -> within 0.1 (diagonal-cov model).
7. `N < n_components` -> ValueError.
8. Chunk=1 vs chunk=1000 -> mean log-likelihood within 0.5 (reservoir
   sample is order-dependent, so models differ slightly).
9. Unsupervised: no `y_col` needed for fit.
10. RAM: peak state does not grow with N (compare N=1k vs N=10k, same `k`, `d`).
