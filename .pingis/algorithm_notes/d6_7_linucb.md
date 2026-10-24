# D.6.7 Disjoint LinUCB contextual bandit

## Paper
Li, Chu, Langford, Schapire, *A Contextual-Bandit Approach to Personalized
News Article Recommendation*, WWW 2010.

## Docs
- `cadw.ml.linucb.fit_linucb(df, X_cols, arm_col=..., reward_col=...,
  alpha=1.0, l2=1.0)`
- `CadwDataFrame.fit_linucb(...)`.
- Model: `theta()` (per-arm ridge coefficients), `predict(df)` -> (n, K)
  UCB scores, `choose(df)` -> greedy argmax arm, `A_`, `b_`.

## Design
- Disjoint per-arm sufficient statistics: A_a += x x^T, b_a += r x,
  finalised as A_a + l2*I; theta_a = A_a^-1 b_a. The update is exactly
  closed form, so the test reference is the analytic ridge solution
  (test_analytic_ridge_reference, atol 1e-9) rather than a library.
- UCB score: x^T theta_a + alpha * sqrt(x^T A_a^-1 x), vectorised per arm.
- State is O(K*d^2 + K*d), independent of n; the fit streams over batches
  and never retains an observation.

## Benchmark (`.pingis/benches/linucb.json`, K=5, d=8)
| n | fit_s | peak_mb | state_b |
|---|---|---|---|
| 10 000 | 0.002 | 4.7 | 3 136 |
| 50 000 | 0.010 | 19.4 | 3 136 |
| 150 000 | 0.040 | 51.1 | 3 136 |
Reference: closed-form ridge per arm (max abs difference < 1e-9 in tests);
high-signal synthetic recovers true theta to max error < 0.1. Sizes chosen
to stay inside the 100 MB batch guard (extra_factor=3.0); the guard was
not relaxed and no run was skipped.

## RAM model
State = K*d*d*8 + K*d*8 bytes (3 136 B measured for K=5, d=8). Fit holds
one in-memory batch (O(batch*d)); every batch passes
check_batch(extra_factor=3.0).
