# D.6.4 Randomized SVD

## Paper
Halko, Martinsson, Tropp, *Finding Structure with Randomness: Probabilistic
Algorithms for Constructing Approximate Matrix Decompositions*, SIAM Review
53(2), 2011. Reference in tests: sklearn `PCA` singular values.

## Docs
- `cadw.ml.randsvd.fit_randomized_svd(df, X_cols, n_components=8,
  n_oversamples=10, n_iter=2)`
- `CadwDataFrame.fit_randomized_svd(...)`.
- Model: `transform(df)`, `singular_values_`, `explained_variance_ratio(df)`.

## Design
- Streaming mean/centering, then randomized range finder (Omega ~ N(0,1),
  seeded), power iterations, QR, and small SVD of the projected matrix.
- Deterministic (fixed seed); state O(n_components*d) after fit.

## Benchmark (`.pingis/benches/randsvd.json`)
| n | d | k | fit_s | sv0 | ratio0 | peak_mb | state_b |
|---|---|---|---|---|---|---|---|
| 10 000 | 20 | 3 | 0.008 | 547.85 | 0.522 | 9.3 | 920 |
| 50 000 | 50 | 5 | 0.110 | 1997.8 | 0.294 | 74.3 | 2 696 |
| 150 000 | 20 | 3 | 0.215 | 2148.3 | 0.655 | 70.8 | 920 |
Reference: leading singular values match sklearn PCA to ~1e-6 relative
(exact in tests); explained variance ratio agrees. Sizes chosen to stay
inside the 100 MB batch guard (extra_factor=3.0); the guard was not relaxed.

## RAM model
Fit holds the centered batch and Omega (O(batch*d + d*k)); after fit state =
singular values + V (O(d*k), 920 B measured). Every batch passes
`check_batch(extra_factor=3.0)`.
