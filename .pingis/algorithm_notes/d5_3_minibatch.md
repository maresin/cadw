# D.5.3 MiniBatch KMeans + streaming EM GMM

## Paper
Sculley, *Web-Scale K-Means Clustering*, WWW 2010 (MiniBatch KMeans).
Dempster, Laird, Rubin, *Maximum Likelihood from Incomplete Data via the EM
Algorithm*, JRSS-B 39(1), 1977. Pedregosa et al., *scikit-learn*, JMLR 12,
2011 (reference implementations used in tests).

## Docs
- `cadw.ml.minibatch.fit_minibatch_kmeans(df, X_cols, n_clusters, batch_size)`
- `cadw.ml.minibatch.fit_minibatch_gmm(df, X_cols, n_components, batch_size)`
- `CadwDataFrame.fit_minibatch_kmeans(...)` / `fit_minibatch_gmm(...)`.
- KMeans: `predict(df)`, `inertia(df)`, `centers_`, `counts_`.
- GMM: `predict(df)`, `predict_proba(df)`, `score(df)`, `score_samples(df)`.

## Design
- KMeans: k-means++ on a bounded 10*batch reservoir, then Sculley
  per-centroid streaming update with per-centroid counts; convergence via
  relative inertia change.
- GMM: diagonal-covariance EM; E-step per mini-batch, M-step accumulated
  from sufficient statistics (Nk, sum r*x, sum r*x^2); seeded from k-means++
  centroids.
- State O(K*d) in both; no materialisation of all n.

## Benchmark (`.pingis/benches/minibatch.json`, k=3, d=2)
| n | model | fit_s | metric | peak_mb | state_b |
|---|---|---|---|---|---|
| 20 000 | kmeans | 0.054 | inertia 167 483 | 9.1 | 328 |
| 20 000 | gmm | 0.082 | score -3.4608 | 2.9 | 376 |
| 100 000 | kmeans | 0.364 | inertia 49 893 | 13.1 | 328 |
| 100 000 | gmm | 0.657 | score -2.548 | 13.0 | 376 |
| 500 000 | kmeans | 1.979 | inertia 249 564 | 63.4 | 328 |
| 500 000 | gmm | 3.113 | score -2.5484 | 66.7 | 376 |
Reference: sklearn MiniBatchKMeans inertia within 1.15x on the same data;
sklearn GaussianMixture(diag) log-likelihood within 0.15 nats. Peak RSS at
n=500k is 67 MB, well under the 2.2 GB budget.

## RAM model
State = O(K*d) both models. Fit holds a bounded 10*batch reservoir
(O(batch*d)); the EM loop keeps only the K*d sufficient statistics. Every
batch passes through check_batch(extra_factor=3.0).
