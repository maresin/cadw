# D.5.4 BIRCH streaming CF-tree clustering

## Paper
Zhang, Ramakrishnan, Livny, *BIRCH: An Efficient Data Clustering Method for
Very Large Databases*, SIGMOD 1996. Reference used in tests: sklearn
`Birch` (scikit-learn, JMLR 12, 2011).

## Docs
- `cadw.ml.birch.fit_birch(df, X_cols, n_clusters=8, threshold=0.5,
  branching_factor=50, max_iter=20)`
- `CadwDataFrame.fit_birch(...)` (mixin in `cadw/dataframe/_ml_bindings.py`).
- Model: `predict(df)`, `centers_`, `to_dict()`, `state_nbytes()`.

## Design
- Streaming CF insertion over Arrow batches. CF = (N, LS, SS) sufficient
  statistic; insertion walks the single-level CF list and absorbs into the
  closest CF whose post-insertion radius stays <= `threshold`.
- When fan-out exceeds `branching_factor` the two closest CFs are merged
  (paper's split policy), keeping the CF count bounded.
- Phase 3 agglomerates leaf CF centroids with weighted k-means++ (weights =
  CF counts), vectorised radius computation per batch.
- No point is retained after insertion; state = O(#CF*(2d+1)) floats.

## Benchmark (`.pingis/benches/birch.json`, d=2, k=4, threshold=0.5)
| n | fit_s | n_cf | clusters | peak_mb | state_b |
|---|---|---|---|---|---|
| 5 000 | 0.286 | 50 | 4 | 1.1 | 4 176 |
| 20 000 | 1.336 | 50 | 4 | 0.0* | 4 176 |
| 50 000 | 3.388 | 50 | 4 | 3.4 | 4 176 |
*RSS sampler floor (delta clamped at 0); sampling noise on a fast run.
Reference: sklearn Birch on the same 4-blob data yields the same cluster
assignment up to label permutation (purity 1.0 in tests). Fit time is
O(n * #CF) with #CF bounded at branching_factor=50, independent of n.

## RAM model
State = #CF * (2d+1) * 8 bytes + centroids (k*d*8) + 512 overhead, constant
in N. Every batch passes `check_batch(extra_factor=3.0)`; model state is
checked via `check_state`.
