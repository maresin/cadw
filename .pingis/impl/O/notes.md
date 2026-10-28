# Section O — algorithm specs (EDA / cleaning / FE)

Conventions same as I/J/M/N. Input for encoders: `batch_iter_factory` of `pa.RecordBatch`
with columns to transform; multi-pass safe.

## O2.KNNImputer
- state_class: O(K*d) complete-row candidate pool (K user-capped), spilled to disk if `K*d*8 > state_limit`
- state_type: partitioned (HashShuffle complete rows by block) + reduce (top-k distances per NaN row block)
- bounded_state: true — NaN block processed per batch; candidate pool on disk
- n_scalability: O(N*d + nan_rows*K*d) time, RAM = block_rows*d*8 + K*d*8 (if K fits) else block_rows*d*8 + chunk*d*8
- primitive_reuse: `hash_shuffle(complete_rows_iter, key_columns=['block_id'], value_columns=feature_cols, num_partitions=P, spill_dir=...)`; `read_partition_batches` for candidate pool; `ReduceOp` per block top-k accumulation; `MemoryGuard.check_state(8*block_rows*d)`; `SpillDir` for pool
- passes: 2 (pass 1 collect complete rows into partitions; pass 2 stream incomplete rows, for each block load its candidate partition and impute)
- algorithm (Troyanskaya 2001): for each NaN row, distance over *observed* dims only (not full d); k nearest complete rows weighted by 1/dist; impute NaN with weighted mean. If block has no complete rows in its partition -> nearest complete partition via centroid distance (bounded)
- edge cases: column all-NaN -> left as NaN, warn; k > pool size -> use pool size
- output: `KNNImputerModel(pool_path, k, n_features, col_names, weights='uniform'|'distance')`; `.transform(batch_iter)` yields imputed batches; `.fit_transform` = fit + transform
- deviation from sklearn: sklearn materializes the training matrix and uses full pairwise distance per query; we keep complete rows on disk and stream. Improvement: partition-based candidate narrowing (block_id = hash(round_robin(row_id))) so per-NaN-row search is O(K) not O(n_complete).

## O3.OneHotEncoder
- state_class: O(V) vocabulary per column (spilled if `V > state_limit`)
- state_type: aggregation (unique values) + elementwise (encode)
- bounded_state: true
- n_scalability: O(N log V) time (sort-based) or O(N) (hash unique), O(1) RAM if spilled
- primitive_reuse: pass 1 `external_sort(key_columns=[col_name])` -> stream unique via `read_sorted`, accumulate per-column vocab; `SpillDir` + `array_to_record_batch` to store vocab when V large; pass 2 stream encode via `np.searchsorted` on vocab (loaded chunk-wise if spilled)
- passes: 2
- algorithm: for each column independently: pass 1 sort & dedup -> vocab; pass 2 encode via `searchsorted`; unseen categories -> extra `_unknown` column (optional, `handle_unknown='ignore'|'error'`)
- edge cases: null -> unknown column; float -> treated as categorical (warn if high cardinality)
- output: `OneHotEncoderModel(categories_: list[np.ndarray], drop=None, handle_unknown='ignore', feature_names_=...)`; `.transform(batch_iter)` yields encoded batches (sparse-like via csr from numpy per batch)
- deviation from sklearn: sklearn stores full V in RAM; we spill vocabulary. Improvement: if user passes `max_categories`, we keep only top-M most frequent (from the same sort-based pass, streaming counts).

## O3.KBinsDiscretizer
- state_class: O(d * n_bins) edges
- state_type: cross_row (Sort) — per-column quantile edges
- bounded_state: true — edges fit in RAM after accumulation (d*n_bins small); sort runs are on disk
- n_scalability: O(N log N * d) time, O(1) RAM per column
- primitive_reuse: `external_sort(key_columns=[col])` per column, `read_sorted` to stream; for exact quantiles, cross-run streaming with two-level (per-batch quantile -> merge quantile); `MemoryGuard.check_state(8*d*n_bins)`
- passes: 1 per column (external sort) + 1 encode pass
- algorithm: exact quantiles via sort (sklearn uses subsampling by default; we offer exact as reference and optional `subsample=int` -> reservoir of that size). Edges midpoints between quantiles; `strategy='uniform'|'quantile'|'kmeans'` (uniform via min/max from Reduce; kmeans via 1-D Lloyd on sorted stream)
- output: `KBinsDiscretizerModel(bin_edges_: list[np.ndarray], encode='ordinal'|'onehot', n_bins, strategy)`; `.transform(batch_iter)`
- deviation from sklearn: sklearn default `subsample=200_000`; we do exact for reference and expose `subsample` optional. Improvement: no memory blowup even for N=1e8 since sort is external.

## O6.StratifiedSplit
- state_class: O(C) fold counters (C = #classes or #strata keys)
- state_type: cross_row (HashShuffle by class label + row_hash)
- bounded_state: true
- n_scalability: O(N) time, O(1) RAM
- primitive_reuse: `hash_shuffle` twice — first by `y` (strata), then per-stratum shuffle by `row_hash` to randomize within stratum; or single shuffle by `y` only + deterministic `row_hash` sort in `read_partition_batches`; `Reduce` counters per fold per stratum; `SpillDir` + `spill_array_writer` to write train/test assignments per fold
- passes: 1 to compute assignments (write fold-id per row to disk), then user calls `.split()` per fold which streams fold batches by reading the assignment file and joining by row_id
- algorithm: for fold f in 0..K-1, for stratum s, assign rows to fold f if `(rank_in_stratum % K) == f` where `rank_in_stratum = cumulative count in hash order` (deterministic with seed)
- edge cases: stratum with < K rows -> distributed round-robin, warn; stratify=None -> random K-fold
- output: `StratifiedSplitter(n_splits, shuffle=True, random_state, stratify_col)`; `.split()` yields (train_ids_path, test_ids_path) or in-memory mask batches via `read_partition_batches`
- deviation from sklearn: sklearn materializes y; we shuffle and stream. Improvement: single-shuffle design (shuffle by `(y, hash(row_id))` in one `hash_shuffle`), then a running per-stratum counter produces fold ids in O(1) RAM.

## O6.ReservoirSample
- state_class: O(K) reservoir
- state_type: elementwise (online)
- bounded_state: true
- n_scalability: O(N) time, O(K) RAM
- primitive_reuse: `MemoryGuard.check_state(8*K*d)`; no shuffle/sort needed; `np.random.default_rng` for vectorized batch sampling
- passes: 1
- algorithm (Vitter R): per batch, `n_new = batch.num_rows`; for each i > K slot we need `rand(0..i)` — vectorized: use geometric skip in big batches (Algorithm L) to avoid one RNG call per row; small batches (< 4*K) use simple rejection
- edge cases: K > N -> return all, warn; K=0 -> raise
- output: `ReservoirSampler(k, random_state)`; `.add(batch_iter)`, `.sample()` -> `pa.RecordBatch` of K rows
- deviation from sklearn: none directly; we add Algorithm L (Li 1994) vectorized for large batches so cost is O(skipped rows) RNG calls instead of O(N).

## O7.GridSearch
- state_class: O(|grid|) parameter tuples + best-score state
- state_type: elementwise (loop over candidates)
- bounded_state: true
- n_scalability: O(|grid| * fit_cost) — delegates to the estimator's out-of-core path; no additional memory
- primitive_reuse: `cadw.cv` (if reusable) + `ReduceOp` `MeanMetricOp` for score aggregation; otherwise estimator's own `fit/predict`
- passes: |grid| independent fits, each with its own internal passes (e.g., 2 for encoders, 1 for bandits)
- algorithm: for `(params, estimator)` in `ParameterGrid(param_grid)`: fit, score via `Reduce` over streamed validation set (never materialized), keep best
- edge cases: deterministic ordering for tie-break; `n_jobs` ignored (out-of-core, single process)
- output: `GridSearchCV(estimator, param_grid, cv, best_params_, best_score_, best_estimator_)`
- deviation from sklearn: sklearn with `n_jobs=-1` parallelizes in RAM; we stay serial with O(1) RAM per fit. Improvement: early-stop a candidate if its running score upper bound falls below `best_score * (1 - tol)` (optional).

## O7.RandomizedSearch
- state_class: O(n_iter)
- state_type: elementwise
- bounded_state: true
- n_scalability: O(n_iter * fit_cost)
- primitive_reuse: same as GridSearch
- algorithm: sample `n_iter` param tuples from distributions via `np.random.default_rng(seed)`; fit/score/keep-best — same as above
- output: `RandomizedSearchCV(estimator, param_distributions, n_iter, cv, best_params_, best_score_, best_estimator_)`
- deviation: same as GridSearch; improvement: cache the sorted key stream once if estimator is stateless-per-batch (e.g., encoders).

## SKIP rationale
See selection.md — PolynomialFeatures (d^2 explosion), IterativeImputer/MICE (multi-pass with per-row missing-ness mask is possible but complex), SMOTE (kNN over imbalanced — same pattern as KNNImputer but not priority), YDataProfiling/Pandera/GreatExpectations/Optuna/SHAPTree/Fairlearn/Evidently (wrappers or tier3_export).
