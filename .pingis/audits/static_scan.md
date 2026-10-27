# Static RAM-risk scan — per-pattern counts

Pattern definitions:
- `win_copy`: `sliding_window_view(...).copy()` — forces O(n*w) dense window array
- `win_index`: fancy-index window materialization (`X[idx]`)
- `full_collect`: `Table.from_batches(batches)` / `list(it)` — collects stream to RAM
- `n_state_alloc`: `np.zeros/empty/full((n,...))` — full-dataset state
- `to_pylist` (ML): per-batch or full column Python conversion

| pattern | count |
|---|---|
| win_copy | 4 |
| win_index | 14 |
| full_collect | 127 |
| to_pylist | 15 |
| n_state_alloc | 19 |
| no_guard_ml | 0 |

## Violating call sites (RAM-risky)

### A. Window materialization (O(n*w) per call)

These functions build a dense (n_w, w) float64 array over the whole series with no MemoryGuard check before allocation:

| file:line | snippet |
|---|---|
| cadw/ml/sub_pca.py:91 | sliding_window_view(..).copy() |
| cadw/ml/s2g.py:239 | sliding_window_view(..).copy() |
| cadw/ml/kshape.py:208 | sliding_window_view(..).copy() |
| cadw/ml/dwt_mlead.py:90 | sliding_window_view(..).copy() |

Additional dense window arrays via fancy indexing:
- `cadw/ml/sub_pca.py:205` `windows = X[idx]` -> (n_w, w) float64
- `cadw/ml/sub_pca.py:207` multivariate reshape
- `cadw/ml/eif.py` `_make_windows` -> `X[idx].reshape(n_w, w*c)`
- `cadw/ml/tspq.py` `_make_windows` -> `X[idx].reshape(n_w, w*c)`
- `cadw/ml/patch_memory_ad.py:218`, `cadw/ml/s2g.py:239`, `cadw/ml/kshape.py:208`, `cadw/ml/dwt_mlead.py:90` (`_sliding_windows`)

Note: `hampel.py:114` / `zms.py:105` use `sliding_window_view` WITHOUT `.copy()` — views only, low risk.

### B. Full-dataset state arrays (O(n) / O(n*K))

| file:line | snippet | in-batch-loop |
|---|---|---|
| cadw/ml/lambdamart.py:89 | codes = np.empty((n, d), dtype=np.uint8) | NO (full vstack of X/y/q before) |
| cadw/ml/bocpd.py:237 | R_history = np.zeros((n, max_run + 1)) | NO (whole series) |
| cadw/ml/gbdt.py:1207 | probs_full = np.full((n, K)) | in _iter_pred per batch |
| cadw/ml/gbdt.py:1326 | grad_all = np.zeros((n_total, K)) | NO (multiclass full state) |
| cadw/ml/gbdt.py:1327 | hess_all = np.zeros((n_total, K)) | NO (multiclass full state) |
| cadw/ml/chains.py:95 | out = np.zeros((n, n_models)) | in predict per batch |
| cadw/ml/naive_bayes.py:273 | log_joint = np.empty((n, C)) | per batch in _iter_pred |
| cadw/ml/gmm.py:39 | logp = np.empty((n, k)) | per batch |
| cadw/ml/minibatch.py:202 | logp = np.empty((n, k)) | per batch |
| cadw/ml/kmeans_ad.py:108 | dists_sq = np.zeros((n, k)) | NO (full window set) |
| cadw/ml/pca.py:129 | proj = np.full((n, n_components)) | per batch in _iter_transform |

## Summary — guard presence

- 24/47 ML modules call `MemoryGuard.check_batch` directly.
- 23/47 delegate to `reduce()` (which calls guard) or accept `guard=` param but do not pass check_batch per batch (still guarded indirectly if the Reduce op does).
- **Critical**: the 9 window-based TS anomaly modules (sub_pca, autoencoder, kmeans_ad, kshape, s2g, patch_memory_ad, dwt_mlead, eif, tspq) do **not** call `guard.check_batch` or `guard.check_state` before constructing O(n*w) windows.
- **Critical**: lambdamart materializes the entire dataset (vstack) — `check_state(n_rows*(d*8+16))` is called but does not include the intermediate list `Xs` which duplicates the same memory during accumulation.
- **Critical**: bocpd holds `R_history (n, max_run+1)` — guard has `est_bytes = 8*n*(max_run+1)*4` check with `_probe_memory_limit_mb(guard)` that only reads a probe; the actual `guard.check_state` is not invoked, only a probe on limit.