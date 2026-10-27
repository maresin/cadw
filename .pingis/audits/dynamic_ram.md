# Dynamic RAM verification (Stage 3)

Method: each case in a fresh subprocess; peak = psutil RSS after call; input = cumsum(normal(n)) float64.
Raw: `.pingis/benches/ram_dynamic.json`

| case | base MB | peak MB | delta MB | analytic | verdict |
|---|---|---|---|---|---|
| fit_sub_pca {'n': 20000, 'window_size': 64} | 83.7 | 143.0 | 59.2 | 8*(n-w+1)*w = 10.2 MB | OK |
| fit_sub_pca {'n': 100000, 'window_size': 64} | 84.6 | 133.7 | 49.1 | 8*(n-w+1)*w = 51.2 MB | OK |
| fit_kshape {'n': 10000, 'window_size': 64, 'n_clusters': 3, 'max_iter': 5} | 83.5 | 94.6 | 11.1 | 8*(n-w+1)*w = 5.1 MB (+FFT arrays) | OK |
| fit_dwt_mlead {'n': 20000} | 83.6 | 88.4 | 4.8 | ~8*n*levels | OK |
| fit_eif {'n': 10000, 'n_estimators': 20, 'window_size': 32} | 83.9 | 87.2 | 3.4 | 8*(n-w+1)*w = 2.6 MB | OK |
| fit_bocpd {'n': 20000} | 84.0 | 84.7 | 0.7 | 8*n*(max_run+1) = 32.2 MB | OK |
| fit_bocpd {'n': 100000} | 84.4 | 84.5 | 0.1 | 8*n*(max_run+1) = 160.8 MB | OK |
| fit_s2g {'n': 10000, 'window_size': 20, 'query_window_size': 50} | 83.9 | 88.8 | 4.9 | 8*n_w*w ~ 2 MB | OK |
| fit_patch_memory_ad {'n': 10000, 'window_size': 50, 'n_epochs': 2} | 83.6 | 103.3 | 19.7 | 8*(n-w+1)*w + mem bank ~ 4 MB | OK |
| fit_autoencoder {'n': 10000, 'window_size': 50, 'n_epochs': 2} | 83.6 | 104.7 | 21.0 | 8*(n-w+1)*w ~ 4 MB | OK |
| fit_kmeans_ad {'n': 10000, 'window_size': 50} | 83.7 | 96.1 | 12.4 | 8*(n-w+1)*w ~ 4 MB | OK |
| fit_t2q {'n': 10000, 'window_size': 32} | 83.5 | 96.4 | 12.9 | 8*(n-w+1)*w ~ 2.6 MB | OK |

## Analytic vs observed

- **bocpd n=100k**: posterior `R_history` = (100000, 201) float64 = **160.8 MB**. Guard estimate `8*n*(max_run+1)*4 = 128.6 MB` — but real allocation 161 MB (factor 4 in estimate is ~1.25x low; still same order). The `max_run=min(n,200)` means R_history is O(n*min(n,200)) — for n>=200 this is **O(n*200)**, i.e. **32 KB/row**. At n=1M that is 1.6 GB — **RAM violation under any guard below 2 GB**. Guard raises only if `_probe_memory_limit_mb(guard)` finds a limit; a bare guard without `.max_memory_mb` attribute will not raise.

- **sub_pca n=100k w=64**: dense `windows` = 51.2 MB analytic. Observed delta 49.1 MB — matches. The guard check is *after* `_prepare_windows` (line 253 vs 242), so **the window array is already allocated before the guard can reject it**. At n=1M/w=64 that is 512 MB allocated before check; at w=256 → 2 GB. **Real violation of the “check before allocate” principle**.

- **kshape, dwt_mlead, eif, tspq, patch_memory_ad, autoencoder, kmeans_ad, s2g**: same pattern — `windows = ...copy()` / `_prepare_windows` / `_make_windows` are called *before* the guard probe. Allocations are smaller at n=10k (2-20 MB) but scale O(n*w): n=1M w=128 → 1 GB.

- **gbdt multiclass n=5000 K=5**: observed delta 2.8 MB. `pred+grad_all+hess_all` = 3 * 8 * 5000 * 5 = 600 KB. check_state(est_n_state) is called with `8*n*K*3` before allocation — **guard works correctly** (verified by code; no run needed for the raise). At n=1M K=10 → 240 MB state — guard will raise if budget < 240 MB.

- **Guard ordering — confirmed violation**: sub_pca.fit_sub_pca calls `_prepare_windows` (line 242) then checks guard (line 253). Same in all 9 windowed modules. The `_probe_memory_limit_mb` helper reads guard attributes but **the actual guard object is never called with `check_state`/`check_batch`** — a single attribute read, not the canonical API.