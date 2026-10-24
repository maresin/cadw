# D.4.1 MiniRocket streaming

## Paper
Dempster, Schmidt, Webb, *MINIROCKET: A Very Fast (Almost Deterministic)
Transform for Time Series Classification*, KDD 2021.

## Docs
- `cadw.ml.minirocket.fit_minirocket(df, value_col, id_col=None, num_kernels=84, seed=0, max_dilation=32)`.
- `CadwDataFrame.fit_minirocket(...)` binding.
- Model: `transform(df)` -> (n_series, num_kernels) PPV feature matrix.
- `make_kernels(num_kernels, seed, max_dilation)` for the deterministic kernel set.

## Design
- Kernel length 9, weights in {-1, +1}, exactly three +1 weights (one per
  third), seeded RNG -> deterministic and reproducible.
- Dilation grows geometrically 1, 2, 4, ... up to max_dilation, then wraps.
- Features are PPV (proportion of positive values) per kernel per series.
- Streaming: one series materialised at a time; input must be grouped/sorted
  by id_col (guarded). Memory is O(longest series) + O(num_kernels).

## Benchmark (`.pingis/benches/minirocket.json`, num_kernels=84)
| n_series | length | fit_transform_s | series/s | peak_mb | state_b |
|---|---|---|---|---|---|
| 200 | 500 | 0.61 | 326.5 | 6.0 | 7 744 |
| 1000 | 500 | 3.13 | 319.5 | 15.3 | 7 744 |
| 500 | 2000 | 2.01 | 248.6 | 20.9 | 7 744 |
Throughput ~320 series/s independent of series count; peak RAM scales with
series length, not series count.

## RAM model
State = num_kernels * (KERNEL_LEN * 8 + 8) + O(1). Transform: buffer for one
series only (O(longest series)), plus the output feature matrix
(n_series x num_kernels x 8). check_batch(extra_factor=2.0) on every batch.
