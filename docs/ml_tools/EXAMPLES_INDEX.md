# Pattern → Reference code

**Purpose:** for any new model, find the existing pattern to copy.
Do not invent. Copy the snippet, adapt to your function.

| Pattern | Reference | State | Test |
|---|---|---|---|
| **Reduce** (aggregation) | `cadw/ops/reduce.py::ReduceOp` + `reduce` | O(d) | `test_reduce.py` |
| **HashShuffle** (partition) | `cadw/ops/shuffle.py::hash_shuffle` | O(n/P) | `test_shuffle.py` |
| **Sort** (global order) | `cadw/ops/sort.py::external_sort` | O(k·rbs) | `test_sort.py` |
| **SpillBuffer** (pa.Table output) | `cadw/core/spill.py::SpillBuffer` | O(batch) | `test_spill_buffer.py` |
| **SpillArray** (numpy output) | `cadw/core/spill.py::SpillArray` | O(batch·K) | `test_spill_buffer.py` |
| **AutoResult** (unified view) | `cadw/core/spill.py::AutoResult` | — | `test_spill_buffer.py` |
| **Facade unwrap** | `cadw/dataframe/_core.py::_wrap_result_table` | — | `test_default_auto.py` |
| **Chunked strided** (sliding window) | `cadw/ml/hampel.py::fit_hampel` | O(chunk·w) | `test_hampel.py` |
| **Chunked multi-pass** (mean→cov→score) | `cadw/ml/tspq.py::fit_t2q` | O(chunk·d) | `test_tspq.py` |
| **Chunked SGD** (epoch permutation) | `cadw/ml/autoencoder.py::fit_autoencoder` | O(n) int64 | `test_autoencoder.py` |
| **Pre-sample + chunked build** | `cadw/ml/eif.py::fit_eif` | O(n_est·psi·d) | `test_eif.py` |
| **Reservoir + chunked Lloyd** | `cadw/ml/kmeans_ad.py::fit_kmeans_ad` | O(S·d) | `test_kmeans_ad.py` |
| **Two-path** (real vs duck guard) | `cadw/ml/kmeans_ad.py::fit_kmeans_ad` | — | `test_kmeans_ad.py` |
| **Refuse-at-guard** (documented) | `cadw/ml/binseg.py::fit_binseg` | O(n) | `test_binseg.py` |
| **Streaming SGD** (linear head) | `cadw/ops/optim.py::sgd_streaming` | O(d·K) | `test_rocket.py` |
| **distinct_count via HashShuffle** | *backlog v0.9.95* | O(n/P) | — |

## Anti-pattern → correct-pattern

| Anti-pattern (G-gate) | Fix |
|---|---|
| `np.zeros((n_total, K))` (G1) | `np.memmap` via `SpillDir` |
| `np.zeros((hyperparam, ...))` (G5) | `SpillDir` or `HashShuffle` |
| `parts.append` + `np.concatenate` (G6) | `SpillArray.finalize()` |
| `sliding_window_view(...).copy()` (G3) | strided + chunked loop |
| `check_state` after `append` (G2) | `check_state` before |
| `len(set(col.to_pylist()))` (G2) | `HashShuffle` + `distinct_count` (backlog) |
