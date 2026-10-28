# CADW primitives reused by reference implementations

All paths relative to repo root `/home/maresin/Dev/sandbox/temp_cadw/Cadw_framework`.
Reference algos live under `.pingis/impl/<Sec>/` and import from these modules
(read-only use; no edits to `cadw/`).

## 1. MemoryGuard — `cadw/core/memory_guard.py`
Class `MemoryGuard(max_memory_bytes=..., safety=..., state_safety=..., buffer_safety=...)`.
Relevant API used by reference impls:
- `.budget_bytes` -> total budget.
- `.available_bytes` -> remaining after reservations.
- `.state_limit_bytes` -> `available_bytes() * state_safety` (0.5 default).
- `.row_size_bytes` -> conservative per-row size after `estimate_row_size*`.
- `.get_batch_rows(batch)` -> max rows per batch to fit budget (used in smoke tests).
- `.check_batch(batch, name=, extra_factor=)` -> raises `MemoryLimitExceeded` if batch too big.
- `.check_state(nbytes, name=)` -> raises if state exceeds state limit.
- `.estimate_state_size(op_name, schema=, d=, bins=)` -> pre-flight state size.
- `.estimate_unique_keys(sample_batch, n_total)` -> `k_est` for shuffle planning.
- `.estimate_row_size_from_batch(batch)` / `.estimate_row_size_from_schema(schema)`.
- `.plan(OpSpec(...), DataChars)` / `.plan_pipeline(ops, DataChars)` -> pick `Params` for
  `reduce` / `hash_shuffle` / `sort` / `groupby` / `join` memory models.
- `.reserve(n)` / `.release(n)` -> manual reservation accounting.
Usage pattern in ref impls: instantiate at top of `fit(...)` with `max_memory_mb=<budget>,
state_safety=0.5, safety=0.7`; call `.check_state()` before allocating any dense state
(O(d^2), O(V), O(K*d)); call `.check_batch(batch)` inside any per-batch loop that does
non-trivial numpy work.

## 2. Reduce — `cadw/ops/reduce.py`
`ReduceOp` ABC: `init()`, `update(state, batch)`, `merge(a,b)`, `finalize(state)`,
`state_nbytes(state)`, optional `estimate_state_nbytes()`, optional `batch_memory_factor()`.
Driver `reduce(batch_iter, op, memory_guard, *, log=True)` -> single pass, RAM bounded by
`batch + state + O(1)`. Pre-flight `check_state(op.estimate_state_nbytes())` before `init()`.
Built-ins used directly: `CountOp`, `SumOp` (`cadw/ops/reduce.py`).
Reference impls subclass `ReduceOp` for per-algorithm states:
- `I/kaplan_meier.py`: `KMCountsOp` (per-unique-time at-risk / events) — actually uses Sort first.
- `I/weibull_aft.py`: `XTWXOp` for sufficient stats (d*d + 2d), Newton step afterwards.
- `J/complement_nb.py`: `CNBClassTermOp` — per-class term counts (C x V) + totals.
- `J/nmf.py`: `XtVOp`, `WHOp` for multiplicative updates.
- `M/bandits.py`: `ArmSumCountOp` for epsilon-greedy / UCB1 / Thompson.
- `N/kernel_density.py`: `LogSumExpOp` for log-likelihood per query batch.
- `O/*`: `UniqueValsOp`, `QuantileEdgesOp`, `StratifiedCountOp`.
Every op overrides `estimate_state_nbytes()` (else MemoryGuard fallback = 1 MiB).
See canonical example `cadw/ml/kmeans.py::KMeansStepOp` (dense `(k,d)` sums via `np.add.at`).

## 3. HashShuffle — `cadw/ops/shuffle.py`
`hash_shuffle(batch_iter, key_columns, value_columns, num_partitions, memory_guard,
spill_dir, *, hash_seed=0, log=True) -> list[Path]`.
Each output IPC file has `value_columns + key_columns + row_id(uint64)`.
Pair reader: `read_partition_batches(path, columns=None) -> Iterator[pa.RecordBatch]`.
Planner picker: `cadw/core/models_shuffle.py::HashShuffleMemoryModel.solve` (P, writer buffer,
batch) — called via `MemoryGuard.plan(OpSpec("hash_shuffle", ...))`.
Reference usages:
- `J/nmf.py`: shuffle `(row_block, x_row)` to update `W` blockwise.
- `O/knn_imputer.py`: shuffle complete rows into partitions so each NaN block only sees
  its partition's candidate pool.
- `O/stratified_split.py`: shuffle by `(class, rand_seed(row_id))`; each partition is a
  class-stratum, split fractions independently.
`num_partitions` is capped by `MemoryGuard.plan` — never materialize all keys at once.

## 4. Sort — `cadw/ops/sort.py`
`external_sort(batch_iter, key_columns, value_columns, memory_guard, spill_dir, *,
ascending=True, nulls_first=False, merge_fan_in=None, read_batch_size=1000, log=True) -> Path`.
Result: single IPC file with `value_columns + row_id`.
Streaming reader: `read_sorted(path, columns=None) -> Iterator[pa.RecordBatch]`.
`compute_fan_in(...)` returns the merge fan-in used by planner.
Planner picker: `cadw/core/models_sort.py::SortMemoryModel.solve`.
Reference usages:
- `I/kaplan_meier.py`: sort `(t, e)` by `t` then Reduce at-risk/event counts per unique `t`.
- `O/kbins_discretizer.py`: per-column external sort to stream quantile edges exactly.

## 5. SpillDir + Arrow IPC adapters — `cadw/core/spill.py`
`SpillDir(guard=..., name=..., path=None, keep_on_error=True)` context manager:
  - `.path(filename)` -> temp path inside spill dir.
  - `.root` -> dir path.
  - `.cleanup_file(path)` -> free early.
  - on `__exit__` removes dir (keeps on error for debugging).
Adapters for `(n, K)` float matrices:
  - `array_schema(shape_or_width, dtype)` -> `pa.Schema` (columns `c0..cK-1`).
  - `array_to_record_batch(arr)` / `record_batch_to_numpy(rb)` -> bidirectional.
  - `spill_array_writer(path, schema)` context-managed `ArrowIPCBatchWriter`.
  - `spill_array_reader(path, columns=None)` context-managed `ArrowIPCBatchReader`
    (closes to force mmap unmap — critical for RSS).
NEVER read all spill files into RAM: iterate via `spill_array_reader` / `read_sorted` /
`read_partition_batches`. This is the METHODOLOGY §5 "iterative state" pattern.
Reference usages:
- `I/kaplan_meier.py`: spill `(t, S(t), n_at_risk, d)` step function when unique times > state limit.
- `J/nmf.py`: spill `W` (`(n,k)`) between iterations; `H` (`(k,V)`) kept in state if `k*V` fits.
- `N/kernel_density.py`: spill K centers when `K*d*8 > state_limit_bytes`, read chunks per batch.
- `O/one_hot_encoder.py`: spill vocabulary when `V > state_limit`; second pass reads vocab chunks.

## 6. IO primitives — `cadw/io/batch_reader.py`, `cadw/io/batch_writer.py`
`ArrowIPCBatchReader(path, columns)` (has `.close()` for eager mmap release),
`ArrowIPCBatchWriter(path, schema)`. Used indirectly through `SpillDir` adapters or
`spill_array_writer/reader`. Reference impls never construct these directly (except through
sort/shuffle return paths, which yield plain `Path`s that we read with `read_sorted` /
`read_partition_batches`).

## 7. Memory models (planner) — `cadw/core/models*.py`
`ReduceMemoryModel`, `HashShuffleMemoryModel`, `SortMemoryModel`, `GroupByMemoryModel`,
`JoinMemoryModel`, registered in `cadw/core/models.py` and selectable via
`MemoryGuard.plan(OpSpec(op_name, params_hint, op_context), DataChars)`.
Reference impls do NOT hardcode batch sizes: they call `guard.plan(...)` or
`guard.get_batch_rows(sample_batch)` and fall back to per-batch iteration of the input
`CadwDataFrame._iter_filtered_batches()` (same pattern as `cadw/ml/kmeans.py::fit_kmeans`).

## 8. Data ingestion (read-only)
- `CadwDataFrame._iter_filtered_batches()` — single streaming entry point used everywhere
  in `cadw/ml/*` (`kmeans.py:129`, `coxph.py`, `tfidf.py`). Reference impls take either a
  `CadwDataFrame` or an `Iterable[pa.RecordBatch]` factory (callable returning a fresh
  iterator) so they can make multiple passes without buffering.
- `DataChars` (`cadw/core/models.py`) constructed via `data_chars_from_sample(sample_batch,
  n_total, k_est=..., avg_key_size_bytes=..., size_of_key_entry_bytes=...)`; sample = first
  batch from `cadw/core/sampling.py::peek_first_batch`.

## 9. Resampling (O/N sections)
`cadw/core/sampling.py` also hosts `data_chars_for_join(...)`. Reservoir / stratified
sampling implemented in `.pingis/impl/O/` reuse only `MemoryGuard.check_state` and
`np.random.default_rng`; no cadw internal sampler to wrap.

## 10. Not used / explicitly avoided
- `pa.Table` / `read_all()`: forbidden by METHODOLOGY §1.1. Reference impls never call them.
- `ArrowIPCBatchReader.batch_size`: ignored by IPC format; batching must be done at the
  source (`_iter_filtered_batches`) or by re-chunking a `RecordBatch` with `pa.Table.slice`
  (allowed because slice is zero-copy — only used for re-chunking a single batch).
