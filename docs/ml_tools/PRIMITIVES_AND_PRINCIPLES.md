# Primitives and Principles — CADW Cheat Sheet

**For:** any session working on `cadw/` (human or AI). **Mandatory
context.** This file is included in every `--mode=tiny` snapshot.

**Version:** 1.0 (2026-10-27, v0.9.87-phase0a-fix1)
**Companion:** `docs/ml_tools/METHODOLOGY_ALGORITHM_SELECTION.md`

---

## §1 Principles — postulates, not conventions

1. **Out-of-core by design.** No more than one batch in RAM. Bounded
   state: RAM does not grow with `n`, given disk.
2. **Base cases, not exceptions.** Unique categories > limit, whole
   column > limit, iterative state > limit, feature matrix > limit
   — ALL must work (with disk). If they don't, it's a CADW bug, not
   user data. See METHODOLOGY §1.1.
3. **No escape hatches.** No `read_all()`, no `collect()`, no
   `to_numpy()` for n-sized results. If a model needs all data in
   RAM, it is not CADW material — refactor to O(batch) state, or
   `described_only: true`.
4. **Three primitives + composition.** `Reduce`, `HashShuffle`,
   `Sort`. Everything heavy is their composition or explicit spill.
   Plus `ArrowIPCBatchWriter` / `ArrowIPCBatchReader` for the base
   I/O layer.
5. **Guard before alloc.** Every n-sized or O(n·k) allocation
   preceded by `guard.check_state` or `guard.check_batch`.
6. **Primitives reuse.** No new SpillArray-like class if
   `ArrowIPCBatchWriter` / `ArrowIPCBatchReader` + existing
   primitives suffice. Extend before duplicating.

---

## §2 Primitives — Reduce, HashShuffle, Sort, Batch I/O

### Reduce
`cadw/ops/reduce.py` + `cadw/core/models_reduce.py`

```
f_reduce = batch*row_size + 2*state_bytes
state_bytes = op.state_nbytes(init_state)   # provided by op
```

Use for: sum, mean, XTX, covariance, histograms, class stats,
Welford. **Never** for iterative per-row state — see §5 pattern 3.

### HashShuffle
`cadw/ops/shuffle.py:327-403`

```python
spill_dir = Path(spill_dir)
spill_dir.mkdir(parents=True, exist_ok=True)
paths = [spill_dir / f"partition_{p:05d}.arrow" for p in range(P)]
writers = [ArrowIPCBatchWriter(str(p), schema) for p in paths]
for batch in source:
    writers[partition_id].write_batch(batch)
```

Use for: groupby (skew isolation), join (build side), dedup, ALS
user/item factors. `groupby.py:384` shows recursive sub-partition
spill for heavy keys.

### Sort
`cadw/ops/sort.py::external_sort`

```
f_sort = max(run_buffer, k*read_buffer + k*heap + read_batch*row_size)
```

Writes runs to disk, k-way merge. Use for: global median, top-k,
k-fold, rank, ALS item-sweep. **v0.9.46-fix2** — ALS had
`_sorted_batches(False)` with sort key `["i"]` (wrong: i = user_pos
in that view); correct is `["u"]`. Production bug, 3 months latent,
found by 1-step comparison. **Lesson:** always verify 1-step vs
manual reference before tuning.

### Batch I/O — the base for iterative spill

`cadw/io/batch_writer.py`:
```python
ArrowIPCBatchWriter(path, schema)       # .write_batch(pa.RecordBatch)
ParquetBatchWriter(path, schema, compression='snappy')
```

`cadw/io/batch_reader.py`:
```python
ArrowIPCBatchReader(path, columns=None, batch_size=None)
  .__iter__ / .__next__      -> pa.RecordBatch (streaming)
  .get_batch(idx)            -> O(1) random access via IPC footer
  .close()                   -> forces mmap unmap
```

**v0.9.31 audit A1:** `ArrowIPCBatchReader.close()` explicitly closes
the mmap because Arrow does not track OS-mapped pages in its pool
accounting (RSS drift 5-13× on N=1M without it). Any code that
holds N readers (e.g. k-way merge) must close them.

**No `SpillArray` class exists.** Use `ArrowIPCBatchWriter` /
`ArrowIPCBatchReader` directly + a `SpillDir` context manager for
lifecycle (planned, see §5 pattern 3).

---

## §3 MemoryGuard API

`cadw/core/memory_guard.py` — 870 lines. Key methods:

```
check_batch(n_bytes)                     # pre-batch alloc
check_state(state_bytes, name=...)       # pre-n-sized state
check_partition_buffers(P, buf_bytes)    # HashShuffle writers
check_partition_writers(...)
reserve(nbytes) / release(nbytes)        # manual accounting
get_read_batch_size() / get_write_batch_size()
get_max_dict_elements()                  # for unique/dict-heavy ops
estimate_row_size_from_batch / from_schema / with_sample
estimate_state_size(op, data_chars)
estimate_unique_keys(col)
```

Limits:
```
budget_bytes, state_limit_bytes, available_bytes, usable_bytes,
_state_limit, _buffer_limit
```

Duck-typed guards (test doubles) with only `max_memory_mb` attribute
still work via the `_probe_memory_limit_mb(guard)` helper — but that
helper is **duplicated in 15 modules**. Consolidation planned
(backlog item, low priority).

**Canonical pattern for a new model:**
```python
if guard is not None:
    est_bytes = 8 * n_pre * d_pre           # analytic shape
    if hasattr(guard, "check_state"):
        guard.check_state(est_bytes, name="model.state")
    else:
        limit_mb = _probe_memory_limit_mb(guard)  # legacy fallback
        ...
# THEN allocate
```

---

## §4 DataChars / OpMemoryModel / Planner

`cadw/core/models.py` (~200 lines):
- `DataChars(row_size_bytes, k_est, avg_key_size_bytes,
  size_of_key_entry_bytes, ...)` — computed from sample.
- `OpMemoryModel` (ABC): `estimate(params, data_chars) -> int`,
  `solve(available_bytes, data_chars, ...) -> Optional[Params]`.
- `Params` (dataclass base).
- `register_model(m)` — global registry used by `planner.plan()`.

**Planner** (`cadw/core/planner.py`) is **predict-only** in v0.1.x.
`spill_boundaries` are reserved for v0.2. Do NOT try to use planner
for runtime decisions — call `MemoryGuard.check_*` directly.

**MemoryModels** (`models_reduce.py`, `models_sort.py`,
`models_shuffle.py`) are analytic formulas for the planner — they
**do not** allocate memory. Real allocation happens in `ops/`.

---

## §5 Spill patterns — three known patterns

### Pattern 1: Ops-with-spill_dir
Groupby / shuffle / dedup accept `spill_dir` parameter. Pattern:
```python
spill_dir = Path(spill_dir)
spill_dir.mkdir(parents=True, exist_ok=True)
paths = [spill_dir / f"partition_{p:05d}.arrow" for p in range(P)]
```
Recursive sub-partition (groupby.py:384) for heavy keys:
```python
sub_spill = spill_dir / f"resplit_d{depth}_{partition_path.stem}"
sub_spill.mkdir(parents=True, exist_ok=True)
```
**Caller** manages cleanup (not the op).

### Pattern 2: External sort
ALS user/item factors:
```python
upath = external_sort(_sorted_batches(True), ["u"], ["u","i","w"],
                      memory_guard, spill_dir / "shuffle_user")
```
`external_sort` internally writes runs to disk, k-way merges.

### Pattern 3a: Memmap for random-access iterative state
When an iterative loop needs **random row access** (e.g. patch_memory_ad
training loop uses `rng.permutation` -> row i at arbitrary times),
ArrowIPC sequential scan is insufficient. Use `np.memmap` via
`SpillDir` for lifecycle:

```python
with SpillDir(guard=guard, name="pma") as sdir:
    X_mm = np.memmap(str(sdir.path("X.bin")), dtype=np.float64,
                     mode="w+", shape=(n, d))
    for chunk, start in chunked_windows(x, w, chunk_nw):
        X_mm[start:start+len(chunk)] = compute(chunk)  # write
    X_mm.flush()
    for i in rng.permutation(n):
        xi = X_mm[i:i+1]        # O(1) random row (OS page cache)
        train_step(xi)
    del X_mm                    # release handle before SpillDir exit
```

Memmap semantics: Python-side peak = O(chunk*d), RSS may grow via OS
page cache but is OS-evictable (not tracked by tracemalloc). Use for
random-access; use ArrowIPC (SpillArray) for sequential patterns.

### Pattern 3b: Sequential ArrowIPC iterative-state spill (phase0b)
For models whose per-row state must survive iterations (GBDT pred,
lambdamart X stack). Pattern:
```python
with SpillDir(guard=mg, name="gbdt_mc") as sdir:
    pred_path = sdir.path("pred_r000.arrow")
    writer = ArrowIPCBatchWriter(pred_path, array_schema((n, K), np.float64))
    for b in df._iter_filtered_batches():
        writer.write_batch(array_to_record_batch(base_pred_b, K))
    writer.close()

    for round_i in range(n_estimators):
        next_path = sdir.path(f"pred_r{round_i+1:03d}.arrow")
        writer = ArrowIPCBatchWriter(next_path, array_schema((n, K), np.float64))
        with ArrowIPCBatchReader(pred_path, columns=[f"c{k}" for k in range(K)]) as reader:
            for b, prev_pa in zip(df._iter_filtered_batches(), reader):
                prev = pa_to_numpy(prev_pa)          # (b, K)
                new = update_pred(prev, b, y_all, K) # (b, K)
                writer.write_batch(array_to_record_batch(new, K))
        writer.close()
        sdir.cleanup_file(pred_path)                 # free round_i
        pred_path = next_path
```

No `read_all()` in this pattern — always through batches or index.

---

## §6 Anti-patterns (blocked by gates G1-G4, METHODOLOGY §8)

| Anti-pattern | Gate | Fix |
|---|---|---|
| `np.zeros((n, K))` for iterative state | G1 | SpillDir + ArrowIPCBatchWriter |
| `sliding_window_view(...).copy()` full-n | G3 | strided + chunked |
| `check_state` after allocation | G2 | move guard up |
| `tier1_core` + `bounded_state:false` | G4 | demote or refactor |
| `read_all()` / `collect()` / `to_numpy()` n-sized | PRINCIPLE §1.3 | refactor |
| `np.vstack(list_of_batches)` with len=n | G1 | SpillDir or pre-allocated Reduce |

---

## §7 Recent decisions — one-line index

Full details in `CHANGELOG.md` (available in `--mode=full`).

```
v0.9.87 (2026-10-27): audit d07072d → METHODOLOGY + SpillDir plan + gates
v0.9.87-fix1c: BOCPD rolling posterior (163MB→0.5MB @ n=100k)
v0.9.87-fix1b/2: 8 windowed modules guard-order + est_bytes fix
v0.9.87-fix1a: sub_pca guard-order fix
v0.9.87-phase0a: METHODOLOGY_ALGORITHM_SELECTION.md + refactor plan
v0.9.86.3: remove HybridETSGBDT (redundant w/ Ensemble, false sota)
v0.9.86.2: attribution fix (M4→Zhang 2003); nonlinear bench
v0.9.85: metrics extension (wape/mqlike/msis/wql/sql/crps)
v0.9.84: GBDT transforms integration + AnomalyPipeline
v0.9.83: GBDTForecaster capability-based transforms
v0.9.82: ensemble + transforms closure (documented)
v0.9.81a: EnsembleForecaster (per-member pipeline)
v0.9.80: AutoForecastPipeline + capability flags
v0.9.79a-b: transforms (Differencer, STL, MSTL, AutoStat)
v0.9.78: OTM + AutoTheta (CV-selected STM/OTM)
v0.9.76: AutoETS (AICc) + 3-role ensemble
v0.9.72: BOCPD + 3-role D3.2 ensemble
v0.9.64: guard false-positive fix (np not flagged as high-risk)
v0.9.50: M3 legacy v2→v3 schema migration (53 cards)
v0.9.49: value_counts multi-chunk schema (int64/double alignment)
v0.9.46-fix2: ALS sort key ["i"]→["u"] (3-month latent bug)
v0.9.31 A1: ArrowIPCBatchReader.close() forces mmap unmap
v0.9.22: GBDT multiclass state formula (n*8*(1+4K), was n*K*8*3)
```

---

## §8 Quick checklist when touching `cadw/`

Before commit:

- [ ] Any n-sized allocation has `guard.check_state` BEFORE it
- [ ] Iterative state (survives between iterations) uses SpillDir pattern
- [ ] Windowed computation uses `strided_view` + chunked iteration
- [ ] No `read_all()` / `collect()` in new code
- [ ] If n-sized state cannot be avoided without new primitive — open
      discussion before coding; do not silently allocate
- [ ] YAML card has `state_class`, `state_type`, `bounded_state`,
      `n_scalability`, `streaming_strategy`, `primitive_reuse`
- [ ] Gate tests pass (`pytest -q tests/test_primitive_reuse.py
      tests/test_guard_before_alloc.py tests/test_no_dense_windows.py
      tests/test_ml_tools_schema.py`)
