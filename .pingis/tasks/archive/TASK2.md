# TASK2.md — Library memory audit (Arrow / pandas / numpy)

**Type:** audit + benchmarks + source review
**Branch:** `pingis/library-memory-audit-v0.8.8`
**Depends on:** `analysis_ram.md`, `analysis_external.md` (v0.8.8 run)
**Output dir:** `_FOR_PINGIS/` only. **No `cadw/` edits.**

---

## Context — why this task exists

CADW's **guarantee** is unconditional out-of-core: for `Reduce`,
`HashShuffle`, `Sort`, `groupby`, `join`, `dedup`, `rank`, `quantile`,
`resample` — operations either fit the MemoryGuard budget or raise
`MemoryLimitExceeded` **before** any O(N) allocation.

**But.** The guarantee today is enforced only on CADW's own
allocations. When CADW calls `pa.Table.from_batches`, `Table.join`,
`Table.group_by`, `Table.sort_by`, `pc.sort_indices(table)`,
`to_pandas()`, or `pc.quantile(arr)` — these allocate inside the
library. MemoryGuard does not see them.

Two prior lessons from `DECISIONS.md §13.4`:

- **Episode 25 (LightGBM adapter)** — `lgb.train` is O(N)-state.
  `MemoryGuard` cannot help: OOM happens inside the library **after**
  control left CADW. → Removed.
- **Episode 26 (`partial_fit` whitelist)** — whitelist is trust,
  not guarantee. Any model with hidden growing state OOMs silently.
  → Removed.

**Arrow is the same risk class.** The current code implicitly trusts
Arrow. That is a policy hole, not a detail. Two examples already in
code:

- **P-D1** (`cadw/ops/dedup.py:84-98`): `_local_dedup` does
  `table.to_pandas()` → `df.sort_values` → `df.drop_duplicates` →
  `Table.from_pandas(df)`. Only `table.nbytes` is checked; the pandas
  copies (up to 5× for string columns) are unchecked. Peak estimate
  vs real peak can differ by 2–3×.
- **P-A3** (`cadw/ops/join.py:436-490,589-593`): `Table.join`
  materializes the entire output before `check_state(merged.nbytes)`.
  The guard is `matches × 256 ≤ 0.3 × avail` — correct for narrow
  schemas, off by 2–3× for wide ones.

**Goal of TASK2:** for every Arrow / pandas call CADW makes (or
proposes to make in `analysis_external.md`), produce either:

1. **A documented peak-memory formula + bound condition**, verified by
   benchmark, and **safe under MemoryGuard's check_state contract**; or
2. **A verdict that the call is unsafe** — must not be used, must be
   replaced with a bounded alternative (per-batch, streaming, custom
   implementation), or must be removed.

The output is **canonical formulas** for `DECISIONS.md §9.2.N` (like
`§9.2.1–§9.2.5` for CADW's own ops) — so a future patch that adds an
Arrow call goes through the same rigor as a patch that adds a new CADW
primitive.

---

## Constraints

- **Do NOT modify `cadw/`.** Analysis only.
- Work in branch `pingis/library-memory-audit-v0.8.8`.
- Output only to `_FOR_PINGIS/` (gitignored) and `.pingis/` (except
  `TASK.md` / `TASK2.md`, kept).
- Do not add new dependencies. Use only `pyarrow`, `pandas`, `numpy`,
  `psutil`, `xxhash` — same as `bench/_lib.py`.
- All measurements via `RSSSampler` (from `bench/_lib.py`) — 20 ms
  sampling thread, `peak_delta_mb` and `peak_abs_mb` both reported.
- Do not modify `merged_cadw_project.py`, do not touch tags, do not
  commit to `master`.

---

## Part 1 — Inventory (calls in CADW today)

Grep every Arrow/pandas call in `cadw/` (all `.py` except `__init__`,
`ml/*` is optional). Categorize:

### 1a. High-risk (allocate O(N), O(K²), O(T) with T = input table)

Known from `analysis_external.md` + my own grep — **verify and extend**:

| Call | Location(s) | Current guard | Claimed peak |
|---|---|---|---|
| `pa.Table.from_batches(list_of_batches, schema)` | `groupby.py:265`, `sort.py:104`, `join.py:109`, `dedup.py:88` | varies | 2× table (list + table copy) |
| `pa.Table.join` (Arrow C++) | `join.py:436-490` | `check_state(merged.nbytes)` AFTER | output table size; inside peak unknown |
| `Table.group_by(...).aggregate(...)` | `groupby.py`, `resample.py` (planned), `dedup.py` (planned) | `check_state(table.nbytes)` | hash table K + result |
| `Table.sort_by` | `dedup.py` (planned), `sort.py` (does NOT use — uses `pc.sort_indices` + `pc.take`) | none | unknown — sort-based, likely 2× table |
| `pc.sort_indices(table)` | `sort.py:_write_run` | buf ≤ 0.2×avail | unknown — output is `int64[N]` = 8×N |
| `pc.sort_indices(array)` | `shuffle.py:283` (shuffle parts) | `check_batch` | output `int64[batch]` |
| `pc.take(table_or_array, order)` | `sort.py:_write_run` (chunked), `shuffle.py`, `join.py` | none explicit | output = taken rows; bounded by chunk |
| `pc.quantile` | **planned** (P-Q2) | none | sorted-based → O(N)? |
| `pc.hash_tdigest` | **planned** (P-Q2) | none | approximate; O(1/ε) claimed |
| `pc.rank` | `rank.py:94` (used for `first`), **planned** for rest (P-R1) | per-batch | sort-based? O(N)? |
| `pc.value_counts` | **planned** (P-G2, P-G5) | none | O(K) hash table + result |
| `pc.dictionary_encode` | `shuffle.py:86` (via `to_pandas`), `estimate_unique_keys` | `check_batch(extra_factor=3.0)` for one path | O(K) dictionary |
| `pc.index_in` / `pc.is_in` | **planned** (P-D1, P-SH1) | none | O(N) bool array + hash set of values |
| `to_pandas()` (DataFrame / Series) | `dedup.py:91` | `check_state(table.nbytes)` before | **string dtype ~3× per value**; no check after |
| `Table.from_pandas(df)` | `dedup.py:93` | none | ~1× Arrow table |
| `pc.binary_join_element_wise` | `estimate_unique_keys`, **planned** (P-G2) | none | string concat; O(N) intermediate |
| `pa.concat_arrays(parts)` | `shift.py`, `rolling.py`, `_RollState`-related | none explicit | output size |
| `pa.Table.from_pylist(rows)` | `groupby._accum_to_table` (heavy path, bounded ≤20 rows) | bounded by heavy-key count | small |
| `np.zeros((d, d))` | `reduce.CovarianceOp.init` | `estimate_state_size` before `init` (P8 fix) | 8d² |

**Extend this table yourself** — grep `cadw/` and check every `.to_pandas()`,
`.from_batches(`, `.group_by(`, `.sort_by(`, `.join(`, `pc.` call.

### 1b. Bounded (peak = O(batch) or O(fixed buffer)) — verify, do not re-test

These are known-safe because CADW already enforces `check_batch` +
small batches. Sanity check, but no benchmark needed:

- Element-wise kernels: `pc.add`, `pc.subtract`, `pc.multiply`,
  `pc.divide`, `pc.floor`, `pc.cast`, `pc.is_valid`, `pc.is_null`,
  `pc.and_kleene`, `pc.or_kleene`, `pc.bit_wise_xor`,
  `pc.bit_wise_and`, `pc.equal`, `pc.not_equal`, `pc.less`,
  `pc.greater`, etc. → O(batch).
- Reductions on a batch: `pc.sum`, `pc.mean`, `pc.min_max`,
  `pc.count_distinct(batch_col)` → O(1) scalar.
- Batch-size-dependent: `.to_pylist()` on batch, `.slice`, `.filter`
  on batch → O(batch × 28-80 B Python objects, `check_batch extra_factor`
  should cover).

**Do NOT benchmark these** — waste of time. But do note if you find a
call site without an adjacent `check_batch`.

---

## Part 2 — Peak-memory benchmarks (the core)

For each **high-risk** call, write
`_FOR_PINGIS/bench/bench_arrow_<name>.py` following `_lib.py`'s
`RSSSampler` pattern. Measure:

- `peak_delta_mb` (peak − baseline)
- `peak_abs_mb`
- wall time
- state size (if stateful — `hash_tdigest`, `dictionary_encode`)

Test matrix:

| Function | Cases | Expected behaviour |
|---|---|---|
| `pa.Table.from_batches(gen, schema)` vs `(list, schema)` | N=100k, 1M, 10M; ncols=3 | gen saves list; peak ~1.0–1.5× T |
| `Table.join(left, right, keys, "inner")` | PK–FK; 1:N; N:M with matches=1M | peak ≈ output size (materialized) |
| `Table.group_by(by).aggregate(specs)` | K=100, 10k, 1M; N=1M | peak ≈ K×key + result |
| `Table.sort_by([("k", "ascending")])` | N=100k, 1M, 10M; int64 key | peak ≈ 2× T (sort buffer + result) |
| `pc.sort_indices(table, sort_keys=...)` | N=100k, 1M | output = N × 8 B; **+ any internal buffer?** |
| `pc.sort_indices(array)` | N=1M | output = N × 8 B |
| `pc.quantile(arr, q=[0.5])` on unsorted col | N=100k, 1M | **is it sort-based (O(N) extra) or approximate?** |
| `pc.hash_tdigest(arr, alpha=0.01)` on col | N=1M, 10M | state ≈ c/α, mergeable? |
| `pc.rank(arr, sort_keys=..., tiebreaker=...)` | N=100k, 1M | sort-based → O(N) int64 result + sort buffer? |
| `pc.value_counts(arr)` | K=100, 10k, N (=unique) | peak ≈ K×key + K×count |
| `pc.index_in(arr, value_set=ids)` | N=1M, K=1k | peak ≈ N bool + K×value |
| `pc.is_in(arr, value_set=ids)` | same | same |
| `to_pandas()` on Table (string key col) | N=100k, 1M, T=1MB, 10MB | peak ≈ 3–5× T for object dtype |
| `Table.from_pandas(df)` | same | peak ≈ 1–2× df |
| `pa.concat_arrays(parts)` | total N=1M | peak ≈ 2× output |

For each: emit **3 rows** in `RESULTS2.md`:

```
| Function | Input | peak_delta | peak_abs | wall | formula (fitted) |
```

Fit a linear formula (e.g. `peak = a×T + b×K×c + d`) using least-squares
across the sizes. Report residuals. If residuals > 30% — the behaviour
is non-linear; document why.

**Critical measurement:** always **restart process** between cases.
`psutil.Process().memory_info().rss` reports peak across the process
lifetime; without restart, `baseline_mb` from a previous case pollutes
`peak_delta`. Use `os.execv` or a `subprocess.run` per case, or clear
intermediate data with `gc.collect()` + `del`.

---

## Part 3 — Source code review (where feasible)

For every call where benchmarks suggest non-obvious peak, **read the
source**. In order of accessibility:

### 3a. Python wrappers in `.venv`
- `pyarrow/compute.py` — Python-side `pc.quantile`, `pc.rank`,
  `pc.value_counts`, `pc.hash_tdigest` — look at `**kwargs`, default
  `memory_pool`, any pre-sorts.
- `pyarrow/__init__.py` — `Table.from_batches`, `Table.join`,
  `Table.group_by`, `Table.sort_by` — Python wrappers around C++.
- `pandas/core/*.py` — `.to_pandas()` → `PyArrowConvert` (`Table.to_pandas`
  wrapper); string/object handling.
- `pyarrow/includes/*.pxd` — `.pxd` files (if present) — Cython
  declarations with C++ signatures (`Table::join`, `Table::group_by`).

### 3b. C++ source (if vendored)
- `pyarrow/_compute.pyx` / `pyarrow/_compute.pxd` — if packaged with
  wheels (rare but check).
- `pyarrow/Table.pxi` / `Table.pyx` — if present.

### 3c. GitHub / docs (if network available)
- **Arrow C++ join** — search `acero/join` on GitHub. Expected:
  `HashJoinNode` — hash table on build side, probe batch
  materializes output; peak = |output| × row_width + |build| hash.
- **`Table.group_by`** — `acero/groupby`. Peak = hash table |K| × key +
  |result| × row.
- **`Table.sort_by`** — `acero/sort`. Peak = |T| + sort buffer |T|
  (int64 indices).
- **`pc.quantile`** — `compute/kernels/vector_selection` or
  `compute/kernels/aggregate`. Verify exact vs approximate; if exact
  it sorts.
- **`pc.hash_tdigest`** — `compute/kernels/tdigest`. Peak = state
  size = O(1/α).
- **`pc.rank`** — `compute/kernels/vector_sort/rank`. Sort-based.
- **`Table.from_batches`** — `C++ table.cc`. Iterates batches,
  concatenates into contiguous buffers.

If network is not available — **say so** in `REPORT2.md`. Benchmarks
+ docs-based reasoning is sufficient; source review is bonus.

---

## Part 4 — Live violations in current code

For each of the following, decide: is peak bounded by MemoryGuard today?

### 4a. **P-D1** (dedup pandas path) — **the strongest live violation**

`cadw/ops/dedup.py:84-98`:

```python
table = pa.Table.from_batches(batches)                       # O(T)
check_state(table.nbytes)                                    # checks Arrow table
df = table.to_pandas()                                       # UNCHECKED: 3× for strings
df = df.sort_values("row_id", ...)                           # UNCHECKED: 1× copy
df = df.drop_duplicates(subset=subset_cols, keep=keep)       # UNCHECKED: 1× copy
df = df.drop(columns=["row_id"])                             # UNCHECKED: 1× copy
return pa.Table.from_pandas(df, schema=out_schema, preserve_index=False)  # UNCHECKED
```

**Test:** create a partition of T=100 MB (string key, 10% dupes).
Measure real peak vs `check_state(table.nbytes)`. Expected: real peak
is **3–5×** `T`. If so, this is **P-D1 S2 correctness** — current guard
is wrong, not just conservative.

**Suggest:** either (a) Arrow-native rewrite
`Table.group_by(subset).aggregate([("row_id", "min")])` + `is_in` +
`filter`; or (b) stopgap: add `check_state(pandas_estimate)` before
`to_pandas` where `pandas_estimate = table.nbytes × 5` (worst case for
object dtype). Report which is safe.

### 4b. **P-A1** (join `_safe_materialize_partition`)

```python
batches = [ipc_reader.get_batch(i).select(keep) for i in range(n)]
table = pa.Table.from_batches(batches)   # 2× T_build
```

**Test:** create an IPC file T=100 MB. Measure peak vs `T`. Expected:
**1.8–2.0×** T. If so, `stat().st_size ≤ 0.5×avail` gives
`peak ≤ 1.0×avail` — no headroom at the limit. Confirm.

**Suggest:** `Table.from_batches((reader.get_batch(i) for i in ...), schema)`
→ peak ~1.0–1.2× T (list eliminated; Arrow still copies into table).

### 4c. **P-A3** (join output `Table.join`)

**Test:** `left.join(right, join_type="inner")` on left=1k rows
right=1k rows, then 100k × 100k. Measure `peak − output_size`. Expected
delta ≈ 0. If delta > 0.5× output, the join holds an internal hash
table (build side) on top of output — needs to be counted in the guard.

**Suggest:** `max_output_rows` should use real
`sum(bit_width // 8 for field in empty_schema) + nullable_overhead`,
not 256 bytes.

### 4d. **P-G3** (groupby skew path)

`groupby._groupby_with_skew` opens P writers without
`check_partition_buffers`. Compare with `shuffle.py:270` which does
call it.

---

## Part 5 — Proposed alternatives in `analysis_external.md` — **verify each**

For each **Arrow-native rewrite** proposed in `analysis_external.md §3`,
answer:

1. Does the proposed call's **peak memory** fit the guard's budget
   (verify via benchmark from Part 2)?
2. If yes — is the formula stable across cases (residuals < 30%)?
3. If no — mark **"reject — needs bounded alternative"**.

Compile a table:

| P-ID | Proposed call | Verified peak | Fit? | Verdict |
|---|---|---|---|---|
| P-R1 | `pc.rank` | ? | ? | accept / reject / needs per-batch |
| P-Q1 | `pc.take(targets)` | ? | ? | ? |
| P-Q2 | `pc.quantile` | ? | ? | ? |
| P-Q2-alt | `pc.hash_tdigest` | ? | ? | ? |
| P-G2/P-G5 | `pc.value_counts` | ? | ? | ? |
| P-H1 | `Table.group_by` | ? | ? | ? |
| P-D1 | `Table.sort_by` + `is_in` | ? | ? | ? |
| P-D1-alt | `Table.group_by(subset) + is_in` | ? | ? | ? |
| P-S2 | `pa.concat_tables + pc.sort_indices(batch)` | ? | ? | ? |
| P-SH1 | `pc.dictionary_encode` + `index_in` | ? | ? | ? |
| P-SH2 | `pc.bit_wise_xor` + `pc.cast` | ? | ? | ? |

`accept` = peak formula fits `check_state` contract under any budget
(peak ≤ K × output, K bounded and small).
`reject` = needs per-batch execution or custom implementation.
`needs research` = formula unstable / not derivable — prototype is
insufficient, needs source review or multiple benchmarks.

---

## Part 6 — Output format

Write these files in `_FOR_PINGIS/`:

### `REPORT2.md` — executive summary

At top:

```
**Run**: YYYY-MM-DD
**CADW version**: v0.8.8 (git describe)
**Arrow version**: 26.0.0
**Findings**: X accept, Y reject, Z needs research, W live violations
```

Body:
- Prior-audit (`REPORT.md`) status: P-D1, P-A1, P-A3, P-G3, P-G5 — still
  live? confirmed?
- New live violations found (if any beyond the 5).
- Table of accepted formulas → ready for `DECISIONS.md §9.2.N`.
- Table of rejected proposals → what CADW must NOT do.
- Table of "needs research" → what is deferred and why.
- Recommendation: which `analysis_external.md` rewrites are safe to
  implement, which are deferred, which are removed from backlog.

### `analysis_arrow_memory.md` — full catalog

For every Arrow call in Part 1a:

```
### Call: <pc.xyz / Table.xyz>

- **Location(s)**: <file:line>
- **Current guard**: <none / check_batch / check_state / stat().st_size>
- **Benchmark**: `bench/bench_arrow_<name>.py`
- **Peak formula (fitted)**: `peak = a × X + b × Y + c` (fitted from
  N ∈ {100k, 1M, 10M})
- **Residuals**: max |measured − fitted| / measured
- **Guard recommendation**: what `check_state`/`check_batch` should be
- **Verdict**: SAFE (fits guard contract) / UNSAFE (reject) /
  NEEDS-RESEARCH / NEEDS-BATCHED (per-batch execution)
```

### `bench/bench_arrow_<name>.py` (× N)

One file per high-risk call. All use `_lib.RSSSampler`. Each:
- generates input at N ∈ {100k, 1M, 10M} (as applicable)
- calls the Arrow function **exactly once** per size
- reports `peak_delta_mb`, `peak_abs_mb`, wall
- writes `bench/_out_arrow_<name>.json`

### `bench/RESULTS2.md` — table of all results

Per-row: function, input, peak_delta, peak_abs, wall, fitted formula,
residuals.

### `analysis_source.md` — source review notes

- What was read (`.venv/pyarrow/*.py`, `.venv/pyarrow/includes/*.pxd`).
- What was not accessible (C++ binary, network) — say so honestly.
- For each high-risk function, if source confirms/denies benchmark:
  explicit "source confirms X" or "source unavailable, benchmark-based
  verdict only".

---

## Part 7 — Recommendations to CADW (feed back into `DECISIONS.md`)

For each accepted formula, produce a **`§9.2.N`-ready entry** in the
same format as `§9.2.1–§9.2.5`:

```
### §9.2.6. `pc.quantile` — Arrow compute

**Call**: `pc.quantile(array, q=[...], interpolation="linear", ...)`.

**Peak formula (verified by benchmark, N ∈ {100k, 1M, 10M})**:

    f_quantile = N × sizeof(element_type) + SORT_INTERNAL

where `SORT_INTERNAL` depends on the backend (measured: ~1.0 × N × 8 B
if sort-based, ~0 if approximate).

**Guard recommendation**: `check_state(f_quantile)` before call.
For out-of-core use, prefer per-batch `pc.quantile` +
`pc.hash_tdigest` (P-Q2 alternative) — full-column `pc.quantile`
rejected for T > 0.5×avail.

**Verdict**: `NEEDS-BATCHED` (full-column rejected).
```

Full template per function: call signature, benchmark, formula,
guard recommendation, verdict.

For rejected proposals — short note + reason. Add to
`DECISIONS.md §11.6` backlog: "rejected by library-memory audit".

---

## Part 8 — Acceptance criteria

This task is **done** when:

1. **Every high-risk call in Part 1a** has:
   - at least one benchmark file with 3 sizes,
   - a fitted formula with residual < 30%,
   - a verdict (SAFE / UNSAFE / NEEDS-BATCHED / NEEDS-RESEARCH).
2. **`REPORT2.md`** includes an "Executive summary" of findings,
   and a "Recommended next step" section for `DECISIONS.md`.
3. **All 5 Part 4 live violations** are confirmed or refuted
   (with measurement, not opinion).
4. **Every proposed rewrite in `analysis_external.md §3`** is
   reclassified based on new measurements: `accept` / `reject` /
   `needs research`.
5. **`DECISIONS.md §9.2.N` draft** is delivered — copy-pasteable
   entries for the formulas that the maintainers want to adopt.
6. **No `cadw/` files modified** — `git diff cadw/` is empty.

If a call cannot be benchmarked (e.g. requires very large memory
to trigger the pathological case) — say so, propose the limit, and
mark as `NEEDS-RESEARCH` with a plan.

---

## Part 9 — Re-run criteria

Re-run `TASK2` when:

- **Arrow or pandas version bumps** (`pyproject.toml`) — kernels may
  change. Run all `bench_arrow_*` and compare to prior `RESULTS2.md`.
- **A new Arrow call is added to `cadw/`** — extend Part 1a; new
  benchmark; new §9.2.N entry.
- **A new Arrow-native rewrite is proposed** (like this run's
  `analysis_external.md`) — Part 5 re-run for the affected P-IDs.
- **~2–3 months** since the last run, whichever comes first.

---

## Part 10 — Self-critique examples (do not repeat)

- **Bench without process restart** — `peak_abs_mb` polluted by prior
  case. **Always** restart process or `del + gc.collect()` between
  cases.
- **Fit formula with one point** — minimum 3 sizes for a 2-parameter
  fit. If formula is 3-parameter (`a×N + b×K + c`), need 3 points that
  vary N and K independently (e.g. N=1M K=100, N=1M K=10k, N=100k K=10k).
- **Trusting docs** — Arrow docs for `pc.hash_tdigest` say "approximate
  quantile with bounded state" but do not state the constant. **Measure
  it**, do not copy the claim.
- **Skipping `to_pandas` case** because it's "obviously bad" — no,
  measure it. Actual peak depends on dtype: int64 → 1×, string →
  3–5×. The 5× isn't a documented fact, it's a benchmark to run.
- **Concluding "safe" without checking worst-case schema** — always
  test with: (a) narrow int64, (b) wide float64, (c) string. Wide
  schema is where 256 B/row guards fail.

---

## Appendix A — Existing benchmarks to extend (do not rewrite)

These already exist in `_FOR_PINGIS/bench/`. Extend, don't duplicate:

- `bench_join.py` — B1a: hash_join_streaming peak vs output size.
  **Extend with**: `Table.join` alone (no shuffle) to isolate
  P-A3 from shuffle cost.
- `bench_suite.py` — B3a/B4/B5/B6/B7. **Extend with**: `to_pandas()`
  on the dedup partition (P-D1 live violation).

## Appendix B — CADW's own formulas to compare against

From `DECISIONS.md §9.2`:

| Model | Formula | Status |
|---|---|---|
| `ReduceMemoryModel` | `batch_size × row_size + 2 × state_bytes` | ✅ |
| `HashShuffleMemoryModel` | `batch × row_size + P × writer_buf + c_hash × batch` | ✅ |
| `SortMemoryModel` | `max(run_buffer, k × read_buffer + k × heap + read_batch × row_size)` | ✅ |
| `GroupByMemoryModel` | `max(f_shuffle, f_local_groupby)` | ✅ |
| `JoinMemoryModel` | `build_partition + probe_batch + output_batch × (1 + c_merge)` | ✅ |

Every new `§9.2.N` for a library call must be **the same class of
thing**: peak RAM in bytes, guarded by `MemoryGuard.check_state` before
the call, with a safety factor.

## Appendix C — Files to read first

- `analysis_ram.md` (prior run, full P-catalog)
- `analysis_external.md` (prior run, per-P feasibility)
- `REPORT.md` (prior run, executive)
- `cadw/ops/join.py`, `cadw/ops/groupby.py`, `cadw/ops/sort.py`,
  `cadw/ops/dedup.py`, `cadw/ops/rank.py`, `cadw/ops/quantile.py`,
  `cadw/ops/resample.py` — call sites
- `cadw/core/memory_guard.py` — what `check_state` actually checks

---

## Appendix D — Explicit non-goals

- **Not** rewriting CADW's operations in terms of Arrow calls. That is
  v0.8.9+ work, based on this audit's verdicts.
- **Not** benchmarking ML paths (`cadw/ml/`) — separate audit if needed.
- **Not** proposing new Arrow functions not already in
  `analysis_external.md §3`.
- **Not** modifying tests to skip or relax. All new findings go to
  `_FOR_PINGIS/`, and adoption follows normal CADW patch process.

---

**Branch**: `pingis/library-memory-audit-v0.8.8`
**Parent task**: `TASK.md` (broader periodic audit)
**Prior runs**: `REPORT.md`, `analysis_ram.md`, `analysis_external.md`
**Sibling tasks**: `TASK2` (this) — library memory; `TASK3` (future) —
ML pipeline memory; `TASK4` (future) — Arrow-native rewrite verification.


