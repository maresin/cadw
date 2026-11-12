# cadw/ops/sort.py
"""
Sort: внешняя сортировка. Два этапа (DECISIONS.md §7):
  1. Генерация run'ов: читаем батчи, сортируем в памяти (pc.sort_indices),
     пишем run в Arrow IPC (все нужные колонки + row_id).
  2. k-way merge: сливаем k run'ов через heap (tie-break — row_id для
     стабильности); при k < числа run'ов — многоуровневый merge.

Инварианты:
  - стабильность: при равных ключах порядок по row_id;
  - пик RAM ≤ k * buffer_size + heap + output_buffer;
  - nulls placement задаётся явно (не зависит от ascending);
  - ключи, которые есть и в value_columns, и в key_columns, не дублируются.
"""
from __future__ import annotations

import gc
import heapq
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter

logger = get_logger("ops.sort")

# On-disk run batch size (rows per IPC batch written by
# _write_run). compute_fan_in MUST use this for the reader side
# of the merge, not read_batch_size (output flush frequency).
_WRITE_RUN_CHUNK = 50_000


def _unique_columns(value_columns: Sequence[str], key_columns: Sequence[str]) -> list[str]:
    """value + key без повторов, порядок: сначала value, затем недостающие key."""
    seen: set[str] = set()
    out: list[str] = []
    for c in list(value_columns) + list(key_columns):
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


class _Key:
    """Ключ сортировки с явной семантикой nulls и per-column asc/desc.

    nulls placement не зависит от ascending (как в Arrow/pandas):
      nulls_first=True  -> null в начале, независимо от asc/desc;
      nulls_first=False -> null в конце.
    """
    __slots__ = ("null_rank", "has_val", "val", "desc")

    def __init__(self, val, descending: bool, nulls_first: bool) -> None:
        self.desc = descending
        if val is None:
            self.has_val = False
            self.null_rank = 0 if nulls_first else 2
            self.val = None
        else:
            self.has_val = True
            self.null_rank = 1
            self.val = val

    def __lt__(self, other: "_Key") -> bool:
        if self.null_rank != other.null_rank:
            return self.null_rank < other.null_rank
        if not self.has_val:          # оба null
            return False
        if self.desc:
            return self.val > other.val
        return self.val < other.val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Key):
            return NotImplemented
        return (
            self.null_rank == other.null_rank
            and (not self.has_val or self.val == other.val)
        )


def _normalize_per_column(x, n: int, name: str):
    if isinstance(x, bool):
        return [x] * n
    xs = list(x)
    if len(xs) != n:
        raise ValueError(f"{name}: ожидалось {n} значений, получено {len(xs)}")
    return xs


def _write_run(
    batches: list[pa.RecordBatch],
    all_columns: Sequence[str],
    key_columns: Sequence[str],
    ascending: list[bool],
    nulls_first: bool,
    spill_dir: Path,
    run_idx: int,
) -> Path:
    """v0.5.2a-fix6 (P5b): batch-wise take/write to avoid materializing
    whole sorted_table. Peak in _write_run = table (via from_batches) +
    order + chunk (small), not table + order + sorted_table (×3).
    """
    table = pa.Table.from_batches(batches)
    null_placement = "at_start" if nulls_first else "at_end"
    sort_keys = [
        (col, "ascending" if asc else "descending", null_placement)
        for col, asc in zip(key_columns, ascending)
    ]
    order = pc.sort_indices(table, sort_keys=sort_keys)

    path = spill_dir / f"run_{run_idx:05d}.arrow"
    writer = ArrowIPCBatchWriter(str(path), table.schema)
    try:
        # Batch-wise: take small chunk, write it, free. No full materialize.
        chunk = _WRITE_RUN_CHUNK
        for i in range(0, len(order), chunk):
            sub_order = order.slice(i, chunk)
            sub_table = table.take(sub_order)
            for b in sub_table.to_batches():
                writer.write_batch(b)
    finally:
        writer.close()
    return path




class _NumpyBackedBuilder:
    """v0.5.2a-fix5: minimal fixed-width builder backed by np.ndarray.

    PyArrow 26 не экспортирует `Float64Builder`/`Int64Builder`/etc.
    (проверено: hasattr(pa, 'Float64Builder') == False). Только
    `StringBuilder` есть в `pa.lib`.

    Peak = capacity × arrow_size (без Python-object overhead: int/float
    в PyArrow = 4-8 B, не 28). Amortized O(n) через geometric grow.

    Null bitmap: bit-packed (np.packbits), LSB-first — совместимо с
    pa.Array.from_buffers.
    """

    __slots__ = ("_dtype", "_arrow_type", "_buf", "_nulls", "_n", "_capacity")

    def __init__(self, arrow_type: pa.DataType, np_dtype):
        import numpy as np
        self._dtype = np_dtype
        self._arrow_type = arrow_type
        self._capacity = 64
        self._buf = np.empty(self._capacity, dtype=np_dtype)
        self._nulls = np.ones(self._capacity, dtype=bool)  # True = null
        self._n = 0

    def _grow(self):
        import numpy as np
        newcap = self._capacity * 2
        new_buf = np.empty(newcap, dtype=self._dtype)
        new_buf[: self._n] = self._buf[: self._n]
        new_nulls = np.ones(newcap, dtype=bool)
        new_nulls[: self._n] = self._nulls[: self._n]
        self._buf = new_buf
        self._nulls = new_nulls
        self._capacity = newcap

    def append(self, value):
        if self._n == self._capacity:
            self._grow()
        self._buf[self._n] = value
        self._nulls[self._n] = False
        self._n += 1

    def append_null(self):
        if self._n == self._capacity:
            self._grow()
        self._buf[self._n] = 0
        self._nulls[self._n] = True
        self._n += 1

    def finish(self) -> pa.Array:
        import numpy as np
        values = self._buf[: self._n].copy()
        # Arrow validity bitmap: bit=1 means VALID, bit=0 means NULL.
        # _nulls[i] = True means NULL (inverted vs Arrow).
        # So packed = packbits(~_nulls).
        # Regression: v0.7.6 diagnostic showed packbits(mask) produced
        # wrong null positions (null↔valid inversion). Fix: packbits(~mask).
        mask = self._nulls[: self._n]
        if mask.any():
            packed = np.packbits(~mask, bitorder="little")
            arr = pa.Array.from_buffers(
                self._arrow_type, self._n,
                [pa.py_buffer(packed), pa.py_buffer(values)],
            )
        else:
            arr = pa.Array.from_buffers(
                self._arrow_type, self._n,
                [None, pa.py_buffer(values)],
            )
        # Reset
        self._capacity = 64
        self._buf = np.empty(self._capacity, dtype=self._dtype)
        self._nulls = np.ones(self._capacity, dtype=bool)
        self._n = 0
        return arr


def _make_builder(dtype: pa.DataType):
    """Create pa.Array.Builder or numpy fallback (v0.5.2a-fix5).

    PyArrow 26 не экспортирует numeric `*Builder` (hasattr(pa,
    'Float64Builder') == False). Fallback chain:
      1. Native pyarrow *Builder (pa / pa.lib / pa.compute) — для
         string/binary/timestamp/date (если есть).
      2. `_NumpyBackedBuilder` — для fixed-width numeric
         (int/uint/float).
      3. BinaryBuilder для string (cast at flush).

    Returns builder (native or numpy-backed) or None.
    """
    import numpy as np
    try:
        import pyarrow.lib as _plib
    except ImportError:
        _plib = None
    try:
        import pyarrow.compute as _pcomp
    except ImportError:
        _pcomp = None

    _mods = [pa, _plib, _pcomp]

    def _try(*names):
        for name in names:
            for mod in _mods:
                if mod is None:
                    continue
                cls = getattr(mod, name, None)
                if cls is not None:
                    try:
                        return cls()
                    except Exception:
                        continue
        return None

    # Numeric — native builder if available; else numpy-backed.
    if pa.types.is_int64(dtype):
        return _try("Int64Builder") or _NumpyBackedBuilder(dtype, np.int64)
    if pa.types.is_int32(dtype):
        return _try("Int32Builder") or _NumpyBackedBuilder(dtype, np.int32)
    if pa.types.is_uint64(dtype):
        return _try("UInt64Builder") or _NumpyBackedBuilder(dtype, np.uint64)
    if pa.types.is_uint32(dtype):
        return _try("UInt32Builder") or _NumpyBackedBuilder(dtype, np.uint32)
    if pa.types.is_float64(dtype):
        return _try("Float64Builder") or _NumpyBackedBuilder(dtype, np.float64)
    if pa.types.is_float32(dtype):
        return _try("Float32Builder") or _NumpyBackedBuilder(dtype, np.float32)
    if pa.types.is_boolean(dtype):
        return _try("BooleanBuilder")

    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        b = _try("StringBuilder", "Utf8Builder",
                 "LargeStringBuilder", "LargeUtf8Builder")
        if b is not None:
            return b
        return _try("BinaryBuilder", "LargeBinaryBuilder")

    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return _try("BinaryBuilder", "LargeBinaryBuilder")

    if pa.types.is_timestamp(dtype):
        for mod in _mods:
            if mod is None:
                continue
            cls = getattr(mod, "TimestampBuilder", None)
            if cls is not None:
                return cls(dtype.unit)
        return _NumpyBackedBuilder(dtype, np.int64)
    if pa.types.is_date32(dtype):
        return _try("Date32Builder") or _NumpyBackedBuilder(dtype, np.int32)
    if pa.types.is_date64(dtype):
        return _try("Date64Builder") or _NumpyBackedBuilder(dtype, np.int64)
    return None


def _flush_builders(builders: dict, schema: pa.Schema) -> pa.RecordBatch:
    """Finish builders → RecordBatch (v0.5.2a-fix2).

    Auto-casts if builder output type differs from schema target
    (e.g. BinaryBuilder used for string → cast to string).
    """
    arrays = []
    for f in schema:
        b = builders[f.name]
        if b is None:
            raise RuntimeError(
                f"_flush_builders: no builder for {f.name} ({f.type}); "
                f"bug in _make_builder (v0.5.2a-fix2)"
            )
        arr = b.finish()
        if arr.type != f.type:
            # e.g. binary → string (utf8 validation); cheap relative to
            # Python-object overhead that P5 was fixing.
            arr = arr.cast(f.type)
        arrays.append(arr)
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


class _RunStream:
    """Per-run buffered reader for Arrow-native k-way merge (v0.9.23c).

    Holds up to `rbs` rows in `buffer` (heap-allocated pa.Table).
    Parent batch from the IPC reader is held as a mmap view — it does
    NOT count toward RSS (pyarrow ipc.open_file is mmap-backed), so
    peak RSS = K * rbs * row_bytes (buffers) + allocations during
    sort. Rows are consumed from the front (sorted runs -> consumed
    rows form a prefix). Refilled when buffer drops below rbs.

    Invariant: for any non-exhausted stream, buffer is sorted by the
    external sort keys (guaranteed by _write_run phase-1 sort).
    """

    __slots__ = (
        "reader", "cur_batch", "cur_pos", "buffer",
        "exhausted", "rbs", "name",
    )

    def __init__(self, path, rbs: int, name: str) -> None:
        self.reader = ArrowIPCBatchReader(str(path), None)
        self.cur_batch: "pa.RecordBatch | None" = None
        self.cur_pos: int = 0
        self.buffer = pa.Table.from_batches([], schema=self.reader.schema)
        self.exhausted: bool = False
        self.rbs: int = rbs
        self.name: str = name

    def _next_batch(self, memory_guard):
        """Return (batch, pos) or None if reader exhausted."""
        if (self.cur_batch is not None
                and self.cur_pos < self.cur_batch.num_rows):
            return self.cur_batch, self.cur_pos
        try:
            b = next(self.reader)
        except StopIteration:
            self.cur_batch = None
            self.cur_pos = 0
            # v0.9.31 (PINGIS AUDIT A1): release mmap'd pages eagerly.
            # Previously the IPC reader (and its mmap) lived until the
            # _RunStream object was garbage-collected — at N=1M with 30+
            # runs, this kept ~30*file_size resident across the whole
            # k-way merge (RSS drift 5-13x over pred_total).
            close = getattr(self.reader, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
            return None
        if memory_guard is not None and b.num_rows > 0:
            memory_guard.check_batch(
                b, name="sort.merge_read[%s]" % self.name,
            )
        self.cur_batch = b
        self.cur_pos = 0
        return b, 0

    def refill(self, memory_guard) -> None:
        """Read up to rbs rows into buffer. Marks exhausted on EOF.

        v0.9.25c (part1_report B1): linear concat. Prior code did
        `self.buffer = pa.concat_tables([self.buffer, chunk])` inside
        the while loop — quadratic, each iteration copies the entire
        accumulated buffer. Measured: 12.9 GB copied @ rbs=200k,
        0.078s vs 0.0003s for list-collect + single concat. Now:
        accumulate parts, one concat_tables(parts) at end.
        """
        parts: list = [self.buffer] if len(self.buffer) else []
        have = len(self.buffer)
        while have < self.rbs:
            r = self._next_batch(memory_guard)
            if r is None:
                self.exhausted = True
                break
            b, pos = r
            need = self.rbs - have
            avail = b.num_rows - pos
            take = min(need, avail)
            if take <= 0:
                continue
            parts.append(pa.Table.from_batches([b.slice(pos, take)]))
            have += take
            self.cur_pos += take
            if self.cur_pos >= b.num_rows:
                self.cur_batch = None
                self.cur_pos = 0
        if len(parts) == 1:
            self.buffer = parts[0]
        elif len(parts) > 1:
            self.buffer = pa.concat_tables(parts)
        # (parts empty + buffer empty = no-op; unchanged)

    def take(self, n: int) -> "pa.Table":
        """Remove and return first n rows from buffer."""
        n = min(n, len(self.buffer))
        if n == 0:
            return self.buffer.slice(0, 0)
        head = self.buffer.slice(0, n)
        self.buffer = self.buffer.slice(n)
        return head

def _merge_keys(
    run_paths: list[Path],
    all_columns: Sequence[str],
    key_columns: Sequence[str],
    value_columns: Sequence[str],
    ascending: list[bool],
    nulls_first: bool,
    out_path: Path,
    read_batch_size: int,
    memory_guard: "MemoryGuard | None" = None,
) -> Path:
    """k-way Arrow-native batched merge (v0.9.23c, AUDIT_FULL_v0922 7.A.1).

    Replaces v0.5.2a per-row Python heapq with batched Arrow sort. Each
    round:
      1. Buffer up to `read_batch_size` rows from each non-exhausted run.
      2. Concat buffers + `__run_idx` tag -> merged Arrow Table.
      3. `pc.sort_indices` (C++) with per-column asc/desc + nulls_first,
         plus `row_id` tie-break (matches heapq's (key, row_id) compare).
      4. Take first `min(rbs, merged.num_rows)` rows -> write output.
      5. Advance each run's buffer cursor by how many of its rows were
         taken. Consumed rows always form a prefix (runs are sorted).

    Semantics shift (v0.9.23c): `read_batch_size` now means "buffer rows
    per run" (was: output flush size). Both give output batches of
    <= read_batch_size rows, so downstream consumers see the same batch
    shape. Callers that set read_batch_size large (e.g. 50000 to match
    _WRITE_RUN_CHUNK) will see proportionally larger merge buffers; the
    check_state at the top bounds peak = K * rbs * row_bytes * (buffers
    + merged + sorted) <= available * state_safety.

    Output identity (byte-for-byte same row order as v0.5.2a heapq):
      - Same per-column asc/desc.
      - Same global nulls_first (applies to all keys).
      - Same row_id tie-break (row_id is monotonic with input order, so
        stable within-run + row_id cross-run == Python stable sort by
        (key..., row_id)).

    Peak: 3 * K * rbs * row_bytes (buffers + merged + sorted). At K=8,
    rbs=1000, row_bytes=640: ~15 MB. Parent IPC batches are mmap-backed
    and don't count toward RSS (only touched pages, bounded by OS).

    Raises ValueError on empty run_paths.
    """
    if not run_paths:
        raise ValueError("_merge_keys: пустой список runs")

    rbs = max(1, read_batch_size)
    K = len(run_paths)
    null_placement = "at_start" if nulls_first else "at_end"

    # Sort spec for pc.sort_indices: per-column order + global null_placement
    # (matches _Key semantics in v0.5.2a), plus row_id tie-break.
    sort_keys = [
        (col, "ascending" if asc else "descending", null_placement)
        for col, asc in zip(key_columns, ascending)
    ] + [("row_id", "ascending", "at_end")]

    streams = [
        _RunStream(p, rbs, name=str(i))
        for i, p in enumerate(run_paths)
    ]

    # Pre-check: conservative upper bound for merge buffer
    if memory_guard is not None:
        row_bytes = max(1, memory_guard.row_size_bytes) + 16
        memory_guard.check_state(
            3 * K * rbs * row_bytes,
            name="sort.merge_buffer(K=%d,rbs=%d)" % (K, rbs),
        )

    out_columns = list(value_columns) + ["row_id"]
    out_schema = pa.schema([
        pa.field(c, streams[0].reader.schema.field(c).type)
        for c in out_columns
    ])
    writer = ArrowIPCBatchWriter(str(out_path), out_schema)

    try:
        while True:
            for s in streams:
                if not s.exhausted:
                    s.refill(memory_guard)

            active = [
                (i, s) for i, s in enumerate(streams) if len(s.buffer) > 0
            ]
            if not active:
                break

            # Tag each row with its source run
            tagged: list = []
            for i, s in active:
                run_idx_arr = pa.array(
                    [i] * len(s.buffer), type=pa.int32(),
                )
                tagged.append(s.buffer.append_column("__run_idx", run_idx_arr))

            merged = tagged[0] if len(tagged) == 1 else pa.concat_tables(tagged)

            # Sort all buffered rows, take top rbs
            order = pc.sort_indices(merged, sort_keys=sort_keys)
            sorted_merged = merged.take(order)
            take_n = min(rbs, sorted_merged.num_rows)
            top = sorted_merged.slice(0, take_n)

            # Write top (drop __run_idx)
            if take_n > 0:
                top_clean = top.select(out_columns)
                for b in top_clean.to_batches():
                    if b.num_rows > 0:
                        writer.write_batch(b)

            # Advance each run by how many of its rows were taken.
            # Arrow-native via pc.value_counts (K = active runs,
            # bounded by fan_in <= 8; safe per §9.2.13).
            # v0.9.23c-fix1: prior np.unique failed because numpy
            # is not imported at module level in sort.py (numpy
            # imported lazily inside _NumpyBackedBuilder only).
            vc = pc.value_counts(top.column("__run_idx"))
            consumed = {
                int(v): int(c)
                for v, c in zip(
                    vc.field("values").to_pylist(),
                    vc.field("counts").to_pylist(),
                )
            }
            for i, s in active:
                s.take(consumed.get(i, 0))
    finally:
        writer.close()
    return out_path



def compute_fan_in(
    available_bytes: int,
    read_batch_size: int,
    row_bytes: int,
    *,
    max_k: int = 256,
) -> int:
    """k (fan-in) for Arrow-native k-way merge.

    v0.9.25c (audit §6.6): must match _merge_keys.check_state(3 * K * rbs
    * row_bytes) <= available * state_safety (=0.5). The factor 3 =
    buffers + merged + sorted (see _merge_keys docstring). Prior code
    used `available // (rbs * row_bytes)` — 3x under-counted the actual
    peak; and caller passed _WRITE_RUN_CHUNK (50000) while _merge_keys
    uses rbs=1000, giving K=2 at 32 MB / 8 cols / 640 B/row — root
    cause of 24.4s @ N=500k/8 cols / 32 MB (audit §6.6).

    Now: state_limit // (3 * rbs * row_bytes), capped at max_k (file
    handle / FD limit — 256 is conservative). Min 2 (merge impossible
    with k < 2; caller (_merge_keys) will fire check_state if 2 runs
    exceed state_limit — documented for wide schemas at tight budget).
    """
    # v0.9.31 (PINGIS AUDIT A1): align with _merge_keys.check_state,
    # which uses (row_bytes + 16) — the 16 B is the __run_idx tag
    # (int32 padded) per row added during merge. Prior mismatch gave
    # K that passed compute_fan_in but failed check_state inside
    # _merge_keys (observed at N=1M: K=21 computed, 6.05 MB needed
    # vs 5.24 MB state_limit → MemoryLimitExceeded).
    state_limit = int(available_bytes * 0.5)
    per_run = 3 * max(1, read_batch_size) * (max(1, row_bytes) + 16)
    k = state_limit // per_run
    return max(2, min(max_k, k))


def external_sort(
    batch_iter: Iterable[pa.RecordBatch],
    key_columns: Sequence[str],
    value_columns: Sequence[str],
    memory_guard: MemoryGuard,
    spill_dir: str | Path,
    *,
    ascending: bool | Sequence[bool] = True,
    nulls_first: bool = False,
    merge_fan_in: int | None = None,
    read_batch_size: int = 1000,
    max_buffer_bytes: int | None = None,
    log: bool = True,
) -> Path:
    """
    Возвращает путь к финальному отсортированному run'у (Arrow IPC).
    Колонки результата: value_columns + row_id.

    Empty-input contract (v0.9.34, A10): if `batch_iter` yields no
    batches, raises ValueError("external_sort: пустой вход"). There
    is no default empty output — pass explicit empty schema if needed
    (see DECISIONS §6.7).
    """
    if not key_columns:
        raise ValueError("key_columns must be non-empty")
    # v0.8.5 (§13.4/126): `row_id` reserved (internal identity column,
    # appended by external_sort / hash_shuffle). User column named
    # `row_id` → duplicate → `KeyError: Field "row_id" exists 2 times in
    # schema` (ArrowIPCBatchReader) or `Multiple matches for
    # FieldRef.Name(row_id)` (pc.sort_indices). Fail-fast here, BEFORE
    # any batch materialization.
    if "row_id" in key_columns or "row_id" in value_columns:
        raise ValueError(
            "external_sort: column name 'row_id' is reserved for CADW "
            "internal identity column. Rename your column before calling. "
            "See DECISIONS §11.5, §13.4/126."
        )
    nk = len(key_columns)
    asc = _normalize_per_column(ascending, nk, "ascending")
    all_columns = _unique_columns(value_columns, key_columns)

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    # --- Фаза 1: генерация run'ов ---
    runs: list[Path] = []
    buf: list[pa.RecordBatch] = []
    buf_bytes = 0
    row_id_offset = 0
    # v0.5.2a-fix6 (P5b): phase-1 peak is 3×buf (buf + from_batches-copy
    # + take-copy). Reducing limit from 0.5× to 0.2× cuts peak from
    # ~3×15MB → ~3×6MB (for 30 MB budget). More runs, but bounded.
    limit = max(1, int(memory_guard.available_bytes() * 0.2))
    # v0.9.225-c-fix3 (P25c): explicit upper bound on Phase 1 buffer.
    # Default None = existing behavior. Consumers that need O(1) peak
    # independent of n pass max_buffer_bytes (e.g. PairwiseLogistic
    # uses 2*max_query_size*(d+2)*8). Write_run peak ~4x limit.
    if max_buffer_bytes is not None:
        mb = int(max_buffer_bytes)
        if mb < 1:
            raise ValueError(
                f"external_sort: max_buffer_bytes must be >= 1; "
                f"got {max_buffer_bytes}")
        limit = min(limit, mb)

    # v0.9.31 (PINGIS AUDIT A1): phase-1 peak = 4x buf (buf + from_batches
    # + sort_indices-int64 + take). The old code accepted any user batch
    # that fits in the raw budget, then immediately built select+append
    # (2x) + _write_run peak (4x), causing 5-13x RSS drift over pred_total.
    # Fix: (a) check_batch extra_factor=2.0 catches real OOM in the
    # select+append step; (b) oversized input batches are split into
    # sub-slices small enough to fit under limit//4, so _write_run never
    # sees a batch bigger than its accounted-for buffer.
    # v0.9.31-fix2 (A1): split threshold = avail//2 (matches the
    # check_batch extra_factor=2.0 threshold); using limit//4 was too
    # aggressive — a 6.29 MB batch split into 12 sub-chunks, creating
    # 12x runs and blowing up the merge-phase mmap working set.
    max_chunk_bytes = max(1, int(memory_guard.available_bytes() * 0.5))

    def _iter_input():
        for b in batch_iter:
            if b.num_rows == 0:
                continue
            # Split oversized batches BEFORE check_batch so that the
            # split sub-batch is what gets checked (v0.9.31-fix1).
            if b.nbytes <= max_chunk_bytes:
                memory_guard.check_batch(
                    b, name="external_sort.batch", extra_factor=2.0,
                )
                yield b
                continue
            per = max(1, int(b.num_rows * max_chunk_bytes / b.nbytes))
            for i in range(0, b.num_rows, per):
                sub = b.slice(i, min(per, b.num_rows - i))
                memory_guard.check_batch(
                    sub, name="external_sort.batch", extra_factor=2.0,
                )
                yield sub

    for batch in _iter_input():
        row_ids = pa.array(
            range(row_id_offset, row_id_offset + batch.num_rows), type=pa.uint64()
        )
        full = batch.select(all_columns).append_column("row_id", row_ids)
        buf.append(full)
        buf_bytes += full.nbytes
        row_id_offset += batch.num_rows
        if buf_bytes >= limit:
            runs.append(
                _write_run(buf, all_columns, key_columns, asc, nulls_first, spill_dir, len(runs))
            )
            buf = []
            buf_bytes = 0

    if buf:
        runs.append(
            _write_run(buf, all_columns, key_columns, asc, nulls_first, spill_dir, len(runs))
        )
    if not runs:
        raise ValueError("external_sort: пустой вход")

    # v0.5.2a-fix6: release phase-1 buffers before merge (frees numpy/Arrow
    # allocations held by builder state / table refs).
    gc.collect()

    # --- Фаза 2: k-way merge (при необходимости — многоуровневый) ---
    row_bytes = max(1, memory_guard.row_size_bytes)
    # v0.9.25c: pass read_batch_size (the rbs that _merge_keys will
    # use), NOT _WRITE_RUN_CHUNK (on-disk chunk, unrelated to merge
    # memory). Prior mis-pass gave K=2 @ 32MB/8cols → 3-4x more merge
    # passes → 60x wall (audit §6.6). Now: K=8 @ 32MB/8cols / 640 B/row.
    k = merge_fan_in or compute_fan_in(
        memory_guard.available_bytes(), read_batch_size, row_bytes
    )

    merge_idx = 0
    while len(runs) > 1:
        next_runs: list[Path] = []
        for i in range(0, len(runs), k):
            group = runs[i:i + k]
            if len(group) == 1:
                next_runs.append(group[0])
                continue
            out = spill_dir / f"merged_{merge_idx:05d}.arrow"
            merge_idx += 1
            _merge_keys(
                group, all_columns, key_columns, value_columns,
                asc, nulls_first, out, read_batch_size, memory_guard,
            )
            next_runs.append(out)
        runs = next_runs

    if log:
        logger.info("external_sort done: rows=%d result=%s", row_id_offset, runs[0])
    return runs[0]


def read_sorted(path: str | Path, columns: list[str] | None = None) -> Iterator[pa.RecordBatch]:
    """Удобный reader для тестов."""
    for b in ArrowIPCBatchReader(str(path), columns):
        yield b