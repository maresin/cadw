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


def _append_to_builders(builders: dict, batch: pa.RecordBatch, pos: int, schema: pa.Schema) -> None:
    """Append one row (via per-column scalar) to builders."""
    for f in schema:
        val = batch.column(f.name)[pos].as_py()
        b = builders[f.name]
        if val is None:
            b.append_null()
        else:
            b.append(val)


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
    """k-way merge. Читаем каждый run батчами, сливаем через heap."""
    if not run_paths:
        raise ValueError("_merge_keys: пустой список runs")

    readers = [ArrowIPCBatchReader(str(p), None) for p in run_paths]
    cur_batches: list[pa.RecordBatch | None] = [None] * len(run_paths)
    cur_pos: list[int] = [0] * len(run_paths)

    def fetch(idx: int) -> tuple[pa.RecordBatch, int] | None:
        while True:
            b = cur_batches[idx]
            if b is not None and cur_pos[idx] < b.num_rows:
                pos = cur_pos[idx]
                cur_pos[idx] += 1
                return b, pos
            try:
                nb = next(readers[idx])
                if memory_guard is not None:
                    memory_guard.check_batch(
                        nb, name="sort.merge_read[%d]" % idx
                    )
                cur_batches[idx] = nb
                cur_pos[idx] = 0
            except StopIteration:
                return None

    def make_key(batch: pa.RecordBatch, pos: int) -> tuple:
        return tuple(
            _Key(batch.column(col)[pos].as_py(), desc, nulls_first)
            for col, desc in zip(key_columns, [not a for a in ascending])
        )

    heap: list[tuple] = []
    for i in range(len(run_paths)):
        r = fetch(i)
        if r is not None:
            b, pos = r
            heapq.heappush(
                heap,
                (make_key(b, pos), int(b.column("row_id")[pos].as_py()), i, b, pos),
            )

    out_columns = list(value_columns) + ["row_id"]
    schema = pa.schema([pa.field(c, readers[0].schema.field(c).type) for c in out_columns])
    writer = ArrowIPCBatchWriter(str(out_path), schema)

    # v0.5.2a (P5): pa.Array.Builder вместо Python-list.
    # Peak = read_batch_size × arrow-size (не ×28-80 per Python object).
    builders: dict = {}
    for f in schema:
        b = _make_builder(f.type)
        if b is None:
            raise NotImplementedError(
                f"external_sort: no Arrow Builder for column {f.name!r} "
                f"type {f.type}; P5 fix covers int/uint/float/bool/string/"
                f"binary/timestamp/date. File an issue for other types."
            )
        builders[f.name] = b

    rows_in_buffer = [0]

    def flush() -> None:
        if rows_in_buffer[0] == 0:
            return
        batch = _flush_builders(builders, schema)
        writer.write_batch(batch)
        rows_in_buffer[0] = 0

    try:
        while heap:
            _, _, run_idx, b, pos = heapq.heappop(heap)
            _append_to_builders(builders, b, pos, schema)
            rows_in_buffer[0] += 1
            if rows_in_buffer[0] >= read_batch_size:
                flush()
            nxt = fetch(run_idx)
            if nxt is not None:
                nb, npos = nxt
                heapq.heappush(
                    heap,
                    (make_key(nb, npos), int(nb.column("row_id")[npos].as_py()), run_idx, nb, npos),
                )
        flush()
    finally:
        writer.close()
    return out_path



def compute_fan_in(available_bytes: int, read_batch_size: int, row_bytes: int) -> int:
    """k (fan-in) для k-way merge: сколько run'ов держать открытыми одновременно.

    Ограничение: k * read_batch_size * row_bytes <= available_bytes.
    Возвращает >= 2 (иначе merge невозможен).
    """
    denom = max(1, read_batch_size * max(1, row_bytes))
    return max(2, available_bytes // denom)


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
    log: bool = True,
) -> Path:
    """
    Возвращает путь к финальному отсортированному run'у (Arrow IPC).
    Колонки результата: value_columns + row_id.
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

    for batch in batch_iter:
        memory_guard.check_batch(batch, name="external_sort.batch")
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
    k = merge_fan_in or compute_fan_in(
        memory_guard.available_bytes(), _WRITE_RUN_CHUNK, row_bytes
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