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
    table = pa.Table.from_batches(batches)
    # null_placement передаётся per-key (per-key API с 25.0.0; глобальный
    # SortOptions.null_placement deprecated).
    null_placement = "at_start" if nulls_first else "at_end"
    sort_keys = [
        (col, "ascending" if asc else "descending", null_placement)
        for col, asc in zip(key_columns, ascending)
    ]
    order = pc.sort_indices(table, sort_keys=sort_keys)
    sorted_table = table.take(order)

    path = spill_dir / f"run_{run_idx:05d}.arrow"
    writer = ArrowIPCBatchWriter(str(path), sorted_table.schema)
    try:
        for b in sorted_table.to_batches():
            writer.write_batch(b)
    finally:
        writer.close()
    return path


def _merge_keys(
    run_paths: list[Path],
    all_columns: Sequence[str],
    key_columns: Sequence[str],
    value_columns: Sequence[str],
    ascending: list[bool],
    nulls_first: bool,
    out_path: Path,
    read_batch_size: int,
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
                cur_batches[idx] = next(readers[idx])
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

    buffer: dict[str, list] = {c: [] for c in out_columns}

    def flush() -> None:
        if not buffer[out_columns[0]]:
            return
        arrays = [pa.array(buffer[c], type=schema.field(c).type) for c in out_columns]
        batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
        writer.write_batch(batch)
        for c in out_columns:
            buffer[c].clear()

    try:
        while heap:
            _, _, run_idx, b, pos = heapq.heappop(heap)
            for c in out_columns:
                buffer[c].append(b.column(c)[pos].as_py())
            if len(buffer[out_columns[0]]) >= read_batch_size:
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
    limit = max(1, memory_guard.available_bytes() // 2)

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

    # --- Фаза 2: k-way merge (при необходимости — многоуровневый) ---
    row_bytes = max(1, memory_guard.row_size_bytes)
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
                asc, nulls_first, out, read_batch_size,
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