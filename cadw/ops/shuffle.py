# cadw/ops/shuffle.py
"""
HashShuffle: разложить поток строк по P партициям так, что все строки
с одинаковым ключом попадают в одну партицию. Партиции не пересекаются
по ключам → результат агрегации = конкатенация (без merge).

Контракт (DECISIONS.md §6):
  - детерминированный хэш (xxhash64, seed) — стабилен между прогонами;
  - каждая строка несёт row_id (uint64) — для восстановления идентичности;
  - одна перестановка (sort_indices) вместо P фильтров;
  - пустой вход → ValueError (без schema нельзя создать writers).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import xxhash

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter

logger = get_logger("ops.shuffle")


def _dedup_columns(cols: Sequence[str]) -> list[str]:
    """Dedupe columns preserving order. FIX §13.4/54: hash_shuffle used to
    write duplicate columns when key ∩ value ≠ ∅ (e.g. groupby(same_col)).
    Caller contract unchanged: we still write key+value semantics; duplicates
    are just collapsed to one Arrow field."""
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _stable_bytes(v) -> bytes:
    """Стабильная сериализация значения в bytes (без зависимости от PYTHONHASHSEED).

    repr() различает типы: repr('1') = "'1'", repr(1) = '1', repr(None) = 'None'.
    Это важно: строка '1' и число 1 не должны попадать в одну партицию.
    """
    return repr(v).encode("utf-8")


def _python_hash_column(arr: pa.Array) -> np.ndarray:
    """Original per-value Python loop. Covers all dtypes. Slow (~32× pandas)."""
    values = arr.to_pylist()
    hashes = [xxhash.xxh64(_stable_bytes(v), seed=0).intdigest() for v in values]
    return np.asarray(hashes, dtype=np.uint64)


def _pandas_hash_column(arr: pa.Array) -> "np.ndarray | None":
    """C-level pandas hashing (v0.8.1). Returns None if unsupported/failed.

    **Types_mapper is REQUIRED for correctness** (not optimization):
    `arr.to_pandas()` without mapper gives `int64` for a batch with no
    nulls and `float64` (NaN) for a batch with nulls. The same value
    (e.g., `1`) would then hash differently across batches → same key
    goes to different partitions → partition-disjoint invariant broken
    → silently wrong results. `types_mapper=Int64Dtype` forces a
    nullable Int64 representation consistently.

    uint64 is excluded (doesn't fit in Int64Dtype, >2^63 overflows).
    """
    import pandas as pd
    t = arr.type
    if pa.types.is_uint64(t):
        return None
    try:
        def _mapper(pt):
            if pa.types.is_integer(pt) and pt.bit_width <= 64:
                # All int8/16/32/64 + uint8/16/32 fit in Int64Dtype.
                return pd.Int64Dtype()
            if pa.types.is_boolean(pt):
                return pd.BooleanDtype()
            return None
        pd_series = arr.to_pandas(types_mapper=_mapper)
        h = pd.util.hash_pandas_object(pd_series, index=False)
        return h.to_numpy(dtype=np.uint64)
    except Exception:
        return None


def _hash_column(arr: pa.Array) -> np.ndarray:
    """Best-effort: pandas C-hash for numeric/timestamp; Python for strings.

    v0.8.1-fix3: strings → Python. Bench v0.8.1:
        int64    39.2 → 2.02 ms  (19.4×)  (pandas)
        float64  39.2 → 0.99 ms  (39.6×)  (pandas)
        string   39.2 → 50.44 ms (0.78×)  (pandas, REGRESSION)
    pd.util.hash_pandas_object for object dtype uses per-element Cython
    (hash_object_array) + categorical detection — ~500 ns/string vs Python
    repr().encode() + xxhash ~390 ns/string. No faster path in pandas 3.x.
    → route strings to Python, keep pandas for numeric/timestamp.
    """
    t = arr.type
    # String → Python path (no benefit from pandas; slight regression)
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return _python_hash_column(arr)
    # Numeric / bool / timestamp / date / time → pandas C-hash (19-39×)
    if (pa.types.is_integer(t) or pa.types.is_floating(t)
            or pa.types.is_boolean(t)
            or pa.types.is_timestamp(t) or pa.types.is_date(t)
            or pa.types.is_time(t)):
        h = _pandas_hash_column(arr)
        if h is not None:
            return h
    # Fallback: struct/list/dict/uint64, or pandas API mismatch.
    return _python_hash_column(arr)


def compute_hashes(
    batch: pa.RecordBatch,
    key_columns: Sequence[str],
    seed: int = 0,
) -> pa.Array:
    """Детерминированный uint64-хэш ключа (boost::hash_combine через NumPy).

    v0.8.1: for numeric/bool/string/timestamp — pandas C-hash (32× faster).
    Fallback: Python loop + xxhash. Seed applied via XOR at the end
    (uniform across dtypes; deterministic; no collision added — XOR is
    a bijection).

    Hash values differ from pre-v0.8.1 (xxhash(repr) vs pandas SipHash).
    Partitions are ephemeral per-call; no persisted spill compatibility
    needed. Partition-disjoint invariant preserved (same value → same hash
    within a call, nulls → same fixed hash).
    """
    if not key_columns:
        raise ValueError("key_columns must be non-empty")

    h_np: np.ndarray | None = None
    for col in key_columns:
        arr = batch.column(col)
        ch = _hash_column(arr)
        if h_np is None:
            h_np = ch
        else:
            # boost::hash_combine (64-битная версия) с wrap-around
            mixed = (
                ch
                + np.uint64(0x9E3779B97F4A7C15)
                + (h_np << np.uint64(6))
                + (h_np >> np.uint64(2))
            )
            h_np = h_np ^ mixed

    assert h_np is not None
    # v0.8.1-fix2: splitmix64 avalanche (ALWAYS, not only when seed).
    # pd.util.hash_pandas_object for object/string dtype shifts << 1
    # (reserves 0 for NA), so all string hashes are EVEN: h & 1 = 0.
    # For P=4 (h & 3) only partitions {0, 2} would be used → 2-way split
    # (not 4-way) → recursion never converges. Avalanche spreads low bits,
    # removing structure. Seed applied via XOR BEFORE avalanche — diffuses
    # cleanly through the finalizer.
    if seed:
        h_np = h_np ^ np.uint64(seed)
    h_np = (h_np ^ (h_np >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    h_np = (h_np ^ (h_np >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    h_np = h_np ^ (h_np >> np.uint64(31))
    return pa.array(h_np, type=pa.uint64())


def _mod_p(hashes: pa.Array, p: int) -> pa.Array:
    """hashes % p без pc.mod (его нет в PyArrow).

    Для степени двойки — bit_wise_and (h & (p-1)); иначе — NumPy uint64
    (float64 теряет точность для 64-битных хэшей).
    """
    if p < 1:
        raise ValueError("p must be >= 1")
    if p == 1:
        return pc.multiply(pc.cast(hashes, pa.uint64()), pa.scalar(0, type=pa.uint64()))
    if p & (p - 1) == 0:  # p — степень двойки
        return pc.bit_wise_and(hashes, pa.scalar(p - 1, type=pa.uint64()))
    h_np = hashes.to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
    return pa.array(h_np % np.uint64(p), type=pa.uint64())


def estimate_num_partitions(
    k_est: int,
    available_bytes: int,
    *,
    bytes_per_key: int = 80,
    safety: float = 1.5,
    groups_per_partition: int | None = None,
) -> int:
    """Оценка P по числу групп (DECISIONS.md §6.4, шаг 1)."""
    if k_est <= 0:
        return 1
    if groups_per_partition is None:
        groups_per_partition = max(1, int(available_bytes * 0.6 / bytes_per_key))
    raw = int(k_est * safety / groups_per_partition)
    return max(1, raw)


def _boundaries(parts_np: np.ndarray) -> list[int]:
    """Индексы, где начинается новая партиция (parts_np отсортирован)."""
    if parts_np.size == 0:
        return [0]
    boundaries = [0]
    prev = parts_np[0]
    for i in range(1, parts_np.size):
        if parts_np[i] != prev:
            boundaries.append(i)
            prev = parts_np[i]
    boundaries.append(parts_np.size)
    return boundaries


def hash_shuffle(
    batch_iter: Iterable[pa.RecordBatch],
    key_columns: Sequence[str],
    value_columns: Sequence[str],
    num_partitions: int,
    memory_guard: MemoryGuard,
    spill_dir: str | Path,
    *,
    hash_seed: int = 0,
    log: bool = True,
) -> list[Path]:
    """
    Возвращает список путей к P файлам партиций (Arrow IPC).
    Каждый файл содержит колонки key + value + row_id (uint64).
    """
    if num_partitions < 1:
        raise ValueError("num_partitions must be >= 1")
    if not key_columns:
        raise ValueError("key_columns must be non-empty")
    # v0.8.5 (§13.4/126): `row_id` reserved — same rationale as in
    # external_sort. Check arguments (not materialized batch) for
    # cheapest fail-fast.
    if "row_id" in key_columns or "row_id" in value_columns:
        raise ValueError(
            "hash_shuffle: column name 'row_id' is reserved for CADW "
            "internal identity column. Rename your column before calling. "
            "See DECISIONS §11.5, §13.4/126."
        )

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    it = iter(batch_iter)
    try:
        first_batch = next(it)
    except StopIteration as exc:
        raise ValueError(
            "hash_shuffle: пустой вход. Для пустого случая передайте schema "
            "явно (см. DECISIONS.md §6.7)."
        ) from exc

    memory_guard.check_batch(first_batch, name="hash_shuffle.first_batch")

    # Схема: key + value + row_id (dedup key ∩ value — §13.4/54)
    all_columns = _dedup_columns(list(key_columns) + list(value_columns))
    fields: list[pa.Field] = []
    for col in all_columns:
        fields.append(first_batch.schema.field(col))
    fields.append(pa.field("row_id", pa.uint64()))
    schema = pa.schema(fields)

    # Контроль памяти на P буферов
    buffer_bytes = memory_guard.recommended_partition_buffer_bytes(num_partitions)
    memory_guard.check_partition_buffers(num_partitions, min(buffer_bytes, 64 * 1024))

    paths = [spill_dir / f"partition_{p:05d}.arrow" for p in range(num_partitions)]
    writers = [ArrowIPCBatchWriter(str(p), schema) for p in paths]

    row_id_offset = 0

    def _process(batch: pa.RecordBatch) -> None:
        nonlocal row_id_offset
        # compute_hashes делает to_pylist() → Python-int (~28 B/значение),
        # для int-колонок это x3-x5 overhead. Учитываем консервативно.
        memory_guard.check_batch(
            batch, name="hash_shuffle.batch", extra_factor=3.0
        )

        hashes = compute_hashes(batch, key_columns, hash_seed)
        parts = _mod_p(hashes, num_partitions)
        order = pc.sort_indices(parts)
        sorted_batch = batch.select(all_columns).take(order)
        sorted_parts = parts.take(order)

        parts_np = sorted_parts.to_numpy(zero_copy_only=False)
        if parts_np.size == 0:
            return

        order_np = order.to_numpy(zero_copy_only=False)   # исходные позиции в batch
        bounds = _boundaries(parts_np)
        for i in range(len(bounds) - 1):
            start, end = bounds[i], bounds[i + 1]
            p = int(parts_np[start])
            piece = sorted_batch.slice(start, end - start)
            row_ids = pa.array(
                (order_np[start:end].astype(np.int64) + row_id_offset).astype(np.uint64),
                type=pa.uint64(),
            )
            piece = piece.append_column("row_id", row_ids)
            writers[p].write_batch(piece)
            if log:
                memory_guard.log_step(
                    "hash_shuffle.write",
                    n_rows=end - start,
                    batch_bytes=piece.nbytes,
                )

        row_id_offset += batch.num_rows

    try:
        _process(first_batch)
        for batch in it:
            _process(batch)
    finally:
        for w in writers:
            w.close()

    if log:
        logger.info(
            "hash_shuffle done: P=%d rows=%d files=%d",
            num_partitions, row_id_offset, len(paths),
        )
    return paths


def read_partition_batches(
    path: str | Path,
    columns: list[str] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Удобный reader для партиции (для тестов и локальной агрегации)."""
    reader = ArrowIPCBatchReader(str(path), columns)
    for batch in reader:
        yield batch