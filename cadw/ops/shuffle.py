# cadw/ops/shuffle.py
"""
HashShuffle: разложить поток строк по P партициям так, что все строки
с одинаковым ключом попадают в одну партицию. Партиции не пересекаются
по ключам → результат агрегации = конкатенация (без merge).

Контракт (DECISIONS.md §6):
  - детерминированный XXH64-based хэш (v0.9.6a: own impl, no pandas);
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


# v0.9.6a: fixed hash for nulls (all nulls → one group, matching
# pandas dropna=False groupby semantics). Magic value chosen after
# splitmix64 avalanche — collision with hashed real value ~2^-64.
_NULL_SENTINEL = np.uint64(0x1BADB002CAFEF00D)

# XXH64 primes (from xxHash reference implementation, XXH64).
_XXH64_P1 = np.uint64(11400714785074694791)
_XXH64_P2 = np.uint64(14029467366897019727)
_XXH64_P3 = np.uint64(1609587929392839161)
_XXH64_P4 = np.uint64(9650029242287828579)
_XXH64_P5 = np.uint64(2870177450012600261)


def _rotl64(x: np.ndarray, r: int) -> np.ndarray:
    """Rotate uint64 array left by r bits (r in [0, 63])."""
    r_u = np.uint64(r)
    return (x << r_u) | (x >> (np.uint64(64) - r_u))


def _vectorized_hash_numeric(arr: pa.Array) -> np.ndarray:
    """XXH64-round over 8-byte uint64 view. Vectorized, no pandas.

    - int/uint/bool/timestamp/date/time: cast to int64 (or uint64)
      → bitcast uint64 → XXH64 round.
    - float: bitcast float64 → int64 → uint64 (preserves NaN bit
      pattern; nulls become NaN via to_numpy → hash(NaN) — matches
      pandas groupby(dropna=False) NaN/null coalescing for floats).
    - nulls in int/temporal: overwritten with _NULL_SENTINEL (own
      group, distinct from any integer value).

    v0.9.6a: own XXH64-based hash. No pandas. Vectorized over the
    batch (O(batch) numpy ops on uint64 arrays).
    """
    n = len(arr)
    t = arr.type

    if pa.types.is_floating(t):
        np_arr = arr.to_numpy(zero_copy_only=False)  # float64, NaN for nulls
        int_view = np_arr.view(np.uint64)
        null_mask = None  # NaN bit pattern hashed; NaN/null coalesce
    else:
        if pa.types.is_boolean(t):
            int_arr = pc.cast(arr, pa.int64())
        elif (pa.types.is_timestamp(t) or pa.types.is_date(t)
              or pa.types.is_time(t)):
            int_arr = pc.cast(arr, pa.int64())
        elif pa.types.is_integer(t):
            # For uint types, keep unsigned representation to avoid
            # int64 overflow (safe=True cast would raise for >= 2^63).
            int_arr = arr  # already an Arrow array; will view below
        else:
            raise NotImplementedError(
                f"_vectorized_hash_numeric: unsupported dtype {t}"
            )
        if int_arr.null_count > 0:
            null_mask = pc.is_null(int_arr).to_numpy(zero_copy_only=False)
            int_arr = pc.fill_null(int_arr, 0)
        else:
            null_mask = None
        np_arr = int_arr.to_numpy(zero_copy_only=False)
        if pa.types.is_unsigned_integer(t) or pa.types.is_boolean(t):
            int_view = np_arr.astype(np.uint64, copy=False)
        else:
            int_view = np_arr.astype(np.int64, copy=False).view(np.uint64)

    # XXH64 round (one 8-byte block, seed=0)
    acc = np.full(n, _XXH64_P5, dtype=np.uint64)
    acc = acc + np.uint64(8)  # += len(block_bytes)
    k1 = int_view * _XXH64_P2
    k1 = _rotl64(k1, 31)
    k1 = k1 * _XXH64_P1
    acc = acc ^ k1
    acc = _rotl64(acc, 27) * _XXH64_P1 + _XXH64_P4

    # XXH64 avalanche (final)
    acc = acc ^ (acc >> np.uint64(33))
    acc = acc * _XXH64_P2
    acc = acc ^ (acc >> np.uint64(29))
    acc = acc * _XXH64_P3
    acc = acc ^ (acc >> np.uint64(32))

    if null_mask is not None:
        acc[null_mask] = _NULL_SENTINEL

    return acc


def _string_hash_column(arr: pa.Array) -> np.ndarray:
    """Per-element xxhash.xxh64 over utf8/binary bytes. No pandas.

    v0.9.6a: no repr()/encode() wrapper — column is homogeneous type
    (all str or all bytes), so no cross-type hash-collision concern.
    Cross-batch consistency: same string → same xxh64 output.
    Nulls → _NULL_SENTINEL (own group).
    """
    n = len(arr)
    out = np.empty(n, dtype=np.uint64)
    for i, v in enumerate(arr.to_pylist()):
        if v is None:
            out[i] = _NULL_SENTINEL
        else:
            data = v.encode("utf-8") if isinstance(v, str) else v
            out[i] = xxhash.xxh64(data, seed=0).intdigest()
    return out


def _stable_bytes(v) -> bytes:
    """Fallback serialization for exotic types (dictionary/struct/list)."""
    return repr(v).encode("utf-8")


def _hash_column(arr: pa.Array) -> np.ndarray:
    """Dispatch: vectorized XXH64 for fixed-width; per-element for strings.

    v0.9.6a: no pandas. Covers all first-class Arrow dtypes
    (int/float/bool/timestamp/date/time/string/binary). Exotic
    (dictionary/struct/list) → Python loop, rare (not a partition key
    column type in practice).
    """
    t = arr.type
    if (pa.types.is_string(t) or pa.types.is_large_string(t)
            or pa.types.is_binary(t) or pa.types.is_large_binary(t)):
        return _string_hash_column(arr)
    if (pa.types.is_integer(t) or pa.types.is_floating(t)
            or pa.types.is_boolean(t)
            or pa.types.is_timestamp(t) or pa.types.is_date(t)
            or pa.types.is_time(t)):
        return _vectorized_hash_numeric(arr)
    # Exotic fallback (dictionary/struct/list)
    values = arr.to_pylist()
    hashes = [xxhash.xxh64(_stable_bytes(v), seed=0).intdigest()
              for v in values]
    return np.asarray(hashes, dtype=np.uint64)


def compute_hashes(
    batch: pa.RecordBatch,
    key_columns: Sequence[str],
    seed: int = 0,
) -> pa.Array:
    """Deterministic uint64 hash of key (boost-combine + splitmix64).

    v0.9.6a: own XXH64-based hash for numeric/string columns; no pandas.
    Seed applied via XOR before final splitmix64 avalanche (uniform
    across dtypes; deterministic; XOR is bijection — no collision added).

    Partition-disjoint invariant: same value → same hash within a call;
    nulls → same fixed sentinel. Splitmix64 avalanche ensures uniform
    low bits for partition assignment (v0.8.1-fix2: pandas string hash
    had even low bits, wasting half the partitions).
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
            # boost::hash_combine (64-bit) with wrap-around
            mixed = (
                ch
                + np.uint64(0x9E3779B97F4A7C15)
                + (h_np << np.uint64(6))
                + (h_np >> np.uint64(2))
            )
            h_np = h_np ^ mixed

    assert h_np is not None
    # Splitmix64 avalanche — uniform low bits for partition assignment.
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