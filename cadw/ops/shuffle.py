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


def _stable_bytes(v) -> bytes:
    """Стабильная сериализация значения в bytes (без зависимости от PYTHONHASHSEED).

    repr() различает типы: repr('1') = "'1'", repr(1) = '1', repr(None) = 'None'.
    Это важно: строка '1' и число 1 не должны попадать в одну партицию.
    """
    return repr(v).encode("utf-8")


def compute_hashes(
    batch: pa.RecordBatch,
    key_columns: Sequence[str],
    seed: int = 0,
) -> pa.Array:
    """Детерминированный uint64-хэш ключа (boost::hash_combine через NumPy).

    Арифметика в np.uint64 даёт wrap-around по определению; pc.add/pc.shift_left
    над uint64 возбуждают ArrowInvalid при переполнении.
    """
    if not key_columns:
        raise ValueError("key_columns must be non-empty")

    h_np: np.ndarray | None = None
    for col in key_columns:
        values = batch.column(col).to_pylist()
        hashes = [xxhash.xxh64(_stable_bytes(v), seed=seed).intdigest() for v in values]
        ch = np.asarray(hashes, dtype=np.uint64)

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

    # Схема: key + value + row_id
    fields: list[pa.Field] = []
    for col in list(key_columns) + list(value_columns):
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
        sorted_batch = batch.select(list(key_columns) + list(value_columns)).take(order)
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