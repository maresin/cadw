"""
Rank: external_sort + потоковое вычисление rank.

Алгоритм (DECISIONS §8, «rank | Sort + локальный ранг»):
  1. external_sort по `by` (ascending).
  2. Потоково: rank = f(позиция, состояние tie-группы). Сохраняем
     (row_id, rank) в файл (порядок — sorted-by-`by`).
  3. external_sort по row_id → rank в исходном порядке.
  4. Zip с исходным DataFrame позиционно.

Streamable методы (v0.3.1):
  - 'first': rank = global position (1-based) в sorted order.
  - 'min':   rank = позиция первого в текущей tie-группе.
  - 'dense': rank = dense counter (1, 2, 3, ... для уникальных значений).

Не streamable (v0.3.2): 'max', 'average', 'last' — требуют знать
конец tie-группы, а значит — буферизацию группы (O(N) при all-equal).

Null values: участвуют как значения (null == null); pandas `skipna=True`
не поддерживается в v0.3.1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pyarrow as pa

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_writer import ArrowIPCBatchWriter

logger = get_logger("ops.rank")

VALID_METHODS = ("first", "min", "dense")


def compute_ranks_to_disk(
    sorted_batches: Iterable[pa.RecordBatch],
    *,
    by: str,
    output_col: str,
    method: str,
    pct: bool,
    total_n: int,
    rank_path: Path,
    memory_guard: MemoryGuard,
    log: bool = False,
) -> Path:
    """Stream sorted batches, write (row_id, rank) to rank_path.

    `sorted_batches` должны содержать колонки [by, "row_id"], отсортированы
    по `by` (ascending; ties — по row_id asc, стабильность из external_sort).

    `total_n` — число строк (для pct). Если 0 и pct=True — ValueError.
    """
    if method not in VALID_METHODS:
        raise NotImplementedError(
            f"rank: method={method!r} not supported in v0.3.1. "
            f"Supported: {VALID_METHODS}. "
            f"'max'/'average'/'last' требуют буферизации tie-группы — v0.3.2."
        )
    if pct and total_n <= 0:
        raise ValueError("rank: pct=True requires total_n > 0")

    rank_dtype = pa.float64() if pct else pa.int64()
    schema = pa.schema([("row_id", pa.uint64()), (output_col, rank_dtype)])
    writer = ArrowIPCBatchWriter(str(rank_path), schema)

    _UNSET = object()
    prev_val = _UNSET
    group_start = 0
    global_pos = 0
    dense_counter = 0

    try:
        for batch in sorted_batches:
            memory_guard.check_batch(batch, name="rank.input")
            vals = batch.column(by).to_pylist()
            row_ids = batch.column("row_id").to_pylist()

            ranks: list = []
            for v in vals:
                global_pos += 1
                if prev_val is _UNSET:
                    group_start = global_pos
                    dense_counter = 1
                elif v != prev_val:
                    group_start = global_pos
                    dense_counter += 1
                # else: continue current tie group

                if method == "first":
                    r = global_pos
                elif method == "min":
                    r = group_start
                else:  # dense
                    r = dense_counter

                ranks.append((r / total_n) if pct else r)
                prev_val = v

            rb = pa.RecordBatch.from_arrays(
                [
                    pa.array(row_ids, type=pa.uint64()),
                    pa.array(ranks, type=rank_dtype),
                ],
                schema=schema,
            )
            writer.write_batch(rb)
    finally:
        writer.close()

    if log:
        logger.info(
            "compute_ranks_to_disk: method=%s pct=%s total_n=%d",
            method, pct, total_n,
        )
    return rank_path
