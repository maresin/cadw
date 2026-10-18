# cadw/ops/groupby.py
"""
GroupBy: корректный при K > памяти.

Алгоритм (DECISIONS.md §6.1, Combine-Shuffle-Reduce):
  1. HashShuffle по ключу группы → P партиций (не пересекаются по ключам).
  2. Для каждой партиции — локальная агрегация в памяти
     (pa.Table.group_by(...).aggregate(...) — векторизованно, без построчных циклов).
  3. Конкатенация результатов партиций → финальный Dataset.
     Merge не нужен: партиции не пересекаются по ключам.

Пиковая RAM:
  - Фаза 1: batch + P буферов записи (см. hash_shuffle).
  - Фаза 2: один батч партиции + словарь агрегатов на партицию
    (ограничен числом групп в партиции, а не в целом датасете).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops.shuffle import hash_shuffle

logger = get_logger("ops.groupby")


_AGG_FUNCS = {"mean", "sum", "count", "min", "max"}


def _local_groupby(
    partition_path: Path,
    by_columns: Sequence[str],
    agg_columns: Sequence[str],
    agg_funcs: Sequence[str] | str,
) -> pa.Table:
    """Локальная агрегация одной партиции. Векторизованно через pa.Table.group_by."""
    if isinstance(agg_funcs, str):
        agg_funcs = [agg_funcs] * len(agg_columns)
    if len(agg_funcs) != len(agg_columns):
        raise ValueError("agg_funcs и agg_columns должны быть одной длины")
    for f in agg_funcs:
        if f not in _AGG_FUNCS:
            raise ValueError(f"Неподдерживаемая агрегация: {f!r}. Доступно: {_AGG_FUNCS}")

    batches = list(ArrowIPCBatchReader(str(partition_path), None))
    if not batches:
        # Пустая партиция — вернуть пустую таблицу с нужной схемой
        fields = [pa.field(c, pa.string()) for c in by_columns]
        for col, f in zip(agg_columns, agg_funcs):
            out_type = pa.int64() if f == "count" else pa.float64()
            fields.append(pa.field(f"{col}_{f}", out_type))
        return pa.Table.from_pylist([], schema=pa.schema(fields))

    table = pa.Table.from_batches(batches)

    if table.num_rows == 0:
        fields = [table.schema.field(c) for c in by_columns]
        for col, f in zip(agg_columns, agg_funcs):
            out_type = pa.int64() if f == "count" else pa.float64()
            fields.append(pa.field(f"{col}_{f}", out_type))
        return pa.Table.from_pylist([], schema=pa.schema(fields))

    # pa.Table.group_by: [(col, "hash"), ...] для нескольких ключей
    agg_specs = []
    for col, f in zip(agg_columns, agg_funcs):
        if f == "mean":
            agg_specs.append((col, "mean"))
        elif f == "sum":
            agg_specs.append((col, "sum"))
        elif f == "count":
            agg_specs.append((col, "count"))
        elif f == "min":
            agg_specs.append((col, "min"))
        elif f == "max":
            agg_specs.append((col, "max"))

    grouped = table.group_by(list(by_columns)).aggregate(agg_specs)
    # Переименовать {col}_{func} — оставляем как есть, downstream знает имена
    return grouped


def groupby_aggregate(
    batch_iter: Iterable[pa.RecordBatch],
    by_columns: Sequence[str] | str,
    agg_columns: Sequence[str] | str,
    agg_funcs: Sequence[str] | str,
    memory_guard: MemoryGuard,
    spill_dir: str | Path,
    *,
    num_partitions: int | None = None,
    hash_seed: int = 0,
    log: bool = True,
) -> pa.Table:
    """
    Out-of-core groupby: HashShuffle + LocalGroupBy + Concat.

    Возвращает pa.Table с колонками:
        by_columns + [f"{col}_{func}" для каждой (col, func)].

    Корректен при K > памяти: число групп в каждой партиции ограничено
    только числом групп с данным хэшем, а не общим K.
    """
    if isinstance(by_columns, str):
        by_columns = [by_columns]
    if isinstance(agg_columns, str):
        agg_columns = [agg_columns]
    if isinstance(agg_funcs, str):
        agg_funcs = [agg_funcs] * len(agg_columns)

    by_columns = list(by_columns)
    agg_columns = list(agg_columns)
    agg_funcs = list(agg_funcs)

    if not by_columns:
        raise ValueError("by_columns must be non-empty")
    if len(agg_columns) != len(agg_funcs):
        raise ValueError("agg_columns и agg_funcs должны быть одной длины")
    for f in agg_funcs:
        if f not in _AGG_FUNCS:
            raise ValueError(f"Неподдерживаемая агрегация: {f!r}. Доступно: {_AGG_FUNCS}")

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    # --- Фаза 1: HashShuffle ---
    # Value columns для партиций: by + agg (после дедупликации)
    # HashShuffle пишет key_columns + value_columns. by_columns уже идут как key,
    # поэтому value_columns — только agg-колонки, которых нет в by (без дублей).
    seen: set[str] = set(by_columns)
    value_columns: list[str] = []
    for c in agg_columns:
        if c not in seen:
            seen.add(c)
            value_columns.append(c)
    if not value_columns:
        # Все agg-колонки совпадают с by — тогда в партицию пойдёт только by.
        # Такой groupby бессмысленен, но не должен падать.
        value_columns = list(by_columns)

    if num_partitions is None:
        num_partitions = 8  # консервативно; вызывающий может передать больше

    partitions = hash_shuffle(
        batch_iter,
        key_columns=by_columns,
        value_columns=value_columns,
        num_partitions=num_partitions,
        memory_guard=memory_guard,
        spill_dir=spill_dir / "shuffle",
        hash_seed=hash_seed,
        log=log,
    )
    if log:
        logger.info("groupby_aggregate: %d partitions", len(partitions))

    # --- Фаза 2: LocalGroupBy по каждой партиции ---
    partial_tables: list[pa.Table] = []
    for i, p in enumerate(partitions):
        t = _local_groupby(p, by_columns, agg_columns, agg_funcs)
        # Пустая таблица без данных не мешает конкатенации
        if t.num_rows > 0:
            partial_tables.append(t)
        if log and t.num_rows > 0:
            memory_guard.log_step(
                "groupby_aggregate.partition",
                batch_id=i,
                n_rows=t.num_rows,
                batch_bytes=t.nbytes,
            )

    # --- Фаза 3: Конкатенация (merge не нужен — партиции не пересекаются по ключам) ---
    if not partial_tables:
        # Пустая схема результата
        fields = [pa.field(c, pa.string()) for c in by_columns]
        for col, f in zip(agg_columns, agg_funcs):
            out_type = pa.int64() if f == "count" else pa.float64()
            fields.append(pa.field(f"{col}_{f}", out_type))
        return pa.Table.from_pylist([], schema=pa.schema(fields))

    result = pa.concat_tables(partial_tables, promote_options="default")
    if log:
        logger.info("groupby_aggregate done: %d rows", result.num_rows)
    return result


def write_groupby_result(
    result_table: pa.Table,
    out_path: str | Path,
) -> Path:
    """Сохранить результат в Arrow IPC (для использования как Dataset)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ArrowIPCBatchWriter(str(out_path), result_table.schema)
    try:
        for batch in result_table.to_batches():
            writer.write_batch(batch)
    finally:
        writer.close()
    return out_path