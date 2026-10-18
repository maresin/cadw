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

import itertools

from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops.shuffle import hash_shuffle, estimate_num_partitions

logger = get_logger("ops.groupby")


_AGG_FUNCS = {"mean", "sum", "count", "min", "max"}


_MAX_DEPTH = 5


def _empty_result_schema(by_columns, agg_columns, agg_funcs) -> pa.Schema:
    """Схема пустого результата агрегации (для пустых партиций)."""
    fields = [pa.field(c, pa.string()) for c in by_columns]
    for col, f in zip(agg_columns, agg_funcs):
        out_type = pa.int64() if f == "count" else pa.float64()
        fields.append(pa.field(f"{col}_{f}", out_type))
    return pa.schema(fields)


def _local_groupby(
    partition_path: Path,
    by_columns: Sequence[str],
    agg_columns: Sequence[str],
    agg_funcs: Sequence[str] | str,
    memory_guard: MemoryGuard | None = None,
    spill_dir: Path | None = None,
    depth: int = 0,
    max_depth: int = _MAX_DEPTH,
) -> pa.Table:
    """Локальная агрегация одной партиции. Векторизованно через pa.Table.group_by."""
    if isinstance(agg_funcs, str):
        agg_funcs = [agg_funcs] * len(agg_columns)
    if len(agg_funcs) != len(agg_columns):
        raise ValueError("agg_funcs и agg_columns должны быть одной длины")
    for f in agg_funcs:
        if f not in _AGG_FUNCS:
            raise ValueError(f"Неподдерживаемая агрегация: {f!r}. Доступно: {_AGG_FUNCS}")

    # 0. Пустая партиция
    batches = list(ArrowIPCBatchReader(str(partition_path), None))
    if not batches:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )
    table = pa.Table.from_batches(batches)
    if table.num_rows == 0:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )

    # 1. Размер партиции = сумма nbytes уже прочитанных батчей
    total_bytes = sum(b.nbytes for b in batches)
    safety_limit = int(memory_guard.available_bytes() * 0.5) if memory_guard else total_bytes

    # 2. Влезает — материализуем и агрегируем
    if total_bytes <= safety_limit:
        agg_specs = [(col, f) for col, f in zip(agg_columns, agg_funcs)]
        return table.group_by(list(by_columns)).aggregate(agg_specs)

    # 3. Не влезает — рекурсивное партиционирование с новым seed
    if memory_guard is None or spill_dir is None:
        raise MemoryLimitExceeded(
            f"_local_groupby: partition too large ({total_bytes} B > {safety_limit} B), "
            f"but memory_guard/spill_dir not provided."
        )
    if depth >= max_depth:
        raise MemoryLimitExceeded(
            f"_local_groupby: depth={depth} >= max_depth={max_depth}; "
            f"partition={partition_path.name} size={total_bytes} B > safety={safety_limit} B. "
            f"Вероятен экстремальный skew (см. DECISIONS §11.4 п.3)."
        )

    def _read_for_resplit():
        for b in ArrowIPCBatchReader(str(partition_path), None):
            cols = [c for c in list(by_columns) + list(agg_columns) if c in b.schema.names]
            yield b.select(cols)

    sub_spill = spill_dir / f"resplit_d{depth}_{partition_path.stem}"
    sub_spill.mkdir(parents=True, exist_ok=True)

    sub_value_cols = [c for c in agg_columns if c not in by_columns]
    sub_paths = hash_shuffle(
        _read_for_resplit(),
        key_columns=list(by_columns),
        value_columns=sub_value_cols,
        num_partitions=4,
        memory_guard=memory_guard,
        spill_dir=sub_spill,
        hash_seed=depth + 1,
        log=False,
    )

    results: list[pa.Table] = []
    for sp in sub_paths:
        r = _local_groupby(
            sp, by_columns, agg_columns, agg_funcs,
            memory_guard=memory_guard, spill_dir=spill_dir,
            depth=depth + 1, max_depth=max_depth,
        )
        if r.num_rows > 0:
            results.append(r)
        try:
            sp.unlink()
        except OSError:
            pass

    if not results:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )
    return pa.concat_tables(results)



def _estimate_k_sample(
    sample_batches: list[pa.RecordBatch],
    by_columns: Sequence[str],
) -> int:
    """Число уникальных ключей в sample. Без экстраполяции (консервативно)."""
    if not sample_batches:
        return 0
    seen: set[tuple] = set()
    cols = list(by_columns)
    for b in sample_batches:
        for i in range(b.num_rows):
            seen.add(tuple(b.column(c)[i].as_py() for c in cols))
    return len(seen)


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
        # Sample первых ~10k строк, оценить K, выбрать P.
        # Бюджет на партицию — половина available: вторая половина нужна
        # на батчи чтения и состояние агрегации.
        sample_cap = 10_000
        sample_iter, rest_iter = itertools.tee(batch_iter, 2)
        sample_batches: list[pa.RecordBatch] = []
        n_sample = 0
        for b in sample_iter:
            sample_batches.append(b)
            n_sample += b.num_rows
            if n_sample >= sample_cap:
                break
        sample_bytes = sum(b.nbytes for b in sample_batches)
        sample_limit = int(memory_guard.available_bytes() * 0.5)
        if sample_bytes > sample_limit:
            raise MemoryLimitExceeded(
                f"groupby_aggregate: sample ({sample_bytes} B) > "
                f"available*0.5 ({sample_limit} B). "
                f"Укажите num_partitions явно, чтобы избежать sample."
            )

        batch_iter = itertools.chain(sample_batches, rest_iter)

        k_est = _estimate_k_sample(sample_batches, by_columns)
        p_est = estimate_num_partitions(
            k_est,
            available_bytes=max(1, memory_guard.available_bytes() // 2),
        )
        num_partitions = max(8, p_est)
        if log:
            logger.info(
                "groupby_aggregate: auto num_partitions=%d (k_est=%d, sample=%d rows)",
                num_partitions, k_est, n_sample,
            )

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
        t = _local_groupby(
            p, by_columns, agg_columns, agg_funcs,
            memory_guard=memory_guard,
            spill_dir=spill_dir,
        )
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