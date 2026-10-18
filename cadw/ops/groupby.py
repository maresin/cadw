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

import numpy as np

from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops.shuffle import hash_shuffle, estimate_num_partitions, compute_hashes, _mod_p, _boundaries

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



_HEAVY_RATIO = 0.05


def _detect_heavy_keys(sample_batches, by_columns, threshold_ratio: float = _HEAVY_RATIO) -> set:
    """Тяжёлые ключи по sample: доля в sample > threshold_ratio."""
    if not sample_batches:
        return set()
    from collections import Counter
    counter: Counter = Counter()
    for b in sample_batches:
        for i in range(b.num_rows):
            key = tuple(b.column(c)[i].as_py() for c in by_columns)
            counter[key] += 1
    total = sum(counter.values())
    if total == 0:
        return set()
    return {k for k, cnt in counter.items() if cnt / total > threshold_ratio}


def _compute_heavy_mask(batch: pa.RecordBatch, by_columns, heavy_keys: set) -> pa.Array:
    n = batch.num_rows
    if not heavy_keys or n == 0:
        return pa.array([False] * n, type=pa.bool_())
    mask = pa.array([False] * n, type=pa.bool_())
    for key in heavy_keys:
        k_mask = pa.array([True] * n, type=pa.bool_())
        for col, v in zip(by_columns, key):
            k_mask = pc.and_(k_mask, pc.equal(batch.column(col), pa.scalar(v)))
        mask = pc.or_(mask, k_mask)
    return mask


def _accumulate_heavy(accum: dict, batch: pa.RecordBatch, by_columns, agg_columns, agg_funcs) -> None:
    # Base-specs без дублей: (col, base_func).
    # mean -> (sum, count); остальное — как есть.
    needed: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for col, f in zip(agg_columns, agg_funcs):
        if f == "mean":
            for bf in ("sum", "count"):
                if (col, bf) not in seen:
                    seen.add((col, bf))
                    needed.append((col, bf))
        else:
            if (col, f) not in seen:
                seen.add((col, f))
                needed.append((col, f))

    select_cols = list(dict.fromkeys(list(by_columns) + list(agg_columns)))
    table = pa.Table.from_batches([batch.select(select_cols)])
    grouped = table.group_by(list(by_columns)).aggregate(needed)

    for row in grouped.to_pylist():
        key = tuple(row[c] for c in by_columns)
        if key not in accum:
            accum[key] = {}
        for col, f in zip(agg_columns, agg_funcs):
            if f == "mean":
                # Внутренние ключи — не пересекаются с явным sum/count.
                sk, ck = f"__mean_sum_{col}", f"__mean_count_{col}"
                accum[key][sk] = accum[key].get(sk, 0.0) + row[f"{col}_sum"]
                accum[key][ck] = accum[key].get(ck, 0) + row[f"{col}_count"]
            else:
                out = f"{col}_{f}"
                new = row[out]
                old = accum[key].get(out)
                if old is None:
                    accum[key][out] = new
                elif f in ("sum", "count"):
                    accum[key][out] = old + new
                elif f == "min":
                    accum[key][out] = min(old, new)
                elif f == "max":
                    accum[key][out] = max(old, new)


def _accum_to_table(accum: dict, by_columns, agg_columns, agg_funcs) -> pa.Table:
    rows = []
    for key, d in accum.items():
        row = dict(zip(by_columns, key))
        for col, f in zip(agg_columns, agg_funcs):
            if f == "mean":
                s = d.get(f"__mean_sum_{col}", 0.0)
                c = d.get(f"__mean_count_{col}", 0)
                row[f"{col}_mean"] = s / c if c > 0 else float("nan")
            else:
                row[f"{col}_{f}"] = d[f"{col}_{f}"]
        rows.append(row)
    if not rows:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )
    return pa.Table.from_pylist(
        rows, schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
    )


def _shuffle_write(writers, batch: pa.RecordBatch, key_columns, value_columns,
                   num_partitions: int, hash_seed: int, row_id_offset: int) -> None:
    hashes = compute_hashes(batch, key_columns, hash_seed)
    parts = _mod_p(hashes, num_partitions)
    order = pc.sort_indices(parts)
    sorted_batch = batch.select(list(key_columns) + list(value_columns)).take(order)
    sorted_parts = parts.take(order)
    parts_np = sorted_parts.to_numpy(zero_copy_only=False)
    order_np = order.to_numpy(zero_copy_only=False)
    if parts_np.size == 0:
        return
    bounds = _boundaries(parts_np)
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        p = int(parts_np[start])
        piece = sorted_batch.slice(start, end - start)
        row_ids = pa.array(
            (order_np[start:end].astype(np.int64) + row_id_offset).astype(np.uint64),
            type=pa.uint64(),
        )
        writers[p].write_batch(piece.append_column("row_id", row_ids))


def _groupby_with_skew(batch_iter, by_columns, agg_columns, agg_funcs, heavy_keys,
                       num_partitions, memory_guard, spill_dir, log) -> pa.Table:
    all_columns: list[str] = []
    seen: set[str] = set()
    for c in list(by_columns) + list(agg_columns):
        if c not in seen:
            seen.add(c)
            all_columns.append(c)

    it = iter(batch_iter)
    try:
        first_batch = next(it)
    except StopIteration:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )
    memory_guard.check_batch(first_batch, name="groupby_with_skew.first", extra_factor=3.0)

    fields = [first_batch.schema.field(c) for c in all_columns]
    fields.append(pa.field("row_id", pa.uint64()))
    schema = pa.schema(fields)
    shuf_dir = spill_dir / "shuffle"
    shuf_dir.mkdir(parents=True, exist_ok=True)
    paths = [shuf_dir / f"partition_{p:05d}.arrow" for p in range(num_partitions)]
    writers = [ArrowIPCBatchWriter(str(p), schema) for p in paths]

    accum: dict = {}
    row_id_offset = 0
    try:
        for batch in itertools.chain([first_batch], it):
            memory_guard.check_batch(
                batch, name="groupby_with_skew.batch", extra_factor=3.0
            )
            heavy_mask = _compute_heavy_mask(batch, by_columns, heavy_keys)
            light_mask = pc.invert(heavy_mask)
            heavy_batch = batch.filter(heavy_mask)
            light_batch = batch.filter(light_mask)
            if heavy_batch.num_rows > 0:
                _accumulate_heavy(accum, heavy_batch, by_columns, agg_columns, agg_funcs)
            if light_batch.num_rows > 0:
                _shuffle_write(
                    writers, light_batch, by_columns, agg_columns,
                    num_partitions, hash_seed=0, row_id_offset=row_id_offset,
                )
            row_id_offset += batch.num_rows
    finally:
        for w in writers:
            w.close()

    light_results: list[pa.Table] = []
    for p in paths:
        if not p.exists() or p.stat().st_size == 0:
            continue
        t = _local_groupby(
            p, by_columns, agg_columns, agg_funcs,
            memory_guard=memory_guard, spill_dir=spill_dir,
        )
        if t.num_rows > 0:
            light_results.append(t)
        try:
            p.unlink()
        except OSError:
            pass

    heavy_result = _accum_to_table(accum, by_columns, agg_columns, agg_funcs)
    all_results = light_results + ([heavy_result] if heavy_result.num_rows > 0 else [])
    if not all_results:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )
    if log:
        logger.info(
            "groupby_with_skew: heavy=%d, light_partitions=%d",
            len(accum), len(light_results),
        )
    return pa.concat_tables(all_results)


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

    # Sample строится ВСЕГДА: он нужен и для auto-P, и для skew detection.
    # Важно: break-and-chain, не tee. С tee chain(sample, rest) дублирует
    # sample (count/sum удваиваются, mean маскирует).
    sample_cap = 10_000
    batch_iter = iter(batch_iter)
    sample_batches: list[pa.RecordBatch] = []
    n_sample = 0
    for b in batch_iter:
        sample_batches.append(b)
        n_sample += b.num_rows
        if n_sample >= sample_cap:
            break
    sample_bytes = sum(b.nbytes for b in sample_batches)
    sample_limit = int(memory_guard.available_bytes() * 0.5)
    skip_skew = False
    if sample_bytes > sample_limit:
        if num_partitions is None:
            raise MemoryLimitExceeded(
                f"groupby_aggregate: sample ({sample_bytes} B) > "
                f"available*0.5 ({sample_limit} B) и num_partitions не задан. "
                f"Передайте num_partitions явно."
            )
        # P задан явно: sample не влезает, skew detection пропускаем,
        # но прочитанное не выбрасываем (batch_iter уже исчерпан sample-циклом).
        skip_skew = True
        if log:
            logger.warning(
                "groupby_aggregate: sample (%d B) > available*0.5 (%d B); "
                "skew detection skipped (num_partitions=%d задан явно)",
                sample_bytes, sample_limit, num_partitions,
            )
    batch_iter = itertools.chain(sample_batches, batch_iter)

    if num_partitions is None:
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

    # --- Skew detection & isolation (§11.4 п.3) ---
    heavy_keys = set() if skip_skew else _detect_heavy_keys(sample_batches, by_columns)
    if heavy_keys:
        if log:
            logger.info(
                "groupby_aggregate: skew detected, %d heavy key(s) -> streaming",
                len(heavy_keys),
            )
        return _groupby_with_skew(
            batch_iter, by_columns, agg_columns, agg_funcs, heavy_keys,
            num_partitions, memory_guard, spill_dir, log,
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