"""
drop_duplicates: HashShuffle по subset + локальный pandas.drop_duplicates.

Алгоритм (DECISIONS §8, «unique, drop_duplicates | HashShuffle + first»):
  1. HashShuffle по subset → P партиций (не пересекаются по subset keys).
  2. Для каждой партиции: read (safe, size-check ДО чтения),
     sort by row_id → pandas.drop_duplicates(subset, keep).
  3. Concat — партиции не пересекаются по subset keys, поэтому
     локальный dedup эквивалентен глобальному.

`keep`:
  - 'first'  — min row_id per key (по исходному порядку).
  - 'last'   — max row_id per key.
  - False    — только ключи с единственным вхождением (drop ALL duplicates).

Ограничения (v0.3.0):
  - Порядок результата: partition-concatenation (не pandas-original order).
  - Recursion при oversized partition: НЕТ; raise MemoryLimitExceeded
    с рекомендацией увеличить num_partitions. §11.5.
"""
from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.ops.shuffle import estimate_num_partitions, hash_shuffle

logger = get_logger("ops.dedup")

_VALID_KEEP = ("first", "last", False)


def _estimate_k_sample(
    batch: pa.RecordBatch,
    subset: list[str],
    n_total: int,
) -> int:
    """Rough estimate: unique count in subset, linear extrapolation."""
    if batch.num_rows == 0:
        return 1
    sub = batch.select(subset)
    df = sub.to_pandas()
    k_sample = int(df.drop_duplicates().shape[0])
    if k_sample <= 0:
        return 1
    if n_total <= batch.num_rows:
        return k_sample
    return max(1, int(k_sample * (n_total / batch.num_rows)))


def _local_dedup(
    partition_path: Path,
    subset_cols: list[str],
    keep,
    memory_guard: MemoryGuard,
) -> pa.Table:
    """Local dedup of one partition. Returns table with original columns
    (row_id dropped). Raises MemoryLimitExceeded if partition > available*0.5."""
    size = partition_path.stat().st_size
    limit = int(memory_guard.available_bytes() * 0.5)
    if size > limit:
        raise MemoryLimitExceeded(
            f"drop_duplicates: partition {partition_path.name} "
            f"size={size} B > limit={limit} B. "
            f"Увеличьте num_partitions или уменьшите subset. "
            f"Recursion не поддерживается (v0.3.0)."
        )

    reader = ipc.open_file(str(partition_path))
    n_batches = reader.num_record_batches
    # Schema from file with row_id stripped (hash_shuffle always appends).
    out_fields = [f for f in reader.schema if f.name != "row_id"]
    out_schema = pa.schema(out_fields)

    if n_batches == 0:
        return pa.Table.from_batches([], schema=out_schema)

    # v0.8.9 (P-A1 for dedup): generator form, schema-correct.
    table = pa.Table.from_batches(
        (reader.get_batch(i).select([f.name for f in out_fields] + ["row_id"])
         for i in range(n_batches)),
        schema=reader.schema,
    )
    memory_guard.check_state(table.nbytes, name=f"dedup({partition_path.name})")
    # v0.8.9 (P-D1): pandas chain (to_pandas + sort_values + drop_duplicates
    # + drop + from_pandas) — measured 3.95-8.29× table.nbytes (Pingis v0.8.8).
    # 5× is a conservative bound covering int and pandas-3-StringDtype paths.
    # Effective partition limit shrinks to avail × 0.5 / 5 = avail × 0.1.
    memory_guard.check_state(
        5 * table.nbytes,
        name=f"dedup.pandas_chain({partition_path.name})",
    )

    df = table.to_pandas()
    # Сорт по row_id (stable) — keep='first'/'last' по исходному порядку.
    df = df.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    df = df.drop_duplicates(subset=subset_cols, keep=keep)
    df = df.drop(columns=["row_id"])

    return pa.Table.from_pandas(df, schema=out_schema, preserve_index=False)


def drop_duplicates(
    batch_iter: Iterable[pa.RecordBatch],
    subset: list[str],
    keep,
    memory_guard: MemoryGuard,
    spill_dir: str | Path,
    *,
    total_rows: int = 0,
    num_partitions: int | None = None,
    log: bool = True,
) -> pa.Table:
    """Out-of-core drop_duplicates.

    - subset: list of columns (non-empty). Dedup key.
    - keep: 'first' | 'last' | False.
    - total_rows: hint for k_est (from df._dataset.total_rows).
    - num_partitions: None — auto from sample.

    Returns pa.Table with original columns (row_id dropped).
    """
    if not subset:
        raise ValueError("drop_duplicates: subset must be non-empty")
    if keep not in _VALID_KEEP:
        raise ValueError(
            f"drop_duplicates: keep must be 'first', 'last', or False; "
            f"got {keep!r}"
        )

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    it = iter(batch_iter)
    try:
        first = next(it)
    except StopIteration as exc:
        raise ValueError("drop_duplicates: empty input") from exc

    for c in subset:
        if c not in first.schema.names:
            raise KeyError(
                f"drop_duplicates: subset column {c!r} not in schema"
            )

    value_cols = [c for c in first.schema.names if c not in subset]

    if num_partitions is None:
        n_total = total_rows if total_rows > 0 else first.num_rows
        k_est = _estimate_k_sample(first, subset, n_total)
        p_est = estimate_num_partitions(
            k_est,
            available_bytes=max(1, memory_guard.available_bytes() // 2),
        )
        num_partitions = max(8, p_est)
        if log:
            logger.info(
                "drop_duplicates: auto P=%d (k_est=%d, n_total=%d)",
                num_partitions, k_est, n_total,
            )

    batches = chain([first], it)
    paths = hash_shuffle(
        batches, subset, value_cols, num_partitions,
        memory_guard, spill_dir / "shuffle", log=log,
    )

    results: list[pa.Table] = []
    for p in paths:
        if not p.exists() or p.stat().st_size == 0:
            continue
        t = _local_dedup(p, subset, keep, memory_guard)
        if t.num_rows > 0:
            results.append(t)

    out_schema = pa.schema(
        [f for f in first.schema if f.name != "row_id"]
    )
    if not results:
        return pa.Table.from_batches([], schema=out_schema)

    return pa.concat_tables(results, promote_options="default")
