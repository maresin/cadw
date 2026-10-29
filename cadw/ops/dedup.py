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

import pyarrow as pa
import pyarrow.ipc as ipc

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.core.spill import SpillBuffer, InMemoryResult, AutoResult
from cadw.ops.shuffle import estimate_num_partitions, hash_shuffle

logger = get_logger("ops.dedup")

_VALID_KEEP = ("first", "last", False)


def _estimate_k_sample(
    batch: pa.RecordBatch,
    subset: list[str],
    n_total: int,
) -> int:
    """Rough K estimate on sample. Arrow-native (v0.9.6c1).

    K_sample via `pc.dictionary_encode` (no pandas). Extrapolate to
    n_total: linear scaling capped at n_total.
    """
    from cadw.core._k_est import estimate_k_sample_arrow

    if batch.num_rows == 0:
        return 1
    k_sample = estimate_k_sample_arrow(batch, subset)
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
    """Local dedup via Arrow-native dictionary_encode + group_by + is_in.

    v0.8.11a (P-D1 close): replaces the v0.8.9 pandas chain. Validated by
    scripts/proto_dedup_arrow.py:
      - Correctness: A==B==C across keep ∈ {first, last, False} ×
        selectivity ∈ {10%, 50%, 90% dupes}.
      - Peak RSS: 0.29× pandas on average; 0.38× table on 1M rows
        (10.9 MB vs 87.7 MB pandas).

    Algorithm:
      1. dictionary_encode each subset column (string → int32 id).
      2. group_by(subset).aggregate(min/max row_id) — keep='first'/'last'.
         For keep=False: aggregate count + min; keep only count==1.
      3. is_in(row_id, keep_rids) → filter original (unencoded) table.

    Order: partition order (same as v0.3.0 pandas path, which sorted by
    row_id inside the partition). See §11.5 note on order.

    Raises MemoryLimitExceeded if partition > available*0.5, or if
    Arrow-native bound exceeds state_limit:
      - 12x table.nbytes for string subset (11.26x measured at K=N,
        +7% safety)
      - 2.5x table.nbytes for numeric subset (1.97x measured at K=N,
        +27% safety)
    See v0.9.3c / P-D2 / 13.4/175.
    """
    import pyarrow.compute as pc

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
    out_fields = [f for f in reader.schema if f.name != "row_id"]
    out_schema = pa.schema(out_fields)

    if n_batches == 0:
        return pa.Table.from_batches([], schema=out_schema)

    # v0.8.11a-fix3: explicit select_schema. `reader.schema` is file
    # column order; `.select(out_fields + ["row_id"])` reorders. In
    # production hash_shuffle always puts row_id last so they coincide,
    # but relying on that is brittle (any other IPC writer breaks).
    # Explicit schema makes _local_dedup agnostic to file column order.
    select_schema = pa.schema(
        list(out_fields) + [reader.schema.field("row_id")]
    )
    table = pa.Table.from_batches(
        (reader.get_batch(i).select([f.name for f in out_fields] + ["row_id"])
         for i in range(n_batches)),
        schema=select_schema,
    )
    memory_guard.check_state(
        table.nbytes, name=f"dedup({partition_path.name})",
    )
    # v0.8.11a: Arrow-native peak ~0.4× table (proto: 10.9/28.6 MB at 1M).
    # 1.5× conservative for all dtypes incl. K≈N (dictionary ≈ table size
    # for all-unique, but no pandas object overhead). No 5× stopgap.
    # v0.9.3c (P-D2, S2): type-based conservative estimate.
    # Bench v0.9.2 audit:
    #   - K=N strings, 400k rows: peak 11.26x table.nbytes
    #     (dictionary_encode copies every string; Arrow StringArray
    #     + int32 indices + BufferBuilder geometric growth).
    #   - K=N int64: peak 1.97x table.nbytes.
    #   - K<<N strings: ~0x (buffers reused).
    # Prior 1.5x bound was extrapolated from int K<<N prototype ->
    # 7.5x underestimate for strings at K=N.
    _has_string_subset = any(
        pa.types.is_string(table.schema.field(c).type)
        or pa.types.is_large_string(table.schema.field(c).type)
        for c in subset_cols
    )
    _dedup_peak_mult = 12 if _has_string_subset else 2.5
    memory_guard.check_state(
        _dedup_peak_mult * table.nbytes,
        name=f"dedup.arrow_native({partition_path.name})",
    )

    # Dictionary-encode subset for compact hash.
    encoded_cols = {
        c: pc.dictionary_encode(table.column(c)) for c in subset_cols
    }
    working = pa.table({
        **{c: encoded_cols[c] for c in subset_cols},
        "row_id": table.column("row_id"),
    })

    if keep in ("first", "last"):
        agg_fn = "min" if keep == "first" else "max"
        agg = working.group_by(subset_cols).aggregate([("row_id", agg_fn)])
        keep_rids = agg.column(f"row_id_{agg_fn}")
    elif keep is False:
        agg = working.group_by(subset_cols).aggregate([
            ("row_id", "count"), ("row_id", "min"),
        ])
        single = agg.filter(pc.equal(agg.column("row_id_count"), 1))
        keep_rids = single.column("row_id_min")
    else:
        raise ValueError(f"dedup: unsupported keep={keep!r}")

    mask = pc.is_in(table.column("row_id"), value_set=keep_rids)
    result = table.filter(mask)
    return result.select([f.name for f in out_fields])


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
    output: str = "auto",
) -> "pa.Table | AutoResult":
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

    out_schema = pa.schema(
        [f for f in first.schema if f.name != "row_id"]
    )

    if output not in ("memory", "auto", "disk"):
        raise ValueError(
            f"drop_duplicates: output must be 'memory', 'auto' or "
            f"'disk'; got {output!r}")

    if output == "memory":
        results: list[pa.Table] = []
        for p in paths:
            if not p.exists() or p.stat().st_size == 0:
                continue
            t = _local_dedup(p, subset, keep, memory_guard)
            if t.num_rows > 0:
                results.append(t)

        if not results:
            return pa.Table.from_batches([], schema=out_schema)

        return pa.concat_tables(results, promote_options="default")

    # output in ("auto", "disk") — SpillBuffer path.
    # v0.9.92-fix: schema from _local_dedup result, not input out_schema.
    # _local_dedup returns hash_shuffle partition order
    # ([*subset, *value_cols]), which may differ from first.schema order.
    # ArrowIPCBatchWriter requires exact schema match → prior code
    # raised ArrowInvalid on any input where hash_shuffle reordered
    # columns (e.g., subset=["v"] on input schema=["a","b","v"]).
    # Mirror memory path: all _local_dedup calls on one hash_shuffle
    # produce identical schema (same reader.schema), so first non-empty
    # t.schema is canonical. Defensive cast if partition schema ever
    # diverges (should not happen).
    buf: "SpillBuffer | None" = None
    buf_schema = None
    for p in paths:
        if not p.exists() or p.stat().st_size == 0:
            continue
        t = _local_dedup(p, subset, keep, memory_guard)
        if t.num_rows > 0:
            if buf is None:
                buf_schema = t.schema
                buf = SpillBuffer(
                    guard=memory_guard, schema=buf_schema,
                    name="dedup",
                    spill_ratio=0.0 if output == "disk" else 0.5,
                )
            for b in t.to_batches():
                if buf_schema is not None and b.schema != buf_schema:
                    b = b.cast(buf_schema)
                buf.write_batch(b)

    if buf is None:
        return InMemoryResult(
            pa.Table.from_batches([], schema=out_schema))

    result = buf.finalize()
    if log:
        logger.info(
            "drop_duplicates done: %d rows (output=%s, spilled=%s)",
            len(result), output, result.is_spilled)
    return result
