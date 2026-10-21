"""Streaming top-k (v0.9.7a, R4: 3946× speedup benchmark).

Streaming top-k algorithm: per-batch in-memory top-k, then final
merge-sort-take on P×k rows. No disk I/O; peak RAM = batch +
O(P × k × row_size), independent of N beyond P = num_batches.

Reference (bench_r4.py, N=2M, k=10): current external_sort +
head → 45.79s / 10.8 MB; streaming top-k → 0.01s / 3.9 MB.
Speedup comes from eliminating disk I/O (external_sort writes
full sorted stream to disk); select_k_unstable adds a further
O(N) vs O(N log N) partial-sort win when available.

Semantics:
  - nulls in `column` sort to end (null_placement="at_end"),
    matching pandas `nlargest`/`nsmallest` (nulls skipped if
    n < n_non_null; returned if n >= n_total).
  - Stable tie-break via `__cadw_row_id` (matches CADW
    `sort_values` external_sort row_id tie-break).
  - Returns pa.Table with ORIGINAL schema (no __cadw_row_id).
"""
from __future__ import annotations

from typing import Iterable

import pyarrow as pa
import pyarrow.compute as pc

from cadw.core.memory_guard import MemoryGuard


_RESERVED_ROW_ID_COL = "__cadw_row_id"
_KINDS = ("largest", "smallest")


def _per_batch_topk(
    table: pa.Table,
    column: str,
    k: int,
    descending: bool,
) -> pa.Table:
    """In-memory top-k of one batch. Stable via __cadw_row_id.

    Prefer pc.select_k_unstable (O(N) partial sort); fallback to
    pc.sort_indices + slice (works for all Arrow types / nulls).
    """
    order = "descending" if descending else "ascending"
    k = min(k, table.num_rows)
    if k <= 0:
        return table.slice(0, 0)
    sort_keys_select = [(column, order), (_RESERVED_ROW_ID_COL, "ascending")]
    sort_keys_full = [
        (column, order, "at_end"),
        (_RESERVED_ROW_ID_COL, "ascending", "at_end"),
    ]
    if hasattr(pc, "select_k_unstable"):
        try:
            idx = pc.select_k_unstable(
                table, k=k, sort_keys=sort_keys_select,
            )
            return table.take(idx)
        except Exception:
            # Fall through to full sort — see module docstring.
            pass
    full_idx = pc.sort_indices(table, sort_keys=sort_keys_full)
    return table.take(full_idx.slice(0, k))


def topk_to_disk(
    batch_iter: Iterable[pa.RecordBatch],
    *,
    column: str,
    n: int,
    kind: str,
    memory_guard: MemoryGuard,
) -> pa.Table:
    """Streaming top-n. Returns pa.Table (original schema).

    Args:
        batch_iter: input batches.
        column: sort key column (must be present in every batch).
        n: target count (>= 1).
        kind: "largest" (descending) | "smallest" (ascending).
        memory_guard: budget + per-batch/state checks.

    Peak RAM = batch_size × row_size (checked per batch) +
    O(P × n × row_size) accumulator (checked after each batch;
    P = num_batches). Independent of N beyond P. Fail-fast
    MemoryLimitExceeded when accumulator estimate exceeds
    state_limit; use smaller n or increase budget.

    Empty input → Table with schema from batch_iter? If empty,
    raises ValueError (no schema without first batch).
    """
    if kind not in _KINDS:
        raise ValueError(
            f"topk_to_disk: kind must be one of {_KINDS}; got {kind!r}"
        )
    if not isinstance(n, int) or n < 1:
        raise ValueError(
            f"topk_to_disk: n must be int >= 1; got {n!r}"
        )

    descending = kind == "largest"
    it = iter(batch_iter)
    try:
        first = next(it)
    except StopIteration as exc:
        raise ValueError("topk_to_disk: empty input") from exc

    if column not in first.schema.names:
        raise KeyError(
            f"topk_to_disk: column {column!r} not in schema "
            f"{first.schema.names}"
        )
    if _RESERVED_ROW_ID_COL in first.schema.names:
        raise ValueError(
            f"topk_to_disk: column name {_RESERVED_ROW_ID_COL!r} "
            f"is reserved for CADW internal tie-break. Rename it "
            f"before calling nlargest/nsmallest."
        )
    out_schema = first.schema

    per_batch: list[pa.Table] = []
    row_id_offset = 0

    def _process(batch: pa.RecordBatch, offset: int) -> None:
        nonlocal row_id_offset
        memory_guard.check_batch(batch, name=f"topk.{kind}.input")
        if batch.num_rows == 0:
            row_id_offset = offset
            return
        rid = pa.array(
            range(offset, offset + batch.num_rows), type=pa.uint64(),
        )
        b = batch.append_column(_RESERVED_ROW_ID_COL, rid)
        t = pa.Table.from_batches([b])
        top = _per_batch_topk(t, column, n, descending)
        per_batch.append(top)
        row_id_offset = offset + batch.num_rows

        # Fail-fast: accumulator + 2× overhead before adding more.
        acc_bytes = sum(tb.nbytes for tb in per_batch)
        memory_guard.check_state(
            2 * acc_bytes,
            name=f"topk.{kind}.accum(P={len(per_batch)},n={n})",
        )

    _process(first, 0)
    for b in it:
        _process(b, row_id_offset)

    if not per_batch:
        return pa.Table.from_batches([], schema=out_schema)

    merged = pa.concat_tables(per_batch)
    order = "descending" if descending else "ascending"
    final_idx = pc.sort_indices(
        merged,
        sort_keys=[
            (column, order, "at_end"),
            (_RESERVED_ROW_ID_COL, "ascending", "at_end"),
        ],
    )
    top_final = merged.take(final_idx.slice(0, n))
    # Drop internal tie-break column; return original schema order.
    return top_final.select(out_schema.names)


__all__ = ["topk_to_disk"]

