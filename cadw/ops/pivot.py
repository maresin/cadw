"""Arrow-native streaming pivot_table (v0.9.6c2, closes R6).

Peak RAM: O(sort_buffer) + O(long_table) + O(K_columns × key_size)
+ O(BATCH_SIZE × K_columns × value_size). Independent of K_index.
No pandas. No dense K_index × K_columns matrix in RAM.

R6 reference: 200k rows / 300x300 dense, previous pandas path
= 26.8 MB RSS (37× useful 720 KB). New streaming path peak ~10 MB
(bounded by groupby intermediate + sort buffer + output batch).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops.groupby import groupby_aggregate
from cadw.ops.sort import external_sort
from cadw.core._k_est import estimate_k_sample_arrow


_VALID_AGG = ("mean", "sum", "count", "min", "max")
_BATCH_ROWS = 1000


def _column_dtype_for_agg(
    aggfunc: str, values_type: pa.DataType,
) -> pa.DataType:
    """Output dtype per aggfunc (null-safe)."""
    if aggfunc == "count":
        return pa.int64()
    if aggfunc in ("mean", "sum"):
        return pa.float64()
    # min / max: preserve input dtype (nullable)
    return values_type


def _column_key_size(dtype: pa.DataType) -> int:
    """Conservative per-value bytes for K_columns check_state."""
    if (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)
            or pa.types.is_timestamp(dtype) or pa.types.is_date(dtype)
            or pa.types.is_time(dtype) or pa.types.is_boolean(dtype)):
        return 8
    if pa.types.is_string(dtype) or pa.types.is_binary(dtype):
        return 32
    return 32


def pivot_to_disk(
    batch_iter: Iterable[pa.RecordBatch],
    *,
    index_cols: Sequence[str],
    columns_col: str,
    values_col: str,
    values_type: pa.DataType,
    aggfunc: str,
    memory_guard: MemoryGuard,
    spill_dir: Path,
) -> tuple[pa.Schema, Path]:
    """Streaming pivot. Returns (out_schema, out_path).

    Caller (CadwDataFrame.pivot_table) reads via ArrowIPCBatchReader
    and wraps in Dataset.from_batches. Intermediate IPC removed by
    caller after Dataset construction.

    Preconditions (validated in caller):
      - index_cols non-empty, columns_col/values_col exist
      - aggfunc in _VALID_AGG
      - values_col type-castable (count allows any, others numeric
        for mean/sum — min/max accept any orderable type)

    Dropna: null in columns_col → row dropped (matches pandas
    pivot_table dropna=True default). Null values → cell null.
    """
    if aggfunc not in _VALID_AGG:
        raise ValueError(
            f"pivot_to_disk: aggfunc must be one of {_VALID_AGG}; "
            f"got {aggfunc!r}"
        )
    index_cols = list(index_cols)
    if not index_cols:
        raise ValueError("pivot_to_disk: index_cols must be non-empty")
    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    value_col_name = f"{values_col}_{aggfunc}"

    def _filtered():
        for b in batch_iter:
            memory_guard.check_batch(b, name="pivot.filter")
            if columns_col in b.schema.names:
                mask = pc.is_valid(b.column(columns_col))
                b = b.filter(mask)
            if b.num_rows > 0:
                yield b

    long_table = groupby_aggregate(
        _filtered(),
        by_columns=index_cols + [columns_col],
        agg_columns=[values_col],
        agg_funcs=[aggfunc],
        memory_guard=memory_guard,
        spill_dir=spill_dir / "groupby",
        log=False,
    )
    if long_table.num_rows == 0:
        raise ValueError(
            "pivot_to_disk: no rows after filtering null columns; "
            "cannot pivot empty data"
        )

    long_batches = list(long_table.to_batches())
    sample = long_batches[0]
    k_columns_est = estimate_k_sample_arrow(sample, [columns_col])
    key_bytes = _column_key_size(
        long_table.schema.field(columns_col).type
    )
    memory_guard.check_state(
        k_columns_est * key_bytes * 2,
        name=f"pivot.columns_est(K~{k_columns_est})",
    )

    sorted_path = external_sort(
        iter(long_batches),
        key_columns=index_cols + [columns_col],
        value_columns=index_cols + [columns_col, value_col_name],
        memory_guard=memory_guard,
        spill_dir=spill_dir / "sort",
        ascending=True,
        log=False,
    )

    columns_seen: dict = {}
    for b in ArrowIPCBatchReader(str(sorted_path), [columns_col]):
        memory_guard.check_batch(b, name="pivot.scan_columns")
        for v in b.column(columns_col).to_pylist():
            if v is not None and v not in columns_seen:
                columns_seen[v] = len(columns_seen)
    try:
        columns_sorted = sorted(columns_seen.keys())
    except TypeError:
        columns_sorted = sorted(columns_seen.keys(), key=str)
    k_columns = len(columns_sorted)
    col_to_pos = {v: i for i, v in enumerate(columns_sorted)}

    value_dtype = _column_dtype_for_agg(aggfunc, values_type)
    out_fields = [
        long_table.schema.field(c) for c in index_cols
    ]
    out_col_names: list = []
    index_set = set(index_cols)
    for c in columns_sorted:
        name = str(c)
        if name in index_set:
            raise ValueError(
                f"pivot_to_disk: output column {name!r} collides "
                f"with index col {name!r}. Rename index col or "
                f"columns values."
            )
        out_col_names.append(name)
        out_fields.append(pa.field(name, value_dtype, nullable=True))
    out_schema = pa.schema(out_fields)

    value_size = _column_key_size(value_dtype)
    batch_bytes_est = _BATCH_ROWS * (k_columns * value_size + 64)
    memory_guard.check_state(
        batch_bytes_est,
        name=f"pivot.output_batch(K={k_columns})",
    )

    out_path = spill_dir / "pivoted.arrow"
    writer = ArrowIPCBatchWriter(str(out_path), out_schema)
    buf: list[dict] = []

    def _flush():
        nonlocal buf
        if not buf:
            return
        rb = pa.RecordBatch.from_pylist(buf, schema=out_schema)
        writer.write_batch(rb)
        buf = []

    _UNSET = object()
    current_key = _UNSET
    current_row: dict = {}

    def _emit(key, row_dict):
        nonlocal buf
        out = dict(zip(index_cols, key))
        for i, name in enumerate(out_col_names):
            out[name] = row_dict.get(i)
        buf.append(out)
        if len(buf) >= _BATCH_ROWS:
            _flush()

    try:
        for b in ArrowIPCBatchReader(str(sorted_path), None):
            # v0.9.22b (AUDIT_ML §1.6): extra_factor=5.0 —
            # per-row to_pylist on K_index cols + columns + values.
            memory_guard.check_batch(
                b, name="pivot.stream_sorted", extra_factor=5.0,
            )
            idx_arrs = [b.column(c).to_pylist() for c in index_cols]
            col_arr = b.column(columns_col).to_pylist()
            v_arr = b.column(value_col_name).to_pylist()
            n = b.num_rows
            for i in range(n):
                key = tuple(arr[i] for arr in idx_arrs)
                if key != current_key:
                    if current_key is not _UNSET:
                        _emit(current_key, current_row)
                    current_key = key
                    current_row = {}
                cval = col_arr[i]
                if cval is None:
                    continue
                current_row[col_to_pos[cval]] = v_arr[i]
        if current_key is not _UNSET:
            _emit(current_key, current_row)
        _flush()
    finally:
        writer.close()

    return out_schema, out_path


__all__ = ["pivot_to_disk"]

