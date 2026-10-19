"""Resample (v0.8.7): bucket aggregation by timestamp/integer `on`.

Algorithm (validated by scripts/proto_resample.py):
  1. Sort by `on` (row_id tie-break from external_sort's heap).
  2. Stream sorted; bucket = `(t_sec // freq) * freq`.
     On bucket change: flush running state, reset, accumulate.
  3. Write output batches (BATCH=1000 buckets) to IPC.

Peak RAM = batch + O(#cols) state + 1 output batch. Independent of N and
bucket size (running aggregate, not `list(values)`).

Semantics:
  - Empty bucket (gap in `on`) → skip (no NaN fill).
  - Null in `on` → skip row (cannot bucket).
  - All-null agg bucket → skip row (all columns None).
  - Null in agg col → skip value (matches pandas skipna).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops.sort import external_sort


_FREQ_RE = re.compile(r'^(\d+)(s|min|m|h|D)$')
_FREQ_UNITS = {'s': 1, 'min': 60, 'm': 60, 'h': 3600, 'D': 86400}
_VALID_AGGS = ('sum', 'mean', 'min', 'max', 'count')
_OUTPUT_BATCH = 1000
_TS_UNIT_FACTOR = {'s': 1, 'ms': 1_000, 'us': 1_000_000, 'ns': 1_000_000_000}


def parse_freq(freq) -> int:
    """Parse freq → seconds. Accepts int or str: '5s', '1min'/'1m', '1h', '1D'."""
    if isinstance(freq, bool):
        raise TypeError("resample: freq must be int or str, not bool")
    if isinstance(freq, int):
        if freq <= 0:
            raise ValueError(f"resample: freq must be > 0; got {freq}")
        return freq
    if not isinstance(freq, str):
        raise TypeError(
            f"resample: freq must be int or str; got {type(freq).__name__}"
        )
    m = _FREQ_RE.match(freq)
    if not m:
        raise ValueError(
            f"resample: freq={freq!r} invalid; expected e.g. "
            f"'5s', '1min', '1h', '1D'"
        )
    n = int(m.group(1))
    unit = m.group(2)
    if n <= 0:
        raise ValueError(
            f"resample: freq multiplier must be > 0; got {n}"
        )
    return n * _FREQ_UNITS[unit]


def _to_unix_seconds(arr: pa.Array) -> pa.Array:
    """Cast timestamp/date/int to int64 unix seconds. Nulls preserved.

    v0.8.7-fix: pyarrow 26 does NOT implement `pc.cast(date32, int64)`
    (`Unsupported cast from date32[day] to int64`). But `cast(date32,
    int32)` works (same bit-width reinterpretation). Route via int32.

    date32 = int32 days since epoch. date64 = int64 ms since epoch.
    """
    t = arr.type
    if pa.types.is_timestamp(t):
        factor = _TS_UNIT_FACTOR[t.unit]
        int_arr = pc.cast(arr, pa.int64())
        return pc.cast(
            pc.floor(pc.divide(int_arr, pa.scalar(factor, type=pa.int64()))),
            pa.int64(),
        )
    if pa.types.is_date32(t):
        # v0.8.7-fix: cast via int32 (pyarrow has no date32→int64 kernel).
        int32_arr = pc.cast(arr, pa.int32())
        int64_arr = pc.cast(int32_arr, pa.int64())
        return pc.multiply(int64_arr, pa.scalar(86400, type=pa.int64()))
    if pa.types.is_date64(t):
        # date64 = int64 ms since epoch; cast to int64 supported.
        int64_ms = pc.cast(arr, pa.int64())
        return pc.cast(
            pc.floor(pc.divide(int64_ms, pa.scalar(1000, type=pa.int64()))),
            pa.int64(),
        )
    if pa.types.is_uint64(t):
        raise TypeError(
            "resample: on column uint64 not supported (overflow risk); "
            "cast to int64 or timestamp"
        )
    if pa.types.is_integer(t):
        return pc.cast(arr, pa.int64())
    raise TypeError(
        f"resample: on column type {t} not supported; expected timestamp, "
        f"date32/date64, or integer"
    )


class _ResampleState:
    """Running aggregate per column: sum/count/min/max. O(1) push."""

    __slots__ = ("agg", "s", "n", "mn", "mx")

    def __init__(self, agg: str):
        self.agg = agg
        self.s = 0.0
        self.n = 0
        self.mn = None
        self.mx = None

    def push(self, v):
        if v is None:
            return
        self.n += 1
        if self.agg in ("sum", "mean"):
            self.s += float(v)
        elif self.agg == "min":
            self.mn = v if self.mn is None else (v if v < self.mn else self.mn)
        elif self.agg == "max":
            self.mx = v if self.mx is None else (v if v > self.mx else self.mx)

    def result(self):
        if self.n == 0:
            return None
        if self.agg == "count":
            return self.n
        if self.agg == "sum":
            return self.s
        if self.agg == "mean":
            return self.s / self.n
        if self.agg == "min":
            return self.mn
        if self.agg == "max":
            return self.mx
        raise RuntimeError(f"_ResampleState: unknown agg {self.agg!r}")

    def reset(self):
        self.s = 0.0
        self.n = 0
        self.mn = None
        self.mx = None


def _bucket_to_arrow(
    bucket_seconds: list[int], on_type: pa.DataType,
) -> pa.Array:
    """Convert list of int64 bucket-seconds → Arrow array of `on_type`.

    v0.8.7-fix: `date32` output via int32 intermediate (same rationale
    as input — no int64→date32 kernel in pyarrow 26; int64→int32 works).
    `date64` direct (int64 ms).
    """
    int_arr = pa.array(bucket_seconds, type=pa.int64())
    if pa.types.is_timestamp(on_type):
        factor = _TS_UNIT_FACTOR[on_type.unit]
        scaled = pc.multiply(int_arr, pa.scalar(factor, type=pa.int64()))
        return pc.cast(scaled, on_type)
    if pa.types.is_date32(on_type):
        days = pc.floor(pc.divide(int_arr, pa.scalar(86400, type=pa.int64())))
        int32_days = pc.cast(days, pa.int32())
        return pc.cast(int32_days, pa.date32())
    if pa.types.is_date64(on_type):
        ms = pc.multiply(int_arr, pa.scalar(1000, type=pa.int64()))
        return pc.cast(ms, pa.date64())
    return pc.cast(int_arr, on_type)


def resample_to_disk(
    batch_iter: Iterable[pa.RecordBatch],
    *,
    on: str,
    on_type: pa.DataType,
    freq_seconds: int,
    agg: str,
    columns: Sequence[str],
    col_types: dict[str, pa.DataType],
    memory_guard: MemoryGuard,
    spill_dir: Path,
) -> tuple[pa.Schema, Path]:
    """Core resample: sort by `on`, stream, bucket transition, write IPC.

    Returns (out_schema, out_path). Caller reads via ArrowIPCBatchReader
    and deletes out_path.
    """
    if agg not in _VALID_AGGS:
        raise ValueError(
            f"resample: agg must be one of {_VALID_AGGS}; got {agg!r}"
        )
    if freq_seconds <= 0:
        raise ValueError(f"resample: freq must be > 0; got {freq_seconds}")

    cols = list(columns)

    # Output schema
    out_fields = [pa.field(on, on_type)]
    for c in cols:
        in_t = col_types[c]
        if agg in ("sum", "mean"):
            ot = pa.float64()
        elif agg == "count":
            ot = pa.int64()
        else:  # min/max
            ot = in_t
        out_fields.append(pa.field(f"{c}_{agg}", ot, nullable=True))
    out_schema = pa.schema(out_fields)

    # Sort by `on`
    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)
    sorted_path = external_sort(
        batch_iter,
        key_columns=[on],
        value_columns=[on] + cols,
        memory_guard=memory_guard,
        spill_dir=spill_dir / "sort_by_on",
        ascending=True,
        log=False,
    )

    out_path = spill_dir / "resampled.arrow"
    writer = ArrowIPCBatchWriter(str(out_path), out_schema)

    states = {c: _ResampleState(agg) for c in cols}
    cur_bucket_sec: int | None = None

    # Output buffer (batch of buckets, not rows)
    buf_on: list[int] = []  # raw int64 seconds; convert at flush
    buf_cols: dict[str, list] = {c: [] for c in cols}

    def _flush_output():
        if not buf_on:
            return
        bucket_arr = _bucket_to_arrow(buf_on, on_type)
        arrays = [bucket_arr]
        for c in cols:
            ftype = out_schema.field(f"{c}_{agg}").type
            arrays.append(pa.array(buf_cols[c], type=ftype))
        writer.write_batch(
            pa.RecordBatch.from_arrays(arrays, schema=out_schema)
        )
        buf_on.clear()
        for c in cols:
            buf_cols[c].clear()

    def _flush_bucket():
        if cur_bucket_sec is None:
            return
        results = {c: states[c].result() for c in cols}
        if all(v is None for v in results.values()):
            return  # all-null bucket, skip (matches pandas dropna=True)
        buf_on.append(cur_bucket_sec)
        for c in cols:
            buf_cols[c].append(results[c])
        if len(buf_on) >= _OUTPUT_BATCH:
            _flush_output()

    try:
        for b in ArrowIPCBatchReader(str(sorted_path), None):
            memory_guard.check_batch(b, name="resample.sorted_input")
            ts_seconds = _to_unix_seconds(b.column(on))
            ts_list = ts_seconds.to_pylist()  # per-batch (OK, matches _RollState)
            col_vals = {c: b.column(c).to_pylist() for c in cols}
            n = b.num_rows
            for i in range(n):
                t = ts_list[i]
                if t is None:
                    continue  # null in on → skip row
                bkt = (t // freq_seconds) * freq_seconds
                if bkt != cur_bucket_sec:
                    _flush_bucket()
                    cur_bucket_sec = bkt
                    for s in states.values():
                        s.reset()
                for c in cols:
                    states[c].push(col_vals[c][i])
        _flush_bucket()
        _flush_output()
    finally:
        writer.close()

    return out_schema, out_path
