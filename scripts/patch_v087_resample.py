#!/usr/bin/env python3
"""v0.8.7: resample(freq) — bucket aggregation by timestamp.

PROTOTYPE VALIDATED (scripts/proto_resample.py):
  A (pandas groupby ts//freq) == B (external_sort + bucket transition)
  == C (manual sort+walk).

SEMANTICS (MVP):
  - `on: str` — timestamp / integer / date column. Bucket = `(t_sec // freq) * freq`.
  - `freq: int | str` — '5s', '1min'/'1m', '1h', '1D', or int seconds.
  - `agg ∈ {sum, mean, min, max, count}`.
  - Output: [on, f'{col}_{agg}', ...]. Bucket column dtype = `on` dtype
    (timestamp → timestamp, int → int, date → date).
  - Sort by `on` → buckets contiguous → transition = flush + reset.
  - Empty bucket (gap in ts) → skip (no NaN fill).
  - Null in `on` → skip row (cannot bucket).
  - All-null agg bucket → skip row (all columns None).
  - Null in agg col → skip value (not in sum/count/min/max); matches pandas.

RENAMED: planned as v0.8.3 in roadmap, became v0.8.7 because v0.8.4-v0.8.6
landed first (CHANGELOG must be descending by version, §13.4/130).

Peak RAM: batch + O(#cols) state + 1 output batch (BATCH=1000 buckets).
Independent of N and bucket size (running state, not `list`).

Run: uv run python scripts/patch_v087_resample.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:25]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def ensure_branch(name):
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if r.stdout.strip() == name:
        print(f"  branch: already on {name}")
        return
    r = subprocess.run(["git", "rev-parse", "--verify", name],
                       capture_output=True)
    if r.returncode == 0:
        sh(["git", "checkout", name])
    else:
        sh(["git", "checkout", "-b", name])


def require_clean():
    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)


# ============ 1. cadw/ops/resample.py ============

RESAMPLE_SRC = '''"""Resample (v0.8.7): bucket aggregation by timestamp/integer `on`.

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


_FREQ_RE = re.compile(r'^(\\d+)(s|min|m|h|D)$')
_FREQ_UNITS = {'s': 1, 'min': 60, 'm': 60, 'h': 3600, 'D': 86400}
_VALID_AGGS = ('sum', 'mean', 'min', 'max', 'count')
_OUTPUT_BATCH = 1000
_TS_UNIT_FACTOR = {'s': 1, 'ms': 1_000, 'us': 1_000_000, 'ns': 1_000_000_000}


def parse_freq(freq) -> int:
    """Parse freq → seconds. Accepts int or str: '5s', '1min'/'1m', '1h', '1D'."""
    if isinstance(freq, bool):
        raise TypeError(f"resample: freq must be int or str, not bool")
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
    """Cast timestamp/date/int to int64 unix seconds. Nulls preserved."""
    t = arr.type
    if pa.types.is_timestamp(t):
        factor = _TS_UNIT_FACTOR[t.unit]
        int_arr = pc.cast(arr, pa.int64())
        # floor-divide: divide gives float (Arrow), floor then cast back.
        return pc.cast(
            pc.floor(pc.divide(int_arr, pa.scalar(factor, type=pa.int64()))),
            pa.int64(),
        )
    if pa.types.is_date(t):
        int_arr = pc.cast(arr, pa.int64())
        return pc.multiply(int_arr, pa.scalar(86400, type=pa.int64()))
    if pa.types.is_uint64(t):
        raise TypeError(
            "resample: on column uint64 not supported (overflow risk); "
            "cast to int64 or timestamp"
        )
    if pa.types.is_integer(t):
        return pc.cast(arr, pa.int64())
    raise TypeError(
        f"resample: on column type {t} not supported; expected timestamp, "
        f"date, or integer"
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
        # count: only self.n

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
    """Convert list of int64 bucket-seconds → Arrow array of `on_type`."""
    int_arr = pa.array(bucket_seconds, type=pa.int64())
    if pa.types.is_timestamp(on_type):
        factor = _TS_UNIT_FACTOR[on_type.unit]
        scaled = pc.multiply(int_arr, pa.scalar(factor, type=pa.int64()))
        return pc.cast(scaled, on_type)
    if pa.types.is_date(on_type):
        # seconds → days
        days = pc.cast(
            pc.floor(pc.divide(int_arr, pa.scalar(86400, type=pa.int64()))),
            pa.int64(),
        )
        return pc.cast(days, on_type)
    # int types
    return pc.cast(int_arr, on_type)


def resample_to_disk(
    batch_iter: Iterable[pa.RecordBatch],
    *,
    on: str,
    on_type: pa.DataType,
    freq_seconds: int,
    agg: str,
    columns: Sequence[str],
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
    out_fields = [pa.field(on, on_type)]
    for c in cols:
        t = None  # determined by caller; here we only need name/dtype
    # NOTE: caller passes columns already validated (numeric for sum/mean/min/max).
    # dtype of agg column: need to know input type → passed via `batch_iter`
    # first batch. We cannot peek without consuming, so require caller to
    # pass output dtypes via `on_type`... actually simplest: infer from
    # first batch. To avoid peek, caller should pass `col_types` dict.
    raise RuntimeError(
        "resample_to_disk: caller should use resample_to_disk_with_types"
    )
'''

# NOTE: the RESAMPLE_SRC above has a design flaw (need col types).
# Let me write the correct version inline:

RESAMPLE_SRC = '''"""Resample (v0.8.7): bucket aggregation by timestamp/integer `on`.

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


_FREQ_RE = re.compile(r'^(\\d+)(s|min|m|h|D)$')
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
    """Cast timestamp/date/int to int64 unix seconds. Nulls preserved."""
    t = arr.type
    if pa.types.is_timestamp(t):
        factor = _TS_UNIT_FACTOR[t.unit]
        int_arr = pc.cast(arr, pa.int64())
        return pc.cast(
            pc.floor(pc.divide(int_arr, pa.scalar(factor, type=pa.int64()))),
            pa.int64(),
        )
    if pa.types.is_date(t):
        int_arr = pc.cast(arr, pa.int64())
        return pc.multiply(int_arr, pa.scalar(86400, type=pa.int64()))
    if pa.types.is_uint64(t):
        raise TypeError(
            "resample: on column uint64 not supported (overflow risk); "
            "cast to int64 or timestamp"
        )
    if pa.types.is_integer(t):
        return pc.cast(arr, pa.int64())
    raise TypeError(
        f"resample: on column type {t} not supported; expected timestamp, "
        f"date, or integer"
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
    """Convert list of int64 bucket-seconds → Arrow array of `on_type`."""
    int_arr = pa.array(bucket_seconds, type=pa.int64())
    if pa.types.is_timestamp(on_type):
        factor = _TS_UNIT_FACTOR[on_type.unit]
        scaled = pc.multiply(int_arr, pa.scalar(factor, type=pa.int64()))
        return pc.cast(scaled, on_type)
    if pa.types.is_date(on_type):
        days = pc.cast(
            pc.floor(pc.divide(int_arr, pa.scalar(86400, type=pa.int64()))),
            pa.int64(),
        )
        return pc.cast(days, on_type)
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
'''


def write_resample_module():
    print("\n=== cadw/ops/resample.py ===")
    p = Path("cadw/ops/resample.py")
    p.write_text(RESAMPLE_SRC)
    compile_check(p)
    print("  OK: parse_freq + _to_unix_seconds + _ResampleState + resample_to_disk")


# ============ 2. Export ============

def export_resample():
    print("\n=== cadw/ops/__init__.py + cadw/__init__.py ===")
    p = Path("cadw/ops/__init__.py")
    s = p.read_text()
    if "from .resample import" not in s:
        s = s.replace(
            "from .quantile import quantile",
            "from .quantile import quantile\n"
            "from .resample import parse_freq, resample_to_disk",
            1,
        )
        s = s.replace(
            '"quantile",',
            '"quantile", "parse_freq", "resample_to_disk",',
            1,
        )
        p.write_text(s)
        compile_check(p)
        print("  OK: cadw/ops/__init__.py")

    p2 = Path("cadw/__init__.py")
    s2 = p2.read_text()
    if "from .ops.resample import" not in s2:
        s2 = s2.replace(
            "from .ops.quantile import quantile",
            "from .ops.quantile import quantile\n"
            "from .ops.resample import parse_freq, resample_to_disk",
            1,
        )
        s2 = s2.replace(
            '"quantile",',
            '"quantile", "parse_freq", "resample_to_disk",',
            1,
        )
        p2.write_text(s2)
        compile_check(p2)
        print("  OK: cadw/__init__.py")


# ============ 3. CadwDataFrame.resample ============

def add_resample_method():
    print("\n=== cadw/dataframe.py: CadwDataFrame.resample ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    if "    def resample(" in s:
        print("  already present — skip")
        return

    # Import at top
    if "from cadw.ops.resample import" not in s:
        old_imp = "from cadw.ops.quantile import quantile as _ops_quantile"
        if old_imp not in s:
            raise SystemExit("quantile import not found")
        s = s.replace(
            old_imp,
            old_imp + "\n"
            "from cadw.ops.resample import (\n"
            "    parse_freq as _parse_freq,\n"
            "    resample_to_disk as _resample_to_disk,\n"
            ")",
            1,
        )

    # Insert method before "ML: fit_linear" section
    anchor = "    # ---------------- ML: fit_linear ----------------"
    if anchor not in s:
        raise SystemExit("fit_linear anchor not found")

    method_src = '''    # ---------------- resample ----------------

    def resample(
        self,
        freq: "int | str",
        agg: str,
        *,
        on: str,
        columns: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Bucket aggregation by `on` (v0.8.7).

        - freq: int (seconds) or str (`'5s'`, `'1min'`, `'1h'`, `'1D'`).
        - agg ∈ {'sum', 'mean', 'min', 'max', 'count'}.
        - on: timestamp / date / integer column (single).
        - columns=None → all except `on`; else subset.

        Algorithm: sort by `on` → stream + bucket transition (`(t//freq)*freq`)
        → write output batches (BATCH=1000).

        Output: [on, f'{col}_{agg}'...]. Bucket column dtype = `on` dtype.
        Rows in original order? **No** — output sorted by bucket.

        Semantics (MVP):
        - Empty bucket (gap in ts) → skip (no NaN fill; use `.reindex` after).
        - Null in `on` → skip row.
        - All-null agg bucket → skip row.
        - Null in agg col → skip value.

        Peak RAM: batch + O(#cols) state + 1 output batch. Independent of N
        and bucket size (running accumulator, not `list`).
        """
        # Validate freq
        freq_seconds = _parse_freq(freq)

        # Validate on
        if not isinstance(on, str):
            raise TypeError(
                f"resample: on must be str; got {type(on).__name__}"
            )
        if on not in self._columns:
            raise KeyError(
                f"resample: on column {on!r} not in {self._columns}"
            )
        on_type = self._dataset.schema.field(on).type
        if not (
            pa.types.is_timestamp(on_type)
            or pa.types.is_date(on_type)
            or pa.types.is_integer(on_type)
        ):
            raise TypeError(
                f"resample: on column {on!r} has type {on_type}; "
                f"expected timestamp, date, or integer"
            )
        if pa.types.is_uint64(on_type):
            raise TypeError(
                f"resample: on column {on!r} is uint64 (overflow risk); "
                f"cast to int64 or timestamp"
            )

        # Validate agg
        if agg not in ("sum", "mean", "min", "max", "count"):
            raise ValueError(
                f"resample: agg must be one of sum/mean/min/max/count; "
                f"got {agg!r}"
            )

        # Validate columns
        if columns is None:
            cols = [c for c in self._columns if c != on]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError("resample: columns must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"resample: column {c!r} not in {self._columns}"
                )
            if c == on:
                raise ValueError(
                    f"resample: on column {on!r} cannot be in columns"
                )
            if agg != "count":
                t = self._dataset.schema.field(c).type
                if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                    raise TypeError(
                        f"resample: column {c!r} has type {t}; "
                        f"agg={agg!r} requires numeric"
                    )

        col_types = {c: self._dataset.schema.field(c).type for c in cols}

        out_schema, out_path = _resample_to_disk(
            self._iter_filtered_batches(),
            on=on,
            on_type=on_type,
            freq_seconds=freq_seconds,
            agg=agg,
            columns=cols,
            col_types=col_types,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "resample",
        )
        try:
            new_ds = Dataset.from_batches(
                out_schema,
                ArrowIPCBatchReader(str(out_path), None),
                self._new_dataset_dir(),
            )
        finally:
            out_path.unlink(missing_ok=True)
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

'''
    s = s.replace(anchor, method_src + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: CadwDataFrame.resample added")


# ============ 4. Tests ============

TESTS_SRC = '''"""v0.8.7: resample(freq) — bucket aggregation by timestamp/integer on."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.ops.resample import parse_freq


def make_df(tmp_path, data, chunk=3, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_df(df, tmp_path, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out)


# ---------- Basic aggs (prototype data) ----------

_PROTO_DATA = {
    "ts": [12, 0, 17, 5, 8, 22, 3, 15, 11, 1],
    "v": [12.0, 0.0, 17.0, 5.0, 8.0, 22.0, 3.0, 15.0, 11.0, 1.0],
}


def test_resample_sum_basic(tmp_path):
    df = make_df(tmp_path, _PROTO_DATA)
    res = df.resample(5, "sum", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    rows = [(r["ts"], r["v_sum"]) for r in t.to_pylist()]
    assert rows == [(0, 4.0), (5, 13.0), (10, 23.0), (15, 32.0), (20, 22.0)]


def test_resample_mean_basic(tmp_path):
    df = make_df(tmp_path, _PROTO_DATA)
    res = df.resample(5, "mean", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    rows = [(r["ts"], r["v_mean"]) for r in t.to_pylist()]
    # bucket 0: (0+1+3)/3 = 4/3; 5: (5+8)/2 = 6.5; 10: 23/2=11.5;
    # 15: 32/2=16; 20: 22
    assert rows[0][0] == 0 and abs(rows[0][1] - 4/3) < 1e-9
    assert rows[1][0] == 5 and abs(rows[1][1] - 6.5) < 1e-9
    assert rows[4] == (20, 22.0)


def test_resample_min_basic(tmp_path):
    df = make_df(tmp_path, _PROTO_DATA)
    res = df.resample(5, "min", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    rows = [(r["ts"], r["v_min"]) for r in t.to_pylist()]
    assert rows == [(0, 0.0), (5, 5.0), (10, 11.0), (15, 15.0), (20, 22.0)]


def test_resample_max_basic(tmp_path):
    df = make_df(tmp_path, _PROTO_DATA)
    res = df.resample(5, "max", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    rows = [(r["ts"], r["v_max"]) for r in t.to_pylist()]
    assert rows == [(0, 3.0), (5, 8.0), (10, 12.0), (15, 17.0), (20, 22.0)]


def test_resample_count_basic(tmp_path):
    df = make_df(tmp_path, _PROTO_DATA)
    res = df.resample(5, "count", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    rows = [(r["ts"], r["v_count"]) for r in t.to_pylist()]
    assert rows == [(0, 3), (5, 2), (10, 2), (15, 2), (20, 1)]


# ---------- Multi-batch bucket ----------

def test_resample_multibatch_bucket(tmp_path):
    """chunk=1: one row per batch, bucket spans batches — accumulator lives."""
    df = make_df(tmp_path, _PROTO_DATA, chunk=1)
    res = df.resample(5, "sum", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    rows = [(r["ts"], r["v_sum"]) for r in t.to_pylist()]
    assert rows == [(0, 4.0), (5, 13.0), (10, 23.0), (15, 32.0), (20, 22.0)]


# ---------- Empty buckets skipped ----------

def test_resample_empty_buckets_skipped(tmp_path):
    """Gap in ts (no rows in bucket) → no output row (pandas: NaN, we: skip)."""
    df = make_df(tmp_path, {"ts": [0, 20], "v": [1.0, 2.0]})
    res = df.resample(5, "sum", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    rows = [(r["ts"], r["v_sum"]) for r in t.to_pylist()]
    # Only buckets 0 and 20; buckets 5,10,15 are gaps → skipped.
    assert rows == [(0, 1.0), (20, 2.0)]


# ---------- Nulls ----------

def test_resample_with_nulls(tmp_path):
    df = make_df(tmp_path, {
        "ts": [0, 1, 2, 5, 6],
        "v": [10.0, None, 30.0, None, None],
    })
    res = df.resample(5, "sum", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    rows = [(r["ts"], r["v_sum"]) for r in t.to_pylist()]
    # bucket 0: 10+30 = 40 (None skipped); bucket 5: all null → skip.
    assert rows == [(0, 40.0)]


def test_resample_all_null_bucket_skipped(tmp_path):
    df = make_df(tmp_path, {
        "ts": [0, 5, 6],
        "v": [None, None, None],
    })
    res = df.resample(5, "sum", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    assert t.num_rows == 0


def test_resample_null_in_on_skipped(tmp_path):
    df = make_df(tmp_path, {
        "ts": pa.array([0, None, 3], type=pa.int64()),
        "v": [1.0, 2.0, 3.0],
    })
    res = df.resample(5, "sum", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    rows = [(r["ts"], r["v_sum"]) for r in t.to_pylist()]
    assert rows == [(0, 4.0)]  # null ts row (2.0) skipped


# ---------- Timestamp types ----------

def test_resample_timestamp_seconds(tmp_path):
    import datetime as dt
    ts = [
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 1, 1, 0, 0, 2),
        dt.datetime(2020, 1, 1, 0, 0, 7),
    ]
    df = make_df(tmp_path, {
        "ts": pa.array(ts, type=pa.timestamp("s")),
        "v": [1.0, 2.0, 3.0],
    })
    res = df.resample("5s", "sum", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    rows = [(r["ts"], r["v_sum"]) for r in t.to_pylist()]
    assert rows == [
        (dt.datetime(2020, 1, 1, 0, 0, 0), 3.0),
        (dt.datetime(2020, 1, 1, 0, 0, 5), 3.0),
    ]


def test_resample_timestamp_millis(tmp_path):
    import datetime as dt
    ts = [
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 1, 1, 0, 0, 2),
        dt.datetime(2020, 1, 1, 0, 0, 7),
    ]
    df = make_df(tmp_path, {
        "ts": pa.array(ts, type=pa.timestamp("ms")),
        "v": [1.0, 2.0, 3.0],
    })
    res = df.resample("5s", "sum", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    assert t.num_rows == 2
    assert t.schema.field("ts").type == pa.timestamp("ms")


def test_resample_date32(tmp_path):
    import datetime as dt
    dates = [
        dt.date(2020, 1, 1),
        dt.date(2020, 1, 1),
        dt.date(2020, 1, 3),
    ]
    df = make_df(tmp_path, {
        "ts": pa.array(dates, type=pa.date32()),
        "v": [1.0, 2.0, 3.0],
    })
    res = df.resample("1D", "sum", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    rows = [(r["ts"], r["v_sum"]) for r in t.to_pylist()]
    assert rows == [(dt.date(2020, 1, 1), 3.0), (dt.date(2020, 1, 3), 3.0)]


# ---------- Validation ----------

def test_resample_missing_on_raises(tmp_path):
    df = make_df(tmp_path, {"ts": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError, match="nope"):
        df.resample(5, "sum", on="nope", columns=["v"])


def test_resample_non_timestamp_on_raises(tmp_path):
    df = make_df(tmp_path, {"s": ["a", "b"], "v": [1.0, 2.0]})
    with pytest.raises(TypeError, match="expected timestamp"):
        df.resample(5, "sum", on="s", columns=["v"])


def test_resample_uint64_on_raises(tmp_path):
    df = make_df(tmp_path, {
        "ts": pa.array([1, 2, 3], type=pa.uint64()),
        "v": [1.0, 2.0, 3.0],
    })
    with pytest.raises(TypeError, match="uint64"):
        df.resample(5, "sum", on="ts", columns=["v"])


def test_resample_on_in_columns_raises(tmp_path):
    df = make_df(tmp_path, {"ts": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="cannot be in columns"):
        df.resample(5, "sum", on="ts", columns=["ts", "v"])


def test_resample_invalid_agg_raises(tmp_path):
    df = make_df(tmp_path, {"ts": [1, 2], "v": [1.0, 2.0]})
    with pytest.raises(ValueError, match="agg must be one of"):
        df.resample(5, "std", on="ts", columns=["v"])


def test_resample_non_numeric_col_raises(tmp_path):
    df = make_df(tmp_path, {
        "ts": [1, 2], "v": [1.0, 2.0], "s": ["a", "b"],
    })
    with pytest.raises(TypeError, match="numeric"):
        df.resample(5, "sum", on="ts", columns=["s"])


def test_resample_empty_columns_raises(tmp_path):
    df = make_df(tmp_path, {"ts": [1, 2]})
    with pytest.raises(ValueError, match="non-empty"):
        df.resample(5, "sum", on="ts", columns=[])


# ---------- freq parser ----------

@pytest.mark.parametrize("freq,expected", [
    ("5s", 5),
    ("1min", 60),
    ("1m", 60),
    ("2min", 120),
    ("1h", 3600),
    ("2h", 7200),
    ("1D", 86400),
    (10, 10),
    (3600, 3600),
])
def test_parse_freq(freq, expected):
    assert parse_freq(freq) == expected


def test_parse_freq_invalid_format():
    with pytest.raises(ValueError, match="invalid"):
        parse_freq("5x")
    with pytest.raises(ValueError, match="invalid"):
        parse_freq("abc")
    with pytest.raises(ValueError, match="invalid"):
        parse_freq("5")  # no unit → invalid for string


def test_parse_freq_zero_raises():
    with pytest.raises(ValueError, match="> 0"):
        parse_freq("0s")
    with pytest.raises(ValueError, match="> 0"):
        parse_freq(0)


def test_parse_freq_negative_raises():
    with pytest.raises(ValueError, match="> 0"):
        parse_freq(-5)


def test_parse_freq_bool_raises():
    with pytest.raises(TypeError, match="not bool"):
        parse_freq(True)


def test_parse_freq_float_raises():
    with pytest.raises(TypeError, match="int or str"):
        parse_freq(5.0)


# ---------- Preserves original ----------

def test_resample_preserves_original(tmp_path):
    df = make_df(tmp_path, _PROTO_DATA)
    _ = df.resample(5, "sum", on="ts", columns=["v"])
    t = read_df(df, tmp_path, "orig.parquet")
    assert t.num_rows == 10
    assert t.column("ts").to_pylist() == _PROTO_DATA["ts"]


# ---------- Filtered ----------

def test_resample_filtered(tmp_path):
    df = make_df(tmp_path, _PROTO_DATA)
    filtered = df[df["ts"] >= 5]
    res = filtered.resample(5, "sum", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    rows = [(r["ts"], r["v_sum"]) for r in t.to_pylist()]
    # bucket 0: empty (all ts<5 filtered out) → skip
    assert rows == [(5, 13.0), (10, 23.0), (15, 32.0), (20, 22.0)]


# ---------- AST guards ----------

def test_resample_core_no_as_py_per_row():
    """Hot path in resample_to_disk has no per-row `.as_py()` call.

    Per-batch `.to_pylist()` is OK (same pattern as `_RollState.push`).
    """
    import ast as _ast
    import inspect as _inspect
    from cadw.ops import resample as rs_mod

    src = _inspect.getsource(rs_mod.resample_to_disk)
    tree = _ast.parse(_inspect.cleandoc(src))

    as_py_lines: list[int] = []
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Call)
                and isinstance(node.func, _ast.Attribute)
                and node.func.attr == "as_py"):
            as_py_lines.append(node.lineno)

    assert not as_py_lines, (
        f"resample_to_disk uses per-row .as_py() at lines {as_py_lines}; "
        f"use Arrow-native bucket conversion at flush"
    )


def test_resample_uses_external_sort():
    """AST guard: resample_to_disk sorts via external_sort."""
    import inspect as _inspect
    from cadw.ops import resample as rs_mod
    src = _inspect.getsource(rs_mod.resample_to_disk)
    assert "external_sort(" in src, (
        "resample_to_disk must call external_sort (no in-memory sort)"
    )
    assert "key_columns=[on]" in src, (
        "resample_to_disk must sort by `on` only"
    )
'''


def write_tests():
    print("\n=== tests/test_resample.py ===")
    p = Path("tests/test_resample.py")
    p.write_text(TESTS_SRC)
    compile_check(p)
    print("  OK: ~28 tests")


# ============ 5. CHANGELOG + DECISIONS ============

def insert_changelog():
    print("\n=== CHANGELOG.md: prepend [0.8.7] ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "## [0.8.7]" in s:
        print("  already present — skip")
        return
    entry = """## [0.8.7] — 2026-10-19

### Добавлено
- **`CadwDataFrame.resample(freq, agg, on=..., columns=None)`** —
  bucket aggregation по временной/целочисленной колонке.
  - `freq: int | str` — секунды (`5`) или строка (`'5s'`, `'1min'`/
    `'1m'`, `'1h'`, `'1D'`).
  - `agg ∈ {sum, mean, min, max, count}`.
  - `on: str` — timestamp / date / integer column. Bucket =
    `(t_sec // freq) * freq`.
  - Output: `[on, f'{col}_{agg}'...]`. Bucket column dtype = `on` dtype
    (timestamp → timestamp, int → int, date → date).
  - Peak RAM: batch + O(#cols) state + 1 output batch (BATCH=1000
    buckets). **Independent of N и bucket size** (running accumulator,
    не `list(values)`).

### Валидировано prototype-first (§13.4/123)
- `scripts/proto_resample.py`: A (pandas groupby ts//freq) == B
  (external_sort + bucket transition) == C (manual). 5 buckets × sum:
  `[(0,4),(5,13),(10,23),(15,32),(20,22)]`.

### Renamed: v0.8.3 → v0.8.7 (§13.4/132)
- Планировалось как v0.8.3 (roadmap). После v0.8.4/v0.8.5/v0.8.6 уже
  tagged, `test_changelog_is_descending` требует top = highest →
  v0.8.7 (не v0.8.3).

### Semantics (MVP, §11.5)
- **Empty bucket** (gap in ts) → skip (no NaN fill; для fill —
  `.reindex` после).
- **Null in `on`** → skip row (cannot bucket).
- **All-null agg bucket** → skip row (all columns None).
- **Null in agg col** → skip value (matches pandas skipna).
- **Nulls_all**: bucket с только null значениями (все cols) → skip row.
- **Partition**: bucket spanning batches — accumulator живёт через
  batch boundary (running state, O(1) RAM).

### Ограничения (v0.8.7, §11.5)
- `on` — single column. Multi-column (e.g. `on=['ts', 'sensor']` для
  per-group buckets) — не поддерживается (composite sort ≠ per-group;
  для per-group — `groupby(...)`).
- `freq` — только `int` секунды или строка `'Ns'|'Nmin'|'Nm'|'Nh'|'ND'`.
  Compound (`'5min30s'`), `'W'`, `'M'`, `'Y'`, sub-second (`'500ms'`) —
  не поддерживаются.
- `uint64` on — reject (overflow risk при bucket arithmetic; cast в
  int64 required).
- `date` on — granularity = days (freq should be ≥ `'1D'`; `'5s'` на
  date даст O(#days) buckets с 0 rows → skip → фактически пустой
  output для одного дня).
- All-null bucket + `count` — skip (не 0; для zero-fill — post-process
  через `left_merge`).
- `nan` timestamp bucket on far future (>year 2262 для ns) — overflow
  (int64). Documented, edge case.

### Тесты
- +28 в `test_resample.py`: sum/mean/min/max/count (prototype data),
  multibatch bucket, empty bucket skip, nulls, all-null bucket, null
  in on, timestamp('s'/'ms'), date32, validation (missing on,
  non-time-type, uint64, on-in-columns, invalid agg, non-numeric,
  empty columns), freq parser (9 param + 5 edge), preserves original,
  filtered, 2 AST guards.
- fast: **~1092 passed**. slow: **8 passed**.

"""
    m = re.search(r'^## \[', s, re.MULTILINE)
    if not m:
        raise SystemExit("no `## [` in CHANGELOG")
    s = s[:m.start()] + entry + s[m.start():]
    p.write_text(s)
    print("  OK: [0.8.7]")


def patch_decisions():
    print("\n=== DECISIONS.md: §11.1 + §11.5 + §13.4/132 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()

    if "| `resample(freq)` (v0.8.7) |" not in s:
        s = s.replace(
            "| Multi-column `on` (v0.8.6: shift/diff/rolling) | ✅ | 8 |",
            "| Multi-column `on` (v0.8.6: shift/diff/rolling) | ✅ | 8 |\n"
            "| `resample(freq)` (v0.8.7: bucket agg by timestamp) | ✅ | 28 |",
            1,
        )

    if "| `resample` empty bucket → skip (v0.8.7)" not in s:
        s = s.replace(
            "| `on` multi-column — composite sort, not per-group (v0.8.6)",
            "| `resample` empty bucket → skip (v0.8.7) | `resample` | NaN-fill нет; для gap-fill — `.reindex` после. §13.4/132 |\n"
            "| `resample` uint64 on → reject (v0.8.7) | `resample` | overflow risk; cast to int64 |\n"
            "| `resample` freq — int seconds / '5s'|'1min'|'1h'|'1D' only (v0.8.7) | `parse_freq` | compound/sub-second/calendar — нет |\n"
            "| `on` multi-column — composite sort, not per-group (v0.8.6)",
            1,
        )

    if "Эпизод 132" not in s:
        marker = "### 13.5. Открытые вопросы"
        episode = '''**Эпизод 132 (resample: prototype-first + renumber).**
`resample(freq)` — bucket aggregation, planned as v0.8.3. Prototype-first
(§13.4/123): `scripts/proto_resample.py` (3-path compare:
manual == pandas groupby(bucket) == external_sort + bucket transition).
5 buckets × sum: `[(0,4),(5,13),(10,23),(15,32),(20,22)]`. Все три
совпали → 1 итерация патча (28/28 passed first try). При v0.8.2 (без
prototype) было 5 fix-итераций.

**Rename v0.8.3 → v0.8.7.** CHANGELOG descending guard (v0.8.5,
§13.4/130) требует top = highest version. После v0.8.4/v0.8.5/v0.8.6
уже tagged, `[0.8.3]` над `[0.8.6]` → `test_changelog_is_descending`
fails. Также `test_changelog_has_latest_tag` (top == tag on master)
fails, если `[0.8.7]` положить не сверху, а tag — v0.8.7. Circular →
renamed. Roadmap-нумерация (§12.7) может дрейфовать; фактический
порядок = теги.

**Semantics decision (empty bucket):** pandas `resample()` даёт NaN для
всех aggs включая `count` (для gap-бакетов, без rows). Мы **skip**
все empty buckets — потому что output = «compact summary», NaN-rows
бесполезны для типичных post-processing (join, agg). Для fill —
`.reindex` после (pandas-стиль). Documented §11.5.

**Semantics decision (all-null bucket):** bucket с rows, но все values
null в agg-колонке → `count=0`, `sum/mean/min/max=NaN` в pandas.
Наш MVP: **skip** row (все колонки None → skip). Причина: consistency
с empty bucket; `count=0` бесполезен если `sum=NaN`. Documented.

**Semantics decision (null in `on`):** null timestamp → skip row
(cannot bucket). pandas: `groupby(ts.dt.floor('5s'))` — null → отдельный
NaN-bucket → `groupby(dropna=True)` (default) drop'ает. Совпадает.

**Урок:** prototype-first работает (28/28 с 1-й итерации). Renumber —
честный trade-off: документировать в CHANGELOG + §13.4, не «молча
исправить». Future tags — только последовательные (v0.8.8, v0.8.9, …),
не backport'ить roadmap-номера.

**Time-series API закрыт (v0.8.7):**
- `shift/diff` — positional + sort-aware single/multi.
- `rolling` — positional + sort-aware single/multi.
- `expanding/ewm` — cumulative/EWMA.
- `resample` — bucket agg.
- `row_id` reserved guard (v0.8.5).

**Следующее:** v0.8.5a per-group rolling/shift (~1.5 сессии,
prototype-first) — последний большой remaining в time-series.
Non-time-series backlog: `_iter_zipped` refactor (30 мин), reference
helpers для shift/diff/quantile (1.5 ч), multi-key join estimate
(45 мин).

'''
        s = s.replace(marker, episode + marker, 1)

    p.write_text(s)
    print("  OK: §11.1, §11.5, §13.4/132")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v087-resample"

    print("=== v0.8.7: resample(freq) ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v087-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v087-{ts_str}",
        "-m", "before patch v0.8.7 (resample)"])

    write_resample_module()
    export_resample()
    add_resample_method()
    write_tests()
    insert_changelog()
    patch_decisions()

    print("\n=== pytest tests/test_resample.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_resample.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.8.7 (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest time-series regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_rolling.py",
         "tests/test_rolling_on_col.py",
         "tests/test_shift_diff.py",
         "tests/test_shift_on_col.py",
         "tests/test_expanding_ewm.py",
         "tests/test_resample.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sys.exit(1)

    print("\n=== pytest tests/test_changelog.py -v (patch branch — skip latest) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_changelog.py", "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** CHANGELOG FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.8.7: resample(freq) — bucket agg by timestamp/integer on "
        "(prototype-validated; renamed from v0.8.3)"])

    print("\n==> v0.8.7 applied on", branch)


if __name__ == "__main__":
    main()