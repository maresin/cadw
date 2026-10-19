#!/usr/bin/env python3
"""prototype: rolling(on=col) — validate algorithm before patching.

DATA: ts=[5,3,1,4,2], v=[50,30,10,40,20], seq_id=[0..4], window=2, agg=sum.
Expected result (original order): [90, 50, None, 70, 30].

NOTE: user column named `seq_id`, NOT `row_id`. `row_id` reserved by
CADW internal (external_sort/hash_shuffle append their own) — user
column named `row_id` causes duplicate in sort_values (§11.5 backlog).

Compares 3 approaches:
    C) Manual: pure Python sorted() + deque.
    A) Reference: df.sort_values('ts').rolling(2,'sum').sort_values('seq_id').
    B) Proposed _rolling_on: external_sort by ts, deque push,
       external_sort by __orig_rid.

Run: uv run python scripts/proto_rolling_on.py
"""
from __future__ import annotations

import sys
import tempfile
from collections import deque
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pyarrow as pa
import pyarrow.parquet as pq

from cadw import CadwDataFrame, MemoryGuard
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops.sort import external_sort


DATA = {
    "ts": [5, 3, 1, 4, 2],
    "v": [50.0, 30.0, 10.0, 40.0, 20.0],
    "seq_id": [0, 1, 2, 3, 4],  # user column; NOT row_id (reserved)
}
WINDOW = 2
AGG = "sum"
EXPECTED = [90.0, 50.0, None, 70.0, 30.0]


def _approx_eq(xs, ys, tol=1e-9):
    if len(xs) != len(ys):
        return False
    for x, y in zip(xs, ys):
        if x is None and y is None:
            continue
        if x is None or y is None:
            return False
        if abs(x - y) > tol:
            return False
    return True


def manual_rolling_on(data, window, agg="sum"):
    """C) ground truth: sort by ts, deque push per row, map back by seq_id."""
    n = len(data["ts"])
    sorted_idx = sorted(
        range(n), key=lambda i: (data["ts"][i], data["seq_id"][i]),
    )
    result = [None] * n
    buf = deque(maxlen=window)
    for orig_i in sorted_idx:
        v = data["v"][orig_i]
        if v is not None:
            buf.append(v)
        pos = data["seq_id"][orig_i]
        if len(buf) < window:
            result[pos] = None
            continue
        if agg == "sum":
            result[pos] = sum(buf)
        elif agg == "mean":
            result[pos] = sum(buf) / len(buf)
        elif agg == "count":
            result[pos] = len(buf)
        elif agg == "min":
            result[pos] = min(buf)
        elif agg == "max":
            result[pos] = max(buf)
    return result


def reference_rolling_on(df, window, agg):
    """A) reference: sort_values + rolling + sort_values back by seq_id."""
    sorted_df = df.sort_values("ts", ascending=True)
    rolled = sorted_df.rolling(window, agg, min_periods=window)
    back = rolled.sort_values("seq_id", ascending=True)
    return back


def proposed_rolling_on(df, td, window, agg):
    """B) proposed algorithm: external_sort → _RollState → external_sort."""
    sorted_path = external_sort(
        df._iter_filtered_batches(),
        key_columns=["ts"],
        value_columns=["ts", "v", "seq_id"],
        memory_guard=df._memory_guard,
        spill_dir=td / "sort1",
        ascending=True,
        log=False,
    )

    out_col = f"v_rolling_{agg}"
    out_schema = pa.schema([
        ("row_id", pa.uint64()),
        (out_col, pa.float64()),
    ])
    rolled_path = td / "rolled.arrow"
    writer = ArrowIPCBatchWriter(str(rolled_path), out_schema)
    buf = deque(maxlen=window)
    try:
        for b in ArrowIPCBatchReader(str(sorted_path), None):
            v_list = b.column("v").to_pylist()
            rid_list = b.column("row_id").to_pylist()
            rolled_vals = []
            for vv in v_list:
                if vv is not None:
                    buf.append(vv)
                if len(buf) < window:
                    rolled_vals.append(None)
                elif agg == "sum":
                    rolled_vals.append(sum(buf))
                elif agg == "mean":
                    rolled_vals.append(sum(buf) / len(buf))
                elif agg == "count":
                    rolled_vals.append(len(buf))
                elif agg == "min":
                    rolled_vals.append(min(buf))
                elif agg == "max":
                    rolled_vals.append(max(buf))
            writer.write_batch(pa.RecordBatch.from_arrays(
                [
                    pa.array(rid_list, type=pa.uint64()),
                    pa.array(rolled_vals, type=pa.float64()),
                ],
                schema=out_schema,
            ))
    finally:
        writer.close()

    def _iter_rolled():
        for b in ArrowIPCBatchReader(str(rolled_path), None):
            names = b.schema.names
            rid_idx = names.index("row_id")
            new_names = (
                ["__orig_rid"]
                + [n for i, n in enumerate(names) if i != rid_idx]
            )
            yield b.rename_columns(new_names)

    sorted_rowid_path = external_sort(
        _iter_rolled(),
        key_columns=["__orig_rid"],
        value_columns=[out_col],
        memory_guard=df._memory_guard,
        spill_dir=td / "sort2",
        ascending=True,
        log=False,
    )
    result = []
    for b in ArrowIPCBatchReader(str(sorted_rowid_path), [out_col]):
        result.extend(b.column(out_col).to_pylist())
    return result


def main():
    print("=" * 60)
    print("rolling(on=col) prototype")
    print("=" * 60)
    print(f"  ts:     {DATA['ts']}")
    print(f"  v:      {DATA['v']}")
    print(f"  seq_id: {DATA['seq_id']}")
    print(f"  window: {WINDOW}, agg={AGG}")
    print(f"  expected (orig order): {EXPECTED}")
    print()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        t = pa.table(DATA)
        p = td / "src.parquet"
        pq.write_table(t, p)
        df = CadwDataFrame.from_parquet(
            p, td / "ds", chunk_size=2,
            memory_guard=MemoryGuard(max_memory_mb=20),
            spill_dir=td / "spill",
        )

        print("--- C) Manual (ground truth) ---")
        result_c = manual_rolling_on(DATA, WINDOW, AGG)
        print(f"  result: {result_c}")
        print()

        print("--- A) Reference: sort_values + rolling + sort_values ---")
        back_df = reference_rolling_on(df, WINDOW, AGG)
        out_p = td / "ref.parquet"
        back_df.write_parquet(out_p)
        ref_table = pq.read_table(out_p)
        result_a = ref_table.column(f"v_rolling_{AGG}").to_pylist()
        print(f"  result: {result_a}")
        print()

        print("--- B) Proposed: external_sort → deque → external_sort ---")
        result_b = proposed_rolling_on(df, td, WINDOW, AGG)
        print(f"  result: {result_b}")
        print()

        print("--- Comparison ---")
        c_eq_a = _approx_eq(result_c, result_a)
        c_eq_b = _approx_eq(result_c, result_b)
        a_eq_b = _approx_eq(result_a, result_b)
        print(f"  C == A (ref):  {c_eq_a}")
        print(f"  C == B (prop): {c_eq_b}")
        print(f"  A == B:        {a_eq_b}")
        print()

        if c_eq_a and c_eq_b and a_eq_b:
            print("  ✅ ALGORITHM VALIDATED — proceed with patch_v084_rolling_on.py")
        else:
            print("  ❌ MISMATCH — debug before patching")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())