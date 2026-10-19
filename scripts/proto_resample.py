#!/usr/bin/env python3
"""prototype: resample(freq) — validate algorithm before patching.

DATA: ts (int64 seconds) = [12, 0, 17, 5, 8, 22, 3, 15, 11, 1]
      v (float64)       = [12, 0, 17, 5, 8, 22, 3, 15, 11, 1]
      freq = 5 (seconds)
      agg  = "sum"

Sorted ts: [0, 1, 3, 5, 8, 11, 12, 15, 17, 22]
Buckets:   [0, 0, 0, 1, 1,  2,  2,  3,  3,  4]
Expected:  [(0, 4), (5, 13), (10, 23), (15, 32), (20, 22)]

Compares 3 approaches:
    C) Manual Python: sort + walk + bucket transition (ground truth).
    A) Reference: pandas groupby(ts // freq).
    B) Proposed: external_sort + stream + bucket transition.

WHY prototype for resample (but not for v0.8.6 multi-on):
    - v0.8.6 — только расширение параметра (list вместо str). `Sort` уже
      multi-key, `_Key` уже `Sequence[str]`. Нет нового алгоритма.
    - resample — новый state: bucket_id = ts // freq, transition detection,
      accumulator reset per bucket, aggregation **уменьшает rows** (не append).
      Другой класс ошибок: off-by-one на границе bucket, multiple batches
      per bucket, пустые buckets (skip or fill NaN — design choice).

HALO: не нужен. Sort by `on` → buckets contiguous → transition только при
смене bucket_id → flush предыдущий + reset. Cross-batch — accumulator
живёт через batch boundary (state O(#cols)).

Run: uv run python scripts/proto_resample.py
"""
from __future__ import annotations

import sys
import tempfile
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
    "ts": [12, 0, 17, 5, 8, 22, 3, 15, 11, 1],
    "v": [12.0, 0.0, 17.0, 5.0, 8.0, 22.0, 3.0, 15.0, 11.0, 1.0],
}
FREQ = 5
AGG = "sum"
EXPECTED = [(0, 4.0), (5, 13.0), (10, 23.0), (15, 32.0), (20, 22.0)]


def _approx_eq_rows(got, expected, tol=1e-9):
    if len(got) != len(expected):
        return False
    for (gb, gv), (eb, ev) in zip(got, expected):
        if gb != eb:
            return False
        if gv is None and ev is None:
            continue
        if gv is None or ev is None:
            return False
        if abs(gv - ev) > tol:
            return False
    return True


def _aggregate(values, agg):
    non_null = [x for x in values if x is not None]
    if not non_null:
        return None
    if agg == "count":
        return len(non_null)
    if agg == "sum":
        return sum(non_null)
    if agg == "mean":
        return sum(non_null) / len(non_null)
    if agg == "min":
        return min(non_null)
    if agg == "max":
        return max(non_null)
    raise ValueError(f"unknown agg: {agg}")


def manual_resample(data, freq, agg):
    """C) ground truth: sort + walk + bucket transition."""
    n = len(data["ts"])
    sorted_idx = sorted(range(n), key=lambda i: data["ts"][i])
    result = []
    cur_bucket = None
    cur_vals = []
    for i in sorted_idx:
        b = (data["ts"][i] // freq) * freq
        if b != cur_bucket:
            if cur_bucket is not None:
                result.append((cur_bucket, _aggregate(cur_vals, agg)))
            cur_bucket = b
            cur_vals = []
        cur_vals.append(data["v"][i])
    if cur_bucket is not None:
        result.append((cur_bucket, _aggregate(cur_vals, agg)))
    return result


def reference_resample(df, freq, agg, col="v"):
    """A) reference: pandas groupby(bucket)."""
    import pandas as pd
    t = df.head(df.count())  # materialize all (small prototype data)
    pdf = t.to_pandas()
    pdf["bucket"] = (pdf["ts"] // freq) * freq
    grouped = pdf.groupby("bucket")[col].agg(agg).reset_index()
    return [(int(r["bucket"]), float(r[col])) for _, r in grouped.iterrows()]


def proposed_resample(df, td, freq, agg, col="v"):
    """B) proposed: external_sort + stream + bucket transition."""
    # Step 1: sort by ts
    sorted_path = external_sort(
        df._iter_filtered_batches(),
        key_columns=["ts"],
        value_columns=["ts", col],
        memory_guard=df._memory_guard,
        spill_dir=td / "sort",
        ascending=True,
        log=False,
    )

    # Step 2: stream + bucket transition, write (bucket_start, agg_value)
    out_schema = pa.schema([
        ("bucket", pa.int64()),
        (f"{col}_{agg}", pa.float64()),
    ])
    out_path = td / "resampled.arrow"
    writer = ArrowIPCBatchWriter(str(out_path), out_schema)

    cur_bucket = None
    cur_vals = []

    def _flush():
        if cur_bucket is None:
            return
        agg_val = _aggregate(cur_vals, agg)
        writer.write_batch(pa.RecordBatch.from_arrays(
            [
                pa.array([cur_bucket], type=pa.int64()),
                pa.array([agg_val], type=pa.float64()),
            ],
            schema=out_schema,
        ))

    try:
        for b in ArrowIPCBatchReader(str(sorted_path), None):
            ts_list = b.column("ts").to_pylist()
            v_list = b.column(col).to_pylist()
            for ts, vv in zip(ts_list, v_list):
                bkt = (ts // freq) * freq
                if bkt != cur_bucket:
                    _flush()
                    cur_bucket = bkt
                    cur_vals = []
                cur_vals.append(vv)
        _flush()
    finally:
        writer.close()

    # Step 3: read result
    result = []
    for b in ArrowIPCBatchReader(str(out_path), None):
        for i in range(b.num_rows):
            result.append((b.column("bucket")[i].as_py(),
                           b.column(f"{col}_{agg}")[i].as_py()))
    return result


def main():
    print("=" * 60)
    print("resample(freq) prototype")
    print("=" * 60)
    print(f"  ts:     {DATA['ts']}")
    print(f"  v:      {DATA['v']}")
    print(f"  freq:   {FREQ}s, agg={AGG}")
    print(f"  expected (by bucket): {EXPECTED}")
    print()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        t = pa.table(DATA)
        p = td / "src.parquet"
        pq.write_table(t, p)
        df = CadwDataFrame.from_parquet(
            p, td / "ds", chunk_size=3,
            memory_guard=MemoryGuard(max_memory_mb=20),
            spill_dir=td / "spill",
        )

        print("--- C) Manual (ground truth) ---")
        result_c = manual_resample(DATA, FREQ, AGG)
        print(f"  result: {result_c}")
        print()

        print("--- A) Reference: pandas groupby(bucket) ---")
        result_a = reference_resample(df, FREQ, AGG)
        print(f"  result: {result_a}")
        print()

        print("--- B) Proposed: external_sort + bucket transition ---")
        result_b = proposed_resample(df, td, FREQ, AGG)
        print(f"  result: {result_b}")
        print()

        print("--- Compare ---")
        c_eq_exp = _approx_eq_rows(result_c, EXPECTED)
        a_eq_c = _approx_eq_rows(result_a, result_c)
        b_eq_c = _approx_eq_rows(result_b, result_c)
        print(f"  C == expected: {c_eq_exp}")
        print(f"  A == C:        {a_eq_c}")
        print(f"  B == C:        {b_eq_c}")
        print()

        if c_eq_exp and a_eq_c and b_eq_c:
            print("  ✅ ALGORITHM VALIDATED — proceed with patch_v083_resample.py")
            print("     (real tests cover: multi-batch buckets, empty buckets,")
            print("      null values, min/max/count/mean, freq parser)")
            return 0
        print("  ❌ MISMATCH — debug before patching")
        if not c_eq_exp:
            print("     C != expected: manual ground truth or EXPECTED wrong")
        if not a_eq_c:
            print("     A != C: pandas reference differs — semantics issue")
        if not b_eq_c:
            print("     B != C: proposed algorithm wrong — debug sort/bucket")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())