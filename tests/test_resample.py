"""v0.8.7: resample(freq) — bucket aggregation by timestamp/integer on."""
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
    """Bucket with all-null values → skipped (not emitted as None-row).

    Note: `v` must be explicit `pa.float64()`. Without type, pyarrow infers
    `null` type (from [None, None, None]) — resample correctly rejects it
    as non-numeric (see test_resample_null_typed_agg_col_raises).
    """
    df = make_df(tmp_path, {
        "ts": [0, 5, 6],
        "v": pa.array([None, None, None], type=pa.float64()),
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

def test_resample_null_typed_agg_col_raises(tmp_path):
    """Column with `null` type (not float64) — TypeError (non-numeric).

    `pa.table({"v": [None, None]})` infers `null` type — has no
    aggregation semantics. For "all-null float64" use `pa.array([...],
    type=pa.float64())`.
    """
    df = make_df(tmp_path, {
        "ts": [0, 5],
        "v": [None, None],  # null type
    })
    with pytest.raises(TypeError, match="numeric"):
        df.resample(5, "sum", on="ts", columns=["v"])


def test_to_unix_seconds_date64_unit():
    """Unit test for date64 branch in _to_unix_seconds.

    Date64 = int64 ms-since-epoch, BUT day-granularity semantics. Cannot
    test through parquet roundtrip (Parquet has no ms-date type;
    date64 → date32 on read, §11.5/§13.4/135). Test helper directly.
    """
    import datetime as dt
    from cadw.ops.resample import _to_unix_seconds

    arr = pa.array(
        [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)],
        type=pa.date64(),
    )
    result = _to_unix_seconds(arr)
    # 2020-01-01 00:00:00 UTC = 1577836800
    # 2020-01-02 00:00:00 UTC = 1577923200
    assert result.to_pylist() == [1577836800, 1577923200]


def test_bucket_to_arrow_date64_unit():
    """Unit test for date64 branch in _bucket_to_arrow."""
    import datetime as dt
    from cadw.ops.resample import _bucket_to_arrow

    seconds = [1577836800, 1577923200]
    arr = _bucket_to_arrow(seconds, pa.date64())
    assert arr.type == pa.date64()
    assert arr.to_pylist() == [dt.date(2020, 1, 1), dt.date(2020, 1, 2)]


def test_to_unix_seconds_date32_unit():
    """Unit test for date32 branch (via int32, §13.4/133)."""
    import datetime as dt
    from cadw.ops.resample import _to_unix_seconds

    arr = pa.array(
        [dt.date(2020, 1, 1), dt.date(2020, 1, 2)],
        type=pa.date32(),
    )
    result = _to_unix_seconds(arr)
    assert result.to_pylist() == [1577836800, 1577923200]


def test_resample_date64_day_freq_through_parquet_limitation(tmp_path):
    """Parquet has no ms-date logical type → date64 becomes date32 on read.

    Real-world `resample(on=date_col)` sees date32, not date64 (after
    parquet roundtrip). Same day-granularity semantics — day-level
    `freq` (e.g. '1D') works correctly. See §11.5, §13.4/135.
    """
    import datetime as dt
    dates = [
        dt.date(2020, 1, 1),
        dt.date(2020, 1, 1),
        dt.date(2020, 1, 3),
    ]
    df = make_df(tmp_path, {
        "ts": pa.array(dates, type=pa.date32()),  # date32 in practice
        "v": [1.0, 2.0, 3.0],
    })
    res = df.resample("1D", "sum", on="ts", columns=["v"])
    t = read_df(res, tmp_path)
    rows = [(r["ts"], r["v_sum"]) for r in t.to_pylist()]
    assert rows == [
        (dt.date(2020, 1, 1), 3.0),
        (dt.date(2020, 1, 3), 3.0),
    ]

