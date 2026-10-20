# v0.9.4b-1: concat / nlargest / nsmallest / dtypes / info /
# memory_usage. New DataFrame-level API surface.

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard, concat


def make_df(tmp_path, data, chunk=10, name="L", mem_mb=50):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def to_rows(df, tmp_path, name="out.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out).to_pylist()


# ============ concat axis=0 ============

def test_concat_axis0_basic(tmp_path):
    a = make_df(tmp_path, {"k": [1, 2, 3], "v": [10, 20, 30]}, name="a")
    b = make_df(tmp_path, {"k": [4, 5], "v": [40, 50]}, name="b")
    out = concat([a, b], axis=0)
    rows = to_rows(out, tmp_path)
    assert len(rows) == 5
    assert [r["k"] for r in rows] == [1, 2, 3, 4, 5]


def test_concat_axis0_single(tmp_path):
    a = make_df(tmp_path, {"k": [1, 2], "v": [10, 20]}, name="a")
    out = concat([a], axis=0)
    rows = to_rows(out, tmp_path)
    assert len(rows) == 2


def test_concat_axis0_schema_mismatch(tmp_path):
    a = make_df(tmp_path, {"k": [1], "v": [10]}, name="a")
    b = make_df(tmp_path, {"k": [2], "w": [20]}, name="b")
    with pytest.raises(ValueError, match="schema mismatch"):
        concat([a, b], axis=0)


def test_concat_axis0_with_filter(tmp_path):
    a = make_df(tmp_path, {"k": [1, 2, 3], "v": [10, 20, 30]}, name="a")
    b = make_df(tmp_path, {"k": [4, 5], "v": [40, 50]}, name="b")
    a_f = a[a["k"] > 1]
    out = concat([a_f, b], axis=0)
    rows = to_rows(out, tmp_path)
    assert [r["k"] for r in rows] == [2, 3, 4, 5]


# ============ concat axis=1 ============

def test_concat_axis1_basic(tmp_path):
    a = make_df(tmp_path, {"x": [1, 2, 3]}, name="a")
    b = make_df(tmp_path, {"y": [10, 20, 30]}, name="b")
    out = concat([a, b], axis=1)
    rows = to_rows(out, tmp_path)
    assert [r["x"] for r in rows] == [1, 2, 3]
    assert [r["y"] for r in rows] == [10, 20, 30]


def test_concat_axis1_single(tmp_path):
    a = make_df(tmp_path, {"x": [1, 2]}, name="a")
    out = concat([a], axis=1)
    rows = to_rows(out, tmp_path)
    assert len(rows) == 2
    assert rows[0]["x"] == 1


def test_concat_axis1_row_mismatch(tmp_path):
    a = make_df(tmp_path, {"x": [1, 2, 3]}, name="a")
    b = make_df(tmp_path, {"y": [10, 20]}, name="b")
    with pytest.raises(ValueError, match="row count mismatch"):
        concat([a, b], axis=1)


def test_concat_axis1_column_collision(tmp_path):
    a = make_df(tmp_path, {"x": [1, 2]}, name="a")
    b = make_df(tmp_path, {"x": [3, 4]}, name="b")
    with pytest.raises(ValueError, match="duplicate column"):
        concat([a, b], axis=1)


def test_concat_empty_list_raises():
    with pytest.raises(ValueError, match="at least one df"):
        concat([], axis=0)


def test_concat_invalid_axis(tmp_path):
    a = make_df(tmp_path, {"x": [1]}, name="a")
    with pytest.raises(ValueError, match="axis must be 0 or 1"):
        concat([a], axis=2)


# ============ nlargest / nsmallest ============

def test_nlargest_basic(tmp_path):
    a = make_df(tmp_path, {"x": [3, 1, 4, 1, 5, 9, 2]}, name="a")
    top = a.nlargest(3, "x")
    rows = to_rows(top, tmp_path)
    assert [r["x"] for r in rows] == [9, 5, 4]


def test_nlargest_ngtN(tmp_path):
    a = make_df(tmp_path, {"x": [1, 2, 3]}, name="a")
    top = a.nlargest(10, "x")
    rows = to_rows(top, tmp_path)
    assert [r["x"] for r in rows] == [3, 2, 1]


def test_nlargest_invalid_n(tmp_path):
    a = make_df(tmp_path, {"x": [1]}, name="a")
    with pytest.raises(ValueError, match="n must be int >= 1"):
        a.nlargest(0, "x")


def test_nlargest_missing_column(tmp_path):
    a = make_df(tmp_path, {"x": [1]}, name="a")
    with pytest.raises(KeyError, match="nope"):
        a.nlargest(1, "nope")


def test_nsmallest_basic(tmp_path):
    a = make_df(tmp_path, {"x": [3, 1, 4, 1, 5, 9, 2]}, name="a")
    bot = a.nsmallest(3, "x")
    rows = to_rows(bot, tmp_path)
    assert [r["x"] for r in rows] == [1, 1, 2]


def test_nsmallest_ngtN(tmp_path):
    a = make_df(tmp_path, {"x": [5, 3, 1]}, name="a")
    bot = a.nsmallest(10, "x")
    rows = to_rows(bot, tmp_path)
    assert [r["x"] for r in rows] == [1, 3, 5]


# ============ dtypes ============

def test_dtypes_basic(tmp_path):
    a = make_df(tmp_path, {"i": [1, 2], "s": ["a", "b"],
                          "f": [1.0, 2.0]}, name="a")
    d = a.dtypes()
    assert d["i"] == "int64"
    assert d["s"] == "string"
    assert d["f"] == "double"


# ============ info ============

def test_info_basic(tmp_path):
    a = make_df(tmp_path, {"i": [1, 2, 3], "s": ["a", "b", "c"]},
                  name="a")
    inf = a.info()
    assert inf["total_rows"] == 3
    assert inf["filtered_rows"] == 3
    assert inf["n_columns"] == 2
    assert inf["columns"]["i"]["dtype"] == "int64"


def test_info_compute_non_null(tmp_path):
    a = make_df(tmp_path, {"i": [1, None, 3]}, name="a")
    inf = a.info(compute_non_null=True)
    assert inf["columns"]["i"]["non_null"] == 2


def test_info_with_filter(tmp_path):
    a = make_df(tmp_path, {"i": [1, 2, 3, 4]}, name="a")
    a_f = a[a["i"] > 2]
    inf = a_f.info()
    assert inf["total_rows"] == 4
    assert inf["filtered_rows"] == 2


# ============ memory_usage ============

def test_memory_usage_basic(tmp_path):
    a = make_df(tmp_path, {"i": [1, 2, 3, 4]}, name="a")
    mu = a.memory_usage()
    assert mu["total"] > 0
    assert mu["per_column"]["i"] > 0


def test_memory_usage_empty(tmp_path):
    a = make_df(tmp_path, {"i": pa.array([], type=pa.int64())},
                 name="a")
    mu = a.memory_usage()
    assert mu["total"] == 0

