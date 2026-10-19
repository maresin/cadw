"""v0.6.2b: shift(k) + diff(k)."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, name="L", data=None, chunk=3):
    if data is None:
        data = {"v": [1, 2, 3, 4, 5], "cat": ["a", "b", "c", "d", "e"]}
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_col(df, tmp_path, col):
    out = tmp_path / "o.parquet"
    df.write_parquet(out)
    return pq.read_table(out).column(col).to_pylist()


# ---------- shift ----------

def test_shift_k1_basic(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(1, columns=["v"])
    assert read_col(res, tmp_path, "v") == [None, 1, 2, 3, 4]


def test_shift_k2(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(2, columns=["v"])
    assert read_col(res, tmp_path, "v") == [None, None, 1, 2, 3]


def test_shift_fill_value(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(2, fill_value=0, columns=["v"])
    assert read_col(res, tmp_path, "v") == [0, 0, 1, 2, 3]


def test_shift_zero_is_noop(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(0, columns=["v"])
    # Same view
    assert res is df or read_col(res, tmp_path, "v") == [1, 2, 3, 4, 5]


def test_shift_preserves_original(tmp_path):
    df = make_df(tmp_path)
    _ = df.shift(1, columns=["v"])
    assert read_col(df, tmp_path, "v") == [1, 2, 3, 4, 5]


def test_shift_string_column(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(1, columns=["cat"])
    assert read_col(res, tmp_path, "cat") == [None, "a", "b", "c", "d"]


def test_shift_string_fill(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(1, fill_value="MISSING", columns=["cat"])
    assert read_col(res, tmp_path, "cat") == [
        "MISSING", "a", "b", "c", "d",
    ]


def test_shift_all_columns_default(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(1)
    assert read_col(res, tmp_path, "v") == [None, 1, 2, 3, 4]
    assert read_col(res, tmp_path, "cat") == [None, "a", "b", "c", "d"]


def test_shift_with_nulls_in_input(tmp_path):
    df = make_df(tmp_path, data={
        "v": [1, None, 3, None, 5],
        "cat": ["a", "b", "c", "d", "e"],
    })
    res = df.shift(1, columns=["v"])
    assert read_col(res, tmp_path, "v") == [None, 1, None, 3, None]


def test_shift_k_ge_n(tmp_path):
    df = make_df(tmp_path)
    res = df.shift(10, fill_value=-1, columns=["v"])
    assert read_col(res, tmp_path, "v") == [-1, -1, -1, -1, -1]


def test_shift_negative_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(NotImplementedError, match="negative k"):
        df.shift(-1, columns=["v"])


def test_shift_missing_column_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.shift(1, columns=["nope"])


def test_shift_non_int_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="k must be int"):
        df.shift(1.5, columns=["v"])


def test_shift_filtered(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 2]  # 3, 4, 5
    res = filtered.shift(1, columns=["v"])
    assert read_col(res, tmp_path, "v") == [None, 3, 4]


# ---------- diff ----------

def test_diff_k1_basic(tmp_path):
    df = make_df(tmp_path)
    res = df.diff(1, columns=["v"])
    # 1, 2-1, 3-2, 4-3, 5-4 = 1,1,1,1,1; first null
    assert read_col(res, tmp_path, "v") == [None, 1, 1, 1, 1]


def test_diff_k2(tmp_path):
    df = make_df(tmp_path, data={"v": [1, 2, 4, 8, 16]})
    res = df.diff(2, columns=["v"])
    # None, None, 4-1, 8-2, 16-4 = None,None,3,6,12
    assert read_col(res, tmp_path, "v") == [None, None, 3, 6, 12]


def test_diff_with_nulls(tmp_path):
    df = make_df(tmp_path, data={"v": [1.0, None, 3.0, 5.0]})
    res = df.diff(1, columns=["v"])
    # diff(1): out[0]=null (fill); out[i]=x[i]-x[i-1]
    # x = [1.0, None, 3.0, 5.0]
    # out[0] = null (fill)
    # out[1] = None - 1.0 = null (null-propagate)
    # out[2] = 3.0 - None = null (null-propagate)
    # out[3] = 5.0 - 3.0 = 2.0
    vals = read_col(res, tmp_path, "v")
    assert vals[0] is None
    assert vals[1] is None
    assert vals[2] is None
    assert abs(vals[3] - 2.0) < 1e-9


def test_diff_all_columns_default(tmp_path):
    df = make_df(tmp_path, data={"a": [1, 3, 6], "b": [10, 20, 40]})
    res = df.diff(1)
    assert read_col(res, tmp_path, "a") == [None, 2, 3]
    assert read_col(res, tmp_path, "b") == [None, 10, 20]


def test_diff_k0_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match=">= 1"):
        df.diff(0, columns=["v"])


def test_diff_preserves_original(tmp_path):
    df = make_df(tmp_path)
    _ = df.diff(1, columns=["v"])
    assert read_col(df, tmp_path, "v") == [1, 2, 3, 4, 5]


def test_diff_subset_only(tmp_path):
    df = make_df(tmp_path, data={
        "a": [1, 2, 3], "b": ["x", "y", "z"],
    })
    res = df.diff(1, columns=["a"])
    assert read_col(res, tmp_path, "a") == [None, 1, 1]
    assert read_col(res, tmp_path, "b") == ["x", "y", "z"]


# ---------- combined ----------

def test_shift_then_diff(tmp_path):
    df = make_df(tmp_path, data={"v": [10, 20, 40, 80]})
    # shift(1) = [None,10,20,40]; diff(1) of shifted = [None,None,10,20]
    res = df.shift(1, columns=["v"]).diff(1, columns=["v"])
    assert read_col(res, tmp_path, "v") == [None, None, 10, 20]


def test_shift_dtypes_preserved(tmp_path):
    df = make_df(tmp_path, data={
        "v_int": [1, 2, 3],
        "v_float": [1.0, 2.0, 3.0],
        "v_str": ["a", "b", "c"],
    })
    res = df.shift(1)
    out = tmp_path / "o.parquet"
    res.write_parquet(out)
    schema = pq.read_schema(out)
    assert schema.field("v_int").type == pa.int64()
    assert schema.field("v_float").type == pa.float64()
    assert schema.field("v_str").type == pa.string()
