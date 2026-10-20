"""v0.9.0a: type ops (astype, dropna, drop, rename) + isnull/notnull."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_table(df, tmp_path, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out)


# ============ isnull / notnull ============

def test_isnull_filter(tmp_path):
    df = make_df(tmp_path, {"a": [1, None, 3, None], "v": [10, 20, 30, 40]})
    filtered = df[df["a"].isnull()]
    assert filtered.count() == 2


def test_notnull_filter(tmp_path):
    df = make_df(tmp_path, {"a": [1, None, 3, None], "v": [10, 20, 30, 40]})
    filtered = df[df["a"].notnull()]
    assert filtered.count() == 2


# ============ astype ============

def test_astype_int_to_float(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    result = df.astype({"v": "float64"})
    t = read_table(result, tmp_path)
    assert t.schema.field("v").type == pa.float64()
    assert t.column("v").to_pylist() == [1.0, 2.0, 3.0]


def test_astype_pa_type(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    result = df.astype({"v": pa.int32()})
    t = read_table(result, tmp_path)
    assert t.schema.field("v").type == pa.int32()


def test_astype_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2]})
    with pytest.raises(KeyError, match="nope"):
        df.astype({"nope": "int64"})


def test_astype_invalid_alias_raises(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2]})
    with pytest.raises(ValueError, match="unknown dtype"):
        df.astype({"v": "not_a_real_type"})


def test_astype_safe_cast_overflow_raises(tmp_path):
    df = make_df(tmp_path, {"v": [100000, 200000]})
    with pytest.raises(pa.lib.ArrowInvalid):
        df.astype({"v": "int8"}).write_parquet(tmp_path / "o.parquet")


def test_astype_unsafe_cast_wraps(tmp_path):
    df = make_df(tmp_path, {"v": [100000]})
    # safe=False → wraps; just verify it doesn't raise
    result = df.astype({"v": "int8"}, safe=False)
    assert result.count() == 1


# ============ dropna ============

def test_dropna_any(tmp_path):
    df = make_df(tmp_path, {
        "a": [1, None, 3, None],
        "b": [10, 20, None, None],
        "v": [1, 2, 3, 4],
    })
    result = df.dropna(how="any")
    assert result.count() == 1  # only row with both non-null


def test_dropna_all(tmp_path):
    df = make_df(tmp_path, {
        "a": [1, None, 3, None],
        "b": [10, 20, None, None],
    })
    result = df.dropna(how="all")
    assert result.count() == 3  # only fully-null row dropped


def test_dropna_subset(tmp_path):
    df = make_df(tmp_path, {
        "a": [1, None, 3],
        "b": [None, 20, None],
    })
    result = df.dropna(subset=["a"])
    assert result.count() == 2  # only non-null 'a'


def test_dropna_invalid_how_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1, None]})
    with pytest.raises(ValueError, match="how"):
        df.dropna(how="some")


def test_dropna_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1, None]})
    with pytest.raises(KeyError, match="nope"):
        df.dropna(subset=["nope"])


# ============ drop ============

def test_drop_single(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = df.drop("b")
    assert "b" not in result.columns
    assert set(result.columns) == {"a", "c"}


def test_drop_multiple(tmp_path):
    df = make_df(tmp_path, {"a": [1], "b": [2], "c": [3]})
    result = df.drop(["a", "c"])
    assert result.columns == ["b"]


def test_drop_missing_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1]})
    with pytest.raises(KeyError, match="nope"):
        df.drop("nope")


# ============ rename ============

def test_rename_single(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2], "b": [3, 4]})
    result = df.rename({"a": "alpha"})
    assert "alpha" in result.columns
    assert "a" not in result.columns
    assert "b" in result.columns
    t = read_table(result, tmp_path)
    assert t.column("alpha").to_pylist() == [1, 2]


def test_rename_multiple(tmp_path):
    df = make_df(tmp_path, {"a": [1], "b": [2], "c": [3]})
    result = df.rename({"a": "x", "c": "z"})
    assert result.columns == ["x", "b", "z"]


def test_rename_missing_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1]})
    with pytest.raises(KeyError, match="nope"):
        df.rename({"nope": "x"})


def test_rename_collision_raises(tmp_path):
    df = make_df(tmp_path, {"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="collides"):
        df.rename({"a": "b"})


# ============ chained ============

def test_astype_then_filter(tmp_path):
    df = make_df(tmp_path, {"v": [1, 2, 3, 4]})
    result = df.astype({"v": "float64"})[df["v"] > 2]
    assert result.count() == 2


def test_dropna_then_astype(tmp_path):
    df = make_df(tmp_path, {"a": [1, None, 3], "v": [10, 20, 30]})
    result = df.dropna(how="any").astype({"v": "int32"})
    assert result.count() == 2
