"""Тесты CadwDataFrame.iloc (v0.4.0b)."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.dataset.index import MaterializedIndex, RangeIndex


def make_df(tmp_path, name="L", data=None, chunk=3):
    if data is None:
        data = {"k": ["a", "b", "c", "d", "e"], "v": [10, 20, 30, 40, 50]}
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- iloc[int] ----------

def test_iloc_int_returns_dict(tmp_path):
    df = make_df(tmp_path)
    row = df.iloc[0]
    assert isinstance(row, dict)
    assert row == {"k": "a", "v": 10}


def test_iloc_int_middle(tmp_path):
    df = make_df(tmp_path)
    assert df.iloc[2] == {"k": "c", "v": 30}


def test_iloc_int_last(tmp_path):
    df = make_df(tmp_path)
    assert df.iloc[4] == {"k": "e", "v": 50}


def test_iloc_int_negative(tmp_path):
    df = make_df(tmp_path)
    assert df.iloc[-1] == {"k": "e", "v": 50}
    assert df.iloc[-5] == {"k": "a", "v": 10}


def test_iloc_int_out_of_range(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(IndexError, match="out of range"):
        df.iloc[10]
    with pytest.raises(IndexError, match="out of range"):
        df.iloc[-10]


# ---------- iloc[int] on filtered view ----------

def test_iloc_int_on_filtered(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15]  # rows: b,c,d,e (v:20,30,40,50)
    assert filtered.iloc[0] == {"k": "b", "v": 20}
    assert filtered.iloc[2] == {"k": "d", "v": 40}
    assert filtered.iloc[-1] == {"k": "e", "v": 50}


# ---------- iloc[slice] ----------

def test_iloc_slice_returns_cadwdf(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[1:4]
    assert isinstance(sub, CadwDataFrame)
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["b", "c", "d"]


def test_iloc_slice_with_step(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[0:5:2]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["a", "c", "e"]


def test_iloc_slice_negative(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[-3:]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["c", "d", "e"]


def test_iloc_slice_on_filtered(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15]  # b,c,d,e
    sub = filtered.iloc[1:3]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["c", "d"]


# ---------- iloc[list] ----------

def test_iloc_list_returns_cadwdf(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[[0, 2, 4]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [(r["k"], r["v"]) for r in rows] == [("a", 10), ("c", 30), ("e", 50)]


def test_iloc_list_duplicates(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[[0, 0, 2]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["a", "a", "c"]


def test_iloc_list_negative(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[[-1, 0]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["e", "a"]


def test_iloc_list_out_of_range(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(IndexError, match="positions out of range"):
        df.iloc[[0, 10]]


def test_iloc_list_empty(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[[]]
    assert isinstance(sub, CadwDataFrame)
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows == []


def test_iloc_list_on_filtered(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15]  # b,c,d,e
    sub = filtered.iloc[[0, 2]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["b", "d"]


# ---------- index preservation ----------

def test_iloc_list_preserves_range_index(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[[1, 3]]
    # Parent had RangeIndex(0,5); result has MaterializedIndex([1, 3])
    assert isinstance(sub.index, MaterializedIndex)
    assert list(sub.index) == [1, 3]


def test_iloc_list_preserves_materialized_index(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=False)
    sub = with_idx.iloc[[0, 2, 4]]
    assert isinstance(sub.index, MaterializedIndex)
    assert list(sub.index) == ["a", "c", "e"]


def test_iloc_list_on_filtered_preserves_parent(tmp_path):
    """v0.4.0d: filtered view + iloc[list] → preserves parent index via raw positions."""
    from cadw.dataset.index import MaterializedIndex
    df = make_df(tmp_path)  # v: 10,20,30,40,50
    filtered = df[df["v"] > 15]  # rows b(20),c(30),d(40),e(50) → raw positions 1,2,3,4
    sub = filtered.iloc[[0, 2]]  # filtered positions 0, 2 → raw 1, 3
    assert isinstance(sub.index, MaterializedIndex)
    assert list(sub.index) == [1, 3]


# ---------- errors ----------

def test_iloc_bool_array_raises(tmp_path):
    df = make_df(tmp_path)
    mask = pa.array([True, False, True, False, True])
    with pytest.raises(NotImplementedError, match="bool_array"):
        df.iloc[mask]


def test_iloc_unsupported_type(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="unsupported key"):
        df.iloc["hello"]


# ---------- AST guard sanity: facade clean ----------

def test_iloc_indexer_is_separate_module():
    """_ILocIndexer живёт в cadw/indexers.py, не в dataframe.py."""
    import cadw.dataframe as df_mod
    import inspect
    src = inspect.getsource(df_mod._core)
    assert "class _ILocIndexer" not in src, (
        "_ILocIndexer должен быть в cadw/indexers.py, не в facade"
    )
