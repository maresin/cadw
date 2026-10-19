"""v0.6.1a: MaterializedMultiIndex + set_index(list) + reset_index explode."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.dataset.index import (
    RangeIndex,
    MaterializedIndex,
    MaterializedMultiIndex,
)


def make_df(tmp_path, name="L", data=None, chunk=3):
    if data is None:
        data = {
            "a": ["x", "x", "y", "y", "z"],
            "b": [1, 2, 1, 2, 1],
            "v": [10, 20, 30, 40, 50],
        }
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- MaterializedMultiIndex ----------

def test_multi_index_nlevels():
    idx = MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1, 2])])
    assert idx.nlevels == 2 and len(idx) == 2


def test_multi_index_getitem_int():
    idx = MaterializedMultiIndex([
        pa.array(["a", "b", "c"]), pa.array([1, 2, 3]),
    ])
    assert idx[0] == ("a", 1) and idx[-1] == ("c", 3)


def test_multi_index_getitem_slice():
    idx = MaterializedMultiIndex([
        pa.array(["a", "b", "c"]), pa.array([1, 2, 3]),
    ])
    sub = idx[1:3]
    assert isinstance(sub, MaterializedMultiIndex)
    assert list(sub) == [("b", 2), ("c", 3)]


def test_multi_index_iter():
    idx = MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1, 2])])
    assert list(idx) == [("a", 1), ("b", 2)]


def test_multi_index_to_arrow():
    idx = MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1, 2])])
    arr = idx.to_arrow()
    assert pa.types.is_struct(arr.type)
    assert arr.type.num_fields == 2
    assert arr[0].as_py() == {"level_0": "a", "level_1": 1}


def test_multi_index_equals():
    a = MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1, 2])])
    b = MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1, 2])])
    c = MaterializedMultiIndex([pa.array(["a", "c"]), pa.array([1, 2])])
    assert a.equals(b) and not a.equals(c)


def test_multi_index_has_duplicates():
    a = MaterializedMultiIndex([pa.array(["a", "b", "a"]), pa.array([1, 2, 1])])
    assert a.has_duplicates()
    b = MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1, 2])])
    assert not b.has_duplicates()


def test_multi_index_get_indexer():
    idx = MaterializedMultiIndex([
        pa.array(["a", "b", "c"]), pa.array([1, 2, 3]),
    ])
    assert idx.get_indexer(("b", 2)) == [1]
    assert idx.get_indexer(("z", 99)) == [-1]
    assert idx.get_indexer([("a", 1), ("c", 3)]) == [0, 2]


def test_multi_index_monotonic():
    a = MaterializedMultiIndex([
        pa.array(["a", "a", "b"]), pa.array([1, 2, 1]),
    ])
    assert a.is_monotonic_increasing()
    assert not a.is_monotonic_decreasing()


def test_multi_index_requires_two_levels():
    with pytest.raises(ValueError, match="at least 2"):
        MaterializedMultiIndex([pa.array(["a"])])


def test_multi_index_length_mismatch():
    with pytest.raises(ValueError, match="level 1"):
        MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1])])


# ---------- set_index multi ----------

def test_set_index_multi_drop_true(tmp_path):
    df = make_df(tmp_path)
    new = df.set_index(["a", "b"], drop=True)
    assert isinstance(new.index, MaterializedMultiIndex)
    assert new.index.nlevels == 2
    assert list(new.index) == [("x", 1), ("x", 2), ("y", 1), ("y", 2), ("z", 1)]
    assert "a" not in new.columns and "b" not in new.columns
    assert "v" in new.columns
    assert "a" in df.columns


def test_set_index_multi_drop_false(tmp_path):
    df = make_df(tmp_path)
    new = df.set_index(["a", "b"], drop=False)
    assert isinstance(new.index, MaterializedMultiIndex)
    assert "a" in new.columns and "b" in new.columns


def test_set_index_multi_inplace(tmp_path):
    df = make_df(tmp_path)
    result = df.set_index(["a", "b"], drop=True, inplace=True)
    assert result is df
    assert isinstance(df.index, MaterializedMultiIndex)


def test_set_index_single_still_works(tmp_path):
    df = make_df(tmp_path)
    new = df.set_index("a", drop=False)
    assert isinstance(new.index, MaterializedIndex)
    assert list(new.index) == ["x", "x", "y", "y", "z"]


# ---------- reset_index multi ----------

def test_reset_index_multi_drop_true(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index(["a", "b"], drop=True)
    restored = with_idx.reset_index(drop=True)
    assert isinstance(restored.index, RangeIndex)
    assert restored.index.stop == 5


def test_reset_index_multi_drop_false(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index(["a", "b"], drop=True)
    restored = with_idx.reset_index(drop=False)
    assert isinstance(restored.index, RangeIndex)
    assert "level_0" in restored.columns and "level_1" in restored.columns
    out = tmp_path / "o.parquet"
    restored.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["level_0"] for r in rows] == ["x", "x", "y", "y", "z"]
    assert [r["level_1"] for r in rows] == [1, 2, 1, 2, 1]


def test_reset_index_multi_collision(tmp_path):
    df = make_df(tmp_path, data={
        "a": ["x", "y"], "b": [1, 2],
        "level_0": [100, 200], "v": [10, 20],
    })
    with_idx = df.set_index(["a", "b"], drop=True)
    restored = with_idx.reset_index(drop=False)
    assert "level_0" in restored.columns
    assert "level_0_0" in restored.columns


# ---------- filter + iloc + parquet ----------

def test_index_multi_with_filter(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index(["a", "b"], drop=False)
    filtered = with_idx[with_idx["v"] > 15]
    idx = filtered.index
    assert isinstance(idx, MaterializedMultiIndex)
    assert list(idx) == [("x", 2), ("y", 1), ("y", 2), ("z", 1)]


def test_iloc_list_preserves_multi_index(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index(["a", "b"], drop=False)
    sub = with_idx.iloc[[0, 2, 4]]
    assert isinstance(sub.index, MaterializedMultiIndex)
    assert list(sub.index) == [("x", 1), ("y", 1), ("z", 1)]


def test_multi_index_via_parquet(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index(["a", "b"], drop=True)
    restored = with_idx.reset_index(drop=False)
    out = tmp_path / "o.parquet"
    restored.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 5
    assert "level_0" in rows[0] and "level_1" in rows[0]
