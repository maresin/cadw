"""v0.6.1b: loc[tuple] — MultiIndex label access."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.dataset.index import (
    MaterializedIndex,
    MaterializedMultiIndex,
    RangeIndex,
)


def make_multi_df(tmp_path, data=None, chunk=2):
    if data is None:
        data = {
            "a": ["x", "x", "y", "y", "z"],
            "b": [1, 2, 1, 2, 1],
            "v": [10, 20, 30, 40, 50],
        }
    t = pa.table(data)
    p = Path(tmp_path) / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, Path(tmp_path) / "ds", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / "spill",
    )
    return df.set_index(["a", "b"], drop=False)


def write_read(df, tmp_path, cols=None):
    out = tmp_path / "o.parquet"
    df.write_parquet(out)
    return pq.read_table(out).to_pylist()


# ---------- loc[tuple] scalar ----------

def test_loc_tuple_scalar(tmp_path):
    df = make_multi_df(tmp_path)
    row = df.loc[("x", 1)]
    assert isinstance(row, dict)
    assert row == {"a": "x", "b": 1, "v": 10}


def test_loc_tuple_scalar_middle(tmp_path):
    df = make_multi_df(tmp_path)
    row = df.loc[("y", 2)]
    assert row == {"a": "y", "b": 2, "v": 40}


def test_loc_tuple_missing_raises(tmp_path):
    df = make_multi_df(tmp_path)
    with pytest.raises(KeyError, match="not in index"):
        df.loc[("x", 99)]


# ---------- loc[tuple_lo:tuple_hi] slice ----------

def test_loc_tuple_slice_inclusive(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[("x", 1):("y", 2)]
    rows = write_read(sub, tmp_path)
    assert [(r["a"], r["b"], r["v"]) for r in rows] == [
        ("x", 1, 10), ("x", 2, 20), ("y", 1, 30), ("y", 2, 40),
    ]


def test_loc_tuple_slice_single(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[("x", 1):("x", 1)]
    rows = write_read(sub, tmp_path)
    assert [(r["a"], r["b"]) for r in rows] == [("x", 1)]


def test_loc_tuple_slice_open_start(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[:("y", 1)]
    rows = write_read(sub, tmp_path)
    assert [(r["a"], r["b"]) for r in rows] == [
        ("x", 1), ("x", 2), ("y", 1),
    ]


def test_loc_tuple_slice_open_stop(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[("y", 2):]
    rows = write_read(sub, tmp_path)
    assert [(r["a"], r["b"]) for r in rows] == [("y", 2), ("z", 1)]


def test_loc_tuple_slice_open_both(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[:]
    rows = write_read(sub, tmp_path)
    assert len(rows) == 5


def test_loc_tuple_slice_reverse_empty(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[("y", 2):("x", 1)]
    rows = write_read(sub, tmp_path)
    assert rows == []


def test_loc_tuple_slice_missing_endpoint_raises(tmp_path):
    df = make_multi_df(tmp_path)
    with pytest.raises(KeyError, match="not in index"):
        df.loc[("x", 99):("y", 1)]
    with pytest.raises(KeyError, match="not in index"):
        df.loc[("x", 1):("y", 99)]


# ---------- loc[list of tuples] ----------

def test_loc_tuple_list(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[[("x", 1), ("y", 2), ("z", 1)]]
    rows = write_read(sub, tmp_path)
    assert [(r["a"], r["b"]) for r in rows] == [
        ("x", 1), ("y", 2), ("z", 1),
    ]


def test_loc_tuple_list_missing_raises(tmp_path):
    df = make_multi_df(tmp_path)
    with pytest.raises(KeyError, match="not in index"):
        df.loc[[("x", 1), ("x", 99)]]


def test_loc_tuple_list_empty(tmp_path):
    df = make_multi_df(tmp_path)
    sub = df.loc[[]]
    rows = write_read(sub, tmp_path)
    assert rows == []


# ---------- filtered view ----------

def test_loc_tuple_filtered(tmp_path):
    df = make_multi_df(tmp_path)
    filtered = df[df["v"] > 15]
    row = filtered.loc[("y", 1)]
    assert row == {"a": "y", "b": 1, "v": 30}
    with pytest.raises(KeyError):
        filtered.loc[("x", 1)]  # not in filtered


# ---------- duplicates → CadwDataFrame ----------

def test_loc_tuple_duplicates_returns_cadwdf(tmp_path):
    df = make_multi_df(tmp_path, data={
        "a": ["x", "x", "y"],
        "b": [1, 1, 2],
        "v": [10, 20, 30],
    })
    sub = df.loc[("x", 1)]
    assert isinstance(sub, CadwDataFrame)
    rows = write_read(sub, tmp_path)
    assert [r["v"] for r in rows] == [10, 20]


def test_loc_tuple_slice_duplicates_raises(tmp_path):
    df = make_multi_df(tmp_path, data={
        "a": ["x", "x", "y"],
        "b": [1, 1, 2],
        "v": [10, 20, 30],
    })
    with pytest.raises(NotImplementedError, match="unique index"):
        df.loc[("x", 1):("y", 2)]


# ---------- single-index regression ----------

def test_loc_single_index_still_works(tmp_path):
    """Single-index loc not broken by multi change."""
    t = pa.table({"k": ["a", "b", "c"], "v": [1, 2, 3]})
    p = Path(tmp_path) / "s.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, Path(tmp_path) / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / "spill",
    ).set_index("k", drop=False)

    assert isinstance(df.index, MaterializedIndex)
    assert df.loc["b"] == {"k": "b", "v": 2}


def test_loc_range_index_still_raises(tmp_path):
    """RangeIndex loc — TypeError (unchanged)."""
    t = pa.table({"v": [1, 2]})
    p = Path(tmp_path) / "r.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, Path(tmp_path) / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / "spill",
    )
    with pytest.raises(TypeError, match="requires MaterializedIndex"):
        df.loc[0]


def test_loc_tuple_on_single_index_raises_keyerror(tmp_path):
    """Tuple label on single-index — KeyError (not tuple index)."""
    t = pa.table({"k": ["a", "b"], "v": [1, 2]})
    p = Path(tmp_path) / "s2.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, Path(tmp_path) / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / "spill",
    ).set_index("k", drop=False)
    with pytest.raises(KeyError):
        df.loc[("a", 1)]


# ---------- multi via iloc list preserves ----------

def test_iloc_list_multi_loc_consistency(tmp_path):
    """iloc[list] and loc[list] agree on same rows."""
    df = make_multi_df(tmp_path)
    by_iloc = write_read(df.iloc[[0, 2, 4]], tmp_path)
    by_loc = write_read(
        df.loc[[("x", 1), ("y", 1), ("z", 1)]], tmp_path,
    )
    assert by_iloc == by_loc
