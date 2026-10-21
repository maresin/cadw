"""Тесты CadwDataFrame.loc / .at / .iat + MaterializedIndex.get_indexer (v0.4.0c)."""
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


# ---------- MaterializedIndex.get_indexer ----------

def test_get_indexer_scalar():
    idx = MaterializedIndex(pa.array([10, 20, 30]))
    assert idx.get_indexer(20) == [1]
    assert idx.get_indexer(99) == [-1]


def test_get_indexer_list():
    idx = MaterializedIndex(pa.array([10, 20, 30]))
    assert idx.get_indexer([10, 30]) == [0, 2]
    assert idx.get_indexer([10, 99, 30]) == [0, -1, 2]


def test_get_indexer_duplicate_first_wins():
    idx = MaterializedIndex(pa.array([10, 20, 10]))
    assert idx.get_indexer(10) == [0]  # first occurrence (MVP)


def test_get_indexer_string():
    idx = MaterializedIndex(pa.array(["x", "y", "z"]))
    assert idx.get_indexer("y") == [1]
    assert idx.get_indexer(["z", "x"]) == [2, 0]


# ---------- loc on RangeIndex raises ----------

def test_loc_on_range_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="requires MaterializedIndex"):
        df.loc["a"]


# ---------- loc[label] ----------

def test_loc_scalar(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    assert df.loc["c"] == {"k": "c", "v": 30}
    assert df.loc["a"] == {"k": "a", "v": 10}


def test_loc_scalar_missing_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not in index"):
        df.loc["zzz"]


def test_loc_scalar_int_label(tmp_path):
    df = make_df(tmp_path, data={
        "id": [100, 200, 300], "v": [1, 2, 3],
    }).set_index("id", drop=False)
    assert df.loc[200] == {"id": 200, "v": 2}


# ---------- loc[lo:hi] (inclusive) ----------

def test_loc_slice_inclusive(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc["b":"d"]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["b", "c", "d"]


def test_loc_slice_single(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc["b":"b"]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["b"]


def test_loc_slice_reverse_returns_empty(tmp_path):
    """loc['d':'b'] — пустой (не ошибка, как pandas)."""
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc["d":"b"]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows == []


def test_loc_slice_missing_lo_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not in index"):
        df.loc["zzz":"d"]


def test_loc_slice_missing_hi_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not in index"):
        df.loc["b":"zzz"]


def test_loc_slice_open_start(tmp_path):
    """v0.4.1: loc[:'c'] — от начала до c включительно."""
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc[:"c"]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["a", "b", "c"]


def test_loc_slice_open_stop(tmp_path):
    """v0.4.1: loc['c':] — от c до конца."""
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc["c":]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["c", "d", "e"]


def test_loc_slice_open_both(tmp_path):
    """v0.4.1: loc[:] — все строки."""
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc[:]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["a", "b", "c", "d", "e"]


def test_loc_slice_open_start_missing_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not in index"):
        df.loc[:"zzz"]


def test_loc_slice_open_stop_missing_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not in index"):
        df.loc["zzz":]


def test_loc_slice_descending(tmp_path):
    """v0.4.1: descending monotonic index — loc['b':'d'] returns d,c,b."""
    df = make_df(tmp_path, data={
        "k": ["e", "d", "c", "b", "a"],
        "v": [50, 40, 30, 20, 10],
    }).set_index("k", drop=False)
    # descending index ['e','d','c','b','a']
    sub = df.loc["b":"d"]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["d", "c", "b"]


def test_loc_slice_descending_open(tmp_path):
    """descending + open: loc[:'c'] → ['e','d','c']; loc['c':] → ['c','b','a']."""
    df = make_df(tmp_path, data={
        "k": ["e", "d", "c", "b", "a"],
        "v": [50, 40, 30, 20, 10],
    }).set_index("k", drop=False)
    sub1 = df.loc[:"c"]
    sub2 = df.loc["c":]
    out1 = tmp_path / "o1.parquet"
    out2 = tmp_path / "o2.parquet"
    sub1.write_parquet(out1)
    sub2.write_parquet(out2)
    assert [r["k"] for r in pq.read_table(out1).to_pylist()] == ["e", "d", "c"]
    assert [r["k"] for r in pq.read_table(out2).to_pylist()] == ["c", "b", "a"]


def test_loc_slice_non_monotonic_raises(tmp_path):
    """Не monotonic index → NotImplementedError на slice."""
    df = make_df(tmp_path, data={
        "k": ["c", "a", "b"],
        "v": [30, 10, 20],
    }).set_index("k", drop=False)
    with pytest.raises(NotImplementedError, match="monotonic"):
        df.loc["a":"b"]


# ---------- v0.4.1: duplicate labels ----------

def test_loc_scalar_unique_returns_dict(tmp_path):
    """Уникальный index: loc[label] → dict."""
    df = make_df(tmp_path).set_index("k", drop=False)
    row = df.loc["c"]
    assert isinstance(row, dict)
    assert row == {"k": "c", "v": 30}


def test_loc_scalar_duplicate_returns_cadwdf(tmp_path):
    """Duplicate labels: loc[label] → CadwDataFrame (pandas)."""
    df = make_df(tmp_path, data={
        "k": ["a", "a", "b", "c"],
        "v": [10, 20, 30, 40],
    }).set_index("k", drop=False)
    sub = df.loc["a"]
    assert isinstance(sub, CadwDataFrame)
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [(r["k"], r["v"]) for r in rows] == [("a", 10), ("a", 20)]


def test_loc_scalar_duplicate_three_matches(tmp_path):
    df = make_df(tmp_path, data={
        "k": ["x", "y", "x", "z", "x"],
        "v": [1, 2, 3, 4, 5],
    }).set_index("k", drop=False)
    sub = df.loc["x"]
    assert isinstance(sub, CadwDataFrame)
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["v"] for r in rows] == [1, 3, 5]


def test_loc_list_with_duplicates(tmp_path):
    """loc[['a']] на duplicate index → все вхождения 'a'."""
    df = make_df(tmp_path, data={
        "k": ["a", "a", "b"],
        "v": [1, 2, 3],
    }).set_index("k", drop=False)
    sub = df.loc[["a"]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [(r["k"], r["v"]) for r in rows] == [("a", 1), ("a", 2)]


def test_loc_slice_with_duplicates_raises(tmp_path):
    """loc[slice] на duplicate index → NotImplementedError (pandas semantics)."""
    df = make_df(tmp_path, data={
        "k": ["a", "a", "b"],
        "v": [1, 2, 3],
    }).set_index("k", drop=False)
    with pytest.raises(NotImplementedError, match="unique index"):
        df.loc["a":"b"]


def test_at_duplicate_label_raises(tmp_path):
    """`.at[label, col]` на duplicate label → KeyError (honest)."""
    df = make_df(tmp_path, data={
        "k": ["a", "a", "b"],
        "v": [1, 2, 3],
    }).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not unique"):
        df.at["a", "v"]


# ---------- v0.4.1: MaterializedIndex monotonic / has_duplicates / get_positions_all ----------

def test_index_monotonic_increasing():
    idx = MaterializedIndex(pa.array([1, 2, 3, 4]))
    assert idx.is_monotonic_increasing()
    assert not idx.is_monotonic_decreasing()
    assert idx.is_monotonic()


def test_index_monotonic_decreasing():
    idx = MaterializedIndex(pa.array([4, 3, 2, 1]))
    assert not idx.is_monotonic_increasing()
    assert idx.is_monotonic_decreasing()
    assert idx.is_monotonic()


def test_index_not_monotonic():
    idx = MaterializedIndex(pa.array([1, 3, 2, 4]))
    assert not idx.is_monotonic_increasing()
    assert not idx.is_monotonic_decreasing()
    assert not idx.is_monotonic()


def test_index_monotonic_single():
    idx = MaterializedIndex(pa.array([5]))
    assert idx.is_monotonic_increasing()
    assert idx.is_monotonic_decreasing()
    assert idx.is_monotonic()


def test_index_has_duplicates_yes():
    idx = MaterializedIndex(pa.array(["a", "b", "a"]))
    assert idx.has_duplicates()


def test_index_has_duplicates_no():
    idx = MaterializedIndex(pa.array(["a", "b", "c"]))
    assert not idx.has_duplicates()


def test_get_positions_all_single():
    idx = MaterializedIndex(pa.array([10, 20, 30]))
    assert idx.get_positions_all(20) == [1]


def test_get_positions_all_multiple():
    idx = MaterializedIndex(pa.array(["a", "b", "a", "c", "a"]))
    assert idx.get_positions_all("a") == [0, 2, 4]


def test_get_positions_all_missing():
    idx = MaterializedIndex(pa.array([10, 20, 30]))
    assert idx.get_positions_all(99) == []


def test_get_positions_all_string_index_type_strict():
    """v0.9.18-fix3 (§13.4/229): string index + int label → no match."""
    idx = MaterializedIndex(pa.array(["1", "2", "3"]))
    assert idx.get_positions_all(1) == []
    assert idx.get_positions_all("1") == [0]


def test_get_positions_all_tuple_label_single_index():
    """v0.9.18-fix3 (§13.4/229): non-scalar label on single index → no match."""
    idx = MaterializedIndex(pa.array(["a", "b", "c"]))
    assert idx.get_positions_all(("a", 1)) == []
    assert idx.get_positions_all(["a"]) == []


# ---------- loc[[labels]] ----------

def test_loc_list(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc[["a", "c", "e"]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["a", "c", "e"]


def test_loc_list_missing_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not in index"):
        df.loc[["a", "zzz"]]


def test_loc_list_duplicates(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc[["a", "a"]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["a", "a"]


def test_loc_list_empty(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc[[]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows == []


# ---------- loc[bool_array] raises ----------

def test_loc_bool_array_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    mask = pa.array([True, False, True, False, True])
    with pytest.raises(NotImplementedError, match="bool_array"):
        df.loc[mask]


# ---------- .at ----------

def test_at_scalar(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    assert df.at["b", "v"] == 20
    assert df.at["e", "k"] == "e"


def test_at_missing_label_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not in index"):
        df.at["zzz", "v"]


def test_at_missing_col_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="column"):
        df.at["a", "nope"]


def test_at_wrong_key_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(TypeError, match="must be \\(label, col\\)"):
        df.at["a"]


def test_at_setitem_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(NotImplementedError, match="immutable"):
        df.at["a", "v"] = 100


# ---------- .iat ----------

def test_iat_scalar(tmp_path):
    df = make_df(tmp_path)
    assert df.iat[0, "v"] == 10
    assert df.iat[4, "k"] == "e"
    assert df.iat[-1, "v"] == 50


def test_iat_missing_col_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="column"):
        df.iat[0, "nope"]


def test_iat_out_of_range_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(IndexError):
        df.iat[10, "v"]


def test_iat_wrong_key_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="must be \\(pos, col\\)"):
        df.iat[0]


def test_iat_setitem_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(NotImplementedError, match="immutable"):
        df.iat[0, "v"] = 100


# ---------- filter limitation ----------

def test_loc_on_filtered_works(tmp_path):
    """v0.4.0d: loc на filtered view работает (через filtered.index)."""
    df = make_df(tmp_path).set_index("k", drop=False)
    filtered = df[df["v"] > 15]  # b, c, d, e
    assert filtered.loc["c"] == {"k": "c", "v": 30}
    assert filtered.loc["e"] == {"k": "e", "v": 50}
    with pytest.raises(KeyError):
        filtered.loc["a"]  # a не в filtered


# ---------- loc preserves index ----------

def test_loc_list_preserves_index(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc[["a", "c", "e"]]
    assert isinstance(sub.index, MaterializedIndex)
    assert list(sub.index) == ["a", "c", "e"]
