"""
CadwIndex (v0.4.0a): RangeIndex, MaterializedIndex, df.index, set_index,
reset_index. Все сценарии без активных фильтров (MVP).
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.dataset.index import CadwIndex, MaterializedIndex, RangeIndex


def make_df(tmp_path, name="L", data=None, chunk=3):
    if data is None:
        data = {"k": ["a", "b", "c", "d", "e"], "v": [1, 2, 3, 4, 5]}
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- RangeIndex class ----------

def test_range_index_len():
    assert len(RangeIndex(0, 10)) == 10
    assert len(RangeIndex(5, 15)) == 10
    assert len(RangeIndex(0, 10, 2)) == 5
    assert len(RangeIndex(0, 0)) == 0


def test_range_index_getitem():
    r = RangeIndex(10, 20)
    assert r[0] == 10
    assert r[5] == 15
    assert r[-1] == 19
    with pytest.raises(IndexError):
        _ = r[100]


def test_range_index_slice():
    r = RangeIndex(0, 100)
    sub = r[10:20]
    assert isinstance(sub, RangeIndex)
    assert sub.start == 10
    assert sub.stop == 20
    assert len(sub) == 10


def test_range_index_iter():
    r = RangeIndex(3, 7)
    assert list(r) == [3, 4, 5, 6]


def test_range_index_invalid():
    with pytest.raises(ValueError):
        RangeIndex(0, 10, step=0)
    with pytest.raises(ValueError):
        RangeIndex(-1, 10)
    with pytest.raises(ValueError):
        RangeIndex(10, 5)


# ---------- MaterializedIndex class ----------

def test_materialized_index_basic():
    idx = MaterializedIndex(pa.array([10, 20, 30]))
    assert len(idx) == 3
    assert idx[0] == 10
    assert idx[2] == 30
    assert list(idx) == [10, 20, 30]


def test_materialized_index_string():
    idx = MaterializedIndex(pa.array(["x", "y", "z"]))
    assert list(idx) == ["x", "y", "z"]
    assert idx[1] == "y"


# ---------- df.index default ----------

def test_default_index_is_range(tmp_path):
    df = make_df(tmp_path)
    idx = df.index
    assert isinstance(idx, RangeIndex)
    assert idx.start == 0
    assert idx.stop == 5
    assert len(idx) == 5


# ---------- set_index ----------

def test_set_index_drop_true(tmp_path):
    df = make_df(tmp_path)
    new = df.set_index("k", drop=True)
    assert isinstance(new.index, MaterializedIndex)
    assert list(new.index) == ["a", "b", "c", "d", "e"]
    assert "k" not in new.columns
    assert "v" in new.columns
    # Исходный df не изменён
    assert "k" in df.columns
    assert isinstance(df.index, RangeIndex)


def test_set_index_drop_false(tmp_path):
    df = make_df(tmp_path)
    new = df.set_index("k", drop=False)
    assert isinstance(new.index, MaterializedIndex)
    assert list(new.index) == ["a", "b", "c", "d", "e"]
    assert "k" in new.columns
    assert "v" in new.columns


def test_set_index_inplace(tmp_path):
    df = make_df(tmp_path)
    result = df.set_index("k", drop=True, inplace=True)
    assert result is df
    assert isinstance(df.index, MaterializedIndex)
    assert "k" not in df.columns


def test_set_index_missing_col_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.set_index("nope")


def test_set_index_wrong_type_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="col must be str"):
        df.set_index(123)


# ---------- reset_index ----------

def test_reset_index_drop_true(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=True)
    restored = with_idx.reset_index(drop=True)
    assert isinstance(restored.index, RangeIndex)
    assert restored.index.start == 0
    assert restored.index.stop == 5
    # k не восстановлен (drop=True)
    assert "k" not in restored.columns
    assert "v" in restored.columns


def test_reset_index_drop_false(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=True)
    restored = with_idx.reset_index(drop=False)
    assert isinstance(restored.index, RangeIndex)
    # Колонка "index" добавлена со значениями
    assert "index" in restored.columns
    # Check values through parquet roundtrip
    out = tmp_path / "out.parquet"
    restored.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["index"] for r in rows] == ["a", "b", "c", "d", "e"]


def test_reset_index_collision_level_0(tmp_path):
    """Если колонка "index" уже есть — используется "level_0"."""
    df = make_df(tmp_path, data={
        "k": ["a", "b"],
        "index": [100, 200],
        "v": [1, 2],
    })
    with_idx = df.set_index("k", drop=True)
    restored = with_idx.reset_index(drop=False)
    assert "level_0" in restored.columns
    assert "index" in restored.columns  # исходная колонка
    out = tmp_path / "out.parquet"
    restored.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["level_0"] for r in rows] == ["a", "b"]
    assert [r["index"] for r in rows] == [100, 200]


def test_reset_index_inplace(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=True)
    result = with_idx.reset_index(drop=True, inplace=True)
    assert result is with_idx
    assert isinstance(with_idx.index, RangeIndex)


def test_reset_index_on_range_index_drop_true_noop(tmp_path):
    df = make_df(tmp_path)
    same = df.reset_index(drop=True)
    assert isinstance(same.index, RangeIndex)
    assert list(same.index) == [0, 1, 2, 3, 4]


def test_reset_index_on_range_index_drop_false_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="already"):
        df.reset_index(drop=False)


# ---------- chained ----------

def test_set_index_then_select_then_reset(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=True)
    # select subset (no filter) — index сохраняется
    sub = with_idx[["v"]]
    assert isinstance(sub.index, MaterializedIndex)
    assert list(sub.index) == ["a", "b", "c", "d", "e"]
    # reset_index
    restored = sub.reset_index(drop=False)
    out = tmp_path / "out.parquet"
    restored.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["index"] for r in rows] == ["a", "b", "c", "d", "e"]
    assert [r["v"] for r in rows] == [1, 2, 3, 4, 5]


# ---------- filter limitation (MVP) ----------

def test_index_with_filter_preserves(tmp_path):
    """v0.4.0d: filtered.index = MaterializedIndex subset."""
    from cadw.dataset.index import MaterializedIndex
    df = make_df(tmp_path)
    filtered = df[df["v"] > 2]  # rows c, d, e → positions 2, 3, 4
    idx = filtered.index
    assert isinstance(idx, MaterializedIndex)
    assert list(idx) == [2, 3, 4]


def test_set_index_with_filter_works(tmp_path):
    """v0.4.0d: set_index при фильтрах — материализует full col."""
    from cadw.dataset.index import MaterializedIndex
    df = make_df(tmp_path)
    filtered = df[df["v"] > 2]  # c, d, e
    result = filtered.set_index("k", drop=False)
    # index values = filtered view k values (через positions)
    assert isinstance(result.index, MaterializedIndex)
    assert list(result.index) == ["c", "d", "e"]
    # columns содержат k, v
    assert set(result.columns) == {"k", "v"}


def test_reset_index_with_filter_works(tmp_path):
    """v0.4.0d: reset_index при фильтрах → RangeIndex(0, filtered_n)."""
    from cadw.dataset.index import RangeIndex
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=True)
    sub = with_idx[["v"]]
    filtered = sub[sub["v"] > 2]  # rows v=30, 40, 50 → filtered_n = 3
    restored = filtered.reset_index(drop=True)
    assert isinstance(restored.index, RangeIndex)
    assert restored.index.start == 0
    assert restored.index.stop == 3


# ---------- after operations: index reset ----------

def test_index_reset_after_groupby(tmp_path):
    df = make_df(tmp_path, data={
        "g": ["A", "A", "B", "B"],
        "v": [1, 2, 3, 4],
    })
    result = df.groupby("g").agg("v", "sum")
    # groupby результат — новый RangeIndex (как pandas)
    assert isinstance(result.index, RangeIndex)
    assert result.index.start == 0
    assert result.index.stop == 2


def test_index_reset_after_sort(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=True)
    sorted_df = with_idx.sort_values("v", ascending=False)
    # sort_values создаёт новый Dataset → RangeIndex reset
    assert isinstance(sorted_df.index, RangeIndex)


# ---------- persistence: index not serialized to parquet ----------

def test_index_not_persisted_in_parquet(tmp_path):
    """Parquet не хранит CadwIndex; write_parquet → reload → RangeIndex."""
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=False)
    out = tmp_path / "out.parquet"
    with_idx.write_parquet(out)
    reloaded = CadwDataFrame.from_parquet(
        out, tmp_path / "ds2", chunk_size=3,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill2",
    )
    assert isinstance(reloaded.index, RangeIndex)
