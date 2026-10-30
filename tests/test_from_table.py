"""v0.9.6b1: CadwDataFrame.from_table + _InMemoryDataset."""
from pathlib import Path

import pyarrow as pa
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.dataset._in_memory import _InMemoryDataset


def test_from_table_basic():
    t = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))
    assert df.columns == ["a", "b"]
    assert df.count() == 3
    assert df.shape == (3, 2)


def test_from_table_is_in_memory_dataset():
    t = pa.table({"a": [1, 2]})
    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))
    assert isinstance(df._dataset, _InMemoryDataset)


def test_from_table_requires_table():
    with pytest.raises(TypeError, match="requires pa.Table"):
        CadwDataFrame.from_table([1, 2, 3])


def test_from_table_iter_batches():
    t = pa.table({"a": [1, 2, 3, 4, 5]})
    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))
    sizes = [b.num_rows for b in df.to_batches(batch_size=2)]
    assert sizes == [2, 2, 1]


def test_from_table_head():
    t = pa.table({"a": list(range(100))})
    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))
    h = df.head(5)
    assert h.num_rows == 5
    assert h.column("a").to_pylist() == [0, 1, 2, 3, 4]


def test_from_table_filter_works():
    t = pa.table({"a": list(range(10))})
    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))
    filtered = df[df["a"] > 5]
    assert filtered.count() == 4


def test_from_table_sort_values_fast_path():
    t = pa.table({"a": [3, 1, 4, 1, 5, 9, 2, 6]})
    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))
    sorted_df = df.sort_values("a")
    assert sorted_df["a"].to_list() == [1, 1, 2, 3, 4, 5, 6, 9]
    # Result is still in-memory (fast path stays in-memory)
    assert isinstance(sorted_df._dataset, _InMemoryDataset)


def test_from_table_sort_values_descending():
    t = pa.table({"a": [3, 1, 4, 1, 5]})
    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))
    sorted_df = df.sort_values("a", ascending=False)
    assert sorted_df["a"].to_list() == [5, 4, 3, 1, 1]


def test_from_table_multi_key_sort():
    t = pa.table({"a": [1, 2, 1, 2], "b": [10, 20, 5, 15]})
    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))
    sorted_df = df.sort_values(["a", "b"])
    assert sorted_df["a"].to_list() == [1, 1, 2, 2]
    assert sorted_df["b"].to_list() == [5, 10, 15, 20]


def test_from_table_groupby_returns_cadwdataframe():
    """v0.9.95: groupby returns CadwDataFrame; may be in-memory
    (from_producer_result via _InMemoryDataset) or disk-backed
    depending on spill. Both acceptable — type is unified."""
    t = pa.table({"g": ["a", "a", "b", "b"], "v": [1.0, 2.0, 3.0, 4.0]})
    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))
    result = df.groupby("g").agg("v", "mean")
    assert isinstance(result, CadwDataFrame)
    assert result.count() == 2


def test_from_table_add_column_raises():
    t = pa.table({"a": [1, 2]})
    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))
    with pytest.raises(NotImplementedError, match="read-only"):
        df._dataset.add_column("b", iter([]))


def test_from_table_write_parquet_roundtrip(tmp_path):
    t = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df = CadwDataFrame.from_table(t, memory_guard=MemoryGuard(max_memory_mb=10))
    out = tmp_path / "o.parquet"
    df.write_parquet(out)
    import pyarrow.parquet as pq
    reread = pq.read_table(out)
    assert reread.num_rows == 3
    assert reread.column("a").to_pylist() == [1, 2, 3]

