"""Тесты inner_join / CadwDataFrame.merge (11-B-1)."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, name, data, chunk=3):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _sorted_by(table, key):
    return sorted(table.to_pylist(), key=lambda r: r[key])


def test_merge_basic(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2, 3, 4], "a": ["x", "y", "z", "w"]})
    right = make_df(tmp_path, "R", {"k": [1, 3, 5], "b": ["p", "q", "r"]})
    result = left.merge(right, on="k")
    head = result.head(20)
    assert head.num_rows == 2
    rows = _sorted_by(head, "k")
    assert rows == [
        {"k": 1, "a": "x", "b": "p"},
        {"k": 3, "a": "z", "b": "q"},
    ]


def test_merge_multi_key(tmp_path):
    left = make_df(tmp_path, "L", {
        "k1": [1, 1, 2], "k2": ["a", "b", "a"], "v": [10, 20, 30],
    })
    right = make_df(tmp_path, "R", {
        "k1": [1, 2], "k2": ["a", "a"], "w": [100, 200],
    })
    result = left.merge(right, on=["k1", "k2"])
    head = result.head(20)
    rows = sorted(head.to_pylist(), key=lambda r: r["v"])
    assert rows == [
        {"k1": 1, "k2": "a", "v": 10, "w": 100},
        {"k1": 2, "k2": "a", "v": 30, "w": 200},
    ]


def test_merge_suffixes(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2], "v": [10, 20]})
    right = make_df(tmp_path, "R", {"k": [1, 2], "v": [100, 200]})
    result = left.merge(right, on="k", suffixes=("_l", "_r"))
    head = result.head(10)
    names = set(head.schema.names)
    assert "v_l" in names and "v_r" in names
    rows = _sorted_by(head, "k")
    assert rows[0]["v_l"] == 10 and rows[0]["v_r"] == 100


def test_merge_empty_right(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2], "a": [10, 20]})
    right_data = {
        "k": pa.array([], type=pa.int64()),
        "b": pa.array([], type=pa.string()),
    }
    right = make_df(tmp_path, "R", right_data)
    result = left.merge(right, on="k")
    assert result.shape[0] == 0


def test_merge_null_not_matched(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, None, 2], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [None, 2], "b": [100, 200]})
    result = left.merge(right, on="k")
    head = result.head(10)
    assert head.num_rows == 1
    rows = head.to_pylist()
    assert rows[0]["k"] == 2
    assert rows[0]["a"] == 30
    assert rows[0]["b"] == 200


def test_merge_cartesian_on_duplicates(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 1, 2], "a": ["x", "y", "z"]})
    right = make_df(tmp_path, "R", {"k": [1, 1], "b": ["p", "q"]})
    result = left.merge(right, on="k")
    # k=1: 2 left × 2 right = 4; k=2: 0
    assert result.shape[0] == 4


def test_merge_after_filter(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [1, 2, 3], "b": [100, 200, 300]})
    filtered = left[left["a"] > 15]  # k=2, 3
    result = filtered.merge(right, on="k")
    assert result.shape[0] == 2


def test_merge_missing_on_raises(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1], "a": [10]})
    right = make_df(tmp_path, "R", {"other": [1], "b": [100]})
    with pytest.raises(KeyError, match="not in right"):
        left.merge(right, on="k")


def test_merge_unsupported_how(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1], "a": [10]})
    right = make_df(tmp_path, "R", {"k": [1], "b": [100]})
    with pytest.raises(NotImplementedError, match="only 'inner'"):
        left.merge(right, on="k", how="left")


def test_merge_requires_cadwdataframe(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1], "a": [10]})
    with pytest.raises(TypeError, match="CadwDataFrame"):
        left.merge("not a df", on="k")
