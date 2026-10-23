"""v0.9.26y2: CadwDataFrame.tail (Pandas parity, §13.4/314)."""
import pyarrow as pa
import pytest

from cadw import CadwDataFrame, MemoryGuard


def _mk(data, mem_mb=20):
    return CadwDataFrame.from_table(
        pa.table(data),
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
    )


def test_tail_basic():
    df = _mk({"x": [1, 2, 3, 4, 5]})
    t = df.tail(2)
    assert t.column("x").to_pylist() == [4, 5]


def test_tail_default_5():
    df = _mk({"x": list(range(10))})
    t = df.tail()
    assert t.column("x").to_pylist() == [5, 6, 7, 8, 9]


def test_tail_n_larger_than_len():
    df = _mk({"x": [1, 2]})
    t = df.tail(10)
    assert t.column("x").to_pylist() == [1, 2]


def test_tail_zero():
    df = _mk({"x": [1, 2, 3]})
    t = df.tail(0)
    assert t.num_rows == 0
    assert t.schema == df._dataset.schema


def test_tail_negative():
    """Pandas: tail(-k) = all rows except first k."""
    df = _mk({"x": [1, 2, 3, 4, 5]})
    t = df.tail(-2)
    assert t.column("x").to_pylist() == [3, 4, 5]


def test_tail_not_int():
    df = _mk({"x": [1]})
    with pytest.raises(TypeError, match="must be int"):
        df.tail(2.5)


def test_tail_filtered():
    df = _mk({"x": [1, 2, 3, 4, 5], "v": [10, 20, 30, 40, 50]})
    filtered = df[df["v"] > 20]  # x = [3, 4, 5]
    t = filtered.tail(2)
    assert t.column("x").to_pylist() == [4, 5]


def test_tail_empty():
    df = _mk({"x": []})
    t = df.tail(5)
    assert t.num_rows == 0


def test_tail_preserves_all_columns():
    df = _mk({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    t = df.tail(2)
    assert t.column("a").to_pylist() == [2, 3]
    assert t.column("b").to_pylist() == ["y", "z"]
