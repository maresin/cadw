"""v0.9.6b1: _ColumnProxy aggregators + materialize."""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def _df(table: pa.Table, *, chunk: int = 5) -> CadwDataFrame:
    # Disk-backed via from_table (in-memory; iteration via to_batches).
    return CadwDataFrame.from_table(
        table, memory_guard=MemoryGuard(max_memory_mb=10),
    )


# ============ scalar aggregators ============


def test_proxy_mean():
    df = _df(pa.table({"v": [1.0, 2.0, 3.0, 4.0, 5.0]}))
    assert df["v"].mean() == pytest.approx(3.0)


def test_proxy_sum():
    df = _df(pa.table({"v": [1.0, 2.0, 3.0, 4.0, 5.0]}))
    assert df["v"].sum() == pytest.approx(15.0)


def test_proxy_min():
    df = _df(pa.table({"v": [3.0, 1.0, 4.0, 2.0]}))
    assert df["v"].min() == 1.0


def test_proxy_max():
    df = _df(pa.table({"v": [3.0, 1.0, 4.0, 2.0]}))
    assert df["v"].max() == 4.0


def test_proxy_std_default_ddof():
    df = _df(pa.table({"v": [1.0, 2.0, 3.0, 4.0, 5.0]}))
    # ddof=1 → sqrt(2.5)
    assert df["v"].std() == pytest.approx(np.sqrt(2.5))


def test_proxy_std_ddof_0():
    df = _df(pa.table({"v": [1.0, 2.0, 3.0, 4.0, 5.0]}))
    # ddof=0 → sqrt(2.0)
    assert df["v"].std(ddof=0) == pytest.approx(np.sqrt(2.0))


def test_proxy_count_skips_nulls():
    df = _df(pa.table({"v": pa.array([1.0, None, 3.0, None, 5.0])}))
    assert df["v"].count() == 3


def test_proxy_aggregators_on_filtered_view():
    t = pa.table({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    df = _df(t)
    filtered = df[df["v"] > 2.0]
    assert filtered["v"].mean() == pytest.approx(4.0)
    assert filtered["v"].sum() == pytest.approx(12.0)


# ============ materialize ============


def test_proxy_to_list():
    df = _df(pa.table({"v": [1, 2, 3, 4, 5]}))
    assert df["v"].to_list() == [1, 2, 3, 4, 5]


def test_proxy_to_list_after_filter():
    df = _df(pa.table({"v": [1, 2, 3, 4, 5]}))
    filtered = df[df["v"] > 2]
    assert filtered["v"].to_list() == [3, 4, 5]


def test_proxy_to_numpy():
    df = _df(pa.table({"v": [1.0, 2.0, 3.0]}))
    arr = df["v"].to_numpy()
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])


def test_proxy_to_numpy_int():
    df = _df(pa.table({"v": [1, 2, 3]}))
    arr = df["v"].to_numpy()
    np.testing.assert_array_equal(arr, [1, 2, 3])


def test_proxy_to_arrow():
    df = _df(pa.table({"v": [1, 2, 3]}))
    arr = df["v"].to_arrow()
    assert isinstance(arr, pa.Array)
    assert arr.to_pylist() == [1, 2, 3]


def test_proxy_to_arrow_multi_chunk():
    # 10 rows; iter_batches yields 5 at a time; to_arrow concatenates.
    t = pa.table({"v": list(range(10))})
    # Force multi-chunk by writing table with 2 chunks
    t2 = pa.concat_tables([t.slice(0, 3), t.slice(3)])
    df = _df(t2)
    arr = df["v"].to_arrow()
    assert arr.to_pylist() == list(range(10))


def test_proxy_to_list_guard_fires_on_tight_budget():
    t = pa.table({"v": list(range(1000))})
    df = CadwDataFrame.from_table(
        t, memory_guard=MemoryGuard(max_memory_bytes=2000),
    )
    with pytest.raises(MemoryLimitExceeded, match="to_list"):
        df["v"].to_list()


def test_proxy_to_arrow_guard_fires_on_tight_budget():
    t = pa.table({"v": list(range(1000))})
    df = CadwDataFrame.from_table(
        t, memory_guard=MemoryGuard(max_memory_bytes=2000),
    )
    with pytest.raises(MemoryLimitExceeded, match="to_arrow"):
        df["v"].to_arrow()


def test_proxy_aggregators_multi_batch():
    """Aggregators work across batch boundaries."""
    t = pa.concat_tables([
        pa.table({"v": [1.0, 2.0, 3.0]}),
        pa.table({"v": [4.0, 5.0]}),
    ])
    df = _df(t)
    assert df["v"].sum() == pytest.approx(15.0)
    assert df["v"].count() == 5

