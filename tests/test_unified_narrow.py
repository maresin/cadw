"""v0.9.94-unified-narrow: CadwDataFrame.from_auto_result /
from_autoarray_result — wrap ML/ops producer output for chaining.

Non-breaking: producers keep returning AutoResult/AutoArrayResult
by default. These adapters let users lift any producer output into
a CadwDataFrame for further chaining.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.spill import AutoResult, AutoArrayResult


def _batches(table, bs=100):
    for i in range(0, table.num_rows, bs):
        yield table.slice(i, bs).to_batches()[0]


def _mk_groupby_auto(table, spill_dir):
    """v0.9.95: groupby_aggregate returns CadwDataFrame now (unified).
    Construct AutoResult manually for from_auto_result testing."""
    from cadw.core.spill import InMemoryResult
    return InMemoryResult(table)


def _mk_groupby_disk(table, spill_dir):
    """v0.9.95: manually construct OnDiskResult via SpillBuffer."""
    from cadw.core.spill import SpillBuffer
    buf = SpillBuffer(
        guard=MemoryGuard(max_memory_mb=64),
        schema=table.schema, name="test_disk", spill_ratio=0.0)
    for b in table.to_batches():
        buf.write_batch(b)
    return buf.finalize()


# ─── from_auto_result ─────────────────────────────────────────


def test_from_auto_result_basic(tmp_path):
    """from_auto_result wraps AutoResult (InMemoryResult | OnDiskResult)
    as CadwDataFrame.

    v0.9.95: groupby_aggregate migrated to CadwDataFrame return;
    no longer AutoResult. Construct InMemoryResult directly to test
    the adapter in isolation.
    """
    table = pa.table({"k": ["a", "b", "c"], "v": [1, 2, 3]})
    result = _mk_groupby_auto(table, tmp_path)
    assert isinstance(result, AutoResult)
    df = CadwDataFrame.from_auto_result(result)
    assert isinstance(df, CadwDataFrame)
    assert df.count() == 3
    assert "k" in df.columns
    assert "v" in df.columns
    batches = list(df._iter_filtered_batches())
    assert sum(b.num_rows for b in batches) == 3


def test_from_auto_result_spilled_safe(tmp_path):
    """OnDiskResult tmpdir must survive Dataset.from_batches
    (eager consumption) — no GC race."""
    table = pa.table({"k": ["a", "b", "c"], "v": [1, 2, 3]})
    result = _mk_groupby_disk(table, tmp_path)
    assert result.is_spilled
    df = CadwDataFrame.from_auto_result(result)
    # Read AFTER original result goes out of scope (if it did) —
    # Dataset owns its own copy in its own dir.
    assert df.count() == 3
    rows = list(df._iter_filtered_batches())
    total = sum(b.num_rows for b in rows)
    assert total == 3


def test_from_auto_result_wrong_type():
    table = pa.table({"a": [1, 2]})
    with pytest.raises(TypeError, match="expected AutoResult"):
        CadwDataFrame.from_auto_result(table)


def test_from_auto_result_custom_spill_dir(tmp_path):
    table = pa.table({"k": ["a", "b"], "v": [1, 2]})
    result = _mk_groupby_auto(table, tmp_path)
    out_dir = tmp_path / "custom_root"
    out_dir.mkdir()
    df = CadwDataFrame.from_auto_result(result, spill_dir=out_dir)
    # Dataset lives under out_dir/datasets/
    assert (out_dir / "datasets").exists()
    assert df.count() == 2


# ─── from_autoarray_result ───────────────────────────────────


def _mk_1d_autoarray(arr, spill_dir):
    """Simulate iforest.score_samples output (1-D AutoArrayResult)."""
    from cadw.core.spill import SpillArray
    buf = SpillArray(guard=MemoryGuard(max_memory_mb=64), width=1,
                     dtype=np.float64, name="test_1d",
                     spill_ratio=0.5)
    buf.write_array(arr)
    return buf.finalize()


def _mk_2d_autoarray(arr, spill_dir):
    """Simulate tfidf.transform_dense output (2-D AutoArrayResult)."""
    from cadw.core.spill import SpillArray
    k = arr.shape[1]
    buf = SpillArray(guard=MemoryGuard(max_memory_mb=64), width=k,
                     dtype=np.float64, name="test_2d",
                     spill_ratio=0.5)
    buf.write_array(arr)
    return buf.finalize()


def test_from_autoarray_result_1d():
    arr = np.array([1.5, 2.5, 3.5, 4.5])
    result = _mk_1d_autoarray(arr, None)
    assert isinstance(result, AutoArrayResult)
    df = CadwDataFrame.from_autoarray_result(result)
    assert df.count() == 4
    assert df.columns == ["c0"]
    # values preserved
    batches = list(df._iter_filtered_batches())
    vals = np.concatenate(
        [b.column("c0").to_numpy(zero_copy_only=False) for b in batches])
    assert np.allclose(vals, arr)


def test_from_autoarray_result_2d_with_names():
    arr = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]])
    result = _mk_2d_autoarray(arr, None)
    df = CadwDataFrame.from_autoarray_result(
        result, col_names=["pc0", "pc1", "pc2"])
    assert df.count() == 2
    assert df.columns == ["pc0", "pc1", "pc2"]


def test_from_autoarray_result_default_names():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = _mk_2d_autoarray(arr, None)
    df = CadwDataFrame.from_autoarray_result(result)
    assert df.columns == ["c0", "c1"]


def test_from_autoarray_result_col_names_mismatch():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = _mk_2d_autoarray(arr, None)
    with pytest.raises(ValueError, match="col_names length"):
        CadwDataFrame.from_autoarray_result(result, col_names=["a", "b", "c"])


def test_from_autoarray_result_wrong_type():
    table = pa.table({"a": [1, 2]})
    with pytest.raises(TypeError, match="expected AutoArrayResult"):
        CadwDataFrame.from_autoarray_result(table)


def test_from_autoarray_result_empty_requires_width():
    from cadw.core.spill import InMemoryArrayResult
    empty = InMemoryArrayResult(np.empty((0, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="pass width="):
        CadwDataFrame.from_autoarray_result(empty)


def test_from_autoarray_result_empty_with_width():
    from cadw.core.spill import InMemoryArrayResult
    empty = InMemoryArrayResult(np.empty((0, 3), dtype=np.float64))
    df = CadwDataFrame.from_autoarray_result(
        empty, width=3, col_names=["a", "b", "c"])
    assert df.count() == 0
    assert df.columns == ["a", "b", "c"]


def test_from_autoarray_result_spilled_safe():
    """OnDiskArrayResult → CadwDataFrame: tmpdir survives eager
    consumption by Dataset.from_batches."""
    from cadw.core.spill import SpillArray
    arr = np.arange(500, dtype=np.float64).reshape(-1, 1)
    buf = SpillArray(guard=MemoryGuard(max_memory_mb=64), width=1,
                     dtype=np.float64, name="test_spill",
                     spill_ratio=0.0)  # force immediate spill
    buf.write_array(arr)
    result = buf.finalize()
    assert result.is_spilled
    df = CadwDataFrame.from_autoarray_result(result)
    assert df.count() == 500
    batches = list(df._iter_filtered_batches())
    vals = np.concatenate(
        [b.column("c0").to_numpy(zero_copy_only=False) for b in batches])
    assert np.allclose(vals, arr.ravel())
