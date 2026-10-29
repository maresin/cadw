"""S1: SpillBuffer / AutoResult / SpillArray / AutoArrayResult (v0.9.90)."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.core.spill import (
    SpillBuffer, InMemoryResult, OnDiskResult,
    SpillArray, InMemoryArrayResult, OnDiskArrayResult,
)


def _make_batches(n_rows, batch_size):
    for i in range(0, n_rows, batch_size):
        end = min(i + batch_size, n_rows)
        yield pa.RecordBatch.from_arrays(
            [pa.array(np.arange(i, end, dtype=np.int64))],
            names=["c0"])


def _make_arrays(n_rows, batch_size, width=2):
    for i in range(0, n_rows, batch_size):
        end = min(i + batch_size, n_rows)
        yield np.arange(
            i * width, end * width, dtype=np.float64,
        ).reshape(-1, width)


def _mg_large():
    return MemoryGuard(max_memory_mb=64)


def _mg_tiny():
    # threshold = 1000 x 0.001 = 1 byte → first batch spills
    return MemoryGuard(max_memory_bytes=1000)


# ── SpillBuffer + InMemoryResult ──────────────────────────────

def test_spill_buffer_small_no_spill():
    mg = _mg_large()
    schema = pa.schema([("c0", pa.int64())])
    buf = SpillBuffer(guard=mg, schema=schema, spill_ratio=0.5)
    for b in _make_batches(1000, 100):
        buf.write_batch(b)
    assert not buf.is_spilled
    result = buf.finalize()
    assert isinstance(result, InMemoryResult)
    assert not result.is_spilled
    assert len(result) == 1000
    tbl = result.to_table()
    assert tbl.num_rows == 1000
    assert list(tbl.column("c0").to_pylist()) == list(range(1000))


def test_spill_buffer_empty():
    mg = _mg_large()
    schema = pa.schema([("c0", pa.int64())])
    buf = SpillBuffer(guard=mg, schema=schema)
    result = buf.finalize()
    assert isinstance(result, InMemoryResult)
    assert len(result) == 0


def test_spill_buffer_guard_none_no_spill():
    schema = pa.schema([("c0", pa.int64())])
    buf = SpillBuffer(guard=None, schema=schema, spill_ratio=0.5)
    for b in _make_batches(10_000, 1000):
        buf.write_batch(b)
    assert not buf.is_spilled
    result = buf.finalize()
    assert isinstance(result, InMemoryResult)
    assert len(result) == 10_000


# ── SpillBuffer + OnDiskResult ────────────────────────────────

def test_spill_buffer_triggers_spill():
    mg = _mg_tiny()
    schema = pa.schema([("c0", pa.int64())])
    buf = SpillBuffer(guard=mg, schema=schema, spill_ratio=0.001)
    for b in _make_batches(10_000, 100):
        buf.write_batch(b)
    assert buf.is_spilled
    result = buf.finalize()
    assert isinstance(result, OnDiskResult)
    assert result.is_spilled
    assert len(result) == 10_000
    with pytest.raises(MemoryLimitExceeded):
        result.to_table()
    total = sum(b.num_rows for b in result)
    assert total == 10_000
    result.close()


def test_spill_buffer_spills_only_once():
    mg = _mg_tiny()
    schema = pa.schema([("c0", pa.int64())])
    buf = SpillBuffer(guard=mg, schema=schema, spill_ratio=0.001)
    for b in _make_batches(500, 100):
        buf.write_batch(b)
    assert buf.is_spilled
    for b in _make_batches(100, 50):
        buf.write_batch(b)
    result = buf.finalize()
    assert isinstance(result, OnDiskResult)
    assert len(result) == 600
    result.close()


def test_spill_buffer_order_preserved():
    mg = _mg_tiny()
    schema = pa.schema([("c0", pa.int64())])
    buf = SpillBuffer(guard=mg, schema=schema, spill_ratio=0.001)
    for b in _make_batches(500, 50):
        buf.write_batch(b)
    result = buf.finalize()
    vals = []
    for b in result:
        vals.extend(b.column("c0").to_pylist())
    assert vals == list(range(500))
    result.close()


def test_spill_buffer_reiterable():
    mg = _mg_tiny()
    schema = pa.schema([("c0", pa.int64())])
    buf = SpillBuffer(guard=mg, schema=schema, spill_ratio=0.001)
    for b in _make_batches(500, 50):
        buf.write_batch(b)
    result = buf.finalize()
    total1 = sum(b.num_rows for b in result)
    total2 = sum(b.num_rows for b in result)
    assert total1 == total2 == 500
    result.close()


def test_spill_buffer_finalize_twice_raises():
    mg = _mg_large()
    schema = pa.schema([("c0", pa.int64())])
    buf = SpillBuffer(guard=mg, schema=schema)
    for b in _make_batches(100, 50):
        buf.write_batch(b)
    buf.finalize()
    with pytest.raises(RuntimeError, match="already finalized"):
        buf.finalize()


def test_spill_buffer_write_after_finalize_raises():
    mg = _mg_large()
    schema = pa.schema([("c0", pa.int64())])
    buf = SpillBuffer(guard=mg, schema=schema)
    for b in _make_batches(100, 50):
        buf.write_batch(b)
    buf.finalize()
    with pytest.raises(RuntimeError, match="already finalized"):
        buf.write_batch(next(_make_batches(10, 10)))


def test_ondisk_result_close_removes_tmpdir():
    mg = _mg_tiny()
    schema = pa.schema([("c0", pa.int64())])
    buf = SpillBuffer(guard=mg, schema=schema, spill_ratio=0.001,
                      name="test_cleanup")
    for b in _make_batches(500, 50):
        buf.write_batch(b)
    result = buf.finalize()
    assert isinstance(result, OnDiskResult)
    tmpdir = result._owned_tmpdir
    assert tmpdir is not None and tmpdir.exists()
    result.close()
    assert not tmpdir.exists()


def test_ondisk_result_context_manager():
    mg = _mg_tiny()
    schema = pa.schema([("c0", pa.int64())])
    buf = SpillBuffer(guard=mg, schema=schema, spill_ratio=0.001)
    for b in _make_batches(100, 50):
        buf.write_batch(b)
    result = buf.finalize()
    assert isinstance(result, OnDiskResult)
    tmpdir = result._owned_tmpdir
    with result:
        total = sum(b.num_rows for b in result)
    assert total == 100
    assert not tmpdir.exists()


# ── SpillArray + InMemoryArrayResult ─────────────────────────

def test_spill_array_small_no_spill():
    mg = _mg_large()
    buf = SpillArray(guard=mg, width=2, dtype=np.float64,
                     spill_ratio=0.5)
    for a in _make_arrays(100, 10, width=2):
        buf.write_array(a)
    assert not buf.is_spilled
    result = buf.finalize()
    assert isinstance(result, InMemoryArrayResult)
    assert len(result) == 100
    arr = result.to_numpy()
    assert arr.shape == (100, 2)


def test_spill_array_empty():
    mg = _mg_large()
    buf = SpillArray(guard=mg, width=3, dtype=np.float64)
    result = buf.finalize()
    assert isinstance(result, InMemoryArrayResult)
    assert len(result) == 0
    assert result.to_numpy().shape == (0, 3)


def test_spill_array_triggers_spill():
    mg = _mg_tiny()
    buf = SpillArray(guard=mg, width=4, dtype=np.float64,
                     spill_ratio=0.001)
    for a in _make_arrays(500, 50, width=4):
        buf.write_array(a)
    assert buf.is_spilled
    result = buf.finalize()
    assert isinstance(result, OnDiskArrayResult)
    assert len(result) == 500
    with pytest.raises(MemoryLimitExceeded):
        result.to_numpy()
    total = sum(arr.shape[0] for arr in result)
    assert total == 500
    result.close()


def test_spill_array_width_validation():
    mg = _mg_large()
    buf = SpillArray(guard=mg, width=3, dtype=np.float64)
    with pytest.raises(ValueError, match="width mismatch"):
        buf.write_array(np.zeros((10, 2), dtype=np.float64))


def test_spill_array_1d_reshape():
    mg = _mg_large()
    buf = SpillArray(guard=mg, width=1, dtype=np.float64)
    buf.write_array(np.array([1.0, 2.0, 3.0]))
    result = buf.finalize()
    assert result.to_numpy().shape == (3, 1)


def test_spill_array_1d_wrong_width():
    mg = _mg_large()
    buf = SpillArray(guard=mg, width=2, dtype=np.float64)
    with pytest.raises(ValueError):
        buf.write_array(np.array([1.0, 2.0, 3.0]))


def test_spill_array_reiterable_spilled():
    mg = _mg_tiny()
    buf = SpillArray(guard=mg, width=2, dtype=np.float64,
                     spill_ratio=0.001)
    for a in _make_arrays(200, 20, width=2):
        buf.write_array(a)
    result = buf.finalize()
    assert isinstance(result, OnDiskArrayResult)
    total1 = sum(a.shape[0] for a in result)
    total2 = sum(a.shape[0] for a in result)
    assert total1 == total2 == 200
    result.close()


def test_ondisk_array_result_close_removes_tmpdir():
    mg = _mg_tiny()
    buf = SpillArray(guard=mg, width=2, dtype=np.float64,
                     spill_ratio=0.001, name="test_arr_cleanup")
    for a in _make_arrays(200, 20, width=2):
        buf.write_array(a)
    result = buf.finalize()
    assert isinstance(result, OnDiskArrayResult)
    tmpdir = result._owned_tmpdir
    assert tmpdir is not None and tmpdir.exists()
    result.close()
    assert not tmpdir.exists()


# ── spill_ratio validation ───────────────────────────────────

def test_spill_ratio_out_of_range():
    mg = _mg_large()
    schema = pa.schema([("c0", pa.int64())])
    # v0.9.90-S2: 0.0 allowed (= spill on first batch); -0.1 invalid.
    with pytest.raises(ValueError, match="spill_ratio"):
        SpillBuffer(guard=mg, schema=schema, spill_ratio=-0.1)
    with pytest.raises(ValueError, match="spill_ratio"):
        SpillBuffer(guard=mg, schema=schema, spill_ratio=1.5)


def test_spill_ratio_zero_spills_immediately():
    """spill_ratio=0.0 -> threshold 0 -> first non-empty batch spills."""
    mg = _mg_large()  # big budget; ratio=0 forces spill anyway
    schema = pa.schema([("c0", pa.int64())])
    buf = SpillBuffer(guard=mg, schema=schema, spill_ratio=0.0)
    for b in _make_batches(100, 10):
        buf.write_batch(b)
    assert buf.is_spilled
    result = buf.finalize()
    assert isinstance(result, OnDiskResult)
    assert len(result) == 100
    result.close()
