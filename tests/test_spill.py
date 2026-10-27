"""Tests for cadw.core.spill — SpillDir + array adapters (v0.9.87)."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from cadw.core.spill import (
    SpillDir,
    array_schema,
    array_to_record_batch,
    record_batch_to_numpy,
    spill_array_reader,
    spill_array_writer,
)


# ── SpillDir lifecycle ──────────────────────────────────────────


def test_spilldir_creates_and_deletes():
    with SpillDir(name="t1") as sdir:
        root = sdir.root
        assert root.exists()
        (sdir.path("x.txt")).write_text("hello")
    assert not root.exists()


def test_spilldir_path_outside_context_raises():
    sdir = SpillDir(name="t2")
    with pytest.raises(RuntimeError, match="outside context"):
        sdir.path("x")


def test_spilldir_root_outside_context_raises():
    sdir = SpillDir(name="t3")
    with pytest.raises(RuntimeError, match="outside context"):
        _ = sdir.root


def test_spilldir_explicit_path_preserved():
    tmp = Path(tempfile.mkdtemp(prefix="cadw_explicit_"))
    try:
        with SpillDir(name="t4", path=tmp) as sdir:
            assert sdir.root == tmp
            (sdir.path("x.txt")).write_text("hello")
        # explicit path is NOT auto-deleted
        assert tmp.exists()
        assert (tmp / "x.txt").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_spilldir_cleanup_file():
    with SpillDir(name="t5") as sdir:
        p = sdir.path("x.txt")
        p.write_text("hello")
        assert p.exists()
        sdir.cleanup_file(p)
        assert not p.exists()
        sdir.cleanup_file(p)  # idempotent — no raise


def test_spilldir_keeps_on_error():
    kept_path = None
    with pytest.raises(RuntimeError, match="boom"):
        with SpillDir(name="t6", keep_on_error=True) as sdir:
            kept_path = sdir.root
            raise RuntimeError("boom")
    assert kept_path is not None and kept_path.exists()
    shutil.rmtree(kept_path, ignore_errors=True)


def test_spilldir_removes_on_error_when_disabled():
    root_ref = None
    with pytest.raises(RuntimeError, match="boom"):
        with SpillDir(name="t7", keep_on_error=False) as sdir:
            root_ref = sdir.root
            raise RuntimeError("boom")
    assert root_ref is not None and not root_ref.exists()


# ── array_schema ────────────────────────────────────────────────


def test_array_schema_2d_shape():
    schema = array_schema((100, 5), np.float64)
    assert len(schema) == 5
    assert schema.names == ["c0", "c1", "c2", "c3", "c4"]
    assert schema.field("c0").type == pa.float64()


def test_array_schema_int_width():
    schema = array_schema(3, np.int64)
    assert len(schema) == 3
    assert schema.field("c0").type == pa.int64()


def test_array_schema_1d_shape():
    schema = array_schema((100,), np.float64)
    assert len(schema) == 1


def test_array_schema_bad_dtype_raises():
    with pytest.raises(ValueError, match="unsupported dtype"):
        array_schema((10, 2), np.complex128)


def test_array_schema_bad_width_raises():
    with pytest.raises(ValueError, match="K must be >= 1"):
        array_schema(0, np.float64)


# ── Array adapters ──────────────────────────────────────────────


def test_array_to_record_batch_roundtrip():
    arr = np.arange(20, dtype=np.float64).reshape(4, 5)
    rb = array_to_record_batch(arr)
    assert isinstance(rb, pa.RecordBatch)
    assert rb.num_rows == 4
    assert rb.num_columns == 5
    back = record_batch_to_numpy(rb)
    np.testing.assert_array_equal(back, arr)


def test_array_to_record_batch_1d_reshaped():
    arr = np.array([1.0, 2.0, 3.0])
    rb = array_to_record_batch(arr)
    assert rb.num_rows == 3
    assert rb.num_columns == 1


def test_array_to_record_batch_bad_ndim_raises():
    with pytest.raises(ValueError, match="ndim=3"):
        array_to_record_batch(np.zeros((2, 3, 4)))


def test_record_batch_to_numpy_bad_type_raises():
    with pytest.raises(TypeError, match="RecordBatch"):
        record_batch_to_numpy([1, 2, 3])


def test_record_batch_to_numpy_int_dtype_preserved():
    arr = np.arange(12, dtype=np.int64).reshape(3, 4)
    rb = array_to_record_batch(arr)
    back = record_batch_to_numpy(rb)
    assert back.dtype == np.int64
    np.testing.assert_array_equal(back, arr)


# ── Integration: write / read patterns ──────────────────────────


def test_spill_array_end_to_end():
    """Write (n, K) in two batches, read back via iterator."""
    n, K = 100, 3
    full = np.arange(n * K, dtype=np.float64).reshape(n, K)

    with SpillDir(name="e2e") as sdir:
        p = sdir.path("data.arrow")
        schema = array_schema((n, K), np.float64)
        with spill_array_writer(p, schema) as w:
            w.write_batch(array_to_record_batch(full[:40]))
            w.write_batch(array_to_record_batch(full[40:]))

        with spill_array_reader(p) as r:
            chunks = [record_batch_to_numpy(rb) for rb in r]
        back = np.vstack(chunks)
        np.testing.assert_array_equal(back, full)


def test_spill_array_random_access():
    """get_batch(idx) returns correct slice (O(1) via IPC footer)."""
    n, K = 30, 2
    full = np.arange(n * K, dtype=np.float64).reshape(n, K)

    with SpillDir(name="ra") as sdir:
        p = sdir.path("data.arrow")
        schema = array_schema((n, K), np.float64)
        with spill_array_writer(p, schema) as w:
            for i in range(0, n, 10):
                w.write_batch(array_to_record_batch(full[i:i+10]))

        with spill_array_reader(p) as r:
            b0 = record_batch_to_numpy(r.get_batch(0))
            b1 = record_batch_to_numpy(r.get_batch(1))
            b2 = record_batch_to_numpy(r.get_batch(2))
        np.testing.assert_array_equal(b0, full[0:10])
        np.testing.assert_array_equal(b1, full[10:20])
        np.testing.assert_array_equal(b2, full[20:30])


def test_spill_array_streaming_does_not_hold_full_in_ram():
    """Verify streaming pattern: per-batch construction, no full array
    materialized at any point (construction or read)."""
    n, K = 100_000, 5
    batch_size = 10_000

    with SpillDir(name="stream") as sdir:
        p = sdir.path("data.arrow")
        schema = array_schema((n, K), np.float64)

        # Write in batches — never materialize full (n, K)
        with spill_array_writer(p, schema) as w:
            for start in range(0, n, batch_size):
                batch = np.arange(
                    start * K, (start + batch_size) * K,
                    dtype=np.float64,
                ).reshape(batch_size, K)
                w.write_batch(array_to_record_batch(batch))

        # Read in batches — verify count and per-batch shape
        total_rows = 0
        with spill_array_reader(p) as r:
            for rb in r:
                arr = record_batch_to_numpy(rb)
                assert arr.shape[1] == K
                total_rows += arr.shape[0]
        assert total_rows == n


def test_spill_array_reader_close_unmaps():
    """Reader context manager closes underlying IPC reader on exit
    (v0.9.31 A1: explicit mmap unmap; without it RSS drift 5-13x
    when many readers held open)."""
    n, K = 10, 2
    arr = np.arange(n * K, dtype=np.float64).reshape(n, K)

    with SpillDir(name="unmap") as sdir:
        p = sdir.path("data.arrow")
        schema = array_schema((n, K), np.float64)
        with spill_array_writer(p, schema) as w:
            w.write_batch(array_to_record_batch(arr))

        reader_ref = None
        with spill_array_reader(p) as r:
            reader_ref = r
            _ = list(r)
        # After context exit, reader._reader should be None (closed)
        assert getattr(reader_ref, "_reader", None) is None

def test_arrow_ipc_reader_get_batch_bounds():
    """get_batch validates idx range and reader state."""
    n, K = 20, 2
    arr = np.arange(n * K, dtype=np.float64).reshape(n, K)

    with SpillDir(name="bounds") as sdir:
        p = sdir.path("data.arrow")
        schema = array_schema((n, K), np.float64)
        with spill_array_writer(p, schema) as w:
            w.write_batch(array_to_record_batch(arr[:10]))
            w.write_batch(array_to_record_batch(arr[10:]))

        with spill_array_reader(p) as r:
            assert r.num_batches == 2
            # in range
            _ = r.get_batch(0)
            _ = r.get_batch(1)
            # out of range
            with pytest.raises(IndexError, match="out of range"):
                r.get_batch(2)
            with pytest.raises(IndexError, match="out of range"):
                r.get_batch(-1)
        # after close
        with pytest.raises(RuntimeError, match="closed"):
            r.get_batch(0)

