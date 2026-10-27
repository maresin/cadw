"""Spill directory + Arrow IPC array adapters.

Companion to cadw/io/batch_writer.py and cadw/io/batch_reader.py.
Provides the "iterative state" spill pattern (METHODOLOGY §5):

  - SpillDir           context manager for temp dir lifecycle
  - array_schema       pa.Schema for (n, K) matrix (columns c0..cK-1)
  - array_to_record_batch   (b, K) numpy -> pa.RecordBatch
  - record_batch_to_numpy   pa.RecordBatch -> (b, K) numpy
  - spill_array_writer      context-managed ArrowIPCBatchWriter
  - spill_array_reader      context-managed ArrowIPCBatchReader

NO read_all(). Access is via batches() / get_batch(idx). See
METHODOLOGY §1.1 (base cases, not exceptions) and PRIMITIVES §1.3
(no escape hatches). If a caller needs all data in RAM, that caller
is out of scope for CADW.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow as pa

from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter

_log = logging.getLogger(__name__)

__all__ = [
    "SpillDir",
    "array_schema",
    "array_to_record_batch",
    "record_batch_to_numpy",
    "spill_array_writer",
    "spill_array_reader",
]


# ─── SpillDir ─────────────────────────────────────────────────────


class SpillDir:
    """Context manager for spill directory lifecycle.

    Usage:
        with SpillDir(guard=mg, name="gbdt_mc") as sdir:
            p = sdir.path("pred_r000.arrow")
            with spill_array_writer(p, array_schema((n, K), np.float64)) as w:
                for batch in df._iter_filtered_batches():
                    w.write_batch(array_to_record_batch(compute(batch)))
            with spill_array_reader(p) as r:
                for rb in r:
                    arr = record_batch_to_numpy(rb)
                    process(arr)
            sdir.cleanup_file(p)   # free early (optional)

    On __exit__ (success, or error with keep_on_error=False):
    directory is removed. With keep_on_error=True (default) and an
    exception, directory is KEPT for debugging; path is logged via
    logging. Directory contents are on disk, not in RAM, so no
    MemoryGuard.check_state is needed at enter time — per-batch
    guard checks are the caller's responsibility (via
    guard.check_batch inside write loop, or via ArrowIPCBatchWriter
    if it takes a guard — see cadw.io.batch_writer).
    """

    def __init__(self, *, guard=None, name="spill",
                 path=None, keep_on_error=True):
        self.guard = guard
        self.name = str(name)
        self._explicit_path = Path(path) if path is not None else None
        self._tmpdir: Optional[Path] = None
        self._owns_dir = False
        self.keep_on_error = bool(keep_on_error)

    def __enter__(self) -> "SpillDir":
        if self._explicit_path is not None:
            self._tmpdir = self._explicit_path
            self._tmpdir.mkdir(parents=True, exist_ok=True)
            self._owns_dir = False
        else:
            self._tmpdir = Path(tempfile.mkdtemp(
                prefix=f"cadw_{self.name}_"))
            self._owns_dir = True
        if self.guard is not None and hasattr(self.guard, "log_step"):
            try:
                self.guard.log_step(
                    "spill_dir.create", path=str(self._tmpdir))
            except Exception:
                pass  # audit logging must not break functionality
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._tmpdir is None:
            return False
        if not self._owns_dir:
            return False
        if exc_type is not None and self.keep_on_error:
            _log.warning(
                "SpillDir %r kept on error at %s (exc: %s)",
                self.name, self._tmpdir, exc_type.__name__)
            return False
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = None
        return False

    def path(self, filename: str) -> Path:
        if self._tmpdir is None:
            raise RuntimeError(
                "SpillDir.path() called outside context manager")
        return self._tmpdir / filename

    @property
    def root(self) -> Path:
        if self._tmpdir is None:
            raise RuntimeError(
                "SpillDir.root accessed outside context manager")
        return self._tmpdir

    def cleanup_file(self, path) -> None:
        """Delete a single file. Frees disk immediately.
        Idempotent: missing file OK."""
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass


# ─── Array adapters ───────────────────────────────────────────────


def _numpy_to_pa_type(dtype) -> pa.DataType:
    """Explicit mapping — deterministic, no dependence on
    pa.from_numpy_dtype availability across pyarrow versions."""
    np_dtype = np.dtype(dtype)
    mapping = {
        np.dtype(np.float64): pa.float64(),
        np.dtype(np.float32): pa.float32(),
        np.dtype(np.int64): pa.int64(),
        np.dtype(np.int32): pa.int32(),
        np.dtype(np.int16): pa.int16(),
        np.dtype(np.int8): pa.int8(),
        np.dtype(np.uint8): pa.uint8(),
        np.dtype(np.bool_): pa.bool_(),
    }
    if np_dtype not in mapping:
        raise ValueError(
            f"unsupported dtype {np_dtype}; supported: "
            f"{sorted(str(d) for d in mapping)}")
    return mapping[np_dtype]


def array_schema(shape_or_width, dtype) -> pa.Schema:
    """Build pa.Schema for (n, K) matrix stored as K columns c0..cK-1.

    shape_or_width: int (K) or tuple/list (n, K). Tuple uses [1].
    dtype: numpy dtype of the matrix.
    """
    if isinstance(shape_or_width, (tuple, list)):
        if len(shape_or_width) == 1:
            # 1D shape (n,) -> single column matrix (n, 1)
            K = 1
        elif len(shape_or_width) == 2:
            K = int(shape_or_width[1])
        else:
            raise ValueError(
                f"expected 1D or 2D shape, got {shape_or_width}")
    else:
        # int arg: explicit width K
        K = int(shape_or_width)
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    pa_type = _numpy_to_pa_type(dtype)
    return pa.schema([(f"c{i}", pa_type) for i in range(K)])


def array_to_record_batch(arr) -> pa.RecordBatch:
    """Convert (b, K) numpy array to pa.RecordBatch.
    (b,) is accepted and reshaped to (b, 1)."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"expected 1D or 2D array, got ndim={arr.ndim}")
    b, K = arr.shape
    if K < 1:
        raise ValueError(f"K must be >= 1, got K={K}")
    arrays = [pa.array(arr[:, i]) for i in range(K)]
    names = [f"c{i}" for i in range(K)]
    return pa.RecordBatch.from_arrays(arrays, names=names)


def record_batch_to_numpy(rb) -> np.ndarray:
    """Convert pa.RecordBatch (columns c0..cK-1) to (b, K) numpy."""
    if not isinstance(rb, pa.RecordBatch):
        raise TypeError(
            f"expected pa.RecordBatch, got {type(rb).__name__}")
    b, K = rb.num_rows, rb.num_columns
    if K == 0:
        return np.empty((b, 0), dtype=np.float64)
    first_type = rb.schema.field(0).type
    if pa.types.is_float64(first_type):
        dtype = np.float64
    elif pa.types.is_float32(first_type):
        dtype = np.float32
    elif pa.types.is_int64(first_type):
        dtype = np.int64
    elif pa.types.is_int32(first_type):
        dtype = np.int32
    elif pa.types.is_int16(first_type):
        dtype = np.int16
    elif pa.types.is_int8(first_type):
        dtype = np.int8
    elif pa.types.is_boolean(first_type):
        dtype = np.bool_
    else:
        raise TypeError(
            f"unsupported column type {first_type}; "
            f"expected float/int/bool")
    out = np.empty((b, K), dtype=dtype)
    for i in range(K):
        col = rb.column(i)
        out[:, i] = col.to_numpy(zero_copy_only=False)
    return out


# ─── Writer / reader context managers ────────────────────────────


@contextmanager
def spill_array_writer(path, schema) -> Iterator[ArrowIPCBatchWriter]:
    """Context-managed ArrowIPCBatchWriter. Guarantees close() on exit
    (even on exception)."""
    w = ArrowIPCBatchWriter(str(path), schema)
    try:
        yield w
    finally:
        w.close()


@contextmanager
def spill_array_reader(
    path, columns=None,
) -> Iterator[ArrowIPCBatchReader]:
    """Context-managed ArrowIPCBatchReader. Guarantees close() on exit —
    which forces mmap unmap (v0.9.31 audit A1; without this, RSS drift
    5-13x on N=1M when many readers kept alive simultaneously)."""
    r = ArrowIPCBatchReader(str(path), columns)
    try:
        yield r
    finally:
        r.close()
