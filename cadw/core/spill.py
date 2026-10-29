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
    # v0.9.90-S1: streaming-output API
    "SpillBuffer", "AutoResult", "InMemoryResult", "OnDiskResult",
    "SpillArray", "AutoArrayResult", "InMemoryArrayResult",
    "OnDiskArrayResult",
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



# ─── SpillBuffer / AutoResult (v0.9.90-S1) ─────────────────────
# Streaming-output API: in-memory accumulator that spills to disk when
# the guard's budget is exhausted. Once spilled, stays spilled.
# See docs/ml_tools/STREAMING_OUTPUT_ARCHITECTURE.md §2.1.


class SpillBuffer:
    """Accumulate pa.RecordBatch; spill to Arrow IPC on demand.

    Threshold = guard.available_bytes() × spill_ratio (default 0.5,
    matching state_safety). guard=None means "no limit" (test/utility
    use). Once spilled, all subsequent batches go straight to disk.

    Lifecycle:
      - finalize() → AutoResult (InMemoryResult or OnDiskResult).
        Calling finalize() twice, or write_batch after finalize,
        raises RuntimeError.
      - OnDiskResult owns the tempdir; caller must close() it (or use
        `with result:` or rely on __del__).
      - SpillBuffer.__exit__ cleans up tempdir only if finalize() was
        not called (early abort path).
    """

    def __init__(self, *, guard, schema: pa.Schema, name: str = "spill",
                 spill_ratio: float = 0.5):
        # v0.9.90-S2-fix: 0.0 allowed as "spill on first batch"
        # (used by output="disk" to force OnDiskResult).
        if not (0.0 <= spill_ratio <= 1.0):
            raise ValueError(
                f"spill_ratio must be in [0, 1]; got {spill_ratio}")
        self.guard = guard
        self.schema = schema
        self.name = str(name)
        self.spill_ratio = float(spill_ratio)
        self._buf: list[pa.RecordBatch] = []
        self._buf_bytes = 0
        self._rows = 0
        self._writer: Optional[ArrowIPCBatchWriter] = None
        self._tmpdir: Optional[Path] = None
        self._path: Optional[Path] = None
        self._spilled = False
        self._finalized = False

    def _threshold(self) -> int:
        if self.guard is None:
            return 1 << 62
        try:
            avail = self.guard.available_bytes()
        except Exception:
            return 1 << 62
        # spill_ratio == 0.0 → threshold 0 → any non-empty batch spills.
        return int(avail * self.spill_ratio)

    def write_batch(self, batch: pa.RecordBatch) -> None:
        if self._finalized:
            raise RuntimeError("SpillBuffer: already finalized")
        if batch.num_rows == 0:
            return
        self._rows += batch.num_rows
        if self._spilled:
            self._writer.write_batch(batch)
            return
        self._buf.append(batch)
        self._buf_bytes += batch.nbytes
        if self._buf_bytes > self._threshold():
            self._spill()

    def _spill(self) -> None:
        if self._tmpdir is None:
            self._tmpdir = Path(tempfile.mkdtemp(prefix=f"cadw_{self.name}_"))
        self._path = self._tmpdir / "output.arrow"
        self._writer = ArrowIPCBatchWriter(str(self._path), self.schema)
        for b in self._buf:
            self._writer.write_batch(b)
        self._buf.clear()
        self._buf_bytes = 0
        self._spilled = True
        if self.guard is not None and hasattr(self.guard, "log_step"):
            try:
                self.guard.log_step(
                    "spill_buffer.spill", path=str(self._path))
            except Exception:
                pass

    def finalize(self) -> "AutoResult":
        if self._finalized:
            raise RuntimeError("SpillBuffer: already finalized")
        self._finalized = True
        if not self._spilled:
            return InMemoryResult(
                pa.Table.from_batches(self._buf, schema=self.schema))
        self._writer.close()
        result = OnDiskResult(
            self._path, self.schema, rows=self._rows,
            _owned_tmpdir=self._tmpdir)
        self._tmpdir = None
        self._path = None
        self._writer = None
        return result

    @property
    def is_spilled(self) -> bool:
        return self._spilled

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._finalized:
            self._cleanup()
        return False

    def _cleanup(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        if self._tmpdir is not None and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = None
        self._path = None


class AutoResult:
    """Unified view over in-memory / on-disk pa.Table result.

    Contract: re-iterable of pa.RecordBatch; len() O(1) in both cases
    (in-memory: num_rows; on-disk: IPC footer). to_table() works only
    for in-memory; on-disk raises MemoryLimitExceeded (METHODOLOGY §1.3).
    """

    @property
    def schema(self) -> pa.Schema:
        raise NotImplementedError

    @property
    def is_spilled(self) -> bool:
        raise NotImplementedError

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def to_table(self) -> pa.Table:
        raise NotImplementedError


class InMemoryResult(AutoResult):
    def __init__(self, table: pa.Table):
        if not isinstance(table, pa.Table):
            raise TypeError(
                f"InMemoryResult: expected pa.Table, got "
                f"{type(table).__name__}")
        self._t = table

    @property
    def schema(self) -> pa.Schema:
        return self._t.schema

    @property
    def is_spilled(self) -> bool:
        return False

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        yield from self._t.to_batches()

    def __len__(self) -> int:
        return self._t.num_rows

    def to_table(self) -> pa.Table:
        return self._t

    # v0.9.92: legacy-test compat. In-memory result delegates
    # pa.Table attributes (.num_rows, .column, .to_pylist, .to_pandas,
    # .shape). Spilled result raises via OnDiskResult.
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._t, name)


class OnDiskResult(AutoResult):
    def __init__(self, path, schema: pa.Schema, rows: int = -1,
                 _owned_tmpdir=None):
        self._path = Path(path)
        self._schema = schema
        self._rows = int(rows)  # -1 = unknown; fallback to reader scan
        self._owned_tmpdir = (
            Path(_owned_tmpdir) if _owned_tmpdir is not None else None)
        self._closed = False

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    @property
    def is_spilled(self) -> bool:
        return True

    @property
    def path(self) -> Path:
        return self._path

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        if self._closed:
            raise RuntimeError("OnDiskResult is closed")
        reader = ArrowIPCBatchReader(str(self._path), None)
        try:
            yield from reader
        finally:
            reader.close()

    def __len__(self) -> int:
        if self._closed:
            raise RuntimeError("OnDiskResult is closed")
        if self._rows >= 0:
            return self._rows
        # Fallback for callers that did not pass `rows`: scan all
        # batches. ArrowIPCBatchReader.total_rows assumes equal batch
        # sizes, which SpillBuffer does not guarantee (last batch may
        # be smaller). O(num_batches), not O(1) — only hit when caller
        # bypassed SpillBuffer.
        reader = ArrowIPCBatchReader(str(self._path), None)
        try:
            return sum(b.num_rows for b in reader)
        finally:
            reader.close()

    def to_table(self) -> pa.Table:
        from cadw.core.memory_guard import MemoryLimitExceeded
        size = self._path.stat().st_size if self._path.exists() else 0
        raise MemoryLimitExceeded(
            f"OnDiskResult.to_table(): result is spilled ({size} B). "
            f"METHODOLOGY §1.3 — no escape hatches. "
            f"Use `for batch in result`.")

    # v0.9.92: spilled result raises on any pa.Table attribute
    # access (num_rows, column, to_pylist, to_pandas, ...). User
    # must iterate via `for b in result`.
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        self.to_table()  # always raises
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owned_tmpdir is not None and self._owned_tmpdir.exists():
            shutil.rmtree(self._owned_tmpdir, ignore_errors=True)
        self._owned_tmpdir = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ─── SpillArray / AutoArrayResult (v0.9.90-S1) ─────────────────
# NumPy analogue of SpillBuffer. See
# STREAMING_OUTPUT_ARCHITECTURE.md §2.2.


class SpillArray:
    """Accumulate (b, K) np.ndarray; spill to Arrow IPC on demand.

    Same spill-on-demand semantics as SpillBuffer, but stores fixed-
    width matrices (all batches must have shape (b, width)).
    """

    def __init__(self, *, guard, width: int, dtype=np.float64,
                 name: str = "spill_array", spill_ratio: float = 0.5):
        if not (0.0 <= spill_ratio <= 1.0):
            raise ValueError(
                f"spill_ratio must be in [0, 1]; got {spill_ratio}")
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        self.guard = guard
        self.width = int(width)
        self.dtype = np.dtype(dtype)
        self.name = str(name)
        self.spill_ratio = float(spill_ratio)
        self._schema = array_schema(self.width, self.dtype)
        self._buf: list[np.ndarray] = []
        self._buf_bytes = 0
        self._rows = 0
        self._writer: Optional[ArrowIPCBatchWriter] = None
        self._tmpdir: Optional[Path] = None
        self._path: Optional[Path] = None
        self._spilled = False
        self._finalized = False

    def _threshold(self) -> int:
        if self.guard is None:
            return 1 << 62
        try:
            avail = self.guard.available_bytes()
        except Exception:
            return 1 << 62
        return int(avail * self.spill_ratio)

    def write_array(self, arr) -> None:
        if self._finalized:
            raise RuntimeError("SpillArray: already finalized")
        arr = np.asarray(arr, dtype=self.dtype)
        if arr.ndim == 1:
            if self.width != 1:
                raise ValueError(
                    f"SpillArray.write_array: 1D array but "
                    f"width={self.width}")
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(
                f"SpillArray.write_array: expected 1D or 2D, "
                f"got ndim={arr.ndim}")
        if arr.shape[1] != self.width:
            raise ValueError(
                f"SpillArray.write_array: width mismatch "
                f"({arr.shape[1]} != {self.width})")
        if arr.shape[0] == 0:
            return
        if self._spilled:
            self._writer.write_batch(array_to_record_batch(arr))
            self._rows += arr.shape[0]
            return
        self._buf.append(arr)
        self._buf_bytes += arr.nbytes
        self._rows += arr.shape[0]
        if self._buf_bytes > self._threshold():
            self._spill()

    def _spill(self) -> None:
        if self._tmpdir is None:
            self._tmpdir = Path(tempfile.mkdtemp(prefix=f"cadw_{self.name}_"))
        self._path = self._tmpdir / "output.arrow"
        self._writer = ArrowIPCBatchWriter(str(self._path), self._schema)
        for arr in self._buf:
            self._writer.write_batch(array_to_record_batch(arr))
        self._buf.clear()
        self._buf_bytes = 0
        self._spilled = True

    def finalize(self) -> "AutoArrayResult":
        if self._finalized:
            raise RuntimeError("SpillArray: already finalized")
        self._finalized = True
        if not self._spilled:
            if self._buf:
                arr = np.concatenate(self._buf, axis=0)
            else:
                arr = np.empty((0, self.width), dtype=self.dtype)
            # v0.9.92: width==1 → 1-D output. Memory-branch of callers
            # (iforest.score_samples, ordinal.predict) return (n,) via
            # np.concatenate(parts_1d) / np.argmax(p, axis=1). The
            # SpillArray stores (b, 1); ravel to match. Without this,
            # np.where / np.array_equal in downstream produce (n, 1)
            # shapes → pyarrow bool cast fails, ordinal predict
            # broadcasting produces (n, n) mask instead of (n,).
            if self.width == 1 and arr.ndim == 2:
                arr = arr.ravel()
            return InMemoryArrayResult(arr)
        self._writer.close()
        result = OnDiskArrayResult(
            self._path, self.width, self.dtype, self._rows,
            _owned_tmpdir=self._tmpdir)
        self._tmpdir = None
        self._path = None
        self._writer = None
        return result

    @property
    def is_spilled(self) -> bool:
        return self._spilled

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._finalized:
            self._cleanup()
        return False

    def _cleanup(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        if self._tmpdir is not None and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = None
        self._path = None


class AutoArrayResult:
    """Unified view over in-memory / on-disk numpy matrix result.

    Contract: re-iterable of (b, K) np.ndarray; len() = total rows.
    to_numpy() only for in-memory; on-disk raises MemoryLimitExceeded.
    """

    @property
    def is_spilled(self) -> bool:
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def to_numpy(self) -> np.ndarray:
        raise NotImplementedError


class InMemoryArrayResult(AutoArrayResult):
    """In-memory numpy result.

    v0.9.92: transparent wrapper (NOT ndarray-subclass). Explicit
    dunders return plain np.ndarray so downstream (pyarrow,
    np.argsort, np.where, np.unique->set) sees standard ndarray
    rather than subclass. Attribute access (.shape, .dtype, .mean,
    .sum, .tolist, .ravel) delegated via __getattr__.

    __array__: conversions (np.asarray, np.array).
    __array_ufunc__: elementwise (np.add, np.multiply, ...).
    __array_function__: reductions/reshaping (np.unique,
      np.concatenate, np.stack, np.sort, np.argsort, np.where).
      Without this, numpy would return subclass from np.unique →
      __iter__ yields whole array → set() hash fails
      (unhashable ndarray). §13.4-class dunder-dispatch pitfall:
      __array__ handles conversions, __array_function__ handles
      functions.

    Spilled results (OnDiskArrayResult) raise MemoryLimitExceeded on
    any numpy access (§1.3). Only small (state_limit-bounded) arrays
    reach this class.
    """

    def __init__(self, arr):
        self._a = np.asarray(arr)

    @property
    def is_spilled(self) -> bool:
        return False

    def __array__(self, dtype=None, copy=None):
        if dtype is None:
            return self._a
        return self._a.astype(
            dtype, copy=bool(copy) if copy is not None else False)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        conv = tuple(
            x._a if isinstance(x, InMemoryArrayResult) else x
            for x in inputs)
        if 'out' in kwargs:
            out = kwargs.pop('out')
            kwargs['out'] = tuple(
                x._a if isinstance(x, InMemoryArrayResult) else x
                for x in out)
        return getattr(ufunc, method)(*conv, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        conv_args = tuple(
            x._a if isinstance(x, InMemoryArrayResult) else x
            for x in args)
        conv_kwargs = {
            k: (v._a if isinstance(v, InMemoryArrayResult) else v)
            for k, v in kwargs.items()}
        return func(*conv_args, **conv_kwargs)

    def __eq__(self, other):
        return self._a == (other._a if isinstance(other, InMemoryArrayResult) else other)
    def __ne__(self, other):
        return self._a != (other._a if isinstance(other, InMemoryArrayResult) else other)
    def __gt__(self, other):
        return self._a > (other._a if isinstance(other, InMemoryArrayResult) else other)
    def __ge__(self, other):
        return self._a >= (other._a if isinstance(other, InMemoryArrayResult) else other)
    def __lt__(self, other):
        return self._a < (other._a if isinstance(other, InMemoryArrayResult) else other)
    def __le__(self, other):
        return self._a <= (other._a if isinstance(other, InMemoryArrayResult) else other)
    def __add__(self, other):
        return self._a + (other._a if isinstance(other, InMemoryArrayResult) else other)
    def __radd__(self, other):
        return other + self._a
    def __sub__(self, other):
        return self._a - (other._a if isinstance(other, InMemoryArrayResult) else other)
    def __rsub__(self, other):
        return other - self._a
    def __mul__(self, other):
        return self._a * (other._a if isinstance(other, InMemoryArrayResult) else other)
    def __rmul__(self, other):
        return other * self._a
    def __truediv__(self, other):
        return self._a / (other._a if isinstance(other, InMemoryArrayResult) else other)
    def __neg__(self):
        return -self._a
    def __abs__(self):
        return abs(self._a)
    def __getitem__(self, idx):
        return self._a[idx]

    def __iter__(self):
        yield self._a

    def __len__(self) -> int:
        return int(self._a.shape[0]) if self._a.ndim > 0 else 1

    def to_numpy(self) -> np.ndarray:
        return self._a

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._a, name)

    def __repr__(self) -> str:
        return (
            f"InMemoryArrayResult(shape={self._a.shape}, "
            f"dtype={self._a.dtype})")


class OnDiskArrayResult(AutoArrayResult):
    def __init__(self, path, width: int, dtype, rows: int,
                 _owned_tmpdir=None):
        self._path = Path(path)
        self._width = int(width)
        self._dtype = np.dtype(dtype)
        self._rows = int(rows)
        self._owned_tmpdir = (
            Path(_owned_tmpdir) if _owned_tmpdir is not None else None)
        self._closed = False

    @property
    def is_spilled(self) -> bool:
        return True

    @property
    def path(self) -> Path:
        return self._path

    def __iter__(self):
        if self._closed:
            raise RuntimeError("OnDiskArrayResult is closed")
        reader = ArrowIPCBatchReader(str(self._path), None)
        try:
            for rb in reader:
                yield record_batch_to_numpy(rb)
        finally:
            reader.close()

    def __len__(self) -> int:
        return self._rows

    def to_numpy(self) -> np.ndarray:
        from cadw.core.memory_guard import MemoryLimitExceeded
        size = self._path.stat().st_size if self._path.exists() else 0
        raise MemoryLimitExceeded(
            f"OnDiskArrayResult.to_numpy(): result is spilled ({size} B, "
            f"{self._rows} rows x {self._width}). "
            f"METHODOLOGY §1.3 — no escape hatches. "
            f"Use `for arr in result`.")

    # v0.9.92: spilled result raises on any numpy access (§1.3).
    def __array__(self, dtype=None, copy=None):
        self.to_numpy()  # always raises
        return np.empty((0, 0))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        self.to_numpy()  # always raises
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owned_tmpdir is not None and self._owned_tmpdir.exists():
            shutil.rmtree(self._owned_tmpdir, ignore_errors=True)
        self._owned_tmpdir = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

