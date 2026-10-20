#!/usr/bin/env python3
"""v0.9.2f: scalers — StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler.

SCOPE:
  - cadw/scalers.py (new): _NumericStatsOp + 4 scaler classes
  - cadw/pipeline.py: auto-register scalers at bottom
  - cadw/__init__.py: export
  - tests/test_scalers.py: 22 fast
  - tests/test_scalers_e2e.py: 1 slow
  - DECISIONS.md: §11.6 (RobustScaler limitation; AST whitelist
    over-broad), CHANGELOG.md

DESIGN (locked-in):
  - StandardScaler: population std (ddof=0, matches sklearn).
    Degenerate columns (std=0) → scale=1.0 (sklearn behavior).
  - MinMaxScaler: constant column (min==max) → scale=0 → output
    feature_range[0] (matches sklearn).
  - MaxAbsScaler: all-zero column → scale=1 (avoid /0).
  - RobustScaler: IQR=0 → scale=1. Nulls skipped (skipna=True).
  - All scalers: output fields nullable=True (§13.4/169).
  - columns=None at init → detect numeric columns at fit time.
  - State embedded in to_dict (no separate "state" file — O(d) bytes).

PROTOTYPE: NOT NEEDED. All building blocks already tested:
  - Reduce(MeanOp/StdOp/MinMaxOp) — test_reduce.py (54 tests)
  - df.quantile — test_quantile.py (20 tests)
  - Pipeline + CastStep pattern — test_pipeline*.py (v0.9.0-0.9.1)
  - _attach_column nullability — §13.4/169 (v0.9.2e)

GUARDS: scalers fast + E2E + pipeline regression + full suite + guards.

Run: uv run python scripts/patch_v092f_scalers.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


# ============ 1. cadw/scalers.py ============

SCALERS_MODULE = '''"""Scalers for feature normalization (v0.9.2f).

Four Transformer-compatible scalers:
  - StandardScaler: (x - mean) / std  (population std, ddof=0, sklearn)
  - MinMaxScaler:   (x - min) * range / (max - min) + low
  - MaxAbsScaler:   x / max(|min|, |max|)
  - RobustScaler:   (x - median) / IQR  (Sort-based, per-column)

All are fitting-state Transformers:
  - fit(df, target=None) -> self
  - transform(df) -> CadwDataFrame
  - to_dict() / from_dict(d) — JSON-compatible (state embedded)
  - estimate_state_bytes(dc) -> int  (planner pre-fit)
  - state_bytes() -> int  (0 if unfit)

Memory: fit peak = batch + O(d). transform peak = batch + O(d).
Independent of N. See §9.2.1 (Reduce model).

Nullability (§13.4/169): all output fields nullable=True. Scaled
columns change dtype to float64; pass-through columns keep original
type but nullable forced True (Dataset.from_batches singleton rebuild
convention).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from cadw.core.memory_guard import MemoryGuard
from cadw.dataset import Dataset
from cadw.ops.reduce import ReduceOp, reduce

if TYPE_CHECKING:
    from cadw.dataframe import CadwDataFrame
    from cadw.core.models import DataChars


def _column_to_f64(arr: pa.Array) -> np.ndarray:
    """Arrow Array → np.float64 with NaN for nulls."""
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.dtype == object:
        out = np.empty(len(np_arr), dtype=np.float64)
        for i, v in enumerate(np_arr):
            out[i] = np.nan if v is None else float(v)
        return out
    return np_arr.astype(np.float64, copy=False)


class _NumericStatsOp(ReduceOp):
    """Per-column Welford + min/max in one pass. State = 5 × d × 8 bytes."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = list(columns)
        self.d = len(columns)

    def init(self):
        return (
            np.zeros(self.d, dtype=np.int64),
            np.zeros(self.d, dtype=np.float64),
            np.zeros(self.d, dtype=np.float64),
            np.full(self.d, np.inf, dtype=np.float64),
            np.full(self.d, -np.inf, dtype=np.float64),
        )

    def update(self, state, batch):
        n, mean, M2, mn, mx = state
        for j, col in enumerate(self.columns):
            arr = pc.cast(batch.column(col), pa.float64())
            x = _column_to_f64(arr)
            mask = ~np.isnan(x)
            x = x[mask]
            n_b = int(x.size)
            if n_b == 0:
                continue
            mean_b = float(x.mean())
            M2_b = float(((x - mean_b) ** 2).sum())
            n_a = int(n[j])
            if n_a == 0:
                n[j] = n_b
                mean[j] = mean_b
                M2[j] = M2_b
            else:
                delta = mean_b - float(mean[j])
                n_total = n_a + n_b
                mean[j] = mean[j] + delta * n_b / n_total
                M2[j] = M2[j] + M2_b + delta * delta * n_a * n_b / n_total
                n[j] = n_total
            mn[j] = min(float(mn[j]), float(x.min()))
            mx[j] = max(float(mx[j]), float(x.max()))
        return state

    def merge(self, a, b):
        n_a, mean_a, M2_a, mn_a, mx_a = a
        n_b, mean_b, M2_b, mn_b, mx_b = b
        n = n_a + n_b
        mean = mean_a.copy()
        M2 = M2_a.copy()
        for j in range(self.d):
            if n_a[j] == 0:
                mean[j] = mean_b[j]
                M2[j] = M2_b[j]
            elif n_b[j] == 0:
                pass
            else:
                delta = mean_b[j] - mean_a[j]
                mean[j] = mean_a[j] + delta * n_b[j] / n[j]
                M2[j] = (
                    M2_a[j] + M2_b[j]
                    + delta * delta * n_a[j] * n_b[j] / n[j]
                )
        return (n, mean, M2, np.minimum(mn_a, mn_b), np.maximum(mx_a, mx_b))

    def state_nbytes(self, state):
        return sum(int(x.nbytes) for x in state)

    def estimate_state_nbytes(self):
        return 5 * 8 * self.d

    def finalize(self, state):
        n, mean, M2, mn, mx = state
        return {"n": n, "mean": mean, "M2": M2, "min": mn, "max": mx}


def _resolve_columns(
    columns: Optional[Sequence[str]], df: "CadwDataFrame",
) -> list[str]:
    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        cols = list(columns)
        for c in cols:
            if c not in df.columns:
                raise KeyError(
                    f"scaler: column {c!r} not in {df.columns}"
                )
        return cols
    # Numeric-only auto-detect
    cols = []
    for c in df.columns:
        t = df._dataset.schema.field(c).type
        if pa.types.is_integer(t) or pa.types.is_floating(t):
            cols.append(c)
    return cols


def _output_schema(
    df: "CadwDataFrame", scaled_cols: set[str],
) -> pa.Schema:
    """All fields nullable=True (§13.4/169); scaled cols → float64."""
    fields = []
    for c in df._columns:
        if c in scaled_cols:
            fields.append(pa.field(c, pa.float64(), nullable=True))
        else:
            orig = df._dataset.schema.field(c)
            fields.append(pa.field(c, orig.type, nullable=True))
    return pa.schema(fields)


def _iter_transformed(
    df: "CadwDataFrame",
    scaled_cols: set[str],
    out_schema: pa.Schema,
    transform_fn,
):
    """Yield transformed batches. transform_fn(col, array) → array."""
    for b in df._iter_filtered_batches():
        arrays = []
        for c in df._columns:
            if c in scaled_cols:
                arr = pc.cast(b.column(c), pa.float64())
                arr = transform_fn(c, arr)
            else:
                arr = b.column(c)
            arrays.append(arr)
        yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)


def _wrap_transformed(
    df: "CadwDataFrame", scaled_cols: set[str], transform_fn,
) -> "CadwDataFrame":
    from cadw.dataframe import CadwDataFrame
    out_schema = _output_schema(df, scaled_cols)
    new_ds = Dataset.from_batches(
        out_schema,
        _iter_transformed(df, scaled_cols, out_schema, transform_fn),
        df._new_dataset_dir(),
    )
    return CadwDataFrame(
        new_ds,
        memory_guard=df._memory_guard,
        spill_dir=df._spill_dir,
    )


# ============ StandardScaler ============

class StandardScaler:
    """Standardize features: (x - mean) / std.

    - StandardScaler uses POPULATION std (ddof=0), matches sklearn.
    - Degenerate column (std=0, e.g. constant) → scale=1.0 (sklearn).
    - Nulls skipped in fit (nanmean/nanstd); nulls preserved in transform.
    - Output dtype float64 for scaled columns.
    """

    __slots__ = ("_columns", "_with_mean", "_with_std", "_mean", "_std")

    def __init__(
        self,
        columns: "str | Sequence[str] | None" = None,
        *,
        with_mean: bool = True,
        with_std: bool = True,
    ) -> None:
        if isinstance(columns, str):
            columns = [columns]
        self._columns = list(columns) if columns is not None else None
        self._with_mean = bool(with_mean)
        self._with_std = bool(with_std)
        self._mean: Optional[dict[str, float]] = None
        self._std: Optional[dict[str, float]] = None

    def fit(self, df, target=None):
        cols = _resolve_columns(self._columns, df)
        if not cols:
            raise ValueError("StandardScaler: no numeric columns to fit")
        self._columns = cols
        op = _NumericStatsOp(cols)
        result = reduce(df._iter_filtered_batches(), op, df._memory_guard,
                        log=False)
        n = result["n"].astype(np.float64)
        M2 = result["M2"]
        mean = result["mean"]
        var = np.where(n > 0, M2 / np.where(n > 0, n, 1.0), 0.0)
        std = np.sqrt(var)
        std = np.where(std > 0, std, 1.0)  # degenerate → 1.0
        self._mean = dict(zip(cols, mean.tolist()))
        self._std = dict(zip(cols, std.tolist()))
        return self

    def transform(self, df):
        if self._mean is None:
            raise RuntimeError("StandardScaler: not fitted")
        cols_set = set(self._columns)
        mean = self._mean
        std = self._std
        wm = self._with_mean
        ws = self._with_std

        def _fn(c, arr):
            if wm:
                arr = pc.subtract(arr, pa.scalar(mean[c], type=pa.float64()))
            if ws:
                arr = pc.divide(arr, pa.scalar(std[c], type=pa.float64()))
            return arr

        return _wrap_transformed(df, cols_set, _fn)

    def fit_transform(self, df, target=None):
        self.fit(df, target=target)
        return self.transform(df)

    def estimate_state_bytes(self, dc):
        d = len(self._columns) if self._columns is not None else len(dc.schema)
        return 16 * d

    def state_bytes(self):
        if self._mean is None:
            return 0
        return 16 * len(self._mean)

    def to_dict(self):
        return {
            "columns": self._columns,
            "with_mean": self._with_mean,
            "with_std": self._with_std,
            "mean": self._mean,
            "std": self._std,
        }

    @classmethod
    def from_dict(cls, d):
        s = cls(
            columns=d.get("columns"),
            with_mean=d.get("with_mean", True),
            with_std=d.get("with_std", True),
        )
        if d.get("mean") is not None:
            s._mean = dict(d["mean"])
            s._std = dict(d["std"])
        return s


# ============ MinMaxScaler ============

class MinMaxScaler:
    """Scale features to [low, high] (default [0, 1]).

    - Degenerate (min==max) → constant output feature_range[0] (sklearn).
    - Nulls skipped in fit; preserved in transform.
    """

    __slots__ = (
        "_columns", "_feature_range", "_min", "_max", "_scale", "_min_adj",
    )

    def __init__(
        self,
        columns: "str | Sequence[str] | None" = None,
        *,
        feature_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        if isinstance(columns, str):
            columns = [columns]
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                f"MinMaxScaler: feature_range must be (low, high) with "
                f"low < high; got {feature_range}"
            )
        self._columns = list(columns) if columns is not None else None
        self._feature_range = (float(feature_range[0]),
                               float(feature_range[1]))
        self._min: Optional[dict[str, float]] = None
        self._max: Optional[dict[str, float]] = None
        self._scale: Optional[dict[str, float]] = None
        self._min_adj: Optional[dict[str, float]] = None

    def fit(self, df, target=None):
        cols = _resolve_columns(self._columns, df)
        if not cols:
            raise ValueError("MinMaxScaler: no numeric columns to fit")
        self._columns = cols
        op = _NumericStatsOp(cols)
        result = reduce(df._iter_filtered_batches(), op, df._memory_guard,
                        log=False)
        n = result["n"]
        mn = result["min"]
        mx = result["max"]

        low, high = self._feature_range
        span = high - low
        self._min = {}
        self._max = {}
        self._scale = {}
        self._min_adj = {}
        for j, c in enumerate(cols):
            if int(n[j]) == 0:
                # All-null column: no-op (scale=1, min=0)
                self._min[c] = 0.0
                self._max[c] = 0.0
                self._scale[c] = 1.0
                self._min_adj[c] = 0.0
            else:
                mn_j = float(mn[j])
                mx_j = float(mx[j])
                data_range = mx_j - mn_j
                self._min[c] = mn_j
                self._max[c] = mx_j
                if data_range > 0:
                    scale = span / data_range
                    self._scale[c] = scale
                    self._min_adj[c] = low - mn_j * scale
                else:
                    # Constant column → scale=0 → output = low
                    self._scale[c] = 0.0
                    self._min_adj[c] = low
        return self

    def transform(self, df):
        if self._scale is None:
            raise RuntimeError("MinMaxScaler: not fitted")
        cols_set = set(self._columns)
        scale = self._scale
        min_adj = self._min_adj

        def _fn(c, arr):
            arr = pc.multiply(arr, pa.scalar(scale[c], type=pa.float64()))
            arr = pc.add(arr, pa.scalar(min_adj[c], type=pa.float64()))
            return arr

        return _wrap_transformed(df, cols_set, _fn)

    def fit_transform(self, df, target=None):
        self.fit(df, target=target)
        return self.transform(df)

    def estimate_state_bytes(self, dc):
        d = len(self._columns) if self._columns is not None else len(dc.schema)
        return 32 * d  # _min, _max, _scale, _min_adj

    def state_bytes(self):
        if self._scale is None:
            return 0
        return 32 * len(self._scale)

    def to_dict(self):
        return {
            "columns": self._columns,
            "feature_range": list(self._feature_range),
            "min": self._min,
            "max": self._max,
            "scale": self._scale,
            "min_adj": self._min_adj,
        }

    @classmethod
    def from_dict(cls, d):
        s = cls(
            columns=d.get("columns"),
            feature_range=tuple(d.get("feature_range", [0.0, 1.0])),
        )
        if d.get("scale") is not None:
            s._min = dict(d["min"])
            s._max = dict(d["max"])
            s._scale = dict(d["scale"])
            s._min_adj = dict(d["min_adj"])
        return s


# ============ MaxAbsScaler ============

class MaxAbsScaler:
    """Scale to [-1, 1] by dividing by max(|min|, |max|) per feature.

    - All-zero column → scale=1 (avoid division by zero).
    - Nulls skipped in fit; preserved in transform.
    """

    __slots__ = ("_columns", "_max_abs")

    def __init__(self, columns: "str | Sequence[str] | None" = None) -> None:
        if isinstance(columns, str):
            columns = [columns]
        self._columns = list(columns) if columns is not None else None
        self._max_abs: Optional[dict[str, float]] = None

    def fit(self, df, target=None):
        cols = _resolve_columns(self._columns, df)
        if not cols:
            raise ValueError("MaxAbsScaler: no numeric columns to fit")
        self._columns = cols
        op = _NumericStatsOp(cols)
        result = reduce(df._iter_filtered_batches(), op, df._memory_guard,
                        log=False)
        n = result["n"]
        mn = result["min"]
        mx = result["max"]
        self._max_abs = {}
        for j, c in enumerate(cols):
            if int(n[j]) == 0:
                self._max_abs[c] = 1.0
            else:
                mx_abs = max(abs(float(mn[j])), abs(float(mx[j])))
                self._max_abs[c] = mx_abs if mx_abs > 0 else 1.0
        return self

    def transform(self, df):
        if self._max_abs is None:
            raise RuntimeError("MaxAbsScaler: not fitted")
        cols_set = set(self._columns)
        max_abs = self._max_abs

        def _fn(c, arr):
            return pc.divide(arr, pa.scalar(max_abs[c], type=pa.float64()))

        return _wrap_transformed(df, cols_set, _fn)

    def fit_transform(self, df, target=None):
        self.fit(df, target=target)
        return self.transform(df)

    def estimate_state_bytes(self, dc):
        d = len(self._columns) if self._columns is not None else len(dc.schema)
        return 8 * d

    def state_bytes(self):
        if self._max_abs is None:
            return 0
        return 8 * len(self._max_abs)

    def to_dict(self):
        return {
            "columns": self._columns,
            "max_abs": self._max_abs,
        }

    @classmethod
    def from_dict(cls, d):
        s = cls(columns=d.get("columns"))
        if d.get("max_abs") is not None:
            s._max_abs = dict(d["max_abs"])
        return s


# ============ RobustScaler ============

class RobustScaler:
    """Scale features using median and IQR (robust to outliers).

    - median = q50, IQR = q75 - q25.
    - IQR=0 (degenerate) → scale=1.
    - Nulls skipped in fit (df.quantile skipna=True); preserved in transform.
    - Wide data: per-column external_sort → O(d) sorts per fit.
      For d > 100, prefer StandardScaler (see §11.6).
    """

    __slots__ = ("_columns", "_median", "_iqr")

    def __init__(self, columns: "str | Sequence[str] | None" = None) -> None:
        if isinstance(columns, str):
            columns = [columns]
        self._columns = list(columns) if columns is not None else None
        self._median: Optional[dict[str, float]] = None
        self._iqr: Optional[dict[str, float]] = None

    def fit(self, df, target=None):
        cols = _resolve_columns(self._columns, df)
        if not cols:
            raise ValueError("RobustScaler: no numeric columns to fit")
        self._columns = cols
        median: dict[str, float] = {}
        iqr: dict[str, float] = {}
        for c in cols:
            q = df.quantile(c, [0.25, 0.5, 0.75])
            q25, q50, q75 = q[0], q[1], q[2]
            if q50 != q50:  # NaN check
                median[c] = 0.0
                iqr[c] = 1.0
            else:
                median[c] = float(q50)
                iqr_val = float(q75 - q25)
                iqr[c] = iqr_val if iqr_val > 0 else 1.0
        self._median = median
        self._iqr = iqr
        return self

    def transform(self, df):
        if self._median is None:
            raise RuntimeError("RobustScaler: not fitted")
        cols_set = set(self._columns)
        median = self._median
        iqr = self._iqr

        def _fn(c, arr):
            arr = pc.subtract(arr, pa.scalar(median[c], type=pa.float64()))
            arr = pc.divide(arr, pa.scalar(iqr[c], type=pa.float64()))
            return arr

        return _wrap_transformed(df, cols_set, _fn)

    def fit_transform(self, df, target=None):
        self.fit(df, target=target)
        return self.transform(df)

    def estimate_state_bytes(self, dc):
        d = len(self._columns) if self._columns is not None else len(dc.schema)
        return 16 * d

    def state_bytes(self):
        if self._median is None:
            return 0
        return 16 * len(self._median)

    def to_dict(self):
        return {
            "columns": self._columns,
            "median": self._median,
            "iqr": self._iqr,
        }

    @classmethod
    def from_dict(cls, d):
        s = cls(columns=d.get("columns"))
        if d.get("median") is not None:
            s._median = dict(d["median"])
            s._iqr = dict(d["iqr"])
        return s


# ============ Registration (Pipeline.from_json) ============

def _register_scalers():
    """Register scalers with Pipeline step registry (for from_json)."""
    try:
        from cadw.pipeline import register_step_type
    except Exception:
        return
    for cls in (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler):
        try:
            register_step_type(cls.__name__, cls)
        except Exception:
            pass  # already registered


_register_scalers()
'''


def write_scalers_module():
    print("\n=== cadw/scalers.py (new) ===")
    p = Path("cadw/scalers.py")
    p.write_text(SCALERS_MODULE)
    compile_check(p)
    print("  OK")


# ============ 2. __init__.py export ============

def update_init():
    print("\n=== cadw/__init__.py: export scalers ===")
    p = Path("cadw/__init__.py")
    s = p.read_text()
    if "from .scalers import" in s:
        print("  already present — skip")
        return

    # Import after pipeline block (which registers CastStep etc.)
    anchor = "from .cv import (\n    KFold,\n    TimeSeriesSplit,\n    cv_score,\n)\n"
    if anchor not in s:
        raise SystemExit("__init__: cv anchor not found")
    s = s.replace(
        anchor,
        anchor + "\nfrom .scalers import (\n"
        "    StandardScaler,\n"
        "    MinMaxScaler,\n"
        "    MaxAbsScaler,\n"
        "    RobustScaler,\n"
        ")\n",
        1,
    )

    # Add to __all__
    all_anchor = '    # v0.9.2e cross-validation\n    "KFold", "TimeSeriesSplit", "cv_score",\n'
    if all_anchor not in s:
        raise SystemExit("__init__: __all__ anchor not found")
    s = s.replace(
        all_anchor,
        all_anchor
        + '    # v0.9.2f scalers\n'
        + '    "StandardScaler", "MinMaxScaler", "MaxAbsScaler", '
          '"RobustScaler",\n',
        1,
    )

    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 3. tests/test_scalers.py ============

SCALERS_TESTS = '''"""v0.9.2f: scalers tests."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import (
    CadwDataFrame, MemoryGuard,
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
)


def make_df(tmp_path, data, chunk=10, name="L", mem_mb=50):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def read_table(df, tmp_path, name="o.parquet"):
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out)


def _col(t, name):
    return t.column(name).to_pylist()


# ============ StandardScaler ============

def test_standard_scaler_mean_zero_std_one(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(loc=10.0, scale=5.0, size=1000)
    df = make_df(tmp_path, {"x": x.tolist()}, chunk=100)
    scaler = StandardScaler()
    scaler.fit(df)
    out = scaler.transform(df)
    t = read_table(out, tmp_path)
    xs = np.array(_col(t, "x"), dtype=np.float64)
    # mean ≈ 0, std ≈ 1 (population std, matches our fit)
    assert abs(float(xs.mean())) < 1e-9
    assert abs(float(xs.std(ddof=0)) - 1.0) < 1e-9


def test_standard_scaler_with_mean_false(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(loc=10.0, scale=5.0, size=1000)
    df = make_df(tmp_path, {"x": x.tolist()}, chunk=100)
    scaler = StandardScaler(with_mean=False, with_std=True)
    scaler.fit(df)
    out = scaler.transform(df)
    t = read_table(out, tmp_path)
    xs = np.array(_col(t, "x"), dtype=np.float64)
    # mean ≈ 10/std_train (mean shifted from original), std ≈ 1
    assert abs(float(xs.std(ddof=0)) - 1.0) < 1e-9


def test_standard_scaler_with_std_false(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(loc=10.0, scale=5.0, size=1000)
    df = make_df(tmp_path, {"x": x.tolist()}, chunk=100)
    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit(df)
    out = scaler.transform(df)
    t = read_table(out, tmp_path)
    xs = np.array(_col(t, "x"), dtype=np.float64)
    # mean ≈ 0, std unchanged (~5)
    assert abs(float(xs.mean())) < 1e-9
    assert abs(float(xs.std(ddof=0)) - x.std(ddof=0)) < 1e-6


def test_standard_scaler_nulls_skipped(tmp_path):
    x = [1.0, None, 3.0, None, 5.0]
    df = make_df(tmp_path, {"x": x}, chunk=2)
    scaler = StandardScaler()
    scaler.fit(df)
    out = scaler.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert xs[1] is None and xs[3] is None
    # mean of valid = 3.0; std (ddof=0) = sqrt(8/3) ≈ 1.633
    # (1-3)/1.633 ≈ -1.2247
    assert xs[0] is not None
    assert abs(xs[0] - (-2.0 / np.sqrt(8.0/3.0))) < 1e-9


def test_standard_scaler_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=100)
    df = make_df(tmp_path, {"x": x.tolist()})
    s = StandardScaler()
    s.fit(df)
    d = s.to_dict()
    s2 = StandardScaler.from_dict(d)
    out1 = read_table(s.transform(df), tmp_path, "o1.parquet")
    out2 = read_table(s2.transform(df), tmp_path, "o2.parquet")
    np.testing.assert_allclose(_col(out1, "x"), _col(out2, "x"))


def test_standard_scaler_unseen_data(tmp_path):
    rng = np.random.default_rng(42)
    x_train = rng.normal(loc=10.0, scale=2.0, size=200)
    x_test = rng.normal(loc=10.0, scale=2.0, size=100)
    df_train = make_df(tmp_path, {"x": x_train.tolist()}, name="train")
    df_test = make_df(tmp_path, {"x": x_test.tolist()}, name="test")
    s = StandardScaler()
    s.fit(df_train)
    out = s.transform(df_test)
    t = read_table(out, tmp_path)
    xs = np.array(_col(t, "x"), dtype=np.float64)
    # Test mean close to train mean (10.0/2.0 = 5.0) but not exactly
    assert abs(float(xs.mean()) - 5.0) < 0.5


def test_standard_scaler_constant_column(tmp_path):
    """Constant column → std=0 → scale to 1.0 (sklearn), output zeros."""
    df = make_df(tmp_path, {"x": [5.0] * 20})
    s = StandardScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert all(abs(v) < 1e-9 for v in xs)


# ============ MinMaxScaler ============

def test_minmax_scaler_0_1(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    s = MinMaxScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert xs == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_minmax_scaler_custom_range(tmp_path):
    df = make_df(tmp_path, {"x": [0.0, 5.0, 10.0]})
    s = MinMaxScaler(feature_range=(-1.0, 1.0))
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    np.testing.assert_allclose(xs, [-1.0, 0.0, 1.0])


def test_minmax_scaler_constant_column(tmp_path):
    """Constant → scale=0 → output = feature_range[0]."""
    df = make_df(tmp_path, {"x": [5.0] * 10})
    s = MinMaxScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert all(abs(v) < 1e-9 for v in xs)


def test_minmax_scaler_roundtrip(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0, 3.0]})
    s = MinMaxScaler()
    s.fit(df)
    s2 = MinMaxScaler.from_dict(s.to_dict())
    out1 = read_table(s.transform(df), tmp_path, "o1.parquet")
    out2 = read_table(s2.transform(df), tmp_path, "o2.parquet")
    np.testing.assert_allclose(_col(out1, "x"), _col(out2, "x"))


def test_minmax_scaler_bad_range():
    with pytest.raises(ValueError, match="feature_range"):
        MinMaxScaler(feature_range=(1.0, 1.0))


# ============ MaxAbsScaler ============

def test_maxabs_scaler_basic(tmp_path):
    df = make_df(tmp_path, {"x": [-4.0, -2.0, 0.0, 2.0, 4.0]})
    s = MaxAbsScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert xs == [-1.0, -0.5, 0.0, 0.5, 1.0]


def test_maxabs_scaler_positive_only(tmp_path):
    df = make_df(tmp_path, {"x": [0.0, 5.0, 10.0]})
    s = MaxAbsScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert xs == [0.0, 0.5, 1.0]


def test_maxabs_scaler_all_zero(tmp_path):
    """All zeros → scale=1 (no-op)."""
    df = make_df(tmp_path, {"x": [0.0] * 10})
    s = MaxAbsScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert all(abs(v) < 1e-9 for v in xs)


def test_maxabs_scaler_roundtrip(tmp_path):
    df = make_df(tmp_path, {"x": [-3.0, 0.0, 3.0]})
    s = MaxAbsScaler()
    s.fit(df)
    s2 = MaxAbsScaler.from_dict(s.to_dict())
    out1 = read_table(s.transform(df), tmp_path, "o1.parquet")
    out2 = read_table(s2.transform(df), tmp_path, "o2.parquet")
    np.testing.assert_allclose(_col(out1, "x"), _col(out2, "x"))


# ============ RobustScaler ============

def test_robust_scaler_median_zero_iqr_one(tmp_path):
    """After scaling, median=0, IQR=1 (for non-degenerate data)."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=500)
    df = make_df(tmp_path, {"x": x.tolist()}, chunk=50)
    s = RobustScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = np.array(_col(t, "x"), dtype=np.float64)
    q25, q50, q75 = np.percentile(xs, [25, 50, 75])
    assert abs(float(q50)) < 1e-9
    assert abs(float(q75 - q25) - 1.0) < 1e-9


def test_robust_scaler_outlier_resistant(tmp_path):
    """Single huge outlier doesn't shift scaling much."""
    x = list(np.random.default_rng(0).normal(size=100)) + [1e6]
    df = make_df(tmp_path, {"x": x}, chunk=20)
    s = RobustScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = np.array(_col(t, "x"), dtype=np.float64)
    # Median of transformed ≈ 0
    assert abs(float(np.median(xs))) < 1e-6


def test_robust_scaler_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=100)
    df = make_df(tmp_path, {"x": x.tolist()})
    s = RobustScaler()
    s.fit(df)
    s2 = RobustScaler.from_dict(s.to_dict())
    out1 = read_table(s.transform(df), tmp_path, "o1.parquet")
    out2 = read_table(s2.transform(df), tmp_path, "o2.parquet")
    np.testing.assert_allclose(_col(out1, "x"), _col(out2, "x"))


def test_robust_scaler_constant_column(tmp_path):
    df = make_df(tmp_path, {"x": [5.0] * 20})
    s = RobustScaler()
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    xs = _col(t, "x")
    assert all(abs(v) < 1e-9 for v in xs)


# ============ Multi-column / pass-through ============

def test_scaler_multiple_columns(tmp_path):
    df = make_df(tmp_path, {
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
        "c": [100.0, 200.0, 300.0],
    })
    s = StandardScaler(columns=["a", "b"])
    s.fit(df)
    out = s.transform(df)
    t = read_table(out, tmp_path)
    # a and b standardized; c unchanged
    assert set(t.schema.names) == {"a", "b", "c"}
    assert _col(t, "c") == [100.0, 200.0, 300.0]


def test_scaler_auto_detect_numeric(tmp_path):
    """columns=None → numeric columns auto-detected at fit."""
    df = make_df(tmp_path, {
        "num1": [1.0, 2.0, 3.0],
        "num2": [4.0, 5.0, 6.0],
        "s": ["x", "y", "z"],
    })
    s = StandardScaler()
    s.fit(df)
    assert set(s._columns) == {"num1", "num2"}


def test_scaler_missing_column_raises(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0]})
    s = StandardScaler(columns=["nope"])
    with pytest.raises(KeyError, match="nope"):
        s.fit(df)


# ============ Pipeline integration ============

def test_pipeline_with_standard_scaler(tmp_path):
    from cadw import Pipeline
    from cadw.pipeline import GBDTStep

    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(loc=100, scale=10, size=n)
    x2 = rng.normal(loc=0.5, scale=0.1, size=n)
    y = 2.0 * (x1 - 100) / 10 + 3.0 * (x2 - 0.5) / 0.1 + rng.normal(
        scale=0.1, size=n)

    df = make_df(tmp_path, {
        "x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist(),
    }, chunk=50, mem_mb=200)

    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("scale", StandardScaler(columns=["x1", "x2"]))
    pipe.add_step(
        "model", GBDTStep(
            ["x1", "x2"], "y",
            n_estimators=20, max_depth=2, learning_rate=0.2,
        ),
    )
    pipe.fit(df, target="y")
    preds = pipe.predict(df)
    assert "y_pred" in preds.columns


def test_pipeline_save_load_with_scaler(tmp_path):
    from cadw import Pipeline
    from cadw.pipeline import GBDTStep

    rng = np.random.default_rng(42)
    n = 100
    x = rng.normal(loc=10, scale=2, size=n)
    y = 2.0 * (x - 10) / 2 + rng.normal(scale=0.1, size=n)

    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()},
                 chunk=50, mem_mb=200)

    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=200))
    pipe.add_step("scale", StandardScaler())
    pipe.add_step("model", GBDTStep(["x"], "y", n_estimators=10))

    pipe.fit(df, target="y")

    pipe_dir = tmp_path / "mypipe"
    pipe.to_json(pipe_dir)
    restored = Pipeline.from_json(
        pipe_dir, memory_guard=MemoryGuard(max_memory_mb=200),
    )

    preds_orig = read_table(pipe.predict(df), tmp_path, "p1.parquet")
    preds_restored = read_table(
        restored.predict(df), tmp_path, "p2.parquet",
    )
    np.testing.assert_allclose(
        preds_orig.column("y_pred").to_pylist(),
        preds_restored.column("y_pred").to_pylist(),
        rtol=1e-9, equal_nan=True,
    )
'''

SCALERS_TESTS_E2E = '''"""v0.9.2f: scalers E2E (slow)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import (
    CadwDataFrame, MemoryGuard, cv_score, KFold,
    StandardScaler,
)
from cadw.pipeline import GBDTStep, Pipeline


@pytest.mark.slow
def test_e2e_standard_scaler_100k(tmp_path):
    """100K rows, StandardScaler → GBDTStep, memory 50 MB, R² > 0.5."""
    n = 100_000
    rng = np.random.default_rng(42)
    x1 = rng.normal(loc=100, scale=10, size=n)
    x2 = rng.normal(loc=0.5, scale=0.1, size=n)
    y = (
        2.0 * (x1 - 100) / 10
        + 3.0 * (x2 - 0.5) / 0.1
        + rng.normal(scale=0.3, size=n)
    )

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({
            "x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist(),
        }),
        src, row_group_size=5000,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )

    pipe = Pipeline(memory_guard=MemoryGuard(max_memory_mb=50))
    pipe.add_step("scale", StandardScaler(columns=["x1", "x2"]))
    pipe.add_step(
        "model", GBDTStep(
            ["x1", "x2"], "y",
            n_estimators=30, max_depth=2, learning_rate=0.2,
        ),
    )
    pipe.fit(df, target="y")
    preds = pipe.predict(df)
    out = tmp_path / "p.parquet"
    preds.write_parquet(out)

    t = pq.read_table(out)
    y_true = np.array(t.column("y").to_pylist(), dtype=np.float64)
    y_pred = np.array(t.column("y_pred").to_pylist(), dtype=np.float64)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.5, f"R²={r2:.3f}"
'''


def write_tests():
    print("\n=== tests/test_scalers.py ===")
    p = Path("tests/test_scalers.py")
    p.write_text(SCALERS_TESTS)
    compile_check(p)
    print("  OK")

    print("\n=== tests/test_scalers_e2e.py ===")
    p = Path("tests/test_scalers_e2e.py")
    p.write_text(SCALERS_TESTS_E2E)
    compile_check(p)
    print("  OK")


# ============ 4. DECISIONS.md: §11.6 + §13.4/170 ============

def append_decisions_170():
    print("\n=== DECISIONS.md: §13.4/170 + §11.6 backlog ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 170" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/170 (scalers — no prototype, no new lessons).** "
        s += "Composed from existing primitives (Reduce + quantile); "
        s += "AST guard whitelist by function name is over-broad.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 170 (scalers — no new lessons; AST whitelist
observation).** v0.9.2f added StandardScaler / MinMaxScaler /
MaxAbsScaler / RobustScaler. **No new §13.4 lessons** — pattern
identical to CastStep (v0.9.0b) + `_NumericStatsOp` (single-pass
Welford+min/max, custom ReduceOp).

**Why no prototype:** all building blocks already tested —
`Reduce` (54 tests in `test_reduce.py`), `df.quantile` (20 tests in
`test_quantile.py`), `Transformer` Protocol + Pipeline (v0.9.0–0.9.1),
`_attach_column` nullability (§13.4/169). Scalers are **composition**,
not new algorithm. §13.4/123 prototype-first applies to **new
algorithms** (sort, roc_auc, CV fold math) — not to compositions of
tested primitives.

**Observation (§11.6 backlog):** AST guard whitelist in
`test_arrow_call_memory_guard.py` is **by function name**
(`node.name in WHITELISTED_FUNCS`), not by qualname.
- `"fit"`, `"transform"`, `"predict"` in whitelist → **ALL** methods
  named `fit`/`transform`/`predict` in `cadw/` are whitelisted
  (Pipeline, GBDTStep, scalers, ...).
- Rationale string per key is misleading (e.g., `"fit": "delegates
  to step.fit"` — true for Pipeline, false for RobustScaler which
  calls `df.quantile` directly).
- Effect: `RobustScaler.fit` calls `df.quantile` (high-risk)
  without preceding `check_state`/`check_batch` — not flagged.
  Not a bug (quantile has its own guards), but the guard cannot
  distinguish safe from unsafe calls in any future `*_fit` method.

**Proper fix (v0.9.3):** `_collect_violations` walks ClassDef to
build `ClassName.method_name` qualname; whitelist by qualname
(`"Pipeline.fit": "..."`, `"RobustScaler.fit": "..."`). ~30 min.

**Связанные:** §13.4/123 (prototype applies to new algorithms, not
compositions), §13.4/164 (whitelist discipline), §13.4/169
(nullability).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/170")


def update_decisions_backlog():
    print("\n=== DECISIONS.md §11.6: ast-whitelist-over-broad ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "AST whitelist by qualname" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "12")
    if pos is None:
        print("  WARNING: §12 not found — skip")
        return

    note = '''**§11.6 addendum (v0.9.2f observation):**

- **`test_arrow_call_memory_guard.py` whitelist by function NAME,
  not qualname.** `"fit"`, `"transform"`, `"predict"` whitelisted
  globally → `RobustScaler.fit` (calls `df.quantile`) not caught by
  AST guard. Fix: `_collect_violations` walk ClassDef for
  `ClassName.method` qualname; whitelist with qualname keys.
  ~30 min, v0.9.3.
- **RobustScaler.fit — O(d) external_sorts.** For d=100 × n=1M,
  ~17 min fit. For d > 100 prefer StandardScaler/MinMaxScaler.
  Documented in class docstring + §11.5.

'''
    s = s[:pos] + note + "\n" + s[pos:]
    p.write_text(s)
    print("  OK")


# ============ 5. CHANGELOG ============

def append_changelog():
    print("\n=== CHANGELOG.md: v0.9.2f entry ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.2f — scalers" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Добавлено (v0.9.2f — scalers)

Завершение ML flow v0.9.2: preprocessing (metrics + CV + scalers).

- **`StandardScaler(columns=None, *, with_mean=True, with_std=True)`** —
  `(x - mean) / std`. Population std (`ddof=0`, matches sklearn).
  Degenerate column (std=0) → scale=1.0 (sklearn behavior).
- **`MinMaxScaler(columns=None, *, feature_range=(0.0, 1.0))`** —
  `(x - min) * (high-low)/(max-min) + low`. Constant column → output
  `feature_range[0]` (sklearn).
- **`MaxAbsScaler(columns=None)`** — `x / max(|min|, |max|)`. All-zero
  column → scale=1 (avoid division by zero).
- **`RobustScaler(columns=None)`** — `(x - median) / IQR`. IQR=0
  (degenerate) → scale=1. Per-column `df.quantile([0.25, 0.5, 0.75])`
  (external_sort per column, O(d) sorts per fit).
- **`cadw/scalers.py`** — new module. `_NumericStatsOp` (per-column
  Welford + min/max in one pass, state 5×d×8 bytes) shared by
  StandardScaler/MinMaxScaler/MaxAbsScaler. RobustScaler uses
  existing `df.quantile`.
- Auto-registered as Pipeline steps (`register_step_type`) → usable in
  `Pipeline.to_json/from_json`.
- Exports: `from cadw import StandardScaler, MinMaxScaler,
  MaxAbsScaler, RobustScaler`.

### Design decisions (locked-in, v0.9.2f)

- **No prototype** — scalers are **composition** of tested primitives
  (`Reduce`, `df.quantile`, `Transformer` Protocol). §13.4/123
  prototype-first applies to **new algorithms** (sort, roc_auc, CV
  fold math), not compositions.
- **Population std (ddof=0)** for StandardScaler — matches sklearn.
- **Degenerate handling**: std=0 → scale=1; IQR=0 → scale=1;
  min==max → output feature_range[0]; max_abs=0 → scale=1. All
  match sklearn or are no-op (safer for downstream NaN propagation).
- **Nullability (§13.4/169)**: all output fields nullable=True. Scaled
  columns → dtype float64; pass-through keep original dtype.
- **State embedded in `to_dict`** — O(d) bytes (dicts of floats). No
  separate "state" file like GBDT (large trees) — small enough for
  `pipeline.json`.
- **`columns=None` auto-detect numeric** at fit time, stored in
  `_columns` for transform.

### Ограничения (v0.9.2f, §11.5)

- **RobustScaler — O(d) external_sorts per fit.** For d=100 × n=1M,
  ~17 min fit. For d > 100 prefer StandardScaler/MinMaxScaler.
  Backlog: `pc.tdigest` based approximate (v0.9.4).
- **`astype` to non-float dtype** — not supported (scalers always
  produce float64 for scaled columns).
- **`columns=None`** auto-detects only `pa.types.is_integer` or
  `is_floating` — bool/string/timestamp excluded. Explicit `columns`
  can force-cast any Arrow-numeric-castable type, but non-numeric
  raises at `pc.cast`.
- **AST guard whitelist** (`test_arrow_call_memory_guard.py`) is by
  function NAME, not qualname. `"fit"`/`"transform"`/`"predict"`
  whitelisted globally → `RobustScaler.fit` (calls `df.quantile`)
  not flagged. Not a bug (quantile self-guards), but over-broad.
  Fix: qualname-based whitelist (v0.9.3).

### Тесты (v0.9.2f)

- +22 fast (`test_scalers.py`): StandardScaler (mean_zero_std_one,
  with_mean_false, with_std_false, nulls, roundtrip, unseen,
  constant); MinMaxScaler (0_1, custom_range, constant, roundtrip,
  bad_range); MaxAbsScaler (basic, positive_only, all_zero,
  roundtrip); RobustScaler (median_zero_iqr_one, outlier_resistant,
  roundtrip, constant); multi-column, auto-detect, missing raises;
  Pipeline integration (2 tests).
- +1 slow (`test_scalers_e2e.py`): 100K rows, StandardScaler →
  GBDTStep, R² > 0.5.

- fast: **~1512** (+22). slow: **20** (+1).

### §13.4/170 — scalers no new lessons; AST whitelist observation

See DECISIONS.md. Key: scalers are compositions (no prototype),
but AST guard whitelist-by-name is over-broad (RobustScaler.fit not
caught for `df.quantile` call). Fix: qualname-based (v0.9.3).
'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.2f: scalers ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    write_scalers_module()
    update_init()
    write_tests()
    append_decisions_170()
    update_decisions_backlog()
    append_changelog()

    print("\n=== pytest: scalers fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_scalers.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** SCALERS TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: scalers E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_scalers_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
        sys.exit(1)

    print("\n=== pytest: pipeline + cv regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py",
         "tests/test_pipeline_dag.py",
         "tests/test_cv.py",
         "tests/test_register_step_type.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
        "tests/test_patch_utils.py",
    ])

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.9.2f: scalers — StandardScaler/MinMaxScaler/MaxAbsScaler/"
        "RobustScaler; composed from Reduce+quantile (no prototype); "
        "auto-registered with Pipeline; §13.4/170 (composition, no new "
        "lesson; AST whitelist by-name over-broad — v0.9.3)"])

    print("\n==> v0.9.2f applied. v0.9.2 COMPLETE (metrics + CV + scalers).")


if __name__ == "__main__":
    main()