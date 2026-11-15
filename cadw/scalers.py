"""Scalers for feature normalization (v0.9.2f).

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
    # v0.9.235: narrow step (per-batch transform, small state)
    is_barrier = False

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
    # v0.9.235: narrow step (per-batch transform, small state)
    is_barrier = False

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
    # v0.9.235: narrow step (per-batch transform, small state)
    is_barrier = False

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
    # v0.9.235: narrow step (per-batch transform, small state)
    is_barrier = False

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
