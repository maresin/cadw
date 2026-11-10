"""Shared Theta/AutoTheta helpers (v0.9.194).

Moved from cadw.ts.auto_theta (DEC-070 primitives-first): avoid
circular import between theta.py (uses helpers in _fit_auto) and
auto_theta.py (shim re-exports for backward compat).

Contents:
  _naive_seasonal_mae  -- seasonally-naive MAE (MASE denominator)
  _mase                -- MASE
  _log_transform_batches, _mk_log_df -- streaming log(y) transform
  _MultiplicativeWrapper -- log-space wrapper (v0.9.193)
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa

from cadw.ts.models import (
    ForecasterBase, _check_df, _extract_series, _iter_target_batches,
)

def _naive_seasonal_mae(y: np.ndarray, s: int) -> float:
    """Naive seasonal MAE denominator for MASE."""
    y = np.asarray(y, dtype=np.float64)
    if s > 1 and y.size > s:
        return float(np.mean(np.abs(y[s:] - y[:-s])))
    return float(np.mean(np.abs(np.diff(y)))) if y.size > 1 else 1.0

def _mase(y_true: np.ndarray, y_pred: np.ndarray,
          denom: float) -> float:
    return float(np.mean(np.abs(y_true - y_pred)) / max(denom, 1e-12))

def _log_transform_batches(df, target_col, *, time_col=None,
                            group_col=None, batch_size=None,
                            save_accum=None, n_counter=None):
    """v0.9.193: streaming log-transform — yields pa.RecordBatch of log(y).

    Per-batch O(batch) RAM. Rejects non-positive y (multiplicative
    semantic requires y > 0). Optionally accumulates raw y into
    `save_accum` (list) if provided (save_y=True legacy path).
    Optionally accumulates total seen count into `n_counter` (list
    with single int element) — useful to preserve total obs seen even
    when inner truncates to W_eff.

    Uses _iter_target_batches (v0.9.186) + pyarrow construction per
    batch. No O(n) numpy materialization.
    """
    import pyarrow as pa
    schema = pa.schema([pa.field(target_col, pa.float64())])
    for arr in _iter_target_batches(
            df, target_col, time_col=time_col, group_col=group_col,
            group_value=None, batch_size=batch_size):
        if arr.size == 0:
            continue
        if np.any(arr <= 0):
            raise ValueError(
                "_MultiplicativeWrapper: y must be > 0")
        if save_accum is not None:
            save_accum.append(arr.copy())
        if n_counter is not None:
            n_counter[0] += int(arr.size)
        log_b = np.log(arr)
        yield pa.RecordBatch.from_arrays(
            [pa.array(log_b, type=pa.float64())], schema=schema)

def _mk_log_df(df, target_col, *, time_col=None, group_col=None,
               save_accum=None):
    """Build a CadwDataFrame from streaming log-transform.

    Eager-to-disk via Dataset.from_batches; RAM O(batch), disk O(n).

    Returns (log_df, n_seen):
      log_df — CadwDataFrame on transformed (log) series.
      n_seen — total rows observed pre-truncation (inner may truncate
               to W_eff; n_seen captures all input rows).
    """
    import pyarrow as pa
    from cadw import CadwDataFrame
    from cadw.dataset.dataset import Dataset

    schema = pa.schema([pa.field(target_col, pa.float64())])
    mg = getattr(df, "_memory_guard", None)
    n_counter = [0]
    ds = Dataset.from_batches(
        schema,
        _log_transform_batches(df, target_col, time_col=time_col,
                               group_col=group_col, save_accum=save_accum,
                               n_counter=n_counter),
        df._new_dataset_dir(),
    )
    log_df = CadwDataFrame(
        ds,
        memory_guard=mg,
        spill_dir=getattr(df, "_spill_dir", None),
    )
    return log_df, n_counter[0]

class _MultiplicativeWrapper(ForecasterBase):
    """Wrap STM or OTM; fit on log(y), predict exp(.).

    Only for y > 0 (checked by caller). Reuses inner class's STL-
    additive decomposition on the log-transformed series, which is
    equivalent to multiplicative decomposition on y.

    v0.9.193 (DEC-070 streaming): no persistent self.y_; log-transform
    is streaming per batch (via _iter_target_batches + Dataset.from_batches).
    max_window/strict propagate to inner Theta/OTM (v0.9.191).
    """

    def __init__(self, inner_factory, seasonal_period: int, *,
                 max_window="auto", strict: bool = False):
        super().__init__()
        self._inner_factory = inner_factory
        self.seasonal_period = int(seasonal_period)
        self.max_window = max_window
        self.strict = strict
        self._inner = None
        self.n_total_ = None
        self.target_col_ = None
        self.time_col_ = None
        self._saved_y_ = None
        self.handles_trend = True
        self.handles_seasonality = True
        self.handles_multi_seasonality = False

    def fit(self, df, target_col, *, time_col=None, group_col=None,
            save_y=False):
        _check_df(df, target_col, time_col, group_col)
        self.target_col_ = target_col
        self.time_col_ = time_col

        save_accum = [] if save_y else None
        log_df, n_seen = _mk_log_df(
            df, target_col, time_col=time_col,
            group_col=group_col, save_accum=save_accum)

        self._inner = self._inner_factory(self.seasonal_period)
        # Propagate max_window/strict to inner (Theta/OTM v0.9.191
        # support these; other forecasters ignore via TypeError).
        try:
            self._inner.fit(
                log_df, target_col,
                max_window=self.max_window, strict=self.strict)
        except TypeError:
            self._inner.fit(log_df, target_col)

        # v0.9.193: n_total_ = total rows seen pre-truncation (inner
        # may have truncated to W_eff; n_total_ preserves total seen).
        self.n_total_ = n_seen
        self._saved_y_ = (
            np.concatenate(save_accum)
            if save_y and save_accum else None
        )
        # v0.9.193: propagate ApproximationInfo from inner Theta/OTM
        # (e.g. window_truncated when max_window < n). Wrapper inherits
        # ApproximationMixin via ForecasterBase; user sees approximation
        # on wrapper, not buried in inner.
        inner_approx = getattr(self._inner, "approximation", None)
        if inner_approx:
            self.approximation = list(inner_approx)
        self._is_fitted = True
        return self

    def predict(self, steps):
        log_f = self._inner.predict(steps)
        return np.exp(log_f)

    def predict_in_sample(self, df=None):
        """v0.9.193: OOC default requires df (streaming log-transform)."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__}.predict_in_sample called before fit")
        if df is not None:
            log_df, _ = _mk_log_df(
                df, self.target_col_, time_col=self.time_col_)
            log_f = self._inner.predict_in_sample(log_df)
            return np.exp(log_f)
        if self._saved_y_ is not None:
            # Legacy path: reconstruct log_df from saved raw y
            import pyarrow as pa
            from cadw import CadwDataFrame
            log_y = np.log(self._saved_y_)
            log_df = CadwDataFrame.from_table(
                pa.table({self.target_col_: pa.array(log_y,
                                                     type=pa.float64())}))
            log_f = self._inner.predict_in_sample(log_df)
            return np.exp(log_f)
        raise TypeError(
            f"{type(self).__name__}.predict_in_sample: df is required "
            f"(OOC default since v0.9.193). Use fit(..., save_y=True) for "
            f"legacy no-df calls (adds O(n) persistent state).")

    def prediction_interval(self, steps, level=0.95):
        lo_log, hi_log = self._inner.prediction_interval(steps, level)
        return np.exp(lo_log), np.exp(hi_log)

    def state_nbytes(self):
        """v0.9.193: inner + O(1) header + optional _saved_y_ (legacy).
        Was: under-reporting self.y_ O(n) (persistent but uncounted)."""
        if not self._is_fitted:
            return 0
        inner = int(self._inner.state_nbytes()) if self._inner else 0
        sz = inner + 256 + 16  # header + n_total_/seasonal_period
        if self._saved_y_ is not None:
            sz += self._saved_y_.nbytes
        return int(sz)

