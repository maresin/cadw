"""AutoTheta: CV-based selection between STM and OTM variants (v0.9.78b).

AutoTheta evaluates 4 candidates and picks the one with lowest
out-of-sample MASE:

    STM (additive decomposition) + STM (multiplicative)
    OTM (additive)               + OTM (multiplicative)

where:
  - STM = ThetaForecaster (fixed w2=0.5, theta=2)
  - OTM = OTMThetaForecaster (weight-optimized w2, theta=2)
  - additive = classic STL (y = T + S + R)
  - multiplicative = STL on log(y) (requires y > 0); forecast = exp(.)

Multiplicative candidates are skipped if any y <= 0.

Cross-validation protocol:
  - Holdout split: last `val_steps` points as validation.
    Default val_steps = max(s, n // 5) where s = seasonal_period.
  - Fit each candidate on train (all but last `val_steps`).
  - Compute MASE on validation (naive seasonal baseline denominator:
    mean |y[t] - y[t-s]| on train).
  - Pick candidate with min MASE.
  - Refit winner on full series.

Fallback: if no candidate can fit (e.g. all fail), warning + Naive.

This wrapper NEVER underperforms the best of {STM, OTM} on validation,
by construction. It can only be worse on the test set (if selection
overfits the validation split — mitigated by holdout not CV folds).

Reference: statsforecast AutoTheta (OTM variant, CV-based).

Pure numpy; no sklearn/statsmodels/scipy (AST-guard enforced).
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from cadw.ts._decomposition import stl_decompose
from cadw.ts.models import (
    ForecasterBase, NaiveForecaster, _check_df, _extract_series,
    _iter_target_batches,
)
from cadw.ts.otm import OTMThetaForecaster
from cadw.ts.theta import ThetaForecaster


def _naive_seasonal_mae(y: np.ndarray, s: int) -> float:
    """Naive seasonal MAE denominator for MASE."""
    y = np.asarray(y, dtype=np.float64)
    if s > 1 and y.size > s:
        return float(np.mean(np.abs(y[s:] - y[:-s])))
    return float(np.mean(np.abs(np.diff(y)))) if y.size > 1 else 1.0


def _mase(y_true: np.ndarray, y_pred: np.ndarray,
          denom: float) -> float:
    return float(np.mean(np.abs(y_true - y_pred)) / max(denom, 1e-12))


# ── Multiplicative wrapper (log-space STM / OTM) ────────────────

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


# ── AutoThetaForecaster ──────────────────────────────────────────

class AutoThetaForecaster(ForecasterBase):
    """CV-selected STM vs OTM, additive vs multiplicative.

    Parameters
    ----------
    seasonal_period : int, default=1
    val_steps : int | None, default=None
        Holdout size for CV. None -> max(s, n // 5).
    n_iter : int, default=2
        STL inner iterations.
    """

    handles_trend: bool = True
    handles_seasonality: bool = True
    handles_multi_seasonality: bool = False

    def __init__(self, seasonal_period: int = 1, *,
                 val_steps: int | None = None, n_iter: int = 2):
        super().__init__()
        if seasonal_period < 1:
            raise ValueError(
                "AutoThetaForecaster: seasonal_period must be >= 1")
        self.seasonal_period = int(seasonal_period)
        self.val_steps = val_steps
        self.n_iter = int(n_iter)
        self.best_model_ = None
        self.selected_config_ = None
        self.cv_scores_ = []
        self.best_mase_ = float("inf")
        self.y_ = None

    def _candidates(self, has_positive: bool) -> list[dict]:
        cands = [
            {"label": "STM_additive",
             "factory": lambda s: ThetaForecaster(
                 seasonal_period=s, n_iter=self.n_iter),
             "note": "STM, additive decomposition"},
            {"label": "OTM_additive",
             "factory": lambda s: OTMThetaForecaster(
                 seasonal_period=s, n_iter=self.n_iter),
             "note": "OTM, additive decomposition"},
        ]
        if has_positive:
            cands.append({
                "label": "STM_multiplicative",
                "factory": lambda s: _MultiplicativeWrapper(
                    lambda ss: ThetaForecaster(
                        seasonal_period=ss, n_iter=self.n_iter), s),
                "note": "STM on log(y) (= multiplicative decomposition)",
            })
            cands.append({
                "label": "OTM_multiplicative",
                "factory": lambda s: _MultiplicativeWrapper(
                    lambda ss: OTMThetaForecaster(
                        seasonal_period=ss, n_iter=self.n_iter), s),
                "note": "OTM on log(y) (= multiplicative decomposition)",
            })
        return cands

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        _check_df(df, target_col, time_col, group_col)
        y = _extract_series(df, target_col, time_col=time_col)
        n = y.size
        if n < 4:
            raise ValueError(
                "AutoThetaForecaster: need at least 4 observations")
        if np.any(~np.isfinite(y)):
            raise ValueError(
                "AutoThetaForecaster: series contains non-finite values")
        s = self.seasonal_period
        if s > 1 and n < 2 * s:
            raise ValueError(
                f"AutoThetaForecaster: need at least 2*seasonal_period="
                f"{2*s} obs for seasonal mode")

        # Determine holdout size
        if self.val_steps is None:
            val_steps = max(s if s > 1 else 0, n // 5)
        else:
            val_steps = int(self.val_steps)
        val_steps = max(1, min(val_steps, n - 4))
        n_train = n - val_steps

        y_train = y[:n_train]
        y_val = y[n_train:]
        denom = _naive_seasonal_mae(y_train, s)

        has_positive = bool((y > 0).all())
        cands = self._candidates(has_positive)

        # CV evaluation on train/val
        best_mase = float("inf")
        best_label = None
        cv_scores = []
        import pyarrow as pa
        from cadw import CadwDataFrame
        train_df = CadwDataFrame.from_table(
            pa.table({target_col: pa.array(y_train)}))
        val_h = y_val.size

        for cand in cands:
            entry = {"label": cand["label"], "note": cand["note"]}
            try:
                model = cand["factory"](s)
                model.fit(train_df, target_col)
                pred = model.predict(val_h)
                mase = _mase(y_val, pred, denom)
                entry["val_mase"] = float(mase)
                if mase < best_mase:
                    best_mase = mase
                    best_label = cand["label"]
            except Exception as e:
                entry["val_mase"] = None
                entry["error"] = f"{type(e).__name__}: {e}"
            cv_scores.append(entry)

        if best_label is None:
            warnings.warn(
                "AutoThetaForecaster: no candidate valid; "
                "falling back to NaiveForecaster(last).",
                stacklevel=2,
            )
            best_model = NaiveForecaster(strategy="last")
            best_model.fit(df, target_col, time_col=time_col,
                           group_col=group_col)
            selected = {"label": "fallback:NaiveForecaster(last)"}
            best_mase = float("inf")
        else:
            # Refit winner on full series
            winner = next(c for c in cands if c["label"] == best_label)
            best_model = winner["factory"](s)
            best_model.fit(df, target_col, time_col=time_col,
                           group_col=group_col)
            selected = {
                "label": best_label,
                "note": winner["note"],
                "val_mase": float(best_mase),
            }

        self.best_model_ = best_model
        self.selected_config_ = selected
        self.cv_scores_ = cv_scores
        self.best_mase_ = float(best_mase)
        self.y_ = y
        self._is_fitted = True
        return self

    def predict(self, steps):
        return self.best_model_.predict(steps)

    def predict_in_sample(self, df=None):
        import inspect
        accepts_df = False
        if df is not None:
            try:
                sig = inspect.signature(self.best_model_.predict_in_sample)
                accepts_df = "df" in sig.parameters
            except (ValueError, TypeError):
                accepts_df = False
        if accepts_df:
            return self.best_model_.predict_in_sample(df)
        return self.best_model_.predict_in_sample()

    def prediction_interval(self, steps, level=0.95):
        if hasattr(self.best_model_, "prediction_interval"):
            return self.best_model_.prediction_interval(steps, level)
        f = self.predict(steps)
        return f, f

    def state_nbytes(self):
        return (int(self.best_model_.state_nbytes())
                if self.best_model_ else 0)

