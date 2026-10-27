"""Hybrid ETS + GBDT forecaster (v0.9.86).

Zhang (2003) M4-winner family: ETS captures trend+seasonality,
GBDT captures nonlinear residual patterns ETS misses.

Final forecast = ETS_fc + GBDT_fc(residuals).
"""
from __future__ import annotations

import numpy as np

from .ets import ETSForecaster
from .models import (
    ForecasterBase,
    GBDTForecaster,
    _check_df,
    _extract_series,
)


class HybridETSGBDTForecaster(ForecasterBase):
    """ETS + GBDT on residuals (M4 winner family, Zhang 2003).

    Final forecast = ETS_fc + GBDT_fc(residuals). ETS captures
    trend+seasonality; GBDT captures nonlinear residual patterns
    ETS leaves behind.

    Parameters
    ----------
    ets_kwargs : dict, optional
        Passed to ETSForecaster. Default: {} (ETS defaults; user
        should pass {'seasonal_period': s} if the series is seasonal).
    gbdt_kwargs : dict, optional
        Passed to GBDTForecaster for the residual model. Default:
        {'mode': 'recursive', 'lags': (1, 2, 3)}.
    min_resid_train : int, default 20
        Minimum finite residuals required to fit residual GBDT. If
        fewer, GBDT is skipped and hybrid == ETS. Prevents overfitting
        the residual model on tiny samples.

    Prediction intervals
    --------------------
    The PI is the ETS PI shifted by the GBDT point forecast (additive
    point shift; GBDT uncertainty is NOT added to the width). If the
    hybrid forecast is badly calibrated on a particular series, use
    AutoTheta instead (calibrated PIs via holdout MASE). Documented
    limitation; proper combined-variance PI is backlog.

    Not supported in v0.9.86
    ------------------------
    - Multi-series (group_col): NotImplementedError.
    - Nested transforms (differencing=, decompose=): wrap Hybrid in
      AutoForecastPipeline if the series needs stationarization;
      ETS handles trend+season natively, GBDT for residual is on the
      stationary residual already.
    """
    handles_trend = True
    handles_seasonality = True
    handles_multi_seasonality = False

    def __init__(
        self,
        ets_kwargs=None,
        gbdt_kwargs=None,
        min_resid_train=20,
    ):
        super().__init__()
        self.ets_kwargs = dict(ets_kwargs or {})
        self.gbdt_kwargs = dict(gbdt_kwargs or {
            'mode': 'recursive', 'lags': (1, 2, 3),
        })
        self.min_resid_train = int(min_resid_train)
        if self.min_resid_train < 3:
            raise ValueError(
                f"HybridETSGBDTForecaster: min_resid_train must be "
                f">= 3; got {min_resid_train}")
        self._ets = None
        self._gbdt = None
        self._y = None
        self._y_hat_ets_in_sample = None
        self._gbdt_skipped = False
        self._is_fitted = False

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        if group_col is not None:
            raise NotImplementedError(
                "HybridETSGBDTForecaster: group_col not supported in "
                "v0.9.86 (single-series). Wrap per-group in a loop.")
        _check_df(df, target_col, time_col, None)
        y = _extract_series(df, target_col, time_col=time_col)
        if len(y) < 10:
            raise ValueError(
                "HybridETSGBDTForecaster: need at least 10 observations; "
                f"got {len(y)}")
        self._y = np.asarray(y, dtype=np.float64)

        # Step 1: fit ETS on y
        self._ets = ETSForecaster(**self.ets_kwargs).fit(
            df, target_col)
        y_hat_ets = np.asarray(
            self._ets.predict_in_sample(), dtype=np.float64)
        if y_hat_ets.shape != self._y.shape:
            raise RuntimeError(
                f"ETS in-sample shape {y_hat_ets.shape} != y shape "
                f"{self._y.shape}; hybrid requires full-length "
                f"in-sample prediction from ETS")
        self._y_hat_ets_in_sample = y_hat_ets

        # Step 2: residual + GBDT
        resid = self._y - y_hat_ets
        n_valid = int(np.isfinite(resid).sum())
        if n_valid < self.min_resid_train:
            self._gbdt = None
            self._gbdt_skipped = True
        else:
            import pyarrow as pa
            import warnings
            from cadw import CadwDataFrame
            resid_df = CadwDataFrame.from_table(
                pa.table({'y': pa.array(resid, type=pa.float64())}))
            try:
                self._gbdt = GBDTForecaster(
                    **self.gbdt_kwargs).fit(resid_df, 'y')
                self._gbdt_skipped = False
            except Exception as exc:
                warnings.warn(
                    f"HybridETSGBDT: GBDT residual fit failed "
                    f"({type(exc).__name__}: {exc}); falling back "
                    f"to ETS-only",
                    stacklevel=2,
                )
                self._gbdt = None
                self._gbdt_skipped = True

        self._is_fitted = True
        return self

    def predict(self, steps):
        if not getattr(self, '_is_fitted', False):
            raise RuntimeError(
                f"{type(self).__name__}.predict called before fit")
        if not isinstance(steps, int):
            raise TypeError(
                f"{type(self).__name__}.predict: steps must be int, "
                f"got {type(steps).__name__}")
        if steps < 1:
            raise ValueError(
                f"{type(self).__name__}.predict: steps must be >= 1, "
                f"got {steps}")
        ets_fc = np.asarray(
            self._ets.predict(steps), dtype=np.float64)
        if self._gbdt is None:
            return ets_fc
        gbdt_fc = np.asarray(
            self._gbdt.predict(steps), dtype=np.float64)
        return ets_fc + gbdt_fc

    def predict_in_sample(self):
        if not getattr(self, '_is_fitted', False):
            raise RuntimeError(
                f"{type(self).__name__}.predict_in_sample called "
                f"before fit")
        ets_in = np.asarray(
            self._ets.predict_in_sample(), dtype=np.float64)
        if self._gbdt is None:
            return ets_in
        gbdt_in = np.asarray(
            self._gbdt.predict_in_sample(), dtype=np.float64)
        out = ets_in.copy()
        valid = np.isfinite(gbdt_in)
        out[valid] = ets_in[valid] + gbdt_in[valid]
        return out

    def prediction_interval(self, steps, level=0.95):
        if not getattr(self, '_is_fitted', False):
            raise RuntimeError(
                f"{type(self).__name__}.prediction_interval called "
                f"before fit")
        ets_lo, ets_hi = self._ets.prediction_interval(
            steps, level=level)
        ets_lo = np.asarray(ets_lo, dtype=np.float64)
        ets_hi = np.asarray(ets_hi, dtype=np.float64)
        if self._gbdt is None:
            return ets_lo, ets_hi
        gbdt_fc = np.asarray(
            self._gbdt.predict(steps), dtype=np.float64)
        return ets_lo + gbdt_fc, ets_hi + gbdt_fc

    def state_nbytes(self):
        total = 0
        try:
            total += int(self._ets.state_nbytes()) if self._ets else 0
        except (AttributeError, TypeError):
            pass
        try:
            total += int(self._gbdt.state_nbytes()) if self._gbdt else 0
        except (AttributeError, TypeError):
            pass
        return total


__all__ = ["HybridETSGBDTForecaster"]
