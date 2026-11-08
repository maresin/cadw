"""Theta forecasting method — STM on STL decomposition (v0.9.77b).

Assimakopoulos & Nikolopoulos (2000), "The theta model: a decomposition
approach to forecasting", Int. J. Forecasting 16(4).

v0.9.77b changes (vs v0.9.77 initial classical decomposition):
  - STL decomposition (_decomposition.stl_decompose) instead of
    classical decomposition. Fixes the two known STM weaknesses:
    * Trend leak into seasonal estimate (classical pos_means include
      trend drift; STL LOESS separates them cleanly).
    * Better trend estimation via LOESS local-linear.

The Theta method (STM variant):
  1. If seasonal_period > 1: STL decompose y into (T, S, R). Work on
     Y_ds = T + R = y - S.
  2. OLS on Y_ds: theta_0 = a + b*t (intercept, slope).
  3. theta_2 = 2*Y_ds - theta_0  (curvature = fluctuations around OLS).
  4. SES on theta_2 with optimal alpha (grid over 19 values).
  5. Forecast_ds = 0.5 * theta_0_forecast + 0.5 * theta_2_forecast.
  6. If seasonal: forecast = forecast_ds + S_forecast (repeat last
     cycle from STL seasonal component).

References:
  - R forecast::thetaf (Hyndman) - STL + theta + reseasonalize.
  - sktime ThetaForecaster - STM (fixed weights 0.5/0.5).
  - statsforecast AutoTheta - OTM (weight optimisation; not in this
    module, in cadw/ts/otm.py as v0.9.77c).
  - Fiorucci et al. (2016) - "Models for optimising the theta method
    and their connection to state space models" (OTM theory).

KNOWN LIMITATION (documented by benchmark):
  STM performs well on M3-like series where seasonality dominates
  trend. On extreme trend-dominated series (trend >90% of variance),
  STM MASE ~1.4 vs AutoETS 0.5, because SES on theta_2 lags behind
  trend by (1-alpha)/alpha * slope per step. STL fixes the decomposition
  part (trend leak) but not the SES lag — that is fundamental to STM.
  OTM (v0.9.77c) fixes this via weight optimisation. For
  trend-dominated series, prefer AutoETS or AutoTheta (when available).

Pure numpy (no sklearn/statsmodels/scipy/torch in this module).
"""
from __future__ import annotations

import math
import numpy as np

from cadw.ts._decomposition import stl_decompose
from cadw.ts.models import (
    ForecasterBase, _check_df, _extract_series,
)

# v0.9.132 (DEC-054): _ols_fit + _erfinv -> cadw.ops.numeric
from cadw.ops.linalg import ols_fit as _ols_fit
from cadw.ops.numeric import erfinv as _erfinv


# ── Helpers ───────────────────────────────────────────────────────



def _ols_forecast(a: float, b: float, n: int,
                  steps: int) -> np.ndarray:
    h = np.arange(1, steps + 1, dtype=np.float64)
    return a + b * (n - 1 + h)


def _ses_fit(y: np.ndarray, alpha: float
             ) -> tuple[float, np.ndarray, float]:
    """Simple exponential smoothing. Returns (final_level, fitted, sse)."""
    n = y.size
    fitted = np.empty(n, dtype=np.float64)
    lvl = float(y[0])
    sse = 0.0
    for t in range(n):
        fitted[t] = lvl
        err = y[t] - lvl
        sse += err * err
        lvl = lvl + alpha * err
    return lvl, fitted, sse


def _best_alpha(y: np.ndarray) -> float:
    """Grid alpha over [0.05, 0.95] step 0.05 (19 candidates)."""
    best_a, best_sse = 0.5, float("inf")
    for a_i in range(1, 20):
        a = a_i / 20.0
        _, _, sse = _ses_fit(y, a)
        if sse < best_sse:
            best_sse = sse
            best_a = a
    return best_a




# ── ThetaForecaster (STM) ─────────────────────────────────────────

class ThetaForecaster(ForecasterBase):
    """Standard Theta Method (STM) with STL decomposition.

    Parameters
    ----------
    seasonal_period : int, default=1
        1 = no seasonality (pure STM on raw series).
    n_iter : int, default=2
        STL inner iterations (only used if seasonal_period > 1).
    """

    handles_trend: bool = True
    handles_seasonality: bool = True
    handles_multi_seasonality: bool = False

    def __init__(self, seasonal_period: int = 1, *,
                 n_iter: int = 2):
        super().__init__()
        if seasonal_period < 1:
            raise ValueError(
                "ThetaForecaster: seasonal_period must be >= 1")
        self.seasonal_period = int(seasonal_period)
        self.n_iter = int(n_iter)
        self.y_ = None
        self.season_ = None
        self.alpha_ = None
        self.sigma2_ = None
        self.fitted_ = None
        self.trend_ = None
        self.slope_ = None
        self.decomp_ = None

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        _check_df(df, target_col, time_col, group_col)
        y = _extract_series(df, target_col, time_col=time_col)
        if y.size < 4:
            raise ValueError(
                "ThetaForecaster: need at least 4 observations")
        if np.any(~np.isfinite(y)):
            raise ValueError(
                "ThetaForecaster: series contains non-finite values")
        s = self.seasonal_period
        if s > 1 and y.size < 2 * s:
            raise ValueError(
                f"ThetaForecaster: need at least 2*seasonal_period="
                f"{2*s} obs for seasonal mode")

        # 1. Optional STL decomposition
        if s > 1:
            decomp = stl_decompose(y, period=s, n_iter=self.n_iter)
            seasonal = decomp.seasonal
            y_ds = y - seasonal  # = T + R
            self.decomp_ = decomp
        else:
            seasonal = None
            y_ds = y.copy()

        # 2. OLS on deseasonalized series
        a, b = _ols_fit(y_ds)
        theta_0_fit = a + b * np.arange(y.size, dtype=np.float64)

        # 3. theta_2 = 2 * y_ds - theta_0_fit
        theta_2 = 2.0 * y_ds - theta_0_fit

        # 4. SES on theta_2
        alpha = _best_alpha(theta_2)
        _, theta_2_fit, _ = _ses_fit(theta_2, alpha)

        # 5. In-sample combine
        fit_ds = 0.5 * theta_0_fit + 0.5 * theta_2_fit

        # 6. Reseasonalize in-sample
        if s > 1 and seasonal is not None:
            fitted = fit_ds + seasonal
        else:
            fitted = fit_ds
        resid = y - fitted
        self.sigma2_ = float(np.mean(resid * resid))

        self.y_ = y
        self.season_ = seasonal
        self.alpha_ = alpha
        self.trend_ = a
        self.slope_ = b
        self.fitted_ = fitted
        self._is_fitted = True
        return self

    def predict(self, steps):
        y = self.y_
        n = y.size
        if self.seasonal_period > 1 and self.season_ is not None:
            y_ds = y - self.season_
        else:
            y_ds = y

        a, b = self.trend_, self.slope_
        theta_0_f = _ols_forecast(a, b, n, steps)

        theta_2 = 2.0 * y_ds - (a + b * np.arange(n, dtype=np.float64))
        lvl, _, _ = _ses_fit(theta_2, self.alpha_)
        theta_2_f = np.full(steps, lvl, dtype=np.float64)

        combined = 0.5 * theta_0_f + 0.5 * theta_2_f

        if self.seasonal_period > 1 and self.season_ is not None:
            s = self.seasonal_period
            idx = (n + np.arange(steps)) % s
            return combined + self.season_[idx]
        return combined

    def predict_in_sample(self):
        return self.fitted_.copy()

    def prediction_interval(self, steps, level=0.95):
        z = math.sqrt(2.0) * _erfinv(2.0 * (0.5 + level / 2.0) - 1.0)
        f = self.predict(steps)
        sig = float(np.sqrt(max(self.sigma2_, 0.0)))
        h = np.arange(1, steps + 1, dtype=np.float64)
        band = z * sig * np.sqrt(h)
        return f - band, f + band

    def state_nbytes(self):
        if self.y_ is None:
            return 0
        return int(self.y_.nbytes + self.fitted_.nbytes + 256)

