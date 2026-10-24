"""Theta-method forecasting (D.4.4, v0.9.38).

Reference: Assimakopoulos, Nikolopoulos, *The theta model: a
decomposition approach to forecasting*, Int. J. Forecasting 16(4),
2000; Hyndman, Athanasopoulos, *Forecasting: Principles and Practice*,
3rd ed., OTexts 2021, ch. 12 (theta method).
See .pingis/algorithm_notes/d4_4_theta.md.

Standard theta decomposition: the series is split into a theta=0 line
(ordinary least squares trend) and theta=2 curve (twice-detrended); the
forecast is the SES forecast of the theta=2 curve plus the OLS line
extrapolated. The fitted SES alpha is chosen by minimising the in-sample
one-step SSE. RAM: O(n) training series + O(1) state.
"""
from __future__ import annotations

import numpy as np

from cadw.ml._logging import log_fit
from cadw.ts.models import ForecasterBase, _check_df, _extract_series


def _ols_trend(y):
    n = y.size
    t = np.arange(n, dtype=np.float64)
    tc = t - t.mean()
    slope = float(np.dot(tc, y - y.mean()) / np.dot(tc, tc)) if n > 1 else 0.0
    intercept = float(y.mean() - slope * t.mean())
    return intercept, slope


def _ses(y, alpha):
    out = np.empty_like(y)
    lvl = float(y[0])
    sse = 0.0
    for t in range(1, y.size):
        pred = lvl
        out[t - 1] = pred
        e = y[t] - pred
        sse += e * e
        lvl += alpha * e
    out[-1] = lvl
    return out, lvl, sse


class ThetaForecaster(ForecasterBase):
    """Classic theta-method forecaster (Assimakopoulos & Nikolopoulos)."""

    def __init__(self, seasonal=False, seasonal_period=1):
        super().__init__()
        if seasonal and seasonal_period < 2:
            raise ValueError("ThetaForecaster: seasonal=True needs seasonal_period >= 2")
        self.seasonal = bool(seasonal)
        self.seasonal_period = int(seasonal_period)
        self.y_ = None
        self.alpha_ = None
        self.intercept_ = None
        self.slope_ = None
        self.sigma2_ = None
        self.fitted_ = None

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        _check_df(df, target_col, time_col, group_col)
        y = _extract_series(df, target_col, time_col=time_col)
        if y.size < 3:
            raise ValueError("ThetaForecaster: need at least 3 observations")
        if np.any(~np.isfinite(y)):
            raise ValueError("ThetaForecaster: series contains non-finite values")
        self.y_ = y
        self.intercept_, self.slope_ = _ols_trend(y)
        line = self.intercept_ + self.slope_ * np.arange(y.size)
        theta2 = 2.0 * y - line
        best = (np.inf, 0.5)
        for a in np.linspace(0.05, 0.95, 19):
            _, _, sse = _ses(theta2, a)
            if sse < best[0]:
                best = (sse, float(a))
        self.alpha_ = best[1]
        fitted, _, sse = _ses(theta2, self.alpha_)
        self.fitted_ = fitted + line
        self.sigma2_ = float(sse / max(1, y.size))
        self._is_fitted = True
        log_fit("theta", loss=sse, state_bytes=self.state_nbytes())
        return self

    def state_nbytes(self):
        if self.y_ is None:
            return 0
        return int(self.y_.nbytes + self.fitted_.nbytes + 256)

    def _predict(self, steps):
        y = self.y_
        n = y.size
        line = self.intercept_ + self.slope_ * np.arange(n)
        theta2 = 2.0 * y - line
        _, lvl, _ = _ses(theta2, self.alpha_)
        h = np.arange(1, steps + 1, dtype=np.float64)
        theta0_fc = self.intercept_ + self.slope_ * (n - 1 + h)
        theta2_fc = np.full(steps, lvl, dtype=np.float64)
        return 0.5 * theta0_fc + 0.5 * theta2_fc

    def _predict_in_sample(self):
        return self.fitted_.copy()
