"""OTM: weight-optimized Theta method (v0.9.78a).

Practical OTM = STM with optimized (w2, alpha) weights, fixing theta=2
(STM convention). Grid search over (w2, alpha) minimizing in-sample MSE
on deseasonalized series.

Deviation from Fiorucci et al. (2016) strict OTM definition:
  - theta is FIXED at 2.0 (STM convention); Fiorucci OTM optimizes it
  - no separate drift term; the OLS trend inside theta_0 captures drift
  - weights optimized on grid (not continuous gradient)
Full Fiorucci OTM (theta optimization + explicit drift parameter) is in
backlog as 'Full OTM'.

Prediction:
  y_ds = y - seasonal (STL, or raw if s=1)
  theta_0 = OLS(y_ds)   (linear extrapolation)
  theta_2 = 2 * y_ds - theta_0
  theta_2_fit = SES(theta_2, alpha)   (level forecast)
  forecast = (1 - w2) * theta_0_f + w2 * theta_2_f  +  seasonal_forecast

Why this fixes STM limitation:
  STM fixes w2 = 0.5. On trend-dominated series optimal w2 ~= 0.1
  (trust OLS trend, ignore SES lag); on M3-like seasonal optimal
  w2 ~= 0.5-0.7 (trust SES). Auto-selection via grid adapts.

Reference: statsforecast AutoTheta (OTM variant, MASE 1.381 fev-bench);
Fiorucci et al. 2016 (theoretical equivalence to SES-with-drift).

Pure numpy; no scipy/statsmodels/sklearn.
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



def _ses(y: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    """SES: returns (final_level, fitted_series)."""
    n = y.size
    fitted = np.empty(n, dtype=np.float64)
    lvl = float(y[0])
    for t in range(n):
        fitted[t] = lvl
        lvl = lvl + alpha * (y[t] - lvl)
    return lvl, fitted




# ── OTMThetaForecaster ───────────────────────────────────────────

class OTMThetaForecaster(ForecasterBase):
    """OTM: weight-optimized STM (theta=2 fixed, (w2, alpha) optimized).

    Parameters
    ----------
    seasonal_period : int, default=1
    n_iter : int, default=2
        STL inner iterations (only if seasonal_period > 1).
    """

    handles_trend: bool = True
    handles_seasonality: bool = True
    handles_multi_seasonality: bool = False

    W2_GRID = tuple(i / 10.0 for i in range(1, 10))   # 0.1 .. 0.9
    ALPHA_GRID = tuple(i / 20.0 for i in range(1, 20))  # 0.05 .. 0.95

    def __init__(self, seasonal_period: int = 1, *, n_iter: int = 2):
        super().__init__()
        if seasonal_period < 1:
            raise ValueError(
                "OTMThetaForecaster: seasonal_period must be >= 1")
        self.seasonal_period = int(seasonal_period)
        self.n_iter = int(n_iter)
        self.y_ = None
        self.season_ = None
        self.w2_ = None
        self.alpha_ = None
        self.trend_ = None
        self.slope_ = None
        self.sigma2_ = None
        self.fitted_ = None
        self.decomp_ = None
        self.selection_mse_ = None

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        _check_df(df, target_col, time_col, group_col)
        y = _extract_series(df, target_col, time_col=time_col)
        if y.size < 4:
            raise ValueError(
                "OTMThetaForecaster: need at least 4 observations")
        if np.any(~np.isfinite(y)):
            raise ValueError(
                "OTMThetaForecaster: series contains non-finite values")
        s = self.seasonal_period
        if s > 1 and y.size < 2 * s:
            raise ValueError(
                f"OTMThetaForecaster: need at least 2*seasonal_period="
                f"{2*s} obs for seasonal mode")

        # v0.9.129: guard before O(n) transients (G2, AP4).
        # Peak = 31*n*8 + 256 (grid search: 19 SES fitted arrays +
        # temporaries). bounded_by_domain: n<=1e6 -> peak<=248 MB.
        _mg = getattr(df, "_memory_guard", None)
        if _mg is not None:
            _mg.check_state(31 * y.size * 8 + 256, name="otm.fit")

        # 1. Decompose
        if s > 1:
            decomp = stl_decompose(y, period=s, n_iter=self.n_iter)
            seasonal = decomp.seasonal
            y_ds = y - seasonal
            self.decomp_ = decomp
        else:
            seasonal = None
            y_ds = y.copy()

        # 2. Theta lines
        a, b = _ols_fit(y_ds)
        n = y.size
        x = np.arange(n, dtype=np.float64)
        theta_0 = a + b * x
        theta_2 = 2.0 * y_ds - theta_0

        # 3. Precompute SES fits per alpha (O(n) each)
        ses_fits = {alpha: _ses(theta_2, alpha)
                    for alpha in self.ALPHA_GRID}

        # 4. Grid search (w2, alpha) minimizing in-sample MSE on y_ds
        best_mse = float("inf")
        best_w2 = 0.5
        best_alpha = 0.5
        for w2 in self.W2_GRID:
            for alpha in self.ALPHA_GRID:
                _, theta_2_fit = ses_fits[alpha]
                fit_ds = (1.0 - w2) * theta_0 + w2 * theta_2_fit
                mse = float(np.mean((y_ds - fit_ds) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_w2 = w2
                    best_alpha = alpha

        # 5. Final in-sample
        _, theta_2_fit_best = ses_fits[best_alpha]
        fit_ds = (1.0 - best_w2) * theta_0 + best_w2 * theta_2_fit_best
        if s > 1 and seasonal is not None:
            fitted = fit_ds + seasonal
        else:
            fitted = fit_ds
        resid = y - fitted
        self.sigma2_ = float(np.mean(resid * resid))

        self.y_ = y
        self.season_ = seasonal
        self.w2_ = best_w2
        self.alpha_ = best_alpha
        self.trend_ = a
        self.slope_ = b
        self.fitted_ = fitted
        self.selection_mse_ = best_mse
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
        x = np.arange(n, dtype=np.float64)
        theta_0 = a + b * x
        theta_2 = 2.0 * y_ds - theta_0
        lvl, _ = _ses(theta_2, self.alpha_)
        h = np.arange(1, steps + 1, dtype=np.float64)
        theta_0_f = a + b * (n - 1 + h)
        theta_2_f = np.full(steps, lvl, dtype=np.float64)
        combined = (1.0 - self.w2_) * theta_0_f + self.w2_ * theta_2_f
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

