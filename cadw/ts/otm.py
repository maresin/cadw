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

from cadw.core.approximation import ApproximationInfo
from cadw.ts._decomposition import stl_decompose
from cadw.ts.models import (
    ForecasterBase, _check_df, _extract_series, _iter_target_batches,
    _collect_last_w,
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
        # v0.9.183 OOC state
        self.last_k_ = None
        self.lvl_ = None
        self.n_train_ = None
        self.r_ = None
        self.season_cycle_ = None
        self.w2_ = None
        self.alpha_ = None
        self.trend_ = None
        self.slope_ = None
        self.sigma2_ = None
        self.selection_mse_ = None
        self.target_col_ = None
        self.time_col_ = None
        self._saved_y_ = None

    def fit(self, df, target_col, *, time_col=None, group_col=None,
            save_y=False,
            max_window: "str | int | None" = "auto",
            strict: bool = False):
        """Fit OTM Theta forecast.

        v0.9.183 OOC state (lvl_, season_cycle_, params).
        v0.9.191 (DEC-072 P4): window path via _collect_last_w:
          - W_eff = min(max_window, W_max, n); W_max from guard
            (OTM grid search holds 19 SES fitted + transients ~ 31 arrays).
          - Bit-identical when n <= W_eff.
          - ApproximationInfo(window_truncated) disclosed otherwise.

        max_window: "auto" (default, guard-derived) | int W | None (full).
        save_y: True forces full path (legacy no-df predict_in_sample).
        strict: True -> refuse when auto-adjust would lose data.
        """
        _check_df(df, target_col, time_col, group_col)
        self.target_col_ = target_col
        self.time_col_ = time_col
        self._clear_approximation()
        s = self.seasonal_period

        if save_y and isinstance(max_window, int):
            raise ValueError(
                "OTMThetaForecaster.fit: save_y=True incompatible with "
                f"max_window=int({max_window}). Use max_window=None.")

        if save_y or max_window is None:
            y = _extract_series(df, target_col, time_col=time_col)
            n_total = int(y.size)
        else:
            mg = getattr(df, "_memory_guard", None)
            if mg is None:
                y = _extract_series(df, target_col, time_col=time_col)
                n_total = int(y.size)
            else:
                # OTM grid search: 19 SES fitted arrays + theta_0/theta_2/ds
                # + fitted ~ 31 f64 arrays per window row.
                unit_bytes = 31 * 8
                W_max = mg.get_window_size(
                    unit_bytes=unit_bytes, target_fraction=0.25)
                if max_window == "auto":
                    W_eff = W_max
                elif isinstance(max_window, int):
                    if strict and max_window > int(df.count()):
                        from cadw.core.memory_guard import MemoryLimitExceeded
                        raise MemoryLimitExceeded(
                            f"OTMThetaForecaster.fit: max_window={max_window}"
                            f" > n_rows; strict=True refuses.",
                            model="OTMThetaForecaster",
                            params={"max_window": max_window, "strict": True},
                        )
                    W_eff = min(int(max_window), W_max)
                else:
                    raise ValueError(
                        f"OTMThetaForecaster: max_window must be "
                        f"'auto'|int|None, got {type(max_window).__name__}")
                if W_eff < 2 * s:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"OTMThetaForecaster.fit: W_eff={W_eff} < "
                        f"2*seasonal_period={2*s}. Increase budget or "
                        f"reduce seasonal_period.",
                        model="OTMThetaForecaster",
                        params={"seasonal_period": s},
                    )
                y_used, n_total, lost = _collect_last_w(
                    df, target_col, W_eff, time_col=time_col)
                if lost is not None:
                    self._record_approximation(ApproximationInfo(
                        type="window_truncated",
                        axis="n",
                        requested=max_window,
                        effective=W_eff,
                        reason=("budget_limit" if max_window == "auto"
                                else "explicit"),
                        lost_range=lost,
                        accuracy_impact=(
                            f"STL + OTM grid fit on last {W_eff} obs"),
                    ))
                y = y_used

        if y.size < 4:
            raise ValueError(
                "OTMThetaForecaster: need at least 4 observations")
        if np.any(~np.isfinite(y)):
            raise ValueError(
                "OTMThetaForecaster: series contains non-finite values")
        if s > 1 and y.size < 2 * s:
            raise ValueError(
                f"OTMThetaForecaster: need at least 2*seasonal_period="
                f"{2*s} obs for seasonal mode")

        _mg = getattr(df, "_memory_guard", None)
        if _mg is not None:
            _mg.check_state(31 * y.size * 8 + 256, name="otm.fit")

        n = int(y.size)

        # 1. Decompose
        if s > 1:
            decomp = stl_decompose(y, period=s, n_iter=self.n_iter)
            seasonal = decomp.seasonal
            y_ds = y - seasonal
            season_cycle = seasonal[:s].copy()
        else:
            seasonal = None
            y_ds = y.copy()
            season_cycle = None

        # 2. Theta lines
        a, b = _ols_fit(y_ds)
        x = np.arange(n, dtype=np.float64)
        theta_0 = a + b * x
        theta_2 = 2.0 * y_ds - theta_0

        # 3. Precompute SES fits per alpha
        ses_fits = {alpha: _ses(theta_2, alpha)
                    for alpha in self.ALPHA_GRID}

        # 4. Grid search (w2, alpha)
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
        lvl_best, theta_2_fit_best = ses_fits[best_alpha]
        fit_ds = (1.0 - best_w2) * theta_0 + best_w2 * theta_2_fit_best
        if s > 1 and seasonal is not None:
            fitted = fit_ds + seasonal
        else:
            fitted = fit_ds
        resid = y - fitted
        self.sigma2_ = float(np.mean(resid * resid))

        # v0.9.183: store O(model) state
        k = 1 if s <= 1 else s
        self.last_k_ = y[-k:].copy()
        self.lvl_ = float(lvl_best)
        self.n_train_ = n
        self.r_ = n % s if s > 1 else 0
        self.season_cycle_ = season_cycle
        self.w2_ = best_w2
        self.alpha_ = best_alpha
        self.trend_ = a
        self.slope_ = b
        self.selection_mse_ = best_mse
        self._saved_y_ = y if save_y else None
        self._is_fitted = True
        return self

    def predict(self, steps):
        """v0.9.183 OOC: uses stored lvl_ (O(1)), not full y_ (was O(n))."""
        n = self.n_train_
        a, b = self.trend_, self.slope_
        h = np.arange(1, steps + 1, dtype=np.float64)
        theta_0_f = a + b * (n - 1 + h)
        theta_2_f = np.full(steps, self.lvl_, dtype=np.float64)
        combined = (1.0 - self.w2_) * theta_0_f + self.w2_ * theta_2_f
        s = self.seasonal_period
        if s > 1 and self.season_cycle_ is not None:
            idx = (self.r_ + np.arange(steps)) % s
            return combined + self.season_cycle_[idx]
        return combined

    def predict_in_sample(self, df=None):
        """v0.9.183: OOC default requires df (or save_y=True legacy)."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__}.predict_in_sample called before fit"
            )
        if df is not None:
            y = _extract_series(df, self.target_col_,
                                 time_col=self.time_col_)
            return self._recompute_in_sample(y)
        if self._saved_y_ is not None:
            return self._recompute_in_sample(self._saved_y_)
        raise TypeError(
            f"{type(self).__name__}.predict_in_sample: df is required "
            f"(OOC default since v0.9.183). Use fit(..., save_y=True) for "
            f"legacy no-df calls (adds O(n) persistent state)."
        )

    def _recompute_in_sample(self, y):
        """Recompute fitted values from y (bit-identical for training y)."""
        s = self.seasonal_period
        if s > 1:
            decomp = stl_decompose(y, period=s, n_iter=self.n_iter)
            seasonal = decomp.seasonal
            y_ds = y - seasonal
        else:
            seasonal = None
            y_ds = y.copy()
        a, b = _ols_fit(y_ds)
        x = np.arange(y.size, dtype=np.float64)
        theta_0 = a + b * x
        theta_2 = 2.0 * y_ds - theta_0
        ses_fits = {alpha: _ses(theta_2, alpha) for alpha in self.ALPHA_GRID}
        best_w2 = 0.5
        best_alpha = 0.5
        best_mse = float("inf")
        for w2 in self.W2_GRID:
            for alpha in self.ALPHA_GRID:
                _, theta_2_fit = ses_fits[alpha]
                fit_ds = (1.0 - w2) * theta_0 + w2 * theta_2_fit
                mse = float(np.mean((y_ds - fit_ds) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_w2 = w2
                    best_alpha = alpha
        _, theta_2_fit_best = ses_fits[best_alpha]
        fit_ds = (1.0 - best_w2) * theta_0 + best_w2 * theta_2_fit_best
        if s > 1 and seasonal is not None:
            return fit_ds + seasonal
        return fit_ds

    def prediction_interval(self, steps, level=0.95):
        z = math.sqrt(2.0) * _erfinv(2.0 * (0.5 + level / 2.0) - 1.0)
        f = self.predict(steps)
        sig = float(np.sqrt(max(self.sigma2_, 0.0)))
        h = np.arange(1, steps + 1, dtype=np.float64)
        band = z * sig * np.sqrt(h)
        return f - band, f + band

    def state_nbytes(self):
        """v0.9.183 OOC: O(seasonal_period) + optional _saved_y_."""
        if not self._is_fitted:
            return 0
        sz = 256
        if self.season_cycle_ is not None:
            sz += self.season_cycle_.nbytes
        sz += 8 * 8  # alpha_, sigma2_, trend_, slope_, lvl_, w2_, mse_, n_, r_
        if self._saved_y_ is not None:
            sz += self._saved_y_.nbytes
        return int(sz)

