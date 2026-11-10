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
    ForecasterBase, _check_df, _extract_series, _iter_target_batches,
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


def _ses_fit_streaming(y_iter, alpha):
    """v0.9.188: streaming simple SES — O(1) RAM (no `fitted`).

    Bit-identical (atol 1e-12) to ``_ses_fit(y, alpha)[0::2]`` (lvl, sse).
    Returns (lvl, sse, n) — NO fitted array (that was the O(n) leaf part).

    Init: first y as lvl (per _ses_fit v0.9.183 convention).

    Parameters
    ----------
    y_iter : iterable of 1-D np.ndarray (batches, ordered)
    alpha : float
    """
    lvl = None
    sse = 0.0
    n_total = 0
    for b in y_iter:
        if b.size == 0:
            continue
        for v in b:
            y_t = float(v)
            if lvl is None:
                lvl = y_t
            else:
                err = y_t - lvl
                sse += err * err
                lvl = lvl + alpha * err
            n_total += 1
    return (float(lvl) if lvl is not None else 0.0, sse, n_total)


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
        # v0.9.183 OOC state
        self.last_k_ = None
        self.lvl_ = None
        self.n_train_ = None
        self.r_ = None
        self.season_cycle_ = None
        self.alpha_ = None
        self.sigma2_ = None
        self.trend_ = None
        self.slope_ = None
        self.target_col_ = None
        self.time_col_ = None
        self._saved_y_ = None

    def fit(self, df, target_col, *, time_col=None, group_col=None,
            save_y=False):
        """Fit Theta (STM) forecast.

        v0.9.183 OOC: save_y=False (default) keeps only O(seasonal_period)
        state (last_k_, lvl_, season_cycle_, params). save_y=True preserves
        legacy y_ (O(n) extra) for predict_in_sample() no-df.
        """
        _check_df(df, target_col, time_col, group_col)
        self.target_col_ = target_col
        self.time_col_ = time_col
        s = self.seasonal_period

        # v0.9.188: s=1 streaming path (DEC-070 P2). No full y_ materialization.
        if s == 1 and not save_y:
            self._fit_streaming_s1(df, target_col, time_col=time_col)
            self._is_fitted = True
            return self

        # s>1 (stl_decompose leaf) or save_y=True: non-streaming path.
        y = _extract_series(df, target_col, time_col=time_col)
        if y.size < 4:
            raise ValueError(
                "ThetaForecaster: need at least 4 observations")
        if np.any(~np.isfinite(y)):
            raise ValueError(
                "ThetaForecaster: series contains non-finite values")
        if s > 1 and y.size < 2 * s:
            raise ValueError(
                f"ThetaForecaster: need at least 2*seasonal_period="
                f"{2*s} obs for seasonal mode")
        n = int(y.size)

        # 1. Optional STL decomposition
        if s > 1:
            decomp = stl_decompose(y, period=s, n_iter=self.n_iter)
            seasonal = decomp.seasonal
            y_ds = y - seasonal  # = T + R
            season_cycle = seasonal[:s].copy()
        else:
            seasonal = None
            y_ds = y.copy()
            season_cycle = None

        # 2. OLS on deseasonalized series
        a, b = _ols_fit(y_ds)
        theta_0_fit = a + b * np.arange(n, dtype=np.float64)

        # 3. theta_2 = 2 * y_ds - theta_0_fit
        theta_2 = 2.0 * y_ds - theta_0_fit

        # 4. SES on theta_2
        alpha = _best_alpha(theta_2)
        lvl, theta_2_fit, _ = _ses_fit(theta_2, alpha)

        # 5. In-sample combine
        fit_ds = 0.5 * theta_0_fit + 0.5 * theta_2_fit

        # 6. Reseasonalize in-sample
        if s > 1 and seasonal is not None:
            fitted = fit_ds + seasonal
        else:
            fitted = fit_ds
        resid = y - fitted
        self.sigma2_ = float(np.mean(resid * resid))

        # v0.9.183: store O(model) state
        k = 1 if s <= 1 else s
        self.last_k_ = y[-k:].copy()
        self.lvl_ = float(lvl)
        self.n_train_ = n
        self.r_ = n % s if s > 1 else 0
        self.season_cycle_ = season_cycle
        self.alpha_ = alpha
        self.trend_ = a
        self.slope_ = b
        self._saved_y_ = y if save_y else None
        self._is_fitted = True
        return self

    def predict(self, steps):
        """v0.9.183 OOC: uses stored lvl_ (O(1)), not full y_ (was O(n))."""
        n = self.n_train_
        a, b = self.trend_, self.slope_
        theta_0_f = _ols_forecast(a, b, n, steps)

        theta_2_f = np.full(steps, self.lvl_, dtype=np.float64)
        combined = 0.5 * theta_0_f + 0.5 * theta_2_f

        s = self.seasonal_period
        if s > 1 and self.season_cycle_ is not None:
            idx = (self.r_ + np.arange(steps)) % s
            return combined + self.season_cycle_[idx]
        return combined

    def _fit_streaming_s1(self, df, target_col, *, time_col=None):
        """v0.9.188: s=1 streaming fit (DEC-070 P2).

        Pass 1: _ols_fit_streaming -> (a, b, n)
        Pass 2: grid alpha on theta_2 = 2*y_i - (a + b*i) via 19
                _ses_fit_streaming running in parallel (O(19) state)
        Final: _ses_fit_streaming(theta_2, best_alpha) -> (lvl, sse)

        State after fit: a, b, alpha, lvl, sigma2, n_train, last_k.
        """
        from cadw.ops.linalg import ols_fit_streaming

        # Pass 1: OLS on raw y (s=1: y_ds = y)
        def _batches():
            return _iter_target_batches(
                df, target_col, time_col=time_col, batch_size=None)

        a, b, n = ols_fit_streaming(_batches())
        if n < 4:
            raise ValueError(
                "ThetaForecaster: need at least 4 observations")
        self.trend_ = float(a)
        self.slope_ = float(b)
        self.n_train_ = int(n)

        # Pass 2: grid search alpha over [0.05, 0.95] step 0.05
        alphas = [i / 20.0 for i in range(1, 20)]
        # State per alpha: (lvl, sse, n_seen)
        states = {al: [None, 0.0, 0] for al in alphas}

        g_idx = 0
        for arr in _batches():
            if arr.size == 0:
                continue
            # theta_2 = 2*y - a - b*i, per-element with global index
            idx = np.arange(g_idx, g_idx + arr.size, dtype=np.float64)
            theta_2_b = 2.0 * arr - (a + b * idx)
            for al in alphas:
                st = states[al]
                lvl = st[0]
                sse = st[1]
                for v in theta_2_b:
                    y_t = float(v)
                    if lvl is None:
                        lvl = y_t
                    else:
                        err = y_t - lvl
                        sse += err * err
                        lvl = lvl + al * err
                st[0] = lvl
                st[1] = sse
                st[2] += arr.size
            g_idx += arr.size

        # Best alpha by sse
        best_alpha = 0.5
        best_sse = float("inf")
        for al in alphas:
            if states[al][1] < best_sse:
                best_sse = states[al][1]
                best_alpha = al

        # Final lvl from best alpha
        best_lvl = states[best_alpha][0]
        self.alpha_ = float(best_alpha)
        self.lvl_ = float(best_lvl) if best_lvl is not None else 0.0

        # sigma2: recompute residuals in-sample (2-pass) — O(n) compute, O(1) RAM
        # resid_t = y_t - (0.5*theta_0_t + 0.5*theta_2_fit_t)
        # theta_0_t = a + b*t; theta_2_fit_t = previous lvl (or first y)
        resid_sse = 0.0
        theta_2_lvl = None
        g_idx = 0
        for arr in _batches():
            if arr.size == 0:
                continue
            idx = np.arange(g_idx, g_idx + arr.size, dtype=np.float64)
            theta_0_b = a + b * idx
            theta_2_b = 2.0 * arr - theta_0_b
            for i_v, v in enumerate(theta_2_b):
                y_t = float(v)
                if theta_2_lvl is None:
                    fit_t2 = y_t  # convention: first pred = first y
                    theta_2_lvl = y_t
                else:
                    fit_t2 = theta_2_lvl
                    err = y_t - theta_2_lvl
                    theta_2_lvl = theta_2_lvl + best_alpha * err
                # combined in-sample pred
                pred = 0.5 * float(theta_0_b[i_v]) + 0.5 * fit_t2
                # actual y at this t is arr[i_v]
                resid_sse += (float(arr[i_v]) - pred) ** 2
            g_idx += arr.size
        self.sigma2_ = resid_sse / max(1, n)

        # last_k_ (for predict; s=1 -> 1 value)
        last_v = None
        for arr in _batches():
            if arr.size > 0:
                last_v = float(arr[-1])
        self.last_k_ = np.array([last_v], dtype=np.float64)
        self.r_ = 0
        self.season_cycle_ = None
        self._saved_y_ = None
        # _is_fitted set by caller

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
        """Recompute fitted values from arbitrary y (bit-identical for
        training y). O(n) transient, no persistent alloc."""
        s = self.seasonal_period
        if s > 1:
            decomp = stl_decompose(y, period=s, n_iter=self.n_iter)
            seasonal = decomp.seasonal
            y_ds = y - seasonal
        else:
            seasonal = None
            y_ds = y.copy()
        a, b = _ols_fit(y_ds)
        theta_0 = a + b * np.arange(y.size, dtype=np.float64)
        theta_2 = 2.0 * y_ds - theta_0
        alpha = _best_alpha(theta_2)
        _, theta_2_fit, _ = _ses_fit(theta_2, alpha)
        fit_ds = 0.5 * theta_0 + 0.5 * theta_2_fit
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
        sz += 8 * 7  # alpha_, sigma2_, trend_, slope_, lvl_, n_train_, r_
        if self._saved_y_ is not None:
            sz += self._saved_y_.nbytes
        return int(sz)

