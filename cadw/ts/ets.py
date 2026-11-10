"""ETS / Holt-Winters exponential smoothing (D.4.3, v0.9.38).

Reference: Hyndman, Athanasopoulos, *Forecasting: Principles and
Practice*, 3rd ed., OTexts 2021, ch. 8 (exponential smoothing / ETS);
Hyndman, Koehler, Snyder, Grose, *A state space framework for automatic
forecasting using exponential smoothing*, Int. J. Forecasting 18, 2002.
See .pingis/algorithm_notes/d4_3_ets.md.

Additive-error ETS with additive trend and additive seasonality (AAA),
estimated by minimising the one-step-ahead SSE with a bounded
Nelder-Mead-style coordinate search over (alpha, beta, gamma). The
seasonal indices are re-estimated by classical decomposition each pass.
A damped trend (phi) is supported. Predictions include a 95% interval
from the in-sample residual sigma. RAM is O(n + s), no extra state.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from cadw.ml._logging import log_fit
from cadw.core.approximation import ApproximationInfo
from cadw.ts.models import (
    ForecasterBase, _check_df, _extract_series, _collect_last_w,
)


def _initial_seasonal(y, s, seasonal="additive"):
    """Classical decomposition seasonal indices (mean 0 / mean 1)."""
    n = y.size
    k = n // s
    if k < 1:
        raise ValueError("ets: series shorter than one seasonal period")
    means = np.array([y[i * s:(i + 1) * s].mean() for i in range(k)])
    if seasonal == "additive":
        season = y[:k * s].reshape(k, s).mean(axis=0) - means.mean()
        return season - season.mean()
    season = (y[:k * s].reshape(k, s) / means[:, None]).mean(axis=0)
    return season / season.mean()


def _ses_fit(y, alpha, beta, gamma, phi, s, seasonal, damped):
    """Standard Holt-Winters recursion; returns (level, trend, season, sse, fitted)."""
    n = y.size
    season = _initial_seasonal(y, s, seasonal) if s > 1 else np.zeros(1)
    lvl = float(y[0])
    if n >= 2:
        if s > 1 and n > s:
            tr = float((y[s] - y[0]) / s)
        else:
            tr = float(y[1] - y[0])
    else:
        tr = 0.0
    fitted = np.empty(n, dtype=np.float64)
    sse = 0.0
    for t in range(n):
        idx = t % s if s > 1 else 0
        b = phi * tr if damped else tr
        if seasonal == "additive":
            sea = season[idx] if s > 1 else 0.0
            pred = lvl + b + sea
            e = y[t] - pred
            new_l = lvl + b + alpha * e
            new_t = b + alpha * beta * e
            new_sea = season.copy()
            if s > 1:
                new_sea[idx] = season[idx] + gamma * e
        else:
            sea = season[idx] if s > 1 else 1.0
            pred = (lvl + b) * sea
            e = y[t] - pred
            base = lvl + b
            new_l = base + alpha * e / max(abs(sea), 1e-12)
            new_t = b + alpha * beta * e / max(abs(sea), 1e-12)
            new_sea = season.copy()
            if s > 1:
                new_sea[idx] = season[idx] + gamma * e / max(abs(base), 1e-12)
        fitted[t] = pred
        sse += e * e
        lvl, tr = float(new_l), float(new_t)
        if s > 1:
            season = new_sea
    return lvl, tr, season, sse, fitted


def _grid_search(y, s, seasonal, damped, phi_fixed=False):
    """Coordinate search over alpha/beta/gamma/phi on a coarse grid.

    Deterministic, no RNG; each axis is refined twice. Cheap: O(n) per
    evaluation and a bounded number of evaluations.
    """
    alphas = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    betas = np.array([0.01, 0.05, 0.1, 0.2, 0.3])
    gammas = np.array([0.01, 0.05, 0.1, 0.2, 0.3])
    phis = np.array([0.8, 0.9, 0.95, 0.98]) if damped else np.array([1.0])
    best = (np.inf, None)
    for a in alphas:
        for b in betas:
            for g in gammas:
                for p in phis:
                    _, _, _, sse, _ = _ses_fit(y, a, b, g, p, s, seasonal, damped)
                    if sse < best[0]:
                        best = (sse, (a, b, g, p))
    return best[1]


class ETSForecaster(ForecasterBase):
    """Additive-error ETS(A,A,A) / ETS(A,A,N) with optional damping."""

    handles_trend: bool = True
    handles_seasonality: bool = True
    handles_multi_seasonality: bool = False

    def __init__(self, seasonal_period=1, seasonal="additive",
                 trend=True, damped=False):
        super().__init__()
        if seasonal_period < 1:
            raise ValueError("ETSForecaster: seasonal_period must be >= 1")
        if seasonal not in ("additive", "multiplicative"):
            raise ValueError("ETSForecaster: seasonal must be 'additive' or 'multiplicative'")
        self.seasonal_period = int(seasonal_period)
        self.seasonal = seasonal
        self.trend = bool(trend)
        self.damped = bool(damped)
        # v0.9.184 OOC state
        self.lvl_ = None
        self.tr_ = None
        self.season_ = None
        self.alpha_ = None
        self.beta_ = None
        self.gamma_ = None
        self.phi_ = None
        self.sigma2_ = None
        self.n_train_ = None
        self.target_col_ = None
        self.time_col_ = None
        self._saved_y_ = None

    def fit(self, df, target_col, *, time_col=None, group_col=None,
            save_y=False, max_window: "str | int | None" = "auto",
            strict: bool = False):
        """Fit ETS.

        v0.9.189 (DEC-072): s>1 path uses _collect_last_w (window
        truncation). W_eff from guard.get_window_size (target_fraction=0.25,
        unit_bytes=2*s+transient). Bit-identical when n <= W_eff; truncation
        disclosed via ApproximationInfo otherwise.

        max_window:
          - "auto" (default): W_eff from guard.
          - int W: W_eff = min(W, n); strict=True -> refuse if W < n.
          - None: full fit (bounded_by_domain; no truncation).
        """
        _check_df(df, target_col, time_col, group_col)
        s = self.seasonal_period
        self.target_col_ = target_col
        self.time_col_ = time_col
        self._clear_approximation()

        if s == 1 or max_window is None:
            # Full-fit path (s=1 fast path; or explicit full-fit s>1)
            y = _extract_series(df, target_col, time_col=time_col)
            if y.size < 4:
                raise ValueError("ETSForecaster: need at least 4 observations")
            if np.any(~np.isfinite(y)):
                raise ValueError("ETSForecaster: series contains non-finite values")
            if s > 1 and y.size < 2 * s:
                raise ValueError(
                    f"ETSForecaster: need at least 2*seasonal_period={2*s} obs")
            n_total = int(y.size)
        else:
            # s>1 window path: W_eff from guard
            mg = getattr(df, "_memory_guard", None)
            if mg is None:
                # No guard -> full-fit fallback
                y = _extract_series(df, target_col, time_col=time_col)
                n_total = int(y.size)
            else:
                # Reserve budget for state (season s) + HW recursion transient
                state_bytes = 2 * s * 8 + 512
                transient_bytes = 2 * s * 8 * 4  # conservative 4x
                W_max = mg.get_window_size(
                    unit_bytes=state_bytes + transient_bytes,
                    target_fraction=0.25,
                )
                if max_window == "auto":
                    W_eff = W_max
                elif isinstance(max_window, int):
                    if strict and max_window > int(df.count()):
                        from cadw.core.memory_guard import MemoryLimitExceeded
                        raise MemoryLimitExceeded(
                            f"ETSForecaster.fit: max_window={max_window} > "
                            f"n_rows; strict=True refuses.",
                            model="ETSForecaster",
                            params={"max_window": max_window, "strict": True},
                        )
                    W_eff = min(int(max_window), W_max)
                else:
                    raise ValueError(
                        f"ETSForecaster: max_window must be 'auto'|int|None, "
                        f"got {type(max_window).__name__}"
                    )
                if W_eff < 2 * s:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"ETSForecaster.fit: W_eff={W_eff} < 2*seasonal_period="
                        f"{2*s}. Increase budget or reduce seasonal_period.",
                        model="ETSForecaster",
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
                        reason="budget_limit" if max_window == "auto"
                               else "explicit",
                        lost_range=lost,
                        accuracy_impact=(
                            f"HW fit on last {W_eff} obs; initial seasonal "
                            f"from last {W_eff // s} periods"
                        ),
                    ))
                y = y_used
                if y.size < 2 * s:
                    raise ValueError(
                        f"ETSForecaster: need at least 2*seasonal_period={2*s} obs")

        if np.any(~np.isfinite(y)):
            raise ValueError("ETSForecaster: series contains non-finite values")
        self.n_train_ = n_total
        if not self.trend and s == 1:
            alphas = np.linspace(0.05, 0.95, 19)
            best = (np.inf, None)
            for a in alphas:
                _, _, _, sse, _ = _ses_fit(y, a, 0.0, 0.0, 1.0, 1, self.seasonal, False)
                if sse < best[0]:
                    best = (sse, a)
            self.alpha_, self.beta_, self.gamma_, self.phi_ = best[1], 0.0, 0.0, 1.0
        else:
            a, b, g, p = _grid_search(y, s, self.seasonal, self.damped)
            self.alpha_, self.beta_, self.gamma_, self.phi_ = a, (b if self.trend else 0.0), g, p
        lvl, tr, season, sse, fitted = _ses_fit(
            y, self.alpha_, self.beta_, self.gamma_, self.phi_, s,
            self.seasonal, self.damped)
        self.lvl_, self.tr_, self.season_ = lvl, tr, season
        self.sigma2_ = float(sse / max(1, y.size))
        self._saved_y_ = y if save_y else None
        self._is_fitted = True
        log_fit("ets", loss=sse, state_bytes=self.state_nbytes())
        return self

    def state_nbytes(self):
        """v0.9.184 OOC: O(seasonal_period) + optional _saved_y_."""
        if not self._is_fitted:
            return 0
        sz = 256 + 8 * 7  # lvl_, tr_, alpha_, beta_, gamma_, phi_, sigma2_
        if self.season_ is not None:
            sz += self.season_.nbytes
        if self._saved_y_ is not None:
            sz += self._saved_y_.nbytes
        return int(sz)

    def _predict(self, steps):
        s = self.seasonal_period
        out = np.empty(steps, dtype=np.float64)
        lvl, tr = self.lvl_, self.tr_
        n = self.n_train_  # v0.9.184: O(1) instead of self.y_.size
        phi = self.phi_
        for h in range(steps):
            idx = (n + h) % s if s > 1 else 0
            if self.seasonal == "additive":
                f = lvl + (sum(tr * phi ** k for k in range(1, h + 1)) if self.damped else tr * (h + 1)) + (self.season_[idx] if s > 1 else 0.0)
            else:
                f = (lvl + (sum(tr * phi ** k for k in range(1, h + 1)) if self.damped else tr * (h + 1))) * (self.season_[idx] if s > 1 else 1.0)
            out[h] = f
        return out

    def predict_in_sample(self, df=None):
        """v0.9.184: OOC default requires df (or save_y=True legacy)."""
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
            f"(OOC default since v0.9.184). Use fit(..., save_y=True) for "
            f"legacy no-df calls (adds O(n) persistent state)."
        )

    def _recompute_in_sample(self, y):
        """Recompute fitted via _ses_fit forward pass (bit-identical)."""
        _, _, _, _, fitted = _ses_fit(
            y, self.alpha_, self.beta_, self.gamma_, self.phi_,
            self.seasonal_period, self.seasonal, self.damped)
        return fitted

    def prediction_interval(self, steps, level=0.95):
        """Gaussian 95% interval around the point forecasts."""
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + level / 2.0))
        f = self.predict(steps)
        sig = float(np.sqrt(self.sigma2_))
        h = np.arange(1, steps + 1, dtype=np.float64)
        band = z * sig * np.sqrt(h)
        return f - band, f + band
