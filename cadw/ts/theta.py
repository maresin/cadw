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
from cadw.core.approximation import ApproximationInfo
from cadw.ts.models import (
    ForecasterBase, _check_df, _extract_series, _iter_target_batches,
    _collect_last_w,
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
                 n_iter: int = 2, params: str = "manual",
                 val_steps=None):
        """v0.9.194: merged Theta+AutoTheta.

        params:
          - "manual" (default): single Theta fit, OOC via max_window
            (v0.9.183 OOC state, v0.9.191 window truncation).
          - "auto": CV-selected STM/OTM x additive/multiplicative
            (was AutoThetaForecaster; now Theta(params="auto")).
        """
        super().__init__()
        if seasonal_period < 1:
            raise ValueError(
                "ThetaForecaster: seasonal_period must be >= 1")
        if params not in ("manual", "auto"):
            raise ValueError(
                f"ThetaForecaster: params must be 'manual' or 'auto'; "
                f"got {params!r}")
        self.seasonal_period = int(seasonal_period)
        self.n_iter = int(n_iter)
        self.params = params
        self.val_steps = val_steps
        # v0.9.183 OOC state (manual mode)
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
        # v0.9.194 auto mode state
        self.best_model_ = None
        self.selected_config_ = None
        self.cv_scores_ = []
        self.best_mase_ = float("inf")

    def fit(self, df, target_col, *, time_col=None, group_col=None,
            save_y=False, max_window="auto", strict=False):
        """v0.9.194 dispatcher: manual vs auto."""
        _check_df(df, target_col, time_col, group_col)
        self.target_col_ = target_col
        self.time_col_ = time_col
        self._clear_approximation()
        if self.params == "auto":
            if save_y:
                raise ValueError(
                    "ThetaForecaster(params='auto').fit: save_y=True is "
                    "not supported in auto mode (CV delegates to best_model_); "
                    "use params='manual' for legacy no-df predict_in_sample.")
            return self._fit_auto(
                df, target_col, time_col=time_col, group_col=group_col,
                max_window=max_window, strict=strict)
        return self._fit_manual(
            df, target_col, time_col=time_col, group_col=group_col,
            save_y=save_y, max_window=max_window, strict=strict)

    def _fit_auto(self, df, target_col, *, time_col=None, group_col=None,
                  max_window="auto", strict=False):
        """CV-selected STM vs OTM, additive vs multiplicative (v0.9.194).

        Migrated from AutoThetaForecaster.fit (v0.9.78-193). Winner refit
        on full df with max_window/strict propagated. ApproximationInfo
        from best_model_ propagated to self.
        """
        import warnings as _warnings
        from cadw.ts._theta_common import (
            _MultiplicativeWrapper, _mase, _naive_seasonal_mae,
        )
        from cadw.ts.otm import OTMThetaForecaster
        import pyarrow as pa
        from cadw import CadwDataFrame

        y = _extract_series(df, target_col, time_col=time_col)
        n = y.size
        if n < 4:
            raise ValueError(
                "ThetaForecaster (AutoTheta alias): need at least 4 "
                "observations")
        if np.any(~np.isfinite(y)):
            raise ValueError(
                "ThetaForecaster (AutoTheta alias): series contains "
                "non-finite values")
        s = self.seasonal_period
        if s > 1 and n < 2 * s:
            raise ValueError(
                f"ThetaForecaster (AutoTheta alias): need at least "
                f"2*seasonal_period={2*s} obs for seasonal mode")

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

        n_iter_self = self.n_iter
        cands = [
            {"label": "STM_additive",
             "factory": lambda s_: ThetaForecaster(
                 seasonal_period=s_, n_iter=n_iter_self),
             "note": "STM, additive decomposition"},
            {"label": "OTM_additive",
             "factory": lambda s_: OTMThetaForecaster(
                 seasonal_period=s_, n_iter=n_iter_self),
             "note": "OTM, additive decomposition"},
        ]
        if has_positive:
            cands.append({
                "label": "STM_multiplicative",
                "factory": lambda s_: _MultiplicativeWrapper(
                    lambda ss: ThetaForecaster(
                        seasonal_period=ss, n_iter=n_iter_self), s_),
                "note": "STM on log(y) (= multiplicative decomposition)",
            })
            cands.append({
                "label": "OTM_multiplicative",
                "factory": lambda s_: _MultiplicativeWrapper(
                    lambda ss: OTMThetaForecaster(
                        seasonal_period=ss, n_iter=n_iter_self), s_),
                "note": "OTM on log(y) (= multiplicative decomposition)",
            })

        train_df = CadwDataFrame.from_table(
            pa.table({target_col: pa.array(y_train)}))
        val_h = y_val.size

        best_mase = float("inf")
        best_label = None
        cv_scores = []
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
            _warnings.warn(
                "ThetaForecaster(params='auto'): no candidate valid; "
                "falling back to NaiveForecaster(last).", stacklevel=2)
            from cadw.ts.models import NaiveForecaster as _Naive
            best_model = _Naive(strategy="last")
            best_model.fit(df, target_col, time_col=time_col,
                           group_col=group_col)
            selected = {"label": "fallback:NaiveForecaster(last)"}
            best_mase = float("inf")
        else:
            winner = next(c for c in cands if c["label"] == best_label)
            best_model = winner["factory"](s)
            try:
                best_model.fit(
                    df, target_col, time_col=time_col,
                    group_col=group_col,
                    max_window=max_window, strict=strict)
            except TypeError:
                best_model.fit(
                    df, target_col, time_col=time_col,
                    group_col=group_col)
            selected = {
                "label": best_label, "note": winner["note"],
                "val_mase": float(best_mase),
            }

        self.best_model_ = best_model
        self.selected_config_ = selected
        self.cv_scores_ = cv_scores
        self.best_mase_ = float(best_mase)
        inner_approx = getattr(best_model, "approximation", None)
        if inner_approx:
            for info in inner_approx:
                self._record_approximation(info)
        self._is_fitted = True
        return self

    def predict(self, steps):
        """v0.9.194 dispatcher."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__}.predict called before fit")
        if not isinstance(steps, int):
            raise TypeError(
                f"{type(self).__name__}.predict: steps must be int")
        if steps < 1:
            raise ValueError(
                f"{type(self).__name__}.predict: steps must be >= 1")
        if self.params == "auto":
            return self.best_model_.predict(steps)
        return self._predict_manual(steps)

    def predict_in_sample(self, df=None):
        """v0.9.194 dispatcher."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__}.predict_in_sample called before fit")
        if self.params == "auto":
            return self._predict_in_sample_auto(df)
        return self._predict_in_sample_manual(df)

    def _predict_in_sample_auto(self, df):
        import inspect
        if df is not None:
            try:
                sig = inspect.signature(
                    self.best_model_.predict_in_sample)
                if "df" in sig.parameters:
                    return self.best_model_.predict_in_sample(df)
            except (ValueError, TypeError):
                pass
        return self.best_model_.predict_in_sample()

    def prediction_interval(self, steps, level=0.95):
        """v0.9.194 dispatcher."""
        if self.params == "auto":
            if hasattr(self.best_model_, "prediction_interval"):
                return self.best_model_.prediction_interval(steps, level)
            f = self.predict(steps)
            return f, f
        return self._prediction_interval_manual(steps, level)

    def state_nbytes(self):
        """v0.9.194 dispatcher."""
        if not self._is_fitted:
            return 0
        if self.params == "auto":
            return self._state_nbytes_auto()
        return self._state_nbytes_manual()

    def _state_nbytes_auto(self):
        if self.best_model_ is None:
            return 0
        sz = 256 + int(self.best_model_.state_nbytes())
        sz += 128 * max(1, len(self.cv_scores_ or []))
        return sz

    def _fit_manual(self, df, target_col, *, time_col=None, group_col=None,
            save_y=False,
            max_window: "str | int | None" = "auto",
            strict: bool = False):
        """Fit Theta (STM) forecast.

        v0.9.183 OOC state (lvl_, season_cycle_, params).
        v0.9.188 s=1 streaming path (no full y).
        v0.9.191 (DEC-072 P4) s>1 window path via _collect_last_w:
          - W_eff = min(max_window, W_max, n); W_max from guard.
          - Bit-identical when n <= W_eff (no truncation).
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

        # v0.9.191: save_y + explicit max_window=int -> semantic conflict
        if save_y and isinstance(max_window, int):
            raise ValueError(
                "ThetaForecaster.fit: save_y=True incompatible with "
                f"max_window=int({max_window}); save_y needs full y for "
                "legacy no-df predict_in_sample. Use max_window=None.")

        # v0.9.188: s=1 streaming path (O(1) state; no window needed).
        if s == 1 and not save_y:
            self._fit_streaming_s1(df, target_col, time_col=time_col)
            self._is_fitted = True
            return self

        # s>1 (or save_y): window vs full.
        if save_y or max_window is None:
            y = _extract_series(df, target_col, time_col=time_col)
            n_total = int(y.size)
        else:
            mg = getattr(df, "_memory_guard", None)
            if mg is None:
                # No guard: full fallback
                y = _extract_series(df, target_col, time_col=time_col)
                n_total = int(y.size)
            else:
                # STL uses 4 arrays (y, trend, seasonal, remainder); unit_bytes
                # = 4 * f64. state_bytes ~ seasonal_period * f64 (constant).
                unit_bytes = 4 * 8
                W_max = mg.get_window_size(
                    unit_bytes=unit_bytes, target_fraction=0.25)
                if max_window == "auto":
                    W_eff = W_max
                elif isinstance(max_window, int):
                    if strict and max_window > int(df.count()):
                        from cadw.core.memory_guard import MemoryLimitExceeded
                        raise MemoryLimitExceeded(
                            f"ThetaForecaster.fit: max_window={max_window} > "
                            f"n_rows; strict=True refuses.",
                            model="ThetaForecaster",
                            params={"max_window": max_window, "strict": True},
                        )
                    W_eff = min(int(max_window), W_max)
                else:
                    raise ValueError(
                        f"ThetaForecaster: max_window must be 'auto'|int|None, "
                        f"got {type(max_window).__name__}")
                if W_eff < 2 * s:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"ThetaForecaster.fit: W_eff={W_eff} < "
                        f"2*seasonal_period={2*s}. Increase budget or "
                        f"reduce seasonal_period.",
                        model="ThetaForecaster",
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
                            f"STL + Theta fit on last {W_eff} obs"),
                    ))
                y = y_used

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

    def _predict_manual(self, steps):
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

    def _predict_in_sample_manual(self, df=None):
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

    def _prediction_interval_manual(self, steps, level=0.95):
        z = math.sqrt(2.0) * _erfinv(2.0 * (0.5 + level / 2.0) - 1.0)
        f = self.predict(steps)
        sig = float(np.sqrt(max(self.sigma2_, 0.0)))
        h = np.arange(1, steps + 1, dtype=np.float64)
        band = z * sig * np.sqrt(h)
        return f - band, f + band

    def _state_nbytes_manual(self):
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

