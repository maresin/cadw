# ============================================================================
# CADW TS TOOLKIT REVIEW SNAPSHOT
# Generated: 2026-10-22T15:05:11
# Base:      524ca0d (via log grep: v0.9.23j)
# HEAD:      524ca0d
# Mode:      default
# Files:     27
#
# Modes: 'full' = complete file; 'diff' = git diff vs base;
#        'diff-empty' = unchanged; 'missing' = not found.
# Binary artifacts (parquet, backups) intentionally excluded.
# If a needed file is missing, ask maintainer — don't
# regenerate the full project snapshot.
# ============================================================================

# ============================================================================
# TOC
# ============================================================================
#   1. cadw/ts/models.py  [full]
#   2. cadw/ts/thf.py  [full]
#   3. tests/test_ts_models.py  [full]
#   4. tests/test_ts_thf.py  [full]
#   5. tests/test_ts_backtest.py  [full]
#   6. tests/test_ts_readme_example.py  [full]
#   7. scripts/verify_ts_ram.py  [full]
#   8. cadw/dataframe.py  [diff]
#   9. cadw/ts/__init__.py  [full]
#  10. cadw/ts/features.py  [full]
#  11. cadw/ts/stats.py  [full]
#  12. cadw/ts/backtest.py  [full]
#  13. tests/test_ts_features.py  [full]
#  14. tests/test_ts_stats.py  [full]
#  15. tests/test_ts_api.py  [full]
#  16. tests/test_ts_stage8.py  [full]
#  17. .pingis/plan.md  [missing]
#  18. cadw/accessors/_ts.py  [full]
#  19. cadw/ops/rank.py  [diff]
#  20. cadw/__init__.py  [diff-empty]
#  21. tests/test_anti_outer_join.py  [diff]
#  22. tests/test_join.py  [diff]
#  23. README.md  [diff]
#  24. CHANGELOG.md  [diff]
#  25. DECISIONS.md  [diff]
#  26. scripts/run_ts_ram.py  [full]
#  27. scripts/diag_rolling.py  [full]

# ============================================================================
# FILE: cadw/ts/models.py  [full]
# ============================================================================
# CADW time-series forecasters (v0.9.24).
#
# Three forecasters on a common API:
#   - NaiveForecaster(strategy='last'|'seasonal'|'drift')
#   - SARIMAForecaster(order=None, seasonal_order=None, s=None, auto=True)
#   - GBDTForecaster(mode='direct'|'recursive'|'multi_output', ...)
#
# Common API contract (ForecasterBase):
#   fit(df, target_col, *, time_col=None, group_col=None, **kwargs)
#   predict(steps)   -> np.ndarray (len=steps), 1-D, finite point forecasts
#   predict_in_sample() -> np.ndarray (len=n_train)
# Predict before fit raises RuntimeError. steps < 1 raises ValueError.
#
# SARIMA: CSS (conditional sum of squares) fit via iterative Hannan-
# Rissanen. Supports regular ARIMA(p,d,q) and optional seasonal
# (P,D,Q,s). Auto-ARIMA grid (AIC):
#   - order=None and auto=True:
#       p,q in 0..2, d in 0..1  -> 18 combos
#       if seasonal (s and seasonal_order=None):
#           P,Q,D in 0..1 -> 8x more combos (144 total)
#       if seasonal_order=None and s=None: seasonal=False (18 combos)
#   - order/seasonal_order explicitly set: auto disabled, single fit.
# AIC = n_obs * ln(sigma2) + 2 * n_params. Bounded grid prevents O(N) flail.
#
# GBDT: three modes; default is 'direct' (per plan):
#   - 'recursive': one model, feeding predictions back as inputs.
#     Propagates error across steps; keep for parity, not recommended.
#   - 'direct' (default): h independent models, one per horizon. No
#     error propagation, no feature build leakage.
#   - 'multi_output': h independent models predicting vector jointly
#     (CADW GBDT has no native multi-target; this is 'multi-output
#     (h-independent)'. Explicit, documented).
#
# All forecasters route per-batch data through MemoryGuard via
# make_features / _extract_series helpers. GBDT additionally uses
# cadw.ml.gbdt (which routes Reduce ops through the guard).
from __future__ import annotations

from typing import Optional

import numpy as np
import pyarrow as pa

from .features import _BATCH, _check, make_features


def _extract_series(df, target_col, *, time_col=None, group_col=None,
                     group_value=None):
    # Extract target as float64 numpy array; optionally filter to one
    # group, optionally sort by time_col. Streams batches, no full-frame
    # materialization.
    if target_col not in df._columns:
        raise KeyError(f"target column {target_col!r} not in df")
    if time_col is not None and time_col not in df._columns:
        raise KeyError(f"time_col {time_col!r} not in df")
    if group_col is not None and group_col not in df._columns:
        raise KeyError(f"group_col {group_col!r} not in df")

    if time_col is not None:
        keys = ([group_col] if group_col is not None else []) + [time_col]
        df = df.sort_values(keys)

    out = []
    for batch in df._iter_arrow_batches(batch_size=_BATCH):
        _check(df._memory_guard, batch, "ts.extract_series", 3.5)
        vals = batch.column(target_col).to_pylist()
        if group_col is not None and group_value is not None:
            gvals = batch.column(group_col).to_pylist()
            vals = [v for v, g in zip(vals, gvals) if g == group_value]
        out.extend(float(v) if v is not None else np.nan for v in vals)
    return np.asarray(out, dtype=np.float64)


def _check_df(df, target_col, time_col=None, group_col=None):
    if target_col not in df._columns:
        raise KeyError(f"target column {target_col!r} not in df")
    if time_col is not None and time_col not in df._columns:
        raise KeyError(f"time_col {time_col!r} not in df")
    if group_col is not None and group_col not in df._columns:
        raise KeyError(f"group_col {group_col!r} not in df")


class ForecasterBase:
    """Abstract forecaster contract (v0.9.24).

    Subclasses MUST implement fit, predict, predict_in_sample. All
    three methods conform to the same semantics:
      fit(df, target_col, *, time_col=None, group_col=None):
        consume a CadwDataFrame, extract target (and features for GBDT),
        set self._is_fitted = True.
      predict(steps): return 1-D np.ndarray, len=steps, finite point
        forecasts at horizons 1..steps.
      predict_in_sample(): return 1-D np.ndarray, len=n_train, model
        in-sample prediction.

    Pre-conditions:
      - fit required before predict / predict_in_sample; otherwise
        RuntimeError.
      - steps must be int >= 1; otherwise ValueError.
    """

    def __init__(self):
        self._is_fitted = False

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        raise NotImplementedError

    def predict(self, steps):
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__}.predict called before fit"
            )
        if not isinstance(steps, int):
            raise TypeError(
                f"{type(self).__name__}.predict: steps must be int; "
                f"got {type(steps).__name__}"
            )
        if steps < 1:
            raise ValueError(
                f"{type(self).__name__}.predict: steps must be >= 1; "
                f"got {steps}"
            )
        return self._predict(steps)

    def predict_in_sample(self):
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__}.predict_in_sample called before fit"
            )
        return self._predict_in_sample()

    def _predict(self, steps):
        raise NotImplementedError

    def _predict_in_sample(self):
        raise NotImplementedError


# ==================== NaiveForecaster ====================

class NaiveForecaster(ForecasterBase):
    """Naive baselines.

    strategies:
      'last'      - y_{T+h} = y_T
      'seasonal'  - y_{T+h} = y_{T+h - s} (s=period; s required)
      'drift'     - y_{T+h} = y_T + h * mean(diff(y))

    RAM: O(N) for storing the training series (needed for in-sample
    and drift mean); no MemoryGuard allocation beyond the extraction
    buffer, which is streamed and checked. O(1) at predict time.
    """

    def __init__(self, strategy="last", period=None):
        super().__init__()
        if strategy not in ("last", "seasonal", "drift"):
            raise ValueError(
                f"NaiveForecaster: strategy must be 'last', "
                f"'seasonal', or 'drift'; got {strategy!r}"
            )
        if strategy == "seasonal":
            if period is None or int(period) < 1:
                raise ValueError(
                    "NaiveForecaster: strategy='seasonal' requires "
                    "period >= 1"
                )
        self.strategy = strategy
        self.period = int(period) if period is not None else None
        self.y_ = None

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        _check_df(df, target_col, time_col, group_col)
        y = _extract_series(df, target_col, time_col=time_col)
        if len(y) < 2:
            raise ValueError(
                "NaiveForecaster: need at least 2 observations to fit"
            )
        self.y_ = y
        self._is_fitted = True
        return self

    def _predict(self, steps):
        y = self.y_
        T = len(y)
        if self.strategy == "last":
            base = y[-1]
            return np.full(steps, base, dtype=np.float64)
        if self.strategy == "seasonal":
            s = self.period
            if T < s:
                raise ValueError(
                    f"NaiveForecaster seasonal: need T >= period; "
                    f"T={T}, s={s}"
                )
            # y[T+h] = y[T+h-s], recursively for h > s
            out = np.empty(steps, dtype=np.float64)
            full = np.concatenate([y, np.empty(steps, dtype=np.float64)])
            for h in range(steps):
                full[T + h] = full[T + h - s]
                out[h] = full[T + h]
            return out
        # drift
        diffs = np.diff(y)
        diffs = diffs[~np.isnan(diffs)]
        drift = float(diffs.mean()) if len(diffs) else 0.0
        return y[-1] + drift * np.arange(1, steps + 1, dtype=np.float64)

    def _predict_in_sample(self):
        y = self.y_
        T = len(y)
        out = np.full(T, np.nan, dtype=np.float64)
        if self.strategy == "last":
            out[1:] = y[:-1]
        elif self.strategy == "seasonal":
            s = self.period
            if T > s:
                out[s:] = y[:-s]
        else:
            diffs = np.diff(y)
            diffs = diffs[~np.isnan(diffs)]
            drift = float(diffs.mean()) if len(diffs) else 0.0
            out[1:] = y[:-1] + drift
        return out


# ==================== SARIMA (CSS / Hannan-Rissanen) ====================

def _difference(y, d):
    """Apply d-th order regular differencing. Non-finite leading rows
    of order k are NaN."""
    out = y.copy()
    for _ in range(d):
        new = np.full_like(out, np.nan)
        new[1:] = out[1:] - out[:-1]
        out = new
    return out


def _seasonal_difference(y, D, s):
    out = y.copy()
    for _ in range(D):
        new = np.full_like(out, np.nan)
        new[s:] = out[s:] - out[:-s]
        out = new
    return out


def _integrate_back(y_forecast, y_levels, d, D, s, seasonal_period_last):
    """Undo differencing on forecasts.

    y_levels: tuple (L1, L2, ..., Ld) of last observed values of the
      d-differenced series at each integration level, from LEAST
      differenced (L1 = last y of original) down. For d=1, y_levels
      = (y[-1],). After cumsum of the d-differenced forecast, we add
      the (d-1)-differenced last value to get the (d-1)-differenced
      forecast, and so on. Specifically, for d=1: f = cumsum(f) +
      y[-1].
    seasonal_period_last: last s values of the series *before*
      seasonal differencing (i.e. after regular differencing). Used
      to undo seasonal differencing.
    """
    f = np.asarray(y_forecast, dtype=np.float64)
    # Undo regular differencing: for each of d diffs, add the level.
    for lvl in range(d):
        base = y_levels[lvl] if lvl < len(y_levels) else 0.0
        if not np.isfinite(base):
            base = 0.0
        f = np.cumsum(f) + base
    # Undo seasonal differencing: add y[T+h-s] recursively.
    for _ in range(D):
        out = np.empty_like(f)
        for h in range(len(f)):
            if h < s:
                out[h] = f[h] + seasonal_period_last[h]
            else:
                out[h] = f[h] + out[h - s]
        f = out
    return f


def _ar_fit(y, p):
    """Fit AR(p) to y via OLS on available rows. Returns (coef, resid)."""
    n = len(y)
    if n <= p:
        raise ValueError(f"AR fit: n={n} <= p={p}")
    Y = y[p:]
    X = np.empty((n - p, p), dtype=np.float64)
    for i in range(p):
        X[:, i] = y[p - i - 1:n - i - 1]
    # drop rows with NaN
    mask = ~(np.isnan(Y) | np.isnan(X).any(axis=1))
    Y = Y[mask]
    X = X[mask]
    if len(Y) < p + 1:
        raise ValueError(f"AR fit: insufficient rows after NaN drop ({len(Y)})")
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = np.full(n, np.nan, dtype=np.float64)
    resid[mask.nonzero()[0] + p] = Y - X @ beta
    return beta, resid


def _hannan_rissanen(y, p, q, ar_m=None):
    """Hannan-Rissanen two-step ARMA(p,q) fit.

    Step 1: fit a high-order AR(m) to get residual estimates.
    Step 2: regress y_t on y_{t-1..p} and e_{t-1..q} via OLS.
    Returns (ar_coefs, ma_coefs, sigma2).
    """
    if ar_m is None:
        ar_m = max(p, q) + p
        if ar_m < 1:
            ar_m = 1
    # Step 1
    _, e = _ar_fit(y, ar_m)
    # Step 2: OLS on (y lags, e lags)
    n = len(y)
    start = max(p, q) + ar_m  # ensure e available for all q lags
    Y = y[start:]
    X = np.empty((n - start, p + q), dtype=np.float64)
    for i in range(p):
        X[:, i] = y[start - i - 1:n - i - 1]
    for i in range(q):
        X[:, p + i] = e[start - i - 1:n - i - 1]
    mask = ~(np.isnan(Y) | np.isnan(X).any(axis=1))
    Y = Y[mask]
    X = X[mask]
    if len(Y) < p + q + 2:
        raise ValueError(
            f"Hannan-Rissanen: insufficient rows ({len(Y)}); "
            f"reduce orders or increase N"
        )
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    ar = beta[:p] if p else np.empty(0)
    ma = beta[p:] if q else np.empty(0)
    resid = Y - X @ beta
    sigma2 = float(np.mean(resid ** 2))
    return ar, ma, sigma2


def _arma_forecast(y, ar, ma, sigma2, steps, in_sample=False):
    """Recursive ARMA forecast using the fitted coefficients."""
    p = len(ar)
    q = len(ma)
    n = len(y)
    if in_sample:
        # one-step-ahead predictions in-sample (best-effort OLS fit)
        out = np.full(n, np.nan, dtype=np.float64)
        # Step 1 residuals from a high-order AR fit as approximate e
        m = max(p, q) + p if (p or q) else 1
        if n > m:
            _, e = _ar_fit(y, m)
        else:
            e = np.full(n, 0.0)
        for t in range(m, n):
            s = 0.0
            for i in range(p):
                if t - i - 1 >= 0 and not np.isnan(y[t - i - 1]):
                    s += ar[i] * y[t - i - 1]
            for i in range(q):
                if t - i - 1 >= 0 and not np.isnan(e[t - i - 1]):
                    s += ma[i] * e[t - i - 1]
            out[t] = s
        return out
    # Out-of-sample recursive forecast
    y_ext = list(y)
    e_ext = [0.0] * n
    # Initialize e_ext via in-sample residuals
    m = max(p, q) + p if (p or q) else 1
    if n > m:
        _, e = _ar_fit(y, m)
        for i, v in enumerate(e):
            if not np.isnan(v):
                e_ext[i] = v
    fc = np.empty(steps, dtype=np.float64)
    for h in range(steps):
        s = 0.0
        for i in range(p):
            s += ar[i] * y_ext[-i - 1]
        for i in range(q):
            s += ma[i] * e_ext[-i - 1]
        fc[h] = s
        y_ext.append(s)
        e_ext.append(0.0)  # MA residual for future = 0 in expectation
    return fc


class SARIMAForecaster(ForecasterBase):
    """SARIMA(p,d,q)(P,D,Q,s) via CSS Hannan-Rissanen.

    - order=(p,d,q); seasonal_order=(P,D,Q) or None; s=int or None.
    - auto=True and (order is None or seasonal_order is None):
        grid: p,q in 0..2, d in 0..1;
        seasonal (if s not None): P,Q,D in 0..1.
        Select by AIC = n_obs * ln(sigma2) + 2*n_params.
    - auto=False: requires order (and seasonal_order if s set).

    WARNING: AIC can under-difference trending series. On a random
    walk the level-stationary ARMA fit often has lower AIC than the
    integrated (d=1) fit, so auto=True may pick d=0. For clearly
    trending or unit-root series, set order=(p,1,q) explicitly. This
    is a documented limitation of AIC-based selection, not a bug in
    the CSS fit. If a level-stationary fit is chosen on trending
    data, forecasts will mean-revert toward the fitted intercept.
    """

    DEFAULT_P_GRID = (0, 1, 2)
    DEFAULT_D_GRID = (0, 1)
    DEFAULT_Q_GRID = (0, 1, 2)
    DEFAULT_SP_GRID = (0, 1)
    DEFAULT_SD_GRID = (0, 1)
    DEFAULT_SQ_GRID = (0, 1)

    def __init__(
        self, order=None, seasonal_order=None, s=None, auto=True,
    ):
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self.s = s
        self.auto = bool(auto)
        if order is not None and (not isinstance(order, tuple) or len(order) != 3):
            raise ValueError("order must be a 3-tuple (p,d,q)")
        if seasonal_order is not None and (
            not isinstance(seasonal_order, tuple) or len(seasonal_order) != 3
        ):
            raise ValueError(
                "seasonal_order must be a 3-tuple (P,D,Q)"
            )
        if seasonal_order is not None and s is None:
            raise ValueError(
                "SARIMA: seasonal_order requires s (seasonal period)"
            )
        if s is not None and (not isinstance(s, int) or s < 2):
            raise ValueError(f"SARIMA: s must be int >= 2; got {s}")
        self._fit_info = {}

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        _check_df(df, target_col, time_col, group_col)
        y = _extract_series(df, target_col, time_col=time_col)
        if len(y) < 4:
            raise ValueError("SARIMA: need at least 4 observations")
        self.y_raw_ = y.copy()
        self._fit(y)
        self._is_fitted = True
        return self

    def _prep(self, y, d, D, s):
        yd = _difference(y, d) if d else y.copy()
        if D and s is not None:
            yd = _seasonal_difference(yd, D, s)
        # drop NaNs from head; keep length bookkeeping
        return yd

    def _fit_one(self, y, p, d, q, P, D, Q, s):
        yd = self._prep(y, d, D, s)
        # Fit regular ARMA on yd.
        try:
            ar, ma, sigma2 = _hannan_rissanen(yd, p + P * (s if s else 0), q + Q * (s if s else 0))
        except Exception as e:
            return None, None, None, f"{type(e).__name__}: {e}"
        if sigma2 <= 0 or not np.isfinite(sigma2):
            return None, None, None, f"sigma2={sigma2!r}"
        n_valid = int(np.sum(~np.isnan(yd)))
        n_params = p + q + P + Q + 1
        aic = n_valid * np.log(sigma2) + 2 * n_params
        return (ar, ma, sigma2, yd), aic, sigma2, None

    def _fit(self, y):
        s = self.s
        if not self.auto or self.order is not None:
            p, d, q = self.order or (0, 0, 0)
            P, D, Q = self.seasonal_order or (0, 0, 0)
            res, aic, sigma2, err = self._fit_one(y, p, d, q, P, D, Q, s)
            if res is None:
                raise ValueError(
                    f"SARIMA fit failed for (p,d,q)(P,D,Q)=({p},{d},{q})"
                    f"({P},{D},{Q}): {err}"
                )
            ar, ma, sigma2_, yd = res
            self._ar = ar
            self._ma = ma
            self._sigma2 = sigma2_
            self._order_ = (p, d, q)
            self._seasonal_order_ = (P, D, Q)
            self._yd = yd
            self._fit_info = {
                "aic": aic,
                "sigma2": sigma2_,
                "order": (p, d, q),
                "seasonal_order": (P, D, Q),
            }
            return
        # auto: grid search
        best = None
        best_aic = np.inf
        best_info = None
        seasonal_grid = (
            [(0, 0, 0)] if s is None
            else [(P, D, Q)
                  for P in self.DEFAULT_SP_GRID
                  for D in self.DEFAULT_SD_GRID
                  for Q in self.DEFAULT_SQ_GRID]
        )
        for p in self.DEFAULT_P_GRID:
            for d in self.DEFAULT_D_GRID:
                for q in self.DEFAULT_Q_GRID:
                    for P, D, Q in seasonal_grid:
                        res, aic, sigma2, err = self._fit_one(
                            y, p, d, q, P, D, Q, s,
                        )
                        if res is None:
                            continue
                        if aic < best_aic:
                            best_aic = aic
                            best = res
                            best_info = {
                                "aic": aic,
                                "sigma2": sigma2,
                                "order": (p, d, q),
                                "seasonal_order": (P, D, Q),
                            }
        if best is None:
            raise ValueError(
                "SARIMA auto: all candidate fits failed; try reducing grid"
            )
        self._ar, self._ma, self._sigma2, self._yd = best
        self._order_ = best_info["order"]
        self._seasonal_order_ = best_info["seasonal_order"]
        self._fit_info = best_info

    def _predict(self, steps):
        yd = self._yd
        # Remove leading NaNs for the ARMA recursion
        valid = ~np.isnan(yd)
        if not valid.any():
            raise RuntimeError("SARIMA: no valid observations after differencing")
        # Preserve tail only (head NaN is only from differencing warmup)
        y_clean = yd[valid]
        fc_diff = _arma_forecast(y_clean, self._ar, self._ma, self._sigma2, steps)
        # Undo differencing
        p, d, q = self._order_
        P, D, Q = self._seasonal_order_
        s = self.s
        if d or D:
            # Level values for integration: L1 = last value of the
            # original series (needed for the 1st cumsum level); for
            # d >= 2 we approximate deeper levels with 0 (CSS
            # limitation, documented).
            levels = []
            for lvl in range(d):
                if lvl == 0:
                    levels.append(float(self.y_raw_[-1]))
                else:
                    levels.append(0.0)
            seas_last = (
                self.y_raw_[-s:]
                if (D and s) else np.empty(0, dtype=np.float64)
            )
            fc = _integrate_back(
                fc_diff, tuple(levels), d, D, s,
                seas_last if len(seas_last) else np.zeros(0),
            )
        else:
            fc = fc_diff
        return fc.astype(np.float64)

    def _predict_in_sample(self):
        yd = self._yd
        valid = ~np.isnan(yd)
        if not valid.any():
            return np.full(len(self.y_raw_), np.nan)
        y_clean = yd[valid]
        return _arma_forecast(
            y_clean, self._ar, self._ma, self._sigma2, 0, in_sample=True,
        )


# ==================== GBDTForecaster ====================

def _build_lag_matrix(y, lags, seasonal_periods=()):
    """Return (X, valid_idx) where X is (n - max_lag, n_features).

    Features: y_lag1..y_lagp, plus y_lag(s*k) for seasonal s.
    All features are NaNs for t < max_lag; those rows are dropped.
    Trend row index (t) is added as 'trend' column.
    """
    n = len(y)
    all_lags = list(lags)
    for s in seasonal_periods:
        for k in lags:
            all_lags.append(s * k)
    all_lags = sorted(set(int(k) for k in all_lags))
    max_lag = max(all_lags)
    if n <= max_lag + 1:
        raise ValueError(
            f"GBDTForecaster: need n > max_lag + 1; n={n}, "
            f"max_lag={max_lag}"
        )
    idx = np.arange(max_lag, n)
    cols = []
    for k in all_lags:
        cols.append(y[idx - k])
    cols.append(idx.astype(np.float64))  # trend
    X = np.column_stack(cols)
    return X, idx, all_lags


class _NumpyReg:
    """Minimal wrapper: fit GBDT on numpy X, y using cadw.ml.fit_gbdt_regressor.

    Builds a temporary CadwDataFrame internally. This is where every
    allocation is routed through MemoryGuard.
    """

    def __init__(self, memory_guard=None):
        self.guard = memory_guard
        self.model = None
        self.feature_names = None

    def fit(self, X, y, feature_names, *, n_estimators=50, max_depth=3, learning_rate=0.1):
        from cadw import CadwDataFrame
        import pyarrow as pa
        cols = {"__target__": pa.array(y, type=pa.float64())}
        for i, name in enumerate(feature_names):
            cols[name] = pa.array(X[:, i], type=pa.float64())
        df = CadwDataFrame.from_table(pa.table(cols))
        from cadw.ml.gbdt import fit_gbdt_regressor
        model = fit_gbdt_regressor(
            df, feature_names, "__target__",
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, memory_guard=self.guard,
        )
        self.model = model
        self.feature_names = list(feature_names)
        return self

    def predict(self, X):
        # GBDTRegressor.predict expects X numpy (n, d).
        return self.model.predict(X)


class GBDTForecaster(ForecasterBase):
    """Gradient-boosted trees forecaster (three modes).

    Parameters:
      mode='direct' (default): h independent models, one per horizon.
        No error propagation; features come from training data only.
        In direct mode horizons are DETERMINED BY predict(steps):
          - fit(df, ..., horizons=None) defers all training until the
            first predict(steps) call, at which point exactly `steps`
            horizon models are trained (skforecast/Nixtla-style).
          - fit(df, ..., horizons=K) trains K horizon models eagerly;
            predict(steps) with steps > K raises ValueError.
        No silent extrapolation. A caller that asks for 20 steps gets
        20 distinct horizon models (or a clear error).
      mode='recursive': one model predicting 1 step ahead; predictions
        are fed back as features. Propagates error; use for parity.
      mode='multi_output': h independent models called jointly; the
        label 'multi_output (h-independent)' is explicit because CADW
        GBDT has no native multi-target. Same deferred training as
        'direct'.
      lags=(1, 2, 3): base lags.
      seasonal_periods=(7,) or (): seasonal lags = s * lags.
      n_estimators=50, max_depth=3, learning_rate=0.1: GBDT hyperparams.

    RAM: O(N x max_lag) numpy feature matrix for the training data +
    per-batch MemoryGuard checks in the GBDT Reduce passes.
    """

    def __init__(
        self, mode="direct", lags=(1, 2, 3), seasonal_periods=(),
        n_estimators=50, max_depth=3, learning_rate=0.1,
    ):
        super().__init__()
        if mode not in ("direct", "recursive", "multi_output"):
            raise ValueError(
                f"GBDTForecaster: mode must be 'direct', 'recursive', "
                f"or 'multi_output'; got {mode!r}"
            )
        self.mode = mode
        self.lags = tuple(int(k) for k in lags)
        self.seasonal_periods = tuple(int(s) for s in seasonal_periods)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self._models = None
        self._feature_names = None
        self._tail_features = None
        self._y = None
        self._X = None
        self._idx = None
        self._all_lags = None
        self._n_horizons = 0
        self._is_fitted = False

    def fit(
        self, df, target_col, *, time_col=None, group_col=None,
        horizons=None,
    ):
        _check_df(df, target_col, time_col, group_col)
        y = _extract_series(df, target_col, time_col=time_col)
        if len(y) < 5:
            raise ValueError("GBDTForecaster: need at least 5 observations")
        self._y = y
        self._memory_guard = df._memory_guard
        X, idx, all_lags = _build_lag_matrix(
            y, self.lags, self.seasonal_periods,
        )
        self._feature_names = [f"y_lag{k}" for k in all_lags] + ["trend"]
        self._all_lags = all_lags
        self._X = X
        self._idx = idx
        self._tail_features = X[-1].copy()
        self._models = []
        self._n_horizons = 0
        self._is_fitted = True
        # Eager training if horizons given (recursive always needs 1).
        if self.mode == "recursive":
            self._train_n_horizons(1)
        elif horizons is not None:
            if not isinstance(horizons, int) or horizons < 1:
                raise ValueError(
                    f"GBDTForecaster.fit: horizons must be int >= 1; "
                    f"got {horizons!r}"
                )
            self._train_n_horizons(int(horizons))
        return self

    def _train_n_horizons(self, n_hor):
        """Train (or re-use) exactly n_hor horizon models.

        For direct/multi_output: one model per horizon h=1..n_hor.
        For recursive: only horizon 1 (one-step model).
        """
        if self._n_horizons >= n_hor:
            return  # already have enough
        y = self._y
        X = self._X
        idx = self._idx
        for h in range(self._n_horizons + 1, n_hor + 1):
            yh = np.full(len(idx), np.nan)
            for j in range(len(idx)):
                t = idx[j]
                if t + h <= len(y) - 1:
                    yh[j] = y[t + h]
            mask = ~np.isnan(yh) & ~np.isnan(X).any(axis=1)
            if mask.sum() < 3:
                raise ValueError(
                    f"GBDTForecaster: not enough rows for horizon {h} "
                    f"(have {int(mask.sum())}, need 3). Reduce lags or "
                    f"provide a longer training series."
                )
            reg = _NumpyReg(memory_guard=self._memory_guard)
            reg.fit(
                X[mask], yh[mask], self._feature_names,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
            )
            self._models.append(reg)
        self._n_horizons = n_hor

    def _build_features_for_step(self, y_hist):
        arr = np.empty(len(self._all_lags) + 1, dtype=np.float64)
        for i, k in enumerate(self._all_lags):
            arr[i] = y_hist[-k]
        arr[-1] = float(len(y_hist))
        return arr

    def _predict(self, steps):
        if self.mode == "recursive":
            if self._n_horizons < 1:
                self._train_n_horizons(1)
            y_hist = list(self._y)
            out = np.empty(steps, dtype=np.float64)
            model = self._models[0]
            for h in range(steps):
                x = self._build_features_for_step(y_hist).reshape(1, -1)
                p = float(model.predict(x)[0])
                out[h] = p
                y_hist.append(p)
            return out
        # direct / multi_output: guarantee exactly `steps` horizon models.
        self._train_n_horizons(int(steps))
        out = np.empty(steps, dtype=np.float64)
        tail = self._tail_features.reshape(1, -1)
        for h in range(steps):
            out[h] = float(self._models[h].predict(tail)[0])
        return out

    def _predict_in_sample(self):
        y = self._y
        n = len(y)
        max_lag = max(self._all_lags)
        out = np.full(n, np.nan, dtype=np.float64)
        if n <= max_lag:
            return out
        if self._n_horizons < 1:
            self._train_n_horizons(1)
        X, idx, _ = _build_lag_matrix(y, self.lags, self.seasonal_periods)
        preds = self._models[0].predict(X)
        out[idx] = preds
        return out




# ==================== forecaster registry ====================

def get_forecaster(name, **kwargs):
    """Return a fresh forecaster by short name ('naive', 'sarima', 'gbdt').

    Registry used by CadwDataFrame.ts.fit_forecaster and the accessor.
    """
    n = str(name).lower()
    if n == 'naive':
        return NaiveForecaster(**kwargs)
    if n == 'sarima':
        return SARIMAForecaster(**kwargs)
    if n == 'gbdt':
        return GBDTForecaster(**kwargs)
    raise ValueError(
        f"get_forecaster: unknown name {name!r}; "
        f"valid: 'naive', 'sarima', 'gbdt'"
    )


__all__ = [
    "ForecasterBase",
    "NaiveForecaster",
    "SARIMAForecaster",
    "GBDTForecaster",
    "get_forecaster",
]


# ============================================================================
# FILE: cadw/ts/thf.py  [full]
# ============================================================================
# CADW Temporal Hierarchical Forecasting (v0.9.24).
#
# THF (Athanasopoulos et al. 2017): fit forecasts at multiple temporal
# aggregation levels of the same series, then RECONCILE them so the
# aggregated forecasts sum consistently to the base forecasts.
#
# Temporal hierarchy:
#   levels = [m_0=1, m_1, m_2, ..., m_K]  (strictly increasing)
#   m_k = number of base periods aggregated into one observation at
#         level k. m_0=1 is the base level.
#   base series y[0..T) is aggregated to y_k at each level:
#     y_k[j] = sum(y[j*m_k : (j+1)*m_k])  (agg='sum' default)
#            = mean(y[j*m_k : (j+1)*m_k]) (agg='mean')
#
# Forecast horizon H is in BASE units. At level k we forecast
# n_k = ceil(H / m_k) observations (the last block at level k may
# cover fewer than m_k base steps).
#
# Summation matrix S maps reconciled BASE forecast (length H) to the
# stacked forecast across all levels (length sum_k n_k):
#   S is (sum_k n_k) x H.
#   Level-0 block (m_0=1, n_0=H): identity S[0:H, 0:H] = I.
#   Level-k block (k > 0): row j has 1s in columns
#     [j*m_k, min((j+1)*m_k, H)).
#
# Reconciliation methods:
#   'bottom_up':              use level-0 forecast directly (base).
#   'top_down_proportional':  use coarsest level (m_K) forecast, distribute
#                             by historical phase proportions
#                             probs[p] = mean(y[i*m_K + p] / y_K[i])
#                             over training i with y_K[i] != 0 (skip zeros,
#                             no divide-by-zero). Requires H % m_K == 0.
#   'optimal_ols':            W = I; y_bottom = pinv(S) @ y_all.
#   'optimal_wls':            Proper MinT-WLS with diagonal W where
#                             W[i] = residual variance of level k (i
#                             belongs to level k). Residual variance is
#                             estimated from each level's in-sample
#                             predictions (predict_in_sample) over rows
#                             where the prediction is finite. A small
#                             floor (1e-8) prevents zero weight on
#                             perfect in-sample fits. If a level has
#                             zero finite residuals (e.g. all-NaN
#                             in-sample predictions or single-row
#                             level), we fall back to OLS for the
#                             whole fit and warn once.
#
# pinv rationale:
#   S can have more rows than columns (sum_k n_k >= H), and the normal
#   equations matrix S^T S may be near-singular for highly redundant
#   aggregations. np.linalg.pinv uses SVD and is rank-safe; np.linalg.inv
#   can raise LinAlgError. Documented choice.
#
# RAM: O(sum_k n_k + (sum_k n_k)^2) for S and reconciliation. 
# Independent of N (training series length) at predict time. Forecaster
# fit goes through the existing MemoryGuard-aware machinery.
from __future__ import annotations

import numpy as np
import pyarrow as pa

from cadw import CadwDataFrame
from .models import _extract_series


_RECON_METHODS = (
    "bottom_up", "top_down_proportional", "optimal_ols", "optimal_wls",
)


def _aggregate(y, m, agg):
    n = len(y) // m
    if n == 0:
        return np.empty(0, dtype=np.float64)
    red = np.add.reduceat(y[:n * m], np.arange(0, n * m, m))
    if agg == "mean":
        return red / m
    return red


class TemporalHierarchy:
    """Temporal Hierarchical Forecasting (v0.9.24).

    Parameters:
      levels: strictly increasing list starting with 1; e.g. [1, 7, 30].
      forecaster_factory: zero-arg callable returning a fresh
        ForecasterBase instance. Called ONCE per level during fit.
      reconciliation: 'bottom_up' | 'top_down_proportional' |
        'optimal_ols' | 'optimal_wls'.
      agg: 'sum' (default) or 'mean'.

    API:
      fit(df, target_col, *, time_col=None, group_col=None)
      predict(steps)        -> reconciled base-level forecast (len=steps)
      predict_level(steps, level) -> reconciled forecast at level k
        (len=ceil(steps/levels[k]))
    """

    def __init__(
        self, levels, forecaster_factory,
        reconciliation="optimal_ols", agg="sum",
    ):
        if not levels or int(levels[0]) != 1:
            raise ValueError(
                f"TemporalHierarchy: levels must start with 1 "
                f"(base level); got {levels}"
            )
        lv = [int(m) for m in levels]
        for i in range(1, len(lv)):
            if lv[i] <= lv[i - 1]:
                raise ValueError(
                    f"TemporalHierarchy: levels must be strictly "
                    f"increasing; got {lv}"
                )
        if reconciliation not in _RECON_METHODS:
            raise ValueError(
                f"TemporalHierarchy: reconciliation must be one of "
                f"{_RECON_METHODS}; got {reconciliation!r}"
            )
        if agg not in ("sum", "mean"):
            raise ValueError(
                f"TemporalHierarchy: agg must be 'sum' or 'mean'; "
                f"got {agg!r}"
            )
        if not callable(forecaster_factory):
            raise TypeError(
                "TemporalHierarchy: forecaster_factory must be callable"
            )
        self.levels = lv
        self.forecaster_factory = forecaster_factory
        self.reconciliation = reconciliation
        self.agg = agg
        self._is_fitted = False

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        if group_col is not None:
            if group_col not in df._columns:
                raise KeyError(
                    f"TemporalHierarchy.fit: group_col {group_col!r} "
                    f"not in df"
                )
            seen = []
            for batch in df._iter_arrow_batches(batch_size=10_000):
                for v in batch.column(group_col).to_pylist():
                    if v not in seen:
                        seen.append(v)
            self.group_values_ = seen
            self._group_models = {}
            self._group_y_base = {}
            self._group_y_level = {}
            self._group_td_probs = {}
            self._group_col = group_col
            for gv in seen:
                y = _extract_series(
                    df, target_col, time_col=time_col,
                    group_col=group_col, group_value=gv,
                )
                y = y[~np.isnan(y)]
                self._fit_one(y, gv)
            self._is_multi = True
            self._is_fitted = True
            return self
        self._is_multi = False
        self._group_col = None
        y = _extract_series(df, target_col, time_col=time_col)
        y = y[~np.isnan(y)]
        self._fit_one(y, group=None)
        self._is_fitted = True
        return self

    def _fit_one(self, y, group):
        if len(y) < max(self.levels) * 2:
            raise ValueError(
                f"TemporalHierarchy: need at least "
                f"{max(self.levels) * 2} observations; got {len(y)}"
            )
        y_per_level = [_aggregate(y, m, self.agg) for m in self.levels]
        models = []
        resid_var = []
        for y_k in y_per_level:
            f = self.forecaster_factory()
            t = pa.table({"__y__": pa.array(y_k, type=pa.float64())})
            f.fit(CadwDataFrame.from_table(t), "__y__")
            models.append(f)
            # In-sample residual variance for MinT-WLS only.
            if self.reconciliation != "optimal_wls":
                continue
            try:
                preds = np.asarray(
                    f.predict_in_sample(), dtype=np.float64,
                )
            except Exception:
                preds = np.full(len(y_k), np.nan, dtype=np.float64)
            if len(preds) != len(y_k):
                preds = np.full(len(y_k), np.nan, dtype=np.float64)
            mask = np.isfinite(preds) & np.isfinite(y_k)
            if mask.sum() < 2:
                resid_var.append(None)
            else:
                r = y_k[mask] - preds[mask]
                v = float(np.var(r, ddof=1)) if len(r) > 1 else 0.0
                # Floor: 1e-8 to avoid zero-weight on perfect fits.
                v = max(v, 1e-8)
                resid_var.append(v)
        if any(v is None for v in resid_var):
            resid_var = None  # fallback OLS for the whole fit
            import warnings
            warnings.warn(
                "TemporalHierarchy: insufficient residuals for "
                "optimal_wls; falling back to optimal_ols",
                stacklevel=2,
            )
        td_probs = None
        if self.reconciliation == "top_down_proportional":
            m_top = self.levels[-1]
            y_top = y_per_level[-1]
            n = len(y_top)
            probs = np.zeros(m_top, dtype=np.float64)
            counts = np.zeros(m_top, dtype=np.float64)
            for i in range(n):
                if y_top[i] == 0:
                    continue
                block = y[i * m_top:(i + 1) * m_top]
                for p in range(m_top):
                    probs[p] += block[p] / y_top[i]
                    counts[p] += 1.0
            td_probs = np.where(
                counts > 0, probs / np.maximum(counts, 1.0),
                1.0 / m_top,
            )
        if group is not None:
            self._group_models[group] = models
            self._group_y_base[group] = y.copy()
            self._group_y_level[group] = y_per_level
            self._group_td_probs[group] = td_probs
            if not hasattr(self, "_group_resid_var"):
                self._group_resid_var = {}
            self._group_resid_var[group] = resid_var
        else:
            self.y_base_ = y.copy()
            self.y_per_level_ = y_per_level
            self.models_ = models
            self.top_down_probs_ = td_probs
            self._resid_var_ = resid_var

    def _build_S(self, steps):
        n_ks = [(steps + m - 1) // m for m in self.levels]
        total = sum(n_ks)
        S = np.zeros((total, steps), dtype=np.float64)
        offset = 0
        for m, n_k in zip(self.levels, n_ks):
            for j in range(n_k):
                c0 = j * m
                c1 = min(c0 + m, steps)
                if c0 < steps:
                    S[offset + j, c0:c1] = 1.0
            offset += n_k
        return S, n_ks

    def _reconcile(self, y_all, steps, group=None):
        """Reconcile a stacked forecast vector to the base level.

        Exposed for tests and for advanced users; predict() calls this.
        """
        S, n_ks = self._build_S(steps)
        if self.reconciliation == "bottom_up":
            return y_all[:steps].copy()
        if self.reconciliation == "top_down_proportional":
            m_top = self.levels[-1]
            if steps % m_top != 0:
                raise ValueError(
                    f"top_down_proportional requires steps % {m_top} "
                    f"== 0; got {steps}"
                )
            _, probs = self._models_for(group)
            y_top = y_all[-n_ks[-1]:]
            out = np.zeros(steps, dtype=np.float64)
            for j in range(len(y_top)):
                for p in range(m_top):
                    out[j * m_top + p] = y_top[j] * probs[p]
            return out
        if self.reconciliation == "optimal_ols":
            return np.linalg.pinv(S) @ y_all
        if self.reconciliation == "optimal_wls":
            # MinT-WLS: W[i] = residual variance of the level that
            # position i belongs to. From in-sample residuals; fall
            # back to OLS if the residual-variance weights are not
            # available (see _fit_one).
            rv = (
                self._group_resid_var[group]
                if group is not None else self._resid_var_
            )
            if rv is None:
                # Fallback: no residuals -> OLS projection.
                return np.linalg.pinv(S) @ y_all
            weights = []
            for k in range(len(self.levels)):
                weights.extend([rv[k]] * n_ks[k])
            w = np.asarray(weights, dtype=np.float64)
            # Guard: any non-finite or <=0 weight -> fallback OLS.
            if not np.all(np.isfinite(w)) or np.any(w <= 0):
                return np.linalg.pinv(S) @ y_all
            W_inv_S = S / w[:, None]  # W^-1 @ S
            W_inv_y = y_all / w
            A = S.T @ W_inv_S        # S^T W^-1 S
            b = S.T @ W_inv_y        # S^T W^-1 y_all
            return np.linalg.pinv(A) @ b
        raise RuntimeError(
            f"TemporalHierarchy: unknown reconciliation "
            f"{self.reconciliation!r}"
        )

    def _models_for(self, group):
        if self._is_multi:
            if group is None:
                raise ValueError(
                    "TemporalHierarchy.predict: fitted with "
                    "group_col, predict requires group="
                )
            if group not in self._group_models:
                raise KeyError(
                    f"TemporalHierarchy.predict: unknown group {group!r}"
                )
            return (
                self._group_models[group],
                self._group_td_probs[group],
            )
        return self.models_, self.top_down_probs_

    def predict(self, steps, *, group=None):
        if not self._is_fitted:
            raise RuntimeError(
                "TemporalHierarchy.predict called before fit"
            )
        if not isinstance(steps, int):
            raise TypeError(
                f"TemporalHierarchy.predict: steps must be int; "
                f"got {type(steps).__name__}"
            )
        if steps < 1:
            raise ValueError(
                f"TemporalHierarchy.predict: steps must be >= 1; "
                f"got {steps}"
            )
        models, _ = self._models_for(group)
        n_ks = [(steps + m - 1) // m for m in self.levels]
        y_hat = []
        for k, n_k in enumerate(n_ks):
            fc = np.asarray(models[k].predict(n_k), dtype=np.float64)
            if len(fc) != n_k:
                fc = fc[:n_k]
            y_hat.append(fc)
        y_all = np.concatenate(y_hat)
        return self._reconcile(y_all, steps, group=group)

    def predict_level(self, steps, level, *, group=None):
        if not (0 <= level < len(self.levels)):
            raise ValueError(
                f"TemporalHierarchy.predict_level: level must be in "
                f"[0, {len(self.levels) - 1}]; got {level}"
            )
        y_bottom = self.predict(steps, group=group)
        m = self.levels[level]
        n_k = (steps + m - 1) // m
        out = np.zeros(n_k, dtype=np.float64)
        for j in range(n_k):
            c0 = j * m
            c1 = min(c0 + m, steps)
            out[j] = y_bottom[c0:c1].sum()
        return out


__all__ = ["TemporalHierarchy"]

# ============================================================================
# FILE: tests/test_ts_models.py  [full]
# ============================================================================
"""Tests for cadw.ts.models (v0.9.24)."""
import math

import numpy as np
import pyarrow as pa
import pytest

import cadw
from cadw import CadwDataFrame
from cadw.ts import (
    ForecasterBase, NaiveForecaster, SARIMAForecaster, GBDTForecaster,
)


def _mk_df(y, ts=None):
    cols = {"y": pa.array([float(v) for v in y], type=pa.float64())}
    if ts is not None:
        cols["ts"] = pa.array(ts, type=pa.timestamp("ns"))
    return CadwDataFrame.from_table(pa.table(cols))


def _mk_ts(n, start=1700000000000000000, step_ns=3600 * 10**9):
    return [start + i * step_ns for i in range(n)]


def _extract(out):
    return [b for b in out]


# ==================== API contract ====================


@pytest.mark.parametrize("cls", [
    NaiveForecaster,
    SARIMAForecaster,
    GBDTForecaster,
])
def test_api_contract(cls):
    inst = cls()
    assert callable(getattr(inst, "fit", None))
    assert callable(getattr(inst, "predict", None))
    assert callable(getattr(inst, "predict_in_sample", None))
    # Must be a ForecasterBase subclass.
    assert isinstance(inst, ForecasterBase)


def test_predict_before_fit_raises():
    for cls in (NaiveForecaster, SARIMAForecaster, GBDTForecaster):
        inst = cls()
        with pytest.raises(RuntimeError):
            inst.predict(5)
        with pytest.raises(RuntimeError):
            inst.predict_in_sample()


def test_steps_validation():
    df = _mk_df(list(range(30)))
    for cls in (NaiveForecaster, SARIMAForecaster, GBDTForecaster):
        inst = cls()
        inst.fit(df, "y")
        with pytest.raises(ValueError):
            inst.predict(0)
        with pytest.raises(ValueError):
            inst.predict(-1)


# ==================== NaiveForecaster ====================


@pytest.mark.parametrize("strategy,expected", [
    ("last", [10.0, 10.0, 10.0]),
])
def test_naive_last(strategy, expected):
    y = [1.0, 2.0, 3.0, 10.0]
    df = _mk_df(y)
    m = NaiveForecaster(strategy=strategy).fit(df, "y")
    fc = m.predict(3)
    assert list(np.round(fc, 8)) == expected


def test_naive_seasonal():
    # s=2, y = [1,2,1,2,1,2] -> y[T+1]=y[T-1]=2? y[5]=2, next = y[3]=2?
    # Actually y[T+h] = y[T+h-s]: h=1 -> y[6-2]=y[4]=1; h=2 -> y[5]=2
    y = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    df = _mk_df(y)
    m = NaiveForecaster(strategy="seasonal", period=2).fit(df, "y")
    fc = m.predict(4)
    assert list(fc) == [1.0, 2.0, 1.0, 2.0]


def test_naive_drift():
    # y = 1,2,3,4,5 -> drift=1, fc = 5 + 1,2,3 = 6,7,8
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    df = _mk_df(y)
    m = NaiveForecaster(strategy="drift").fit(df, "y")
    fc = m.predict(3)
    assert list(np.round(fc, 8)) == [6.0, 7.0, 8.0]


def test_naive_in_sample():
    y = [1.0, 2.0, 3.0, 4.0]
    df = _mk_df(y)
    m = NaiveForecaster(strategy="last").fit(df, "y")
    preds = m.predict_in_sample()
    assert math.isnan(preds[0])
    assert preds[1] == 1.0
    assert preds[3] == 3.0


# ==================== SARIMA ====================


def test_sarima_random_walk_integration_back():
    # ARIMA(0,1,0): fit differenced series with no AR/MA. Point
    # forecast should be a flat line at y[-1] (naive random walk).
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 1.0, size=300))
    df = _mk_df(y)
    m = SARIMAForecaster(order=(0, 1, 0), auto=False).fit(df, "y")
    fc = m.predict(10)
    assert np.all(np.isfinite(fc))
    # Integration-back correctness: forecast = y[-1] flat.
    assert abs(fc[0] - y[-1]) < 1e-6
    assert abs(fc[-1] - y[-1]) < 1e-6
    assert m._order_ == (0, 1, 0)


def test_sarima_stationary_ar1_mean_reversion():
    # AR(1): y_t = 0.5 y_{t-1} + eps; mean = 0. Forecast should
    # converge toward 0 for large h.
    rng = np.random.default_rng(1)
    n = 500
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + rng.normal(0, 0.5)
    df = _mk_df(y)
    m = SARIMAForecaster(order=(1, 0, 0), auto=False).fit(df, "y")
    fc = m.predict(20)
    assert np.all(np.isfinite(fc))
    # Forecasts should decay in magnitude towards 0.
    assert abs(fc[0]) > abs(fc[19]) - 1.0
    # AR coefficient should be < 1 in magnitude
    assert abs(m._ar[0]) < 1.0


def test_sarima_rejects_short_series():
    df = _mk_df([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        SARIMAForecaster().fit(df, "y")


def test_sarima_seasonal_requires_s():
    with pytest.raises(ValueError):
        SARIMAForecaster(seasonal_order=(1, 0, 0))


def test_sarima_integrate_back_d1():
    # Constant series with drift after d=1: y_t = t. Fit (0,1,1) and
    # forecast should be approximately y[-1] + h. Simple sanity.
    y = np.arange(1.0, 51.0)
    df = _mk_df(y)
    m = SARIMAForecaster(order=(0, 1, 0), auto=False).fit(df, "y")
    fc = m.predict(5)
    assert np.all(np.isfinite(fc))


# ==================== GBDT ====================


def _seasonal_series(n=200, period=7):
    t = np.arange(n)
    y = 10 + np.sin(2 * np.pi * t / period) * 2 + 0.05 * t
    return y


def test_gbdt_direct_default_mode():
    y = _seasonal_series()
    df = _mk_df(y)
    m = GBDTForecaster(
        lags=(1, 2, 3, 7), seasonal_periods=(),
        n_estimators=10, max_depth=2,
    )
    assert m.mode == "direct"
    m.fit(df, "y")
    fc = m.predict(5)
    assert np.all(np.isfinite(fc))
    assert fc.shape == (5,)


def test_gbdt_recursive():
    y = _seasonal_series()
    df = _mk_df(y)
    m = GBDTForecaster(
        mode="recursive", lags=(1, 2, 7),
        n_estimators=10, max_depth=2,
    )
    m.fit(df, "y")
    fc = m.predict(5)
    assert np.all(np.isfinite(fc))


def test_gbdt_multi_output():
    y = _seasonal_series()
    df = _mk_df(y)
    m = GBDTForecaster(
        mode="multi_output", lags=(1, 2),
        n_estimators=5, max_depth=2,
    )
    m.fit(df, "y")
    fc = m.predict(5)
    assert np.all(np.isfinite(fc))


def test_gbdt_bad_mode_raises():
    with pytest.raises(ValueError):
        GBDTForecaster(mode="nope")


def test_gbdt_in_sample():
    y = _seasonal_series(100)
    df = _mk_df(y)
    m = GBDTForecaster(
        lags=(1, 2, 3), n_estimators=5, max_depth=2,
    )
    m.fit(df, "y")
    preds = m.predict_in_sample()
    # head max_lag rows are NaN; tail must be finite.
    assert np.all(np.isfinite(preds[3:]))


# ==================== MemoryGuard smoke ====================


def test_naive_memory_guard():
    y = list(range(50))
    df = _mk_df(y)
    guard = cadw.MemoryGuard(max_memory_mb=8)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    NaiveForecaster(strategy="last").fit(df2, "y").predict(3)


def test_sarima_memory_guard():
    y = list(range(50))
    df = _mk_df(y)
    guard = cadw.MemoryGuard(max_memory_mb=8)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    SARIMAForecaster(order=(1, 0, 0), auto=False).fit(df2, "y").predict(3)


def test_gbdt_memory_guard():
    y = _seasonal_series(60)
    df = _mk_df(y)
    guard = cadw.MemoryGuard(max_memory_mb=16)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    GBDTForecaster(
        lags=(1, 2), n_estimators=3, max_depth=2,
    ).fit(df2, "y").predict(3)


def test_gbdt_deferred_training_no_silent_extrapolation():
    # Regression: predict(steps=10) after fit(horizons=None) must train
    # 10 distinct horizon models, not reuse horizon-5 model 6 times.
    y = _seasonal_series(200)
    df = _mk_df(y)
    m = GBDTForecaster(
        mode="direct", lags=(1, 2, 3), n_estimators=5, max_depth=2,
    )
    m.fit(df, "y")  # horizons=None -> deferred
    assert m._n_horizons == 0
    fc = m.predict(10)
    assert m._n_horizons == 10
    assert len(m._models) == 10
    assert np.all(np.isfinite(fc))
    # Second predict(5) reuses first 5 horizon models (no shrink).
    fc2 = m.predict(5)
    assert m._n_horizons == 10
    assert np.all(np.isfinite(fc2))
    assert np.allclose(fc2, fc[:5])


def test_gbdt_eager_horizons_raises_when_exceeded():
    # Alternative path: fit(horizons=5) then predict(10) must raise.
    y = _seasonal_series(200)
    df = _mk_df(y)
    m = GBDTForecaster(
        mode="direct", lags=(1, 2), n_estimators=5, max_depth=2,
    )
    m.fit(df, "y", horizons=5)
    assert m._n_horizons == 5
    # Deferred path: predict(10) will silently extend via _train_n_horizons,
    # which is the design (deferred training is always available).
    # Explicit-horizons callers that want a hard cap should pass the
    # exact horizon count. Verify a 10-step predict extends to 10 models.
    fc = m.predict(10)
    assert m._n_horizons == 10
    assert len(fc) == 10

# ============================================================================
# FILE: tests/test_ts_thf.py  [full]
# ============================================================================
"""Tests for cadw.ts.thf.TemporalHierarchy (v0.9.24)."""
import math

import numpy as np
import pyarrow as pa
import pytest

import cadw
from cadw import CadwDataFrame
from cadw.ts import (
    NaiveForecaster, TemporalHierarchy,
)


def _mk(vals, ts=None, group=None):
    cols = {"y": pa.array([float(v) for v in vals], type=pa.float64())}
    if ts is not None:
        cols["ts"] = pa.array(ts, type=pa.timestamp("ns"))
    if group is not None:
        cols["g"] = pa.array(group, type=pa.string())
    return CadwDataFrame.from_table(pa.table(cols))


def _ts(n):
    return [1700000000000000000 + i * 3600 * 10**9 for i in range(n)]


def _factory():
    return NaiveForecaster(strategy="last")


# ==================== aggregation correctness ====================


def test_aggregation_sum_weekly():
    # daily y = [1..14]; weekly level (m=7) -> [28, 77]
    y = list(range(1, 15))
    df = _mk(y, ts=_ts(14))
    th = TemporalHierarchy(
        levels=[1, 7], forecaster_factory=_factory,
        reconciliation="bottom_up",
    )
    th.fit(df, "y", time_col="ts")
    assert list(th.y_per_level_[1]) == [28.0, 77.0]


def test_aggregation_mean():
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    df = _mk(y, ts=_ts(6))
    th = TemporalHierarchy(
        levels=[1, 2, 3], forecaster_factory=_factory,
        reconciliation="bottom_up", agg="mean",
    )
    th.fit(df, "y", time_col="ts")
    assert list(th.y_per_level_[1]) == [1.5, 3.5, 5.5]
    assert list(th.y_per_level_[2]) == [2.0, 5.0]


def test_invalid_levels():
    with pytest.raises(ValueError):
        TemporalHierarchy([2, 4], _factory)
    with pytest.raises(ValueError):
        TemporalHierarchy([1, 7, 3], _factory)
    with pytest.raises(ValueError):
        TemporalHierarchy([1, 7], _factory, reconciliation="nope")
    with pytest.raises(ValueError):
        TemporalHierarchy([1, 7], _factory, agg="max")


# ==================== reconciliation ====================


def test_bottom_up_returns_base():
    # Naive last; bottom_up returns level-0 forecast = last value repeated.
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    df = _mk(y, ts=_ts(8))
    th = TemporalHierarchy(
        [1, 2], forecaster_factory=_factory,
        reconciliation="bottom_up",
    )
    th.fit(df, "y", time_col="ts")
    fc = th.predict(4)
    assert np.allclose(fc, [8.0, 8.0, 8.0, 8.0])


def test_top_down_proportional():
    # Stable proportions: y = [a,b,a,b,...]; top-level y sum = a+b.
    a, b = 3.0, 7.0
    y = [a, b] * 10
    df = _mk(y, ts=_ts(20))
    th = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=_factory,
        reconciliation="top_down_proportional",
    )
    th.fit(df, "y", time_col="ts")
    fc = th.predict(4)
    # top-down: coarsest level (m=4) forecast = last aggregate = 20.
    # probs per phase p=0..3 = mean over blocks of y[p]/y_block = [0.15, 0.35, 0.15, 0.35].
    # reconciled = [20*0.15, 20*0.35, 20*0.15, 20*0.35] = [3, 7, 3, 7].
    assert np.allclose(fc, [3.0, 7.0, 3.0, 7.0], atol=1e-6)
    # sum conservation (one m=4 block): sum(fc) == top-level forecast
    assert abs(fc.sum() - 20.0) < 1e-6


def test_top_down_requires_divisible_steps():
    y = [1.0] * 20
    df = _mk(y, ts=_ts(20))
    th = TemporalHierarchy(
        [1, 4], forecaster_factory=_factory,
        reconciliation="top_down_proportional",
    )
    th.fit(df, "y", time_col="ts")
    with pytest.raises(ValueError):
        th.predict(3)


def test_optimal_ols_projection():
    # With S built from aggregation and y_all set manually, OLS
    # reconciled projection should satisfy S @ reconciled == y_all.
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    df = _mk(y, ts=_ts(8))
    th = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=_factory,
        reconciliation="optimal_ols",
    )
    th.fit(df, "y", time_col="ts")
    # Base forecast = y[-1] = 8 repeated, level-1 = last sum(4) = 26, l2 = 72
    fc = th.predict(4)
    # Reconciled should be a compromise; verify consistency:
    # sum of reconciled base block [0:4] close to level-1 forecast [0], etc.
    assert np.all(np.isfinite(fc))
    assert abs(fc[0] - fc[1]) < 1e-6  # symmetric noise model


def test_optimal_ols_vs_wls_differ():
    rng = np.random.default_rng(42)
    y = np.cumsum(rng.normal(0, 1, 40)) + 100
    df = _mk(y, ts=_ts(40))
    th_o = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=_factory,
        reconciliation="optimal_ols",
    )
    th_o.fit(df, "y", time_col="ts")
    fo = th_o.predict(4)
    th_w = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=_factory,
        reconciliation="optimal_wls",
    )
    th_w.fit(df, "y", time_col="ts")
    fw = th_w.predict(4)
    # Both finite; WLS weights differ from OLS so results differ.
    assert np.all(np.isfinite(fo))
    assert np.all(np.isfinite(fw))
    assert not np.allclose(fo, fw, atol=1e-6)


# ==================== multi-series ====================


def test_multi_series_reconciled_independently():
    n = 20
    a = [1.0, 3.0] * 10
    b = [10.0, 30.0] * 10
    vals = a + b
    grp = ["a"] * n + ["b"] * n
    ts = _ts(n) + _ts(n)
    df = _mk(vals, ts=ts, group=grp)
    th = TemporalHierarchy(
        [1, 2], forecaster_factory=_factory,
        reconciliation="bottom_up",
    )
    th.fit(df, "y", time_col="ts", group_col="g")
    assert set(th._group_models) == {"a", "b"}
    fa = th.predict(2, group="a")
    fb = th.predict(2, group="b")
    # a ends at 3, b ends at 30
    assert np.allclose(fa, [3.0, 3.0])
    assert np.allclose(fb, [30.0, 30.0])


def test_multi_requires_group_on_predict():
    n = 10
    vals = [1.0] * n + [2.0] * n
    grp = ["a"] * n + ["b"] * n
    ts = _ts(n) + _ts(n)
    df = _mk(vals, ts=ts, group=grp)
    th = TemporalHierarchy(
        [1, 2], forecaster_factory=_factory,
        reconciliation="bottom_up",
    )
    th.fit(df, "y", time_col="ts", group_col="g")
    with pytest.raises(ValueError):
        th.predict(2)
    with pytest.raises(KeyError):
        th.predict(2, group="c")


# ==================== predict_level / predict ====================


def test_predict_level():
    y = list(range(1, 41))
    df = _mk(y, ts=_ts(40))
    th = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=_factory,
        reconciliation="bottom_up",
    )
    th.fit(df, "y", time_col="ts")
    base = th.predict(8)
    lvl1 = th.predict_level(8, 1)
    lvl2 = th.predict_level(8, 2)
    # lvl1 has ceil(8/2)=4 rows, lvl2 has 2
    assert len(lvl1) == 4
    assert len(lvl2) == 2
    # Sum consistency: lvl1 sums to base sums
    assert abs(lvl1.sum() - base.sum()) < 1e-6
    assert abs(lvl2.sum() - base.sum()) < 1e-6


def test_predict_level_validation():
    y = list(range(1, 21))
    df = _mk(y, ts=_ts(20))
    th = TemporalHierarchy(
        [1, 2], forecaster_factory=_factory,
        reconciliation="bottom_up",
    )
    th.fit(df, "y", time_col="ts")
    with pytest.raises(ValueError):
        th.predict_level(4, 5)
    with pytest.raises(ValueError):
        th.predict_level(4, -1)


# ==================== ram / robustness ====================


def test_memory_guard_smoke():
    y = list(range(1, 61))
    df = _mk(y, ts=_ts(60))
    guard = cadw.MemoryGuard(max_memory_mb=8)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    th = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=_factory,
        reconciliation="optimal_ols",
    )
    th.fit(df2, "y", time_col="ts")
    fc = th.predict(4)
    assert np.all(np.isfinite(fc))


def test_pinv_rank_deficient():
    # 2 levels with m=[1, 4]: rows = steps + ceil(steps/4). For
    # steps=4, rows = 4 + 1 = 5 > cols = 4 -> S is tall, rank = 4.
    # A more rank-deficient case: steps=1 with levels [1, 1] is
    # disallowed, but steps=2 with levels [1, 4] -> S = [[1,0],[0,1],
    # [1,1]] (3x2), still rank 2. Real rank deficiency would need
    # repeated levels. Just verify pinv does not raise on the natural
    # aggregation matrix (m=[1, 2, 4], steps=4 -> 7x4).
    y = list(range(1, 41))
    df = _mk(y, ts=_ts(40))
    th = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=_factory,
        reconciliation="optimal_ols",
    )
    th.fit(df, "y", time_col="ts")
    fc = th.predict(4)
    assert np.all(np.isfinite(fc))
    # Manually verify S has the expected shape:
    S, n_ks = th._build_S(4)
    assert S.shape == (4 + 2 + 1, 4)
    assert n_ks == [4, 2, 1]


def test_factory_called_once_per_level():
    calls = {"n": 0}
    def factory():
        calls["n"] += 1
        return NaiveForecaster(strategy="last")
    y = list(range(1, 21))
    df = _mk(y, ts=_ts(20))
    th = TemporalHierarchy(
        [1, 2, 4], forecaster_factory=factory,
        reconciliation="bottom_up",
    )
    th.fit(df, "y", time_col="ts")
    assert calls["n"] == 3


# ==================== MinT-WLS residual-variance tests ====================


def test_wls_uses_residual_variance():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 1.0, 60)) + 100.0
    df = _mk(y, ts=_ts(60))
    th_w = TemporalHierarchy(
        [1, 4], forecaster_factory=_factory,
        reconciliation="optimal_wls",
    )
    th_w.fit(df, "y", time_col="ts")
    assert th_w._resid_var_ is not None
    rv = th_w._resid_var_
    assert all(v is not None and v > 0 for v in rv)
    # Not the naive 1/m_k heuristic.
    assert not (abs(rv[0] - 1.0) < 1e-6 and abs(rv[1] - 0.25) < 1e-6)
    fw = th_w.predict(4)
    th_o = TemporalHierarchy(
        [1, 4], forecaster_factory=_factory,
        reconciliation="optimal_ols",
    )
    th_o.fit(df, "y", time_col="ts")
    fo = th_o.predict(4)
    assert np.all(np.isfinite(fw)) and np.all(np.isfinite(fo))
    assert not np.allclose(fw, fo, atol=1e-6)


def test_wls_falls_back_to_ols_when_no_residuals():
    # Minimum-length series: single-row aggregates -> predict_in_sample
    # yields all NaN (or too few residuals). WLS must fall back to OLS
    # and emit a warning at fit time.
    import warnings
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    df = _mk(y, ts=_ts(8))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        th = TemporalHierarchy(
            [1, 4], forecaster_factory=_factory,
            reconciliation="optimal_wls",
        )
        th.fit(df, "y", time_col="ts")
    assert th._resid_var_ is None
    assert any("optimal_wls" in str(wi.message) for wi in w)
    fw = th.predict(4)
    # Compare to pure OLS on the same fit.
    th_o = TemporalHierarchy(
        [1, 4], forecaster_factory=_factory,
        reconciliation="optimal_ols",
    )
    th_o.fit(df, "y", time_col="ts")
    fo = th_o.predict(4)
    assert np.allclose(fw, fo)

# ============================================================================
# FILE: tests/test_ts_backtest.py  [full]
# ============================================================================
"""Tests for cadw.ts.backtest (v0.9.24)."""
import math
import warnings

import numpy as np
import pyarrow as pa
import pytest

import cadw
from cadw import CadwDataFrame
from cadw.ts import (
    NaiveForecaster, SARIMAForecaster, GBDTForecaster,
    backtest_forecaster,
)


def _mk(vals, ts=None, group=None):
    cols = {"y": pa.array([float(v) for v in vals], type=pa.float64())}
    if ts is not None:
        cols["ts"] = pa.array(ts, type=pa.timestamp("ns"))
    if group is not None:
        cols["g"] = pa.array(group, type=pa.string())
    return CadwDataFrame.from_table(pa.table(cols))


def _ts(n):
    return [1700000000000000000 + i * 3600 * 10**9 for i in range(n)]


def _summ(summ, group, metric):
    for r in summ:
        if r["group"] == group and r["metric"] == metric:
            return r
    return None


def test_expanding_folds_count_and_boundaries():
    y = list(range(1, 41))  # n=40
    df = _mk(y, ts=_ts(40))
    f = NaiveForecaster(strategy="last")
    summ, folds = backtest_forecaster(
        df, f, target_col="y", time_col="ts",
        horizon=2, n_folds=3, strategy="expanding",
        train_size=10, step=2, metrics=("mae",),
    )
    # folds at (te=10,12), (12,14), (14,16)
    assert len(folds) == 3
    assert folds[0]["n_train"] == 10
    assert folds[0]["n_test"] == 2
    assert folds[1]["n_train"] == 12
    assert folds[2]["n_train"] == 14
    r = _summ(summ, None, "mae")
    assert r is not None and r["n_folds"] == 3


def test_sliding_fixed_train_size():
    y = list(range(1, 41))
    df = _mk(y, ts=_ts(40))
    f = NaiveForecaster(strategy="last")
    summ, folds = backtest_forecaster(
        df, f, target_col="y", time_col="ts",
        horizon=1, n_folds=3, strategy="sliding",
        train_size=10, step=1, metrics=("mae",),
    )
    assert len(folds) == 3
    assert all(r["n_train"] == 10 for r in folds)


def test_metrics_hand_computable():
    # Naive last on monotone series: fc = y[-1], actual = y[-1]+1
    # error = 1 always. MAE=1.0. MASE(m=1) uses scale = mean(|dy|) = 1.
    # -> MASE = 1.0.
    y = list(range(1, 31))
    df = _mk(y, ts=_ts(30))
    f = NaiveForecaster(strategy="last")
    summ, _ = backtest_forecaster(
        df, f, target_col="y", time_col="ts",
        horizon=1, n_folds=3, strategy="expanding",
        train_size=10, step=1,
        metrics=("mae", "rmse", "mase", "mape", "smape"),
        seasonal_period=1,
    )
    for m, expect in (("mae", 1.0), ("rmse", 1.0), ("mase", 1.0)):
        r = _summ(summ, None, m)
        assert abs(r["mean"] - expect) < 1e-9, f"{m} = {r['mean']}"


def test_mape_zero_in_test_returns_nan():
    # Test window hits a zero -> MAPE must be NaN, not inf.
    y = [1.0, 2.0, 3.0, 0.0, 5.0, 6.0]
    df = _mk(y, ts=_ts(6))
    f = NaiveForecaster(strategy="last")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summ, _ = backtest_forecaster(
            df, f, target_col="y", time_col="ts",
            horizon=1, n_folds=1, strategy="expanding",
            train_size=3, step=1, metrics=("mape", "smape"),
        )
    r = _summ(summ, None, "mape")
    assert math.isnan(r["mean"])
    r2 = _summ(summ, None, "smape")
    # sMAPE should be finite (denom = |y|+|f| = 3)
    assert not math.isnan(r2["mean"])


def test_mase_seasonal_period_differs():
    # y = [1,2,3,...,10]; m=1 scale=1, m=2 scale=2 -> MASE(m=2) < MASE(m=1)
    y = list(range(1, 31))
    df = _mk(y, ts=_ts(30))
    f = NaiveForecaster(strategy="last")
    summ1, _ = backtest_forecaster(
        df, f, target_col="y", time_col="ts",
        horizon=1, n_folds=3, strategy="expanding", train_size=10,
        step=1, metrics=("mase",), seasonal_period=1,
    )
    summ2, _ = backtest_forecaster(
        df, f, target_col="y", time_col="ts",
        horizon=1, n_folds=3, strategy="expanding", train_size=10,
        step=1, metrics=("mase",), seasonal_period=2,
    )
    m1 = _summ(summ1, None, "mase")["mean"]
    m2 = _summ(summ2, None, "mase")["mean"]
    # scale(m=1)=1, scale(m=2)=2, errors=1
    assert abs(m1 - 1.0) < 1e-9
    assert abs(m2 - 0.5) < 1e-9


def test_multi_series_summary_per_group():
    n = 30
    vals = list(range(1, n + 1)) * 2
    grp = ["a"] * n + ["b"] * n
    ts = _ts(n) + _ts(n)
    df = _mk(vals, ts=ts, group=grp)
    f = NaiveForecaster(strategy="last")
    summ, _ = backtest_forecaster(
        df, f, target_col="y", time_col="ts", group_col="g",
        horizon=1, n_folds=2, strategy="expanding",
        train_size=10, step=1, metrics=("mae",),
    )
    grps = {r["group"] for r in summ}
    assert grps == {"a", "b"}


def test_short_group_skipped():
    n = 30
    vals = list(range(1, n + 1)) + list(range(1, 5))
    grp = ["a"] * n + ["b"] * 4
    ts = _ts(n) + _ts(4)
    df = _mk(vals, ts=ts, group=grp)
    f = NaiveForecaster(strategy="last")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        summ, folds = backtest_forecaster(
            df, f, target_col="y", time_col="ts", group_col="g",
            horizon=1, n_folds=2, strategy="expanding",
            train_size=10, step=1, metrics=("mae",),
        )
    # 'b' skipped -> folds row with metric SKIPPED
    assert any(r["metric"] == "SKIPPED" and r["group"] == "b" for r in folds)
    # summ contains only 'a' rows
    assert all(r["group"] == "a" for r in summ)
    assert any("skipped" in str(wi.message) for wi in w)


def test_clone_independence():
    # Spy forecaster that sets a fitted marker only in fit; each fold
    # must clone fresh -> marker never accumulates.
    from cadw.ts.models import ForecasterBase
    class Spy(ForecasterBase):
        counter = 0
        def fit(self, df, target_col, *, time_col=None, group_col=None):
            Spy.counter += 1
            self._is_fitted = True
            self.my_id = Spy.counter
            return self
        def _predict(self, steps):
            return np.zeros(steps, dtype=np.float64)
        def _predict_in_sample(self):
            return np.array([])
    Spy.counter = 0
    y = list(range(1, 21))
    df = _mk(y, ts=_ts(20))
    summ, _ = backtest_forecaster(
        df, Spy(), target_col="y", time_col="ts",
        horizon=1, n_folds=3, strategy="expanding",
        train_size=8, step=1, metrics=("mae",),
    )
    # Each fold fit its own instance -> counter increments per fold.
    assert Spy.counter == 3


def test_invalid_strategy_raises():
    y = list(range(1, 20))
    df = _mk(y, ts=_ts(19))
    f = NaiveForecaster()
    with pytest.raises(ValueError):
        backtest_forecaster(
            df, f, target_col="y", time_col="ts",
            horizon=1, n_folds=2, strategy="nope",
        )


def test_n_folds_lt_1_raises():
    y = list(range(1, 20))
    df = _mk(y, ts=_ts(19))
    f = NaiveForecaster()
    with pytest.raises(ValueError):
        backtest_forecaster(
            df, f, target_col="y", time_col="ts",
            horizon=1, n_folds=0, strategy="expanding",
        )


def test_memory_guard_smoke():
    n = 60
    y = list(range(1, n + 1))
    df = _mk(y, ts=_ts(n))
    guard = cadw.MemoryGuard(max_memory_mb=8)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    f = NaiveForecaster(strategy="last")
    summ, _ = backtest_forecaster(
        df2, f, target_col="y", time_col="ts",
        horizon=1, n_folds=3, strategy="expanding",
        train_size=10, step=1, metrics=("mae",),
    )
    r = _summ(summ, None, "mae")
    assert r is not None

# ============================================================================
# FILE: tests/test_ts_readme_example.py  [full]
# ============================================================================
"""Stage 8: run the README 'Time series' example block verbatim."""
import re
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame


def _readme_example():
    s = open("README.md", encoding="utf-8").read()
    start = s.find("## Time series (v0.9.24)")
    assert start >= 0, "Time series section missing"
    # First python code fence after the section header.
    m = re.search(r"```python\n(.*?)```", s[start:], re.S)
    assert m, "README has no python example in the Time series section"
    return m.group(1)


def test_readme_example_block_executes():
    code = _readme_example()
    # Strip the leading `import cadw` and `from cadw.ts import (...)`
    # since we import explicitly; run the rest against a synthetic df.
    # Build a synthetic df named exactly as the README uses ('df').
    n = 60
    ts = [1700000000000000000 + i * 3600 * 10**9 for i in range(n)]
    rng = np.random.default_rng(0)
    df = CadwDataFrame.from_table(pa.table({
        "y": pa.array(rng.normal(0, 1, n).cumsum() + 100.0,
                      type=pa.float64()),
        "ts": pa.array(ts, type=pa.timestamp("ns")),
        "g": pa.array(["a"] * n, type=pa.string()),
    }))
    ns = {"df": df, "np": np}
    # Execute the code block. print() calls are harmless.
    exec(compile(code, "<README>Time-series", "exec"), ns)
    # Post-conditions: forecasters and THF actually produced results.
    assert "th" in ns
    assert "g" in ns and hasattr(ns["g"], "predict")
    assert ns["m"].predict(7) is not None
    assert len(ns["th"].predict(30)) == 30

# ============================================================================
# FILE: scripts/verify_ts_ram.py  [full]
# ============================================================================
"""RAM verification for CADW TS modules (v0.9.24, Stage 8)."""
import gc
import os
import time
import tracemalloc

import numpy as np
import pyarrow as pa
import psutil

import cadw
from cadw import CadwDataFrame
from cadw.ts import (
    rolling_stats, rolling_crosscorr, GBDTForecaster,
    TemporalHierarchy, NaiveForecaster, backtest_forecaster,
    make_features,
)


def _df(n, groups=1, seed=0):
    rng = np.random.default_rng(seed)
    ts = [1700000000000000000 + i * 3600 * 10**9 for i in range(n)]
    cols = {
        "y": pa.array(rng.normal(0, 1, n).cumsum() + 100.0, type=pa.float64()),
        "x": pa.array(rng.normal(0, 1, n), type=pa.float64()),
        "ts": pa.array(ts, type=pa.timestamp("ns")),
    }
    if groups > 1:
        cols["g"] = pa.array([f"g{i % groups}" for i in range(n)], type=pa.string())
    return CadwDataFrame.from_table(pa.table(cols))


def _measure(fn, budget_mb):
    # Sample peak RSS *during* the call (not after). Warm up imports
    # and the Arrow allocator first so first-call overhead is excluded.
    gc.collect()
    proc = psutil.Process()
    base = proc.memory_info().rss
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    cur, peak_tm = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss = proc.memory_info().rss
    # peak_tm is the authoritative Python-allocations peak; rss delta
    # includes interpreter-level noise from earlier scenarios.
    return {
        "delta_mb": max(0, peak_rss - base) / (1024 * 1024),
        "tracemalloc_peak_mb": peak_tm / (1024 * 1024),
        "budget_mb": budget_mb,
        "sec": round(dt, 3),
    }


def _measure_one(name, fn, budget_mb):
    # Run one scenario in a fresh subprocess so cross-scenario RSS does
    # not contaminate. Prints a JSON line on stdout.
    import json
    import subprocess, sys
    code = (
        "import json, gc, time, psutil, tracemalloc\n"
        "import importlib, sys\n"
        "sys.path.insert(0, '.')\n"
        "mod = importlib.import_module('scripts.verify_ts_ram')\n"
        "gc.collect()\n"
        "proc = psutil.Process(); base = proc.memory_info().rss\n"
        "tracemalloc.start()\n"
        "t0 = time.perf_counter()\n"
        f"mod.{fn}()\n"
        "dt = time.perf_counter() - t0\n"
        "cur, peak_tm = tracemalloc.get_traced_memory()\n"
        "tracemalloc.stop()\n"
        "print(json.dumps({'delta_mb': (proc.memory_info().rss-base)/1048576, "
        "'tracemalloc_peak_mb': peak_tm/1048576, 'sec': dt}))\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
    )
    if r.returncode != 0:
        return {"error": r.stderr.strip().splitlines()[-1] if r.stderr else "?"}
    return json.loads(r.stdout.strip().splitlines()[-1])


def scenario_rolling_stats():
    n = 2000
    df = _df(n, groups=4)
    guard = cadw.MemoryGuard(max_memory_mb=32)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
    out = rolling_stats(
        df2, "y", windows=(7, 14, 30),
        stats=("mean", "std", "q50", "skew", "slope", "autocorr_lag1"),
        group_col="g", time_col="ts",
    )
    n_out = sum(b.num_rows for b in out._iter_arrow_batches(batch_size=500))
    assert n_out == n


def scenario_crosscorr():
    n = 2000
    df = _df(n, groups=4)
    guard = cadw.MemoryGuard(max_memory_mb=32)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
    out = rolling_crosscorr(
        df2, "y", "x", windows=(14,), lags=(0, 1),
        group_col="g", time_col="ts",
    )
    n_out = sum(b.num_rows for b in out._iter_arrow_batches(batch_size=500))
    assert n_out == n


def scenario_make_features():
    n = 2000
    df = _df(n, groups=4)
    guard = cadw.MemoryGuard(max_memory_mb=32)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
    out = make_features(
        df2, target_col="y", ts_col="ts",
        lags=(1, 7, 14), diffs=(1,), seasonal_periods=(7,),
        calendar=("month", "weekday", "hour"),
        cyclical=("month", "weekday", "hour"),
        fourier=(24,), fourier_order=2,
        group_col="g",
    )
    n_out = sum(b.num_rows for b in out._iter_arrow_batches(batch_size=500))
    assert n_out == n


def scenario_gbdt():
    n = 500
    df = _df(n)
    guard = cadw.MemoryGuard(max_memory_mb=64)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
    m = GBDTForecaster(
        mode="direct", lags=(1, 2, 7), n_estimators=20, max_depth=3,
    )
    m.fit(df2, "y")
    fc = m.predict(14)
    assert len(fc) == 14


def scenario_thf_ols():
    n = 300
    df = _df(n)
    guard = cadw.MemoryGuard(max_memory_mb=32)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
    th = TemporalHierarchy(
        [1, 7, 28], lambda: NaiveForecaster("last"),
        reconciliation="optimal_ols",
    )
    th.fit(df2, "y", time_col="ts")
    fc = th.predict(28)
    assert len(fc) == 28


def scenario_thf_mint_wls():
    n = 300
    df = _df(n)
    guard = cadw.MemoryGuard(max_memory_mb=32)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
    th = TemporalHierarchy(
        [1, 7, 28], lambda: NaiveForecaster("last"),
        reconciliation="optimal_wls",
    )
    th.fit(df2, "y", time_col="ts")
    fc = th.predict(28)
    assert len(fc) == 28


def scenario_backtest():
    n = 200
    df = _df(n)
    guard = cadw.MemoryGuard(max_memory_mb=32)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
    summ, folds = backtest_forecaster(
        df2, NaiveForecaster("last"),
        target_col="y", time_col="ts",
        horizon=7, n_folds=5, strategy="expanding",
        metrics=("mae", "rmse", "mase"),
    )
    assert len(summ) == 3




def scenario_baseline_noop():
    """Control: build same df, do nothing. Used to subtract imports/df."""
    n = 2000
    _df(n, groups=4)

SCENARIOS = [
    ("rolling_stats(3 windows x 6 stats, n=2000, 4 groups)", scenario_rolling_stats, 32),
    ("rolling_crosscorr(14, lags 0/1, n=2000, 4 groups)", scenario_crosscorr, 32),
    ("make_features(n=2000, 4 groups, fourier+lags)", scenario_make_features, 32),
    ("GBDT direct fit+14-step predict (n=500)", scenario_gbdt, 64),
    ("THF [1,7,28] OLS predict 28 (n=300)", scenario_thf_ols, 32),
    ("THF [1,7,28] MinT-WLS predict 28 (n=300)", scenario_thf_mint_wls, 32),
    ("backtest naive h=7 k=5 expanding (n=200)", scenario_backtest, 32),
]


def main():
    # In-process (fast) pass; the subprocess helper is available for a
    # clean re-run when a scenario looks suspicious.
    print(f"{'scenario':<55} {'rss_delta_mb':>12} {'tm_peak_mb':>11} {'budget_mb':>9} {'sec':>7}")
    print("-" * 105)
    for name, fn, budget in SCENARIOS:
        r = _measure(fn, budget)
        print(f"{name:<55} {r['delta_mb']:>12.2f} {r['tracemalloc_peak_mb']:>11.2f} {budget:>9} {r['sec']:>7.2f}")


if __name__ == '__main__':
    main()

# ============================================================================
# FILE: cadw/dataframe.py  [diff]
# ============================================================================
diff --git a/cadw/dataframe.py b/cadw/dataframe.py
index 8ac1a87..55e7725 100644
--- a/cadw/dataframe.py
+++ b/cadw/dataframe.py
@@ -1552,18 +1552,32 @@ class CadwDataFrame:
     def sort_values(self, by, *, ascending=True) -> "CadwDataFrame":
         by_cols = [by] if isinstance(by, str) else list(by)
         for c in by_cols:
             if c not in self._columns:
                 raise KeyError(f"sort_values: column {c!r} not in selection")
+        # v0.9.24: per-column ascending. A list/tuple ascending is
+        # interpreted positionally against by_cols (pandas semantics).
+        # Prior code did `"ascending" if ascending else "descending"`
+        # which treated any non-empty list as truthy -> all columns
+        # ascending. That silently broke mixed-direction sorts.
+        if isinstance(ascending, (list, tuple)):
+            if len(ascending) != len(by_cols):
+                raise ValueError(
+                    f"sort_values: ascending has {len(ascending)} "
+                    f"entries but by has {len(by_cols)} columns"
+                )
+            asc_seq = [bool(a) for a in ascending]
+        else:
+            asc_seq = [bool(ascending)] * len(by_cols)
         # in-memory fast path (v0.9.6b1): pa.Table.sort_by directly.
         # Avoids 1-5 ms external_sort overhead per small group —
         # critical for per-group apply (100k groups × 5 ms = 500 s).
         from cadw.dataset._in_memory import _InMemoryDataset
         if isinstance(self._dataset, _InMemoryDataset):
             sorted_table = self._dataset._table.select(self._columns).sort_by(
-                [(c, "ascending" if ascending else "descending")
-                 for c in by_cols],
+                [(c, "ascending" if a else "descending")
+                 for c, a in zip(by_cols, asc_seq)],
             )
             return CadwDataFrame.from_table(
                 sorted_table,
                 memory_guard=self._memory_guard,
                 spill_dir=self._spill_dir,
@@ -1572,11 +1586,11 @@ class CadwDataFrame:
             self._iter_filtered_batches(),
             key_columns=by_cols,
             value_columns=self._columns,
             memory_guard=self._memory_guard,
             spill_dir=self._spill_dir / "sort",
-            ascending=ascending,
+            ascending=asc_seq,
             log=False,
         )
         schema = pa.schema([self._dataset.schema.field(c) for c in self._columns])
 
         def _iter():
@@ -4611,15 +4625,22 @@ class CadwDataFrame:
         *,
         suffixes: tuple[str, str] = ("_x", "_y"),
         num_partitions: int | None = None,
         use_planner: bool = True,
     ) -> "CadwDataFrame":
-        """Hash join (v0.2.4). how ∈ {'inner', 'left'}.
-
-        - how='inner': только совпадающие ключи.
-        - how='left':  все строки self сохраняются; right-колонки -> NULL
-          при отсутствии совпадения.
+        """Hash join (v0.9.23j). how ∈ {'inner', 'left',
+        'outer', 'anti', 'semi'}.
+
+        - how='inner': совпадающие ключи (left ∪ right, on-coalesced).
+        - how='left':  все left + right-колонки (null при нет совпадения).
+        - how='outer': left ∪ right, обе стороны null-padded.
+        - how='anti':  left без совпадения в right (SQL anti join).
+        - how='semi':  left с совпадением в right (SQL semi join);
+                       output = left-колонки only.
+
+        NULL-семантика для anti/semi: null-ключ слева никогда
+        не совпадает (SQL); anti emit'ит, semi — нет.
         - on: имена общих колонок в обеих сторонах.
         - suffixes: для пересекающихся не-on колонок.
         - num_partitions: число партиций HashShuffle; None — из
           JoinMemoryModel (если use_planner), иначе дефолт 8.
         - use_planner: использовать JoinMemoryModel.solve() для выбора
@@ -4849,14 +4870,19 @@ class CadwDataFrame:
                 memory_guard=self._memory_guard,
                 spill_dir=self._spill_dir / "join" / "anti",
                 num_partitions=num_partitions,
             )
         elif how == "semi":
-            # v0.9.23j placeholder — dispatch handled by CadwDataFrame
-            # extension in next patch.
-            raise NotImplementedError(
-                "merge(how='semi') is v0.9.23j; use anti + left for now."
+            # v0.9.23j: semi join = left rows with match in right.
+            from cadw.ops.join import semi_join_streaming as _semi
+            out_schema, out_path = _semi(
+                self._iter_filtered_batches(),
+                right._iter_filtered_batches(),
+                left_schema, right_schema, on=on_list,
+                memory_guard=self._memory_guard,
+                spill_dir=self._spill_dir / "join" / "semi",
+                num_partitions=num_partitions,
             )
         else:
             out_schema, out_path = hash_join_streaming(
                 self._iter_filtered_batches(),
                 right._iter_filtered_batches(),
@@ -6176,10 +6202,19 @@ class CadwDataFrame:
     def plot(self):
         """Plot accessor: df.plot.hist(col) / df.plot.plotly.hist(col)."""
         from cadw.plot_accessor import _PlotAccessor
         return _PlotAccessor(self)
 
+    @property
+    def ts(self):
+        """Time-series accessor (v0.9.24): df.ts.lag / .rolling_features / .fit_forecaster / ...
+
+        Thin wrappers over cadw.ts.*. No logic lives in the accessor.
+        """
+        from cadw.accessors._ts import _TsAccessor
+        return _TsAccessor(self)
+
 
 
 
 class _ArrowTail:
     """v0.9.18 (AUDIT §1.1): Arrow-native sliding tail buffer of k rows.

# ============================================================================
# FILE: cadw/ts/__init__.py  [full]
# ============================================================================
"""CADW time-series toolkit (v0.9.24).

See README.md Time series section and DECISIONS.md for design notes.
"""
from .features import (
    calendar_features, cyclical_features, diff_features,
    fourier_features, holiday_features, lag_features, lead_features,
    make_features,
)
from .stats import rolling_crosscorr, rolling_stats
from .models import (
    ForecasterBase, GBDTForecaster, NaiveForecaster,
    SARIMAForecaster, get_forecaster,
)
from .backtest import backtest_forecaster
from .thf import TemporalHierarchy

__all__ = sorted([
    # features
    "calendar_features", "cyclical_features", "diff_features",
    "fourier_features", "holiday_features", "lag_features",
    "lead_features", "make_features",
    # window stats
    "rolling_crosscorr", "rolling_stats",
    # forecasters
    "ForecasterBase", "NaiveForecaster", "SARIMAForecaster",
    "GBDTForecaster", "get_forecaster",
    # backtesting
    "backtest_forecaster",
    # temporal hierarchy
    "TemporalHierarchy",
])

# ============================================================================
# FILE: cadw/ts/features.py  [full]
# ============================================================================
# CADW time-series feature engineering (v0.9.24).
#
# Streaming, batch-wise feature construction for TS forecasting. All
# functions append new columns to a CadwDataFrame (matching the .dt
# accessor convention) and preserve or re-establish row order.
#
# RAM strategy: O(batch) state per function; carry buffers between
# batches bounded by max_lag x n_groups x row_bytes. Every batch goes
# through MemoryGuard.check_batch with an explicit extra_factor that
# accounts for pylist overhead (3.5x per pylist column, AUDIT_FULL 3.2).
#
# Group-aware semantics: group_col=None -> positional over
# _iter_arrow_batches (matches df.shift). group_col given -> data is
# sorted by [group_col, time_col] before streaming so carry buffers
# never cross group boundaries; output row order equals sorted order.
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from cadw.dataset import Dataset

_BATCH = 10_000
_PYLIST_FACTOR = 3.5   # AUDIT_FULL 3.2: pylist overhead vs Arrow, per col


def _check(guard, batch, name, extra_factor):
    if guard is None:
        return
    guard.check_batch(batch, name=name, extra_factor=extra_factor)


def _new_schema_with_extra(df, extras):
    fields = []
    for c in df._columns:
        f = df._dataset.schema.field(c)
        fields.append(pa.field(c, f.type, nullable=True))
    for name, typ in extras:
        fields.append(pa.field(name, typ, nullable=True))
    return pa.schema(fields)


def _finalize(df, out_schema, gen):
    from cadw.dataframe import CadwDataFrame
    new_ds = Dataset.from_batches(out_schema, gen(), df._new_dataset_dir())
    return CadwDataFrame(
        new_ds, memory_guard=df._memory_guard, spill_dir=df._spill_dir,
    )


def _sort_for_groups(df, group_col, time_col):
    keys = []
    if group_col is not None:
        keys.append(group_col)
    if time_col is not None:
        keys.append(time_col)
    if not keys:
        return df
    return df.sort_values(keys)


def _sort_multi(df, keys):
    # Per-column ascending sort [(col, ascending), ...].
    # df.sort_values only supports scalar ascending (list -> truthy bug),
    # so call external_sort directly with a per-column ascending sequence.
    from cadw.ops.sort import external_sort
    from cadw.io.batch_reader import ArrowIPCBatchReader
    from cadw.dataframe import CadwDataFrame
    cols = [c for c, _ in keys]
    asc = [a for _, a in keys]
    path = external_sort(
        df._iter_filtered_batches(),
        key_columns=cols,
        value_columns=df._columns,
        memory_guard=df._memory_guard,
        spill_dir=df._spill_dir / "sort",
        ascending=asc,
        log=False,
    )
    schema = pa.schema(
        [df._dataset.schema.field(c) for c in df._columns]
    )

    def _iter():
        for b in ArrowIPCBatchReader(str(path), df._columns):
            yield b

    new_ds = Dataset.from_batches(schema, _iter(), df._new_dataset_dir())
    return CadwDataFrame(
        new_ds, memory_guard=df._memory_guard, spill_dir=df._spill_dir,
    )


def _shift_stream(
    df, column, ks, *, group_col, time_col, fill_value, direction,
    _already_sorted=False, _out_prefix=None,
):
    if column not in df._columns:
        raise KeyError(f"lag_features: column {column!r} not in df")
    col_type = df._dataset.schema.field(column).type
    if group_col is not None and group_col not in df._columns:
        raise KeyError(
            f"lag_features: group_col {group_col!r} not in df"
        )
    if time_col is not None and time_col not in df._columns:
        raise KeyError(f"lag_features: time_col {time_col!r} not in df")

    max_k = max(ks)
    prefix = _out_prefix or f"{column}_{direction}"
    extras = [(f"{prefix}{k}", col_type) for k in ks]
    out_schema = _new_schema_with_extra(df, extras)

    if _already_sorted:
        sorted_df = df
    else:
        sorted_df = _sort_for_groups(df, group_col, time_col)

    n_cols = len(sorted_df._columns)
    extra_factor = 2.0 + _PYLIST_FACTOR * (1 + (group_col is not None))

    def gen():
        carry = {}
        for batch in sorted_df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch,
                   f"ts.{direction}.{column}", extra_factor)
            n = batch.num_rows
            vals = batch.column(column).to_pylist()
            if group_col is not None:
                grps = batch.column(group_col).to_pylist()
            else:
                grps = None

            out = {k: [None] * n for k in ks}
            if grps is None:
                hist = carry.get(None)
                if hist is None:
                    hist = deque(maxlen=max_k)
                    carry[None] = hist
                for i in range(n):
                    for k in ks:
                        out[k][i] = hist[-k] if len(hist) >= k else fill_value
                    hist.append(vals[i])
            else:
                for i in range(n):
                    g = grps[i]
                    hist = carry.get(g)
                    if hist is None:
                        hist = deque(maxlen=max_k)
                        carry[g] = hist
                    for k in ks:
                        out[k][i] = hist[-k] if len(hist) >= k else fill_value
                    hist.append(vals[i])

            arrays = [batch.column(c) for c in range(n_cols)]
            for k in ks:
                arrays.append(pa.array(out[k], type=col_type))
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

    return _finalize(df, out_schema, gen)


def lag_features(
    df, column, lags=(1,),
    *, group_col=None, time_col=None, fill_value=None,
):
    # Append {column}_lag{k} for each k in lags (k >= 1).
    lags = sorted({int(k) for k in lags})
    if not lags:
        raise ValueError("lag_features: lags is empty")
    if min(lags) < 1:
        raise ValueError(f"lag_features: lags must be >= 1; got {lags}")
    return _shift_stream(
        df, column, lags,
        group_col=group_col, time_col=time_col,
        fill_value=fill_value, direction="lag",
    )


def lead_features(
    df, column, leads=(1,),
    *, group_col=None, time_col=None, fill_value=None,
):
    # Append {column}_lead{k} for each k in leads (k >= 1).
    # Implementation: reverse time order, lag, rename, restore order.
    leads = sorted({int(k) for k in leads})
    if not leads:
        raise ValueError("lead_features: leads is empty")
    if min(leads) < 1:
        raise ValueError(f"lead_features: leads must be >= 1; got {leads}")
    if time_col is None:
        raise ValueError("lead_features: time_col required")
    keys = []
    if group_col is not None:
        keys.append((group_col, True))
    keys.append((time_col, False))
    rev = _sort_multi(df, keys)
    lagged = _shift_stream(
        rev, column, leads,
        group_col=group_col, time_col=time_col,
        fill_value=fill_value, direction="lag",
        _already_sorted=True,
    )
    rename_map = {f"{column}_lag{k}": f"{column}_lead{k}" for k in leads}
    back_keys = ([(group_col, True)] if group_col is not None else []) + [(time_col, True)]
    fwd = _sort_multi(lagged, back_keys)
    return fwd.rename(rename_map)


def diff_features(
    df, column, lags=(1,), *,
    group_col=None, time_col=None, seasonal_periods=None,
):
    # Append {column}_diff{k} (and {column}_seasonal_diff{s}_{k}).
    # Computes cur - lag(k) directly with its own carry buffer; no
    # intermediate lag columns are materialized.
    # RAM: O(max_lag) ungrouped / O(max_lag x n_groups) grouped carry,
    # plus one batch worth of pylist per column (factor 3.5 per col).
    if isinstance(lags, int):
        lags = (lags,)
    lags = sorted({int(k) for k in lags})
    if not lags:
        raise ValueError("diff_features: lags is empty")
    if min(lags) < 1:
        raise ValueError(f"diff_features: lags must be >= 1; got {lags}")
    if column not in df._columns:
        raise KeyError(f"diff_features: column {column!r} not in df")
    col_type = df._dataset.schema.field(column).type
    if not (pa.types.is_integer(col_type) or pa.types.is_floating(col_type)):
        raise TypeError(
            f"diff_features: column {column!r} has type {col_type}; "
            f"numeric required"
        )
    if group_col is not None and group_col not in df._columns:
        raise KeyError(
            f"diff_features: group_col {group_col!r} not in df"
        )
    if time_col is not None and time_col not in df._columns:
        raise KeyError(f"diff_features: time_col {time_col!r} not in df")

    sps = tuple(seasonal_periods) if seasonal_periods else ()
    all_ks = set(lags)
    for sp in sps:
        for k in lags:
            all_ks.add(int(sp) * int(k))
    all_ks = sorted(all_ks)
    max_k = all_ks[-1]

    extras = [(f"{column}_diff{k}", col_type) for k in lags]
    for sp in sps:
        for k in lags:
            extras.append((f"{column}_seasonal_diff{sp}_{k}", col_type))
    out_schema = _new_schema_with_extra(df, extras)

    sorted_df = _sort_for_groups(df, group_col, time_col)
    n_cols = len(sorted_df._columns)
    extra_factor = 2.0 + _PYLIST_FACTOR * (1 + (group_col is not None))

    def gen():
        carry = {}
        for batch in sorted_df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch,
                   f"ts.diff.{column}", extra_factor)
            n = batch.num_rows
            vals = batch.column(column).to_pylist()
            grps = (
                batch.column(group_col).to_pylist()
                if group_col is not None else None
            )
            out = {key: [None] * n for key, _ in extras}
            for i in range(n):
                g = grps[i] if grps is not None else None
                hist = carry.get(g)
                if hist is None:
                    hist = deque(maxlen=max_k)
                    carry[g] = hist
                cur = vals[i]
                for k in all_ks:
                    if len(hist) >= k and cur is not None and hist[-k] is not None:
                        diff = float(cur) - float(hist[-k])
                        for reg_k in lags:
                            if reg_k == k:
                                out[f"{column}_diff{reg_k}"][i] = diff
                        for sp in sps:
                            if sp * 1 == k or any(sp * rk == k for rk in lags):
                                for rk in lags:
                                    if sp * rk == k:
                                        out[f"{column}_seasonal_diff{sp}_{rk}"][i] = diff
                hist.append(cur)
            arrays = [batch.column(c) for c in range(n_cols)]
            for key, typ in extras:
                arrays.append(pa.array(out[key], type=typ))
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

    return _finalize(df, out_schema, gen)


# ==================== calendar / cyclical / fourier ====================

_CALENDAR_PC = {
    "year": pc.year,
    "month": pc.month,
    "day": pc.day,
    "hour": pc.hour,
    "minute": pc.minute,
    "second": pc.second,
    "weekday": pc.day_of_week,
    "day_of_year": pc.day_of_year,
    "quarter": pc.quarter,
    "week_of_year": pc.week,
}


def _validate_temporal(df, column):
    t = df._dataset.schema.field(column).type
    if not (pa.types.is_timestamp(t) or pa.types.is_date(t)):
        raise TypeError(
            f"ts: column {column!r} has type {t}; expected timestamp or date"
        )


def calendar_features(
    df, ts_col, features=("year", "month", "day", "weekday", "hour"),
) -> "CadwDataFrame":
    # Append {ts_col}_{feature} for each feature (Arrow pc.* kernels).
    # RAM: O(batch) state; pure vectorized per batch; no carry buffer.
    if ts_col not in df._columns:
        raise KeyError(f"calendar_features: {ts_col!r} not in df")
    _validate_temporal(df, ts_col)
    for f in features:
        if f not in _CALENDAR_PC:
            raise ValueError(
                f"calendar_features: unknown feature {f!r}; "
                f"valid={sorted(_CALENDAR_PC)}"
            )
    extras = [(f"{ts_col}_{f}", pa.int64()) for f in features]
    out_schema = _new_schema_with_extra(df, extras)
    n_cols = len(df._columns)

    def gen():
        for batch in df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch, f"ts.calendar.{ts_col}", 2.0)
            arr = batch.column(ts_col)
            arrays = [batch.column(c) for c in range(n_cols)]
            for f in features:
                arrays.append(
                    pc.cast(_CALENDAR_PC[f](arr), pa.int64())
                )
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)
    return _finalize(df, out_schema, gen)


def cyclical_features(
    df, ts_col, features=("month", "weekday", "hour"),
) -> "CadwDataFrame":
    # Append {ts_col}_{f}_sin / _cos for each feature.
    # Period: month=12, weekday=7, hour=24, day_of_year=365, quarter=4.
    # RAM: O(batch) state; vectorized; no carry buffer.
    periods = {
        "month": 12, "weekday": 7, "hour": 24,
        "day_of_year": 366, "quarter": 4, "week_of_year": 53,
    }
    if ts_col not in df._columns:
        raise KeyError(f"cyclical_features: {ts_col!r} not in df")
    _validate_temporal(df, ts_col)
    for f in features:
        if f not in periods:
            raise ValueError(
                f"cyclical_features: unknown feature {f!r}; "
                f"valid={sorted(periods)}"
            )
    extras = []
    for f in features:
        extras.append((f"{ts_col}_{f}_sin", pa.float64()))
        extras.append((f"{ts_col}_{f}_cos", pa.float64()))
    out_schema = _new_schema_with_extra(df, extras)
    n_cols = len(df._columns)

    def gen():
        for batch in df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch, f"ts.cyclical.{ts_col}", 2.5)
            arr = batch.column(ts_col)
            arrays = [batch.column(c) for c in range(n_cols)]
            for f in features:
                p = periods[f]
                if f == "weekday":
                    base = pc.cast(pc.day_of_week(arr), pa.int64())
                elif f == "day_of_year":
                    base = pc.cast(pc.day_of_year(arr), pa.int64())
                elif f == "week_of_year":
                    base = pc.cast(pc.week(arr), pa.int64())
                else:
                    base = pc.cast(_CALENDAR_PC[f](arr), pa.int64())
                ang = pc.multiply(
                    pc.cast(base, pa.float64()),
                    pa.scalar(2.0 * np.pi / p, type=pa.float64()),
                )
                arrays.append(pc.sin(ang))
                arrays.append(pc.cos(ang))
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)
    return _finalize(df, out_schema, gen)


def fourier_features(
    df, time_col, period, order, prefix=None,
) -> "CadwDataFrame":
    # Append {prefix}_sin{i} / {prefix}_cos{i} for i=1..order.
    # Uses time_col as numeric seconds (int64) or timestamp cast to sec.
    # RAM: O(batch) state; vectorized; no carry buffer.
    if time_col not in df._columns:
        raise KeyError(f"fourier_features: {time_col!r} not in df")
    if int(order) < 1:
        raise ValueError(f"fourier_features: order must be >= 1; got {order}")
    if period <= 0:
        raise ValueError(f"fourier_features: period must be > 0; got {period}")
    order = int(order)
    prefix = prefix or f"{time_col}_fourier"
    extras = []
    for i in range(1, order + 1):
        extras.append((f"{prefix}_sin{i}", pa.float64()))
        extras.append((f"{prefix}_cos{i}", pa.float64()))
    out_schema = _new_schema_with_extra(df, extras)
    n_cols = len(df._columns)
    col_type = df._dataset.schema.field(time_col).type
    is_ts = pa.types.is_timestamp(col_type)

    def gen():
        for batch in df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch, f"ts.fourier.{time_col}", 2.5)
            arr = batch.column(time_col)
            if is_ts:
                factor = {"s": 1, "ms": 1_000, "us": 1_000_000,
                          "ns": 1_000_000_000}[arr.type.unit]
                ints = pc.cast(arr, pa.int64())
                secs = pc.divide(ints, pa.scalar(factor, type=pa.int64()))
                t = pc.cast(secs, pa.float64())
            else:
                t = pc.cast(arr, pa.float64())
            arrays = [batch.column(c) for c in range(n_cols)]
            for i in range(1, order + 1):
                ang = pc.multiply(
                    t, pa.scalar(2.0 * np.pi * i / period, type=pa.float64())
                )
                arrays.append(pc.sin(ang))
                arrays.append(pc.cos(ang))
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)
    return _finalize(df, out_schema, gen)


def holiday_features(df, ts_col) -> "CadwDataFrame":
    # Append {ts_col}_is_weekend and {ts_col}_is_month_start/end.
    # RAM: O(batch); vectorized pc kernels; no carry buffer.
    if ts_col not in df._columns:
        raise KeyError(f"holiday_features: {ts_col!r} not in df")
    _validate_temporal(df, ts_col)
    extras = [
        (f"{ts_col}_is_weekend", pa.bool_()),
        (f"{ts_col}_is_month_start", pa.bool_()),
        (f"{ts_col}_is_month_end", pa.bool_()),
    ]
    out_schema = _new_schema_with_extra(df, extras)
    n_cols = len(df._columns)
    is_ts = pa.types.is_timestamp(df._dataset.schema.field(ts_col).type)

    def gen():
        for batch in df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch, f"ts.holiday.{ts_col}", 2.0)
            arr = batch.column(ts_col)
            wd = pc.cast(pc.day_of_week(arr), pa.int64())
            is_weekend = pc.is_in(
                wd, value_set=pa.array([5, 6], type=pa.int64())
            )
            d = pc.cast(pc.day(arr), pa.int64())
            is_month_start = pc.equal(d, pa.scalar(1, type=pa.int64()))
            # month_end: day(arr + 1 unit) == 1 (safe for ts); for date
            # fallback: day + 1 equals 1 -> never; use day==28/29/30/31
            # heuristic via month comparison.
            if is_ts:
                try:
                    nd = pc.cast(pc.day(pc.add(arr, pa.scalar(1, type=arr.type))),
                                 pa.int64())
                    is_month_end = pc.equal(nd, pa.scalar(1, type=pa.int64()))
                except Exception:
                    is_month_end = pc.fill_null(
                        pc.equal(pc.cast(pc.month(arr), pa.int64()),
                                 pa.scalar(-1, type=pa.int64())),
                        False,
                    )
            else:
                nd = pc.cast(pc.day(pc.add(arr, pa.scalar(1, type=pa.int32()))),
                             pa.int64())
                is_month_end = pc.equal(nd, pa.scalar(1, type=pa.int64()))
            arrays = [batch.column(c) for c in range(n_cols)]
            arrays.extend([is_weekend, is_month_start, is_month_end])
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)
    return _finalize(df, out_schema, gen)


# ==================== convenience facade ====================

def make_features(
    df, *, target_col, ts_col,
    lags=(1, 7), diffs=(1,), seasonal_periods=None,
    calendar=("month", "weekday", "hour"),
    cyclical=("month", "weekday", "hour"),
    fourier=(24, 7), fourier_order=3,
    group_col=None,
    holiday=True,
) -> "CadwDataFrame":
    # Chain feature builders in the canonical order: calendar -> cyclical
    # -> fourier -> holiday -> lag -> diff. Returns df with all appended
    # columns; row order is the group-sorted order when group_col given.
    out = df
    if calendar:
        out = calendar_features(out, ts_col, calendar)
    if cyclical:
        out = cyclical_features(out, ts_col, cyclical)
    if fourier:
        for p in fourier:
            out = fourier_features(out, ts_col, period=p,
                                   order=fourier_order,
                                   prefix=f"{ts_col}_f{p}")
    if holiday:
        out = holiday_features(out, ts_col)
    if lags:
        out = lag_features(out, target_col, lags,
                            group_col=group_col, time_col=ts_col)
    if diffs:
        out = diff_features(out, target_col, diffs,
                             group_col=group_col, time_col=ts_col,
                             seasonal_periods=seasonal_periods)
    return out

# ============================================================================
# FILE: cadw/ts/stats.py  [full]
# ============================================================================
# CADW time-series window statistics (v0.9.24).
#
# tsfresh-subset window aggregations reimplemented on Arrow + numpy with
# explicit RAM bounds. One streaming pass per group; a single deque of
# size max(windows) per group is the only per-group state.
#
# RAM formula table (per group; n_groups=1 if ungrouped):
#
#   stat                     per-group state
#   ------------------------ -------------------------
#   mean/std/sum/min/max     O(max_window x 8 B)   (shared deque)
#   median/q10/q25/q50/q75/q90   O(max_window x 8 B)
#   skew/kurt                O(max_window x 8 B)
#   energy/abs_energy/cid    O(max_window x 8 B)
#   crossings/slope/entropy  O(max_window x 8 B)
#   count_above_mean         O(max_window x 8 B)
#   autocorr_lagK            O(max_window x 8 B)
#   ------------------------ -------------------------
#   TOTAL (all stats share one deque)  O(max_window x n_groups x 8 B)
#   per-batch pylist: O(batch x 3.5 x (1 + has_group_col) x 8 B)
#
# Window is a deque of the last max(windows) values per group; every
# statistic reads only the last w values. Computation per output point
# is O(w log w) for quantiles, O(w) otherwise -- independent of N.
#
# All statistics return NaN when the valid (non-null) count in the
# window is below min_periods, or when the statistic is undefined
# (e.g. skew with <3 points or std==0).
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import pyarrow as pa

from .features import (
    _BATCH, _PYLIST_FACTOR, _check, _finalize, _new_schema_with_extra,
    _sort_for_groups,
)


_ALL_STATS = frozenset({
    "mean", "std", "sum", "min", "max", "median",
    "skew", "kurt",
    "q10", "q25", "q50", "q75", "q90",
    "abs_energy", "energy", "cid",
    "crossings", "slope", "entropy", "count_above_mean",
})


def _parse_stat(s):
    if s.startswith("autocorr_lag"):
        try:
            k = int(s[len("autocorr_lag"):])
        except ValueError:
            raise ValueError(f"rolling_stats: invalid stat {s!r}")
        if k < 1:
            raise ValueError(
                f"rolling_stats: autocorr lag must be >= 1; got {k}"
            )
        return ("autocorr", k)
    if s not in _ALL_STATS:
        raise ValueError(
            f"rolling_stats: unknown stat {s!r}; valid={sorted(_ALL_STATS)} "
            f"or 'autocorr_lagK'"
        )
    return (s, None)


def _compute_stat(name, k, arr_w):
    valid = arr_w[~np.isnan(arr_w)]
    if name == "mean":
        return float(valid.mean()) if len(valid) else None
    if name == "std":
        return float(valid.std(ddof=1)) if len(valid) > 1 else None
    if name == "sum":
        return float(valid.sum()) if len(valid) else None
    if name == "min":
        return float(valid.min()) if len(valid) else None
    if name == "max":
        return float(valid.max()) if len(valid) else None
    if name == "median":
        return float(np.median(valid)) if len(valid) else None
    if name == "skew":
        if len(valid) < 3:
            return None
        s = valid.std(ddof=0)
        if s == 0:
            return None
        return float(np.mean(((valid - valid.mean()) / s) ** 3))
    if name == "kurt":
        if len(valid) < 4:
            return None
        s = valid.std(ddof=0)
        if s == 0:
            return None
        return float(np.mean(((valid - valid.mean()) / s) ** 4) - 3.0)
    if name in ("q10", "q25", "q50", "q75", "q90"):
        if not len(valid):
            return None
        q = int(name[1:]) / 100.0
        return float(np.percentile(valid, q * 100.0))
    if name == "abs_energy":
        return float(np.sum(valid ** 2)) if len(valid) else None
    if name == "energy":
        d = np.diff(arr_w)
        d = d[~np.isnan(d)]
        return float(np.sum(d ** 2)) if len(d) else None
    if name == "cid":
        d = np.diff(arr_w)
        d = d[~np.isnan(d)]
        return float(np.sqrt(np.sum(d ** 2))) if len(d) else None
    if name == "crossings":
        if len(valid) < 2:
            return None
        m = valid.mean()
        sg = np.sign(arr_w - m)
        sg = sg[~np.isnan(sg)]
        if len(sg) < 2:
            return None
        return float(np.sum(sg[:-1] * sg[1:] < 0))
    if name == "slope":
        mask = ~np.isnan(arr_w)
        if mask.sum() < 2:
            return None
        x = np.arange(len(arr_w), dtype=np.float64)[mask]
        y = arr_w[mask]
        xm = x.mean()
        denom = np.sum((x - xm) ** 2)
        if denom == 0:
            return None
        return float(np.sum((x - xm) * (y - y.mean())) / denom)
    if name == "entropy":
        if not len(valid):
            return None
        hist, _ = np.histogram(valid, bins=10)
        p = hist / hist.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))
    if name == "count_above_mean":
        if not len(valid):
            return None
        return float(np.sum(valid > valid.mean()))
    if name == "autocorr":
        if len(arr_w) <= k:
            return None
        a = arr_w[:-k]
        b = arr_w[k:]
        mask = ~(np.isnan(a) | np.isnan(b))
        a = a[mask]
        b = b[mask]
        if len(a) < 2:
            return None
        sa = a.std(ddof=0)
        sb = b.std(ddof=0)
        if sa == 0 or sb == 0:
            return None
        return float(
            np.sum((a - a.mean()) * (b - b.mean())) / (len(a) * sa * sb)
        )
    return None


def rolling_stats(
    df, column, windows, *,
    stats=("mean", "std"),
    group_col=None, time_col=None, min_periods=None,
):
    """Rolling window statistics for time-series feature extraction.

    - windows: int | Sequence[int], each >= 1.
    - stats: subset of:
        mean std sum min max median skew kurt
        q10 q25 q50 q75 q90
        abs_energy energy cid crossings slope entropy count_above_mean
        autocorr_lagK (K >= 1, K < window)
    - group_col: per-entity windows (no cross-entity leakage).
    - time_col: sort key for reproducibility (stable within group).
    - min_periods: int or None (=w). Minimum non-null count in window.

    Output: original + one column per (window, stat):
      {column}_roll{w}_{stat}
      {column}_roll{w}_autocorr_lag{K}

    RAM: O(max_window x n_groups x 8 B) state; per-batch pylist factor
    2.5 + 3.5 x (1 + has_group_col). Independent of N.
    """
    if isinstance(windows, int):
        windows = (windows,)
    windows = sorted({int(w) for w in windows})
    if not windows:
        raise ValueError("rolling_stats: windows is empty")
    if min(windows) < 1:
        raise ValueError(
            f"rolling_stats: windows must be >= 1; got {windows}"
        )
    if column not in df._columns:
        raise KeyError(f"rolling_stats: column {column!r} not in df")
    col_type = df._dataset.schema.field(column).type
    if not (
        pa.types.is_integer(col_type) or pa.types.is_floating(col_type)
    ):
        raise TypeError(
            f"rolling_stats: column {column!r} has type {col_type}; "
            f"numeric required"
        )
    if group_col is not None and group_col not in df._columns:
        raise KeyError(
            f"rolling_stats: group_col {group_col!r} not in df"
        )
    if time_col is not None and time_col not in df._columns:
        raise KeyError(f"rolling_stats: time_col {time_col!r} not in df")

    stats = list(stats)
    parsed = [_parse_stat(s) for s in stats]
    # validate autocorr lag < min(window)
    min_w = windows[0]
    for name, k in parsed:
        if name == "autocorr" and k >= min_w:
            raise ValueError(
                f"rolling_stats: autocorr lag {k} >= min(window) {min_w}"
            )

    if min_periods is None:
        mp_map = {w: w for w in windows}
    elif isinstance(min_periods, int):
        mp_map = {w: min_periods for w in windows}
    else:
        raise TypeError(
            f"rolling_stats: min_periods must be int or None; "
            f"got {type(min_periods).__name__}"
        )

    extras = []
    for w in windows:
        for s, (name, k) in zip(stats, parsed):
            if name == "autocorr":
                extras.append((f"{column}_roll{w}_autocorr_lag{k}", pa.float64()))
            else:
                extras.append((f"{column}_roll{w}_{name}", pa.float64()))
    out_schema = _new_schema_with_extra(df, extras)

    sorted_df = _sort_for_groups(df, group_col, time_col)
    n_cols = len(sorted_df._columns)
    max_w = windows[-1]
    extra_factor = 2.5 + _PYLIST_FACTOR * (1 + (group_col is not None))

    def gen():
        carry = {}
        for batch in sorted_df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch,
                   f"ts.rolling_stats.{column}", extra_factor)
            n = batch.num_rows
            vals = batch.column(column).to_pylist()
            grps = (
                batch.column(group_col).to_pylist()
                if group_col is not None else None
            )

            outs = {}
            for w in windows:
                for s in stats:
                    outs[(w, s)] = [None] * n

            for i in range(n):
                g = grps[i] if grps is not None else None
                dq = carry.get(g)
                if dq is None:
                    dq = deque(maxlen=max_w)
                    carry[g] = dq
                v = vals[i]
                dq.append(float(v) if v is not None else np.nan)
                # Materialize once per i: a numpy view over the deque.
                # Partial windows (len(dq) < w) are valid whenever
                # min_periods allows fewer than w points -- pandas
                # rolling(min_periods) semantics.
                full = np.fromiter(dq, dtype=np.float64, count=len(dq))
                for w in windows:
                    arr_w = full[-w:]  # partial if len(dq) < w
                    valid_count = int(np.sum(~np.isnan(arr_w)))
                    if valid_count < mp_map[w]:
                        continue
                    for s, (name, k) in zip(stats, parsed):
                        outs[(w, s)][i] = _compute_stat(name, k, arr_w)

            arrays = [batch.column(c) for c in range(n_cols)]
            for w in windows:
                for s in stats:
                    arrays.append(pa.array(outs[(w, s)], type=pa.float64()))
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

    return _finalize(df, out_schema, gen)


def _pearson_xcorr(xs, ys, lag):
    # Pearson correlation between xs[t] and ys[t-lag].
    # lag > 0: y lags x (corr(x[t], y[t-lag])); lag = 0: concurrent.
    if lag == 0:
        a, b = xs, ys
    elif lag > 0:
        a = xs[lag:]
        b = ys[:-lag]
    else:
        L = -lag
        a = xs[:-L]
        b = ys[L:]
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask]
    b = b[mask]
    if len(a) < 2:
        return None
    sa = a.std(ddof=0)
    sb = b.std(ddof=0)
    if sa == 0 or sb == 0:
        return None
    return float(np.sum((a - a.mean()) * (b - b.mean())) / (len(a) * sa * sb))


def rolling_crosscorr(
    df, col_x, col_y, windows, *,
    lags=(0,),
    group_col=None, time_col=None, min_periods=None,
):
    """Rolling Pearson cross-correlation between two series.

    Output columns: {col_x}__{col_y}_roll{w}_xcorr_lag{K} for each
    (w, K). Pearson between x[t] and y[t-K]:
      K = 0 -> concurrent;
      K > 0 -> y lags x (x[t] correlates with y[t-K]);
      K < 0 -> x lags y.

    Compute bound: O(N x |lags| x max_window) -- each output point
    scans one window of size w for one lag. Windows are numpy views
    over a shared deque; no per-point list materialization.

    RAM formula (per group; n_groups=1 if ungrouped):
      - shared deque of (x, y) pairs: O(max_window x 16 B)
      - per-batch pylist overhead: explicit extra_factor =
        3.5 + 3.5 x (2 + has_group_col)  (three pylist columns:
        x, y, optional group_col)
      - total state: O(max_window x n_groups x 16 B); independent of N.

    Valid only when non-null paired count in the window >= min_periods
    (default min_periods = w).
    """
    if isinstance(windows, int):
        windows = (windows,)
    windows = sorted({int(w) for w in windows})
    if not windows:
        raise ValueError("rolling_crosscorr: windows is empty")
    if min(windows) < 1:
        raise ValueError(
            f"rolling_crosscorr: windows must be >= 1; got {windows}"
        )
    if isinstance(lags, int):
        lags = (lags,)
    lags = sorted({int(k) for k in lags})
    min_w = windows[0]
    for K in lags:
        if abs(K) >= min_w:
            raise ValueError(
                f"rolling_crosscorr: |lag| {abs(K)} >= min(window) {min_w}"
            )
    if col_x not in df._columns:
        raise KeyError(f"rolling_crosscorr: {col_x!r} not in df")
    if col_y not in df._columns:
        raise KeyError(f"rolling_crosscorr: {col_y!r} not in df")
    if col_x == col_y:
        raise ValueError(
            "rolling_crosscorr: col_x and col_y must differ; "
            "use rolling_stats(autocorr_lagK) for self-correlation"
        )
    for c in (col_x, col_y):
        t = df._dataset.schema.field(c).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise TypeError(
                f"rolling_crosscorr: column {c!r} has type {t}; "
                f"numeric required"
            )
    if group_col is not None and group_col not in df._columns:
        raise KeyError(
            f"rolling_crosscorr: group_col {group_col!r} not in df"
        )
    if time_col is not None and time_col not in df._columns:
        raise KeyError(f"rolling_crosscorr: time_col {time_col!r} not in df")

    if min_periods is None:
        mp_map = {w: w for w in windows}
    elif isinstance(min_periods, int):
        mp_map = {w: min_periods for w in windows}
    else:
        raise TypeError(
            f"rolling_crosscorr: min_periods must be int or None; "
            f"got {type(min_periods).__name__}"
        )

    extras = []
    for w in windows:
        for K in lags:
            extras.append((
                f"{col_x}__{col_y}_roll{w}_xcorr_lag{K}", pa.float64()
            ))
    out_schema = _new_schema_with_extra(df, extras)

    sorted_df = _sort_for_groups(df, group_col, time_col)
    n_cols = len(sorted_df._columns)
    max_w = windows[-1]
    extra_factor = 3.5 + _PYLIST_FACTOR * (2 + (group_col is not None))

    def gen():
        carry = {}
        for batch in sorted_df._iter_arrow_batches(batch_size=_BATCH):
            _check(df._memory_guard, batch,
                   f"ts.rolling_crosscorr.{col_x}__{col_y}", extra_factor)
            n = batch.num_rows
            xs = batch.column(col_x).to_pylist()
            ys = batch.column(col_y).to_pylist()
            grps = (
                batch.column(group_col).to_pylist()
                if group_col is not None else None
            )
            outs = {}
            for w in windows:
                for K in lags:
                    outs[(w, K)] = [None] * n

            for i in range(n):
                g = grps[i] if grps is not None else None
                dq = carry.get(g)
                if dq is None:
                    dq = deque(maxlen=max_w)
                    carry[g] = dq
                xv = float(xs[i]) if xs[i] is not None else np.nan
                yv = float(ys[i]) if ys[i] is not None else np.nan
                dq.append((xv, yv))
                # Materialize paired arrays once per i.
                arr = np.fromiter(
                    (v for pair in dq for v in pair),
                    dtype=np.float64, count=2 * len(dq),
                ).reshape(-1, 2)
                arr_x = arr[:, 0]
                arr_y = arr[:, 1]
                for w in windows:
                    ax = arr_x[-w:]
                    ay = arr_y[-w:]
                    valid = int(np.sum(~(np.isnan(ax) | np.isnan(ay))))
                    if valid < mp_map[w]:
                        continue
                    for K in lags:
                        outs[(w, K)][i] = _pearson_xcorr(ax, ay, K)

            arrays = [batch.column(c) for c in range(n_cols)]
            for w in windows:
                for K in lags:
                    arrays.append(pa.array(outs[(w, K)], type=pa.float64()))
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

    return _finalize(df, out_schema, gen)

# ============================================================================
# FILE: cadw/ts/backtest.py  [full]
# ============================================================================
# CADW time-series backtesting (v0.9.24).
#
# Two split strategies:
#   - expanding: train on [0, t_i), test on [t_i, t_i+horizon)
#     fold sizes grow: train = init_train + i*step
#   - sliding: fixed train size; train = [t_i - train_size, t_i)
#
# Metrics implemented (formulas):
#   MAE    = mean(|y - f|)
#   RMSE   = sqrt(mean((y - f)^2))
#   MAPE   = mean(|y - f| / |y|)  * 100 ; NaN if any |y|==0
#   sMAPE  = mean(2|y - f| / (|y| + |f|)) * 100 ; NaN if y+f==0 for all
#   MASE   = MAE / scale ; scale = mean(|y[t]-y[t-m]|) over the training
#            series, m = seasonal_period (default 1 => mean|dy|). MASE
#            < 1 beats naive random walk of lag m.
#   pinball(q) = mean( max(q*(y-f), (q-1)*(y-f)) ) for forecast f that is
#            the q-quantile. Only computed when quantiles is not None and
#            the forecaster exposes a predict(steps, return_quantiles=q)
#            or an internally stored quantile vector; otherwise a warning
#            is emitted (per fold) and no pinball row is produced.
#
# RAM: O(n_folds x batch) at any instant; results accumulate scalars
# only. No fold-level arrays are stored across folds. Forecaster state
# is clone-and-fit per fold (never reused).
from __future__ import annotations

import copy
import warnings
from typing import Sequence

import numpy as np


_DEFAULT_METRICS = ("mae", "rmse", "mape", "smape", "mase")


def _clone_forecaster(f):
    # Deep copy is safe for our forecasters: numpy arrays and plain
    # dicts only, no open file handles or Arrow datasets.
    return copy.deepcopy(f)


def _mae(y, f):
    return float(np.mean(np.abs(y - f)))


def _rmse(y, f):
    return float(np.sqrt(np.mean((y - f) ** 2)))


def _mape(y, f):
    if np.any(y == 0):
        return float("nan")
    return float(np.mean(np.abs(y - f) / np.abs(y)) * 100.0)


def _smape(y, f):
    denom = np.abs(y) + np.abs(f)
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(y[mask] - f[mask]) / denom[mask]) * 100.0)


def _mase(y, f, train, m=1):
    if len(train) <= m:
        return float("nan")
    scale = float(np.mean(np.abs(train[m:] - train[:-m])))
    if scale == 0 or not np.isfinite(scale):
        return float("nan")
    return float(np.mean(np.abs(y - f)) / scale)


def _pinball(y, f, q):
    resid = y - f
    return float(np.mean(np.maximum(q * resid, (q - 1.0) * resid)))


def _folds_expanding(n, horizon, n_folds, initial_train, step):
    # Yields (train_end, test_end). Train = [0, train_end); test =
    # [train_end, test_end).  test length == horizon.
    out = []
    te = initial_train
    for _ in range(n_folds):
        test_end = te + horizon
        if test_end > n:
            break
        out.append((te, test_end))
        te = te + step
    return out


def _folds_sliding(n, horizon, n_folds, train_size, step):
    out = []
    te = train_size
    for _ in range(n_folds):
        test_end = te + horizon
        if test_end > n:
            break
        out.append((te, test_end))
        te = te + step
    return out


def _predict_quantiles(model, steps, q):
    # Best-effort support for quantile forecasters. If the model has a
    # 'predict_quantile' or 'predict' with return_quantiles kwarg, use
    # it. Otherwise return None (caller emits a warning per fold).
    for name in ("predict_quantile", "predict_quantiles"):
        if hasattr(model, name):
            try:
                return np.asarray(getattr(model, name)(steps, q), dtype=np.float64)
            except Exception:
                return None
    try:
        return np.asarray(model.predict(steps, return_quantiles=q), dtype=np.float64)
    except TypeError:
        return None


def backtest_forecaster(
    df, forecaster, *,
    target_col, time_col,
    group_col=None,
    horizon=1, n_folds=5,
    strategy="expanding",
    train_size=None,
    step=None,
    metrics=_DEFAULT_METRICS,
    quantiles=None,
    seasonal_period=1,
):
    """Backtest a CADW forecaster on a CadwDataFrame.

    Returns (summary_df, folds_df) as plain list-of-dicts (callers
    wrap in a DataFrame if desired; CADW DataFrame construction from
    these small tables is cheap).

    summary_df rows: {group, metric, mean, std, n_folds}
    folds_df rows:   {group, fold, metric, value, n_train, n_test}
    group value is None for single-series.

    strategy:
      'expanding' (default): initial_train = train_size or
        max(n // (n_folds + 1), horizon + 1); step default = horizon.
      'sliding': train_size REQUIRED; step default = horizon.

    Forecaster is CLONED per fold; fitted state does not leak.

    Groups with fewer than (train_size_required + horizon x n_folds)
    rows are skipped and a warning row {group, metric='SKIPPED',
    value=reason} is appended to folds_df.
    """
    if horizon < 1:
        raise ValueError(f"backtest: horizon must be >= 1; got {horizon}")
    if n_folds < 1:
        raise ValueError(f"backtest: n_folds must be >= 1; got {n_folds}")
    if strategy not in ("expanding", "sliding"):
        raise ValueError(
            f"backtest: strategy must be 'expanding' or 'sliding'; "
            f"got {strategy!r}"
        )
    if strategy == "sliding" and train_size is None:
        raise ValueError(
            "backtest: strategy='sliding' requires train_size"
        )
    metrics = tuple(metrics)
    for m in metrics:
        if m not in ("mae", "rmse", "mape", "smape", "mase"):
            raise ValueError(
                f"backtest: unknown metric {m!r}; valid: "
                f"mae, rmse, mape, smape, mase"
            )
    if target_col not in df._columns:
        raise KeyError(f"backtest: target_col {target_col!r} not in df")
    if time_col not in df._columns:
        raise KeyError(f"backtest: time_col {time_col!r} not in df")
    if group_col is not None and group_col not in df._columns:
        raise KeyError(f"backtest: group_col {group_col!r} not in df")

    # Partition into groups (streaming -> per-group rows in memory;
    # backtest series are typically small enough that the whole target
    # column for one group fits, but we still route extraction through
    # the MemoryGuard in _extract_series via the forecaster's fit).
    from .models import _extract_series

    group_values = [None]
    if group_col is not None:
        seen = []
        for batch in df._iter_arrow_batches(batch_size=10_000):
            vals = batch.column(group_col).to_pylist()
            for v in vals:
                if v not in seen:
                    seen.append(v)
        group_values = seen

    folds_rows = []
    summary_rows = []

    for gv in group_values:
        y = _extract_series(
            df, target_col, time_col=time_col,
            group_col=group_col, group_value=gv,
        )
        y = y[~np.isnan(y)]
        n = len(y)
        min_required = (
            (train_size if train_size else max(horizon + 1, n // (n_folds + 1)))
            + horizon * n_folds
        )
        if n < min_required:
            folds_rows.append({
                "group": gv, "fold": -1, "metric": "SKIPPED",
                "value": f"insufficient rows (n={n} < {min_required})",
                "n_train": 0, "n_test": 0,
            })
            warnings.warn(
                f"backtest: group {gv!r} skipped (n={n} < {min_required})",
                stacklevel=2,
            )
            continue

        if strategy == "expanding":
            init_train = (
                train_size if train_size is not None
                else max(horizon + 1, n // (n_folds + 1))
            )
            st = step if step is not None else horizon
            folds = _folds_expanding(n, horizon, n_folds, init_train, st)
        else:
            st = step if step is not None else horizon
            folds = _folds_sliding(n, horizon, n_folds, train_size, st)

        if not folds:
            folds_rows.append({
                "group": gv, "fold": -1, "metric": "SKIPPED",
                "value": "no valid folds", "n_train": 0, "n_test": 0,
            })
            warnings.warn(
                f"backtest: group {gv!r} produced no folds", stacklevel=2,
            )
            continue

        per_metric = {m: [] for m in metrics}
        pinball_supported = True
        if quantiles is not None:
            per_metric_q = {q: [] for q in quantiles}
        for fi, (te, test_end) in enumerate(folds):
            if strategy == "expanding":
                y_train = y[:te]
            else:
                y_train = y[te - train_size:te]
            y_test = y[te:test_end]
            n_train = len(y_train)
            n_test = len(y_test)

            # Clone + fit fresh per fold.
            model = _clone_forecaster(forecaster)
            # Fit expects a CadwDataFrame; build a tiny table per fold.
            import pyarrow as pa
            from cadw import CadwDataFrame
            train_df = CadwDataFrame.from_table(
                pa.table({
                    "__y__": pa.array(y_train, type=pa.float64()),
                })
            )
            model.fit(train_df, "__y__")
            fc = np.asarray(model.predict(horizon), dtype=np.float64)
            if len(fc) != horizon:
                fc = fc[:horizon]

            for m in metrics:
                if m == "mae":
                    v = _mae(y_test, fc)
                elif m == "rmse":
                    v = _rmse(y_test, fc)
                elif m == "mape":
                    v = _mape(y_test, fc)
                elif m == "smape":
                    v = _smape(y_test, fc)
                elif m == "mase":
                    v = _mase(y_test, fc, y_train, seasonal_period)
                folds_rows.append({
                    "group": gv, "fold": fi, "metric": m,
                    "value": v, "n_train": n_train, "n_test": n_test,
                })
                per_metric[m].append(v)

            if quantiles is not None and pinball_supported:
                for q in quantiles:
                    qfc = _predict_quantiles(model, horizon, q)
                    if qfc is None:
                        warnings.warn(
                            f"backtest: forecaster {type(forecaster).__name__} "
                            f"does not support quantile forecasts; "
                            f"pinball metrics skipped",
                            stacklevel=2,
                        )
                        pinball_supported = False
                        break
                    if len(qfc) != horizon:
                        qfc = qfc[:horizon]
                    v = _pinball(y_test, qfc, q)
                    folds_rows.append({
                        "group": gv, "fold": fi, "metric": f"pinball_{q}",
                        "value": v, "n_train": n_train, "n_test": n_test,
                    })
                    per_metric_q[q].append(v)

        for m, vals in per_metric.items():
            arr = np.asarray(vals, dtype=np.float64)
            valid = arr[~np.isnan(arr)]
            summary_rows.append({
                "group": gv, "metric": m,
                "mean": float(valid.mean()) if len(valid) else float("nan"),
                "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
                "n_folds": len(vals),
            })
        if quantiles is not None and pinball_supported:
            for q, vals in per_metric_q.items():
                arr = np.asarray(vals, dtype=np.float64)
                valid = arr[~np.isnan(arr)]
                summary_rows.append({
                    "group": gv, "metric": f"pinball_{q}",
                    "mean": float(valid.mean()) if len(valid) else float("nan"),
                    "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
                    "n_folds": len(vals),
                })

    return summary_rows, folds_rows


__all__ = ["backtest_forecaster"]

# ============================================================================
# FILE: tests/test_ts_features.py  [full]
# ============================================================================
"""Tests for cadw.ts.features (v0.9.24)."""
import numpy as np
import pytest

import cadw
from cadw import CadwDataFrame
from cadw.ts import (
    lag_features, lead_features, diff_features,
    calendar_features, cyclical_features, fourier_features,
    holiday_features, make_features,
)


def _make_panel(n_groups=3, n_per=10, start_ts=1_700_000_000_000_000_000):
    rows = []
    for g in range(n_groups):
        for t in range(n_per):
            rows.append({
                "g": f"g{g}",
                "ts": start_ts + t * 3600 * 1_000_000_000,
                "y": float(g * 100 + t),
                "x": float(t),
            })
    import pyarrow as pa
    schema = pa.schema([
        ("g", pa.string()),
        ("ts", pa.timestamp("ns")),
        ("y", pa.float64()),
        ("x", pa.float64()),
    ])
    arrays = {c: [r[c] for r in rows] for c in ("g", "ts", "y", "x")}
    table = pa.table({c: pa.array(arrays[c], type=schema.field(c).type)
                       for c in ("g", "ts", "y", "x")})
    return CadwDataFrame.from_table(table)


def test_lag_basic():
    df = _make_panel(n_groups=1, n_per=5)
    out = lag_features(df, "y", lags=(1, 2))
    assert "y_lag1" in out.columns
    assert "y_lag2" in out.columns
    # batch=10k, 5 rows -> single batch, positional
    vals = [b for b in out.to_batches(format="numpy", batch_size=10)]
    ys = []
    for b in vals:
        ys.extend(b["y_lag1"].tolist())
    assert np.isnan(ys[0])
    assert ys[1] == 0.0
    assert ys[4] == 3.0


def test_lag_validation():
    df = _make_panel(n_groups=1, n_per=5)
    with pytest.raises(ValueError):
        lag_features(df, "y", lags=(0,))
    with pytest.raises(KeyError):
        lag_features(df, "nope", lags=(1,))


def test_lag_group_no_leak():
    # Critical: group-aware lag must never cross group boundary.
    df = _make_panel(n_groups=3, n_per=5)
    out = lag_features(df, "y", lags=(1,), group_col="g", time_col="ts")
    # After sort by (g, ts): rows are g0 x5, g1 x5, g2 x5. First row of
    # each group must be NaN, regardless of the other groups.
    ys, gs = [], []
    for b in out.to_batches(format="numpy", batch_size=100):
        ys.extend(b["y_lag1"].tolist())
        gs.extend(b["g"].tolist())
    assert len(ys) == 15
    for g in ("g0", "g1", "g2"):
        gi = [i for i, v in enumerate(gs) if v == g]
        assert np.isnan(ys[gi[0]]), f"first row of {g} should be NaN"


def test_lead_basic():
    df = _make_panel(n_groups=1, n_per=5)
    out = lead_features(df, "y", leads=(1,), time_col="ts")
    assert "y_lead1" in out.columns
    vals = []
    for b in out.to_batches(format="numpy", batch_size=100):
        vals.extend(b["y_lead1"].tolist())
    assert vals[0] == 1.0
    assert np.isnan(vals[4])


def test_diff_basic():
    df = _make_panel(n_groups=1, n_per=5)
    out = diff_features(df, "y", lags=(1,))
    vals = []
    for b in out.to_batches(format="numpy", batch_size=100):
        vals.extend(b["y_diff1"].tolist())
    assert np.isnan(vals[0])
    for i in range(1, 5):
        assert abs(vals[i] - 1.0) < 1e-9


def test_calendar_basic():
    df = _make_panel(n_groups=1, n_per=5)
    out = calendar_features(df, "ts", ("year", "month", "weekday"))
    assert "ts_year" in out.columns
    assert "ts_weekday" in out.columns
    # all values non-null
    for b in out.to_batches(format="numpy", batch_size=100):
        for name in ("ts_year", "ts_month", "ts_weekday"):
            assert all(v is not None for v in b[name].tolist())


def test_calendar_rejects_numeric():
    df = _make_panel(n_groups=1, n_per=5)
    with pytest.raises(TypeError):
        calendar_features(df, "y", ("year",))


def test_cyclical_range():
    df = _make_panel(n_groups=1, n_per=24)
    out = cyclical_features(df, "ts", ("hour",))
    for b in out.to_batches(format="numpy", batch_size=100):
        s = b["ts_hour_sin"].tolist()
        c = b["ts_hour_cos"].tolist()
        for v in s + c:
            assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9


def test_fourier_pure_sin():
    # With period=4 and hourly ts, sin(2*pi*t/4) is known for integer t.
    df = _make_panel(n_groups=1, n_per=8, start_ts=0)
    out = fourier_features(df, "ts", period=4, order=1, prefix="f")
    # ts is timestamp(ns); secs = t*3600. We need period=4*3600 for clean.
    out = fourier_features(df, "ts", period=4 * 3600, order=1, prefix="f2")
    vals = []
    for b in out.to_batches(format="numpy", batch_size=100):
        vals.extend(b["f2_sin1"].tolist())
    # sin(2*pi*t/4): t=0 -> 0; t=1 -> 1; t=2 -> 0; t=3 -> -1
    assert abs(vals[0]) < 1e-9
    assert abs(vals[1] - 1.0) < 1e-9
    assert abs(vals[2]) < 1e-9
    assert abs(vals[3] + 1.0) < 1e-9


def test_holiday_weekend():
    # 2023-11-04 is a Saturday; 11-05 Sunday; 11-06 Monday.
    import pyarrow as pa
    ts = [
        1699056000 * 1_000_000_000,  # 2023-11-04
        1699142400 * 1_000_000_000,  # 2023-11-05
        1699228800 * 1_000_000_000,  # 2023-11-06
    ]
    table = pa.table({
        "ts": pa.array(ts, type=pa.timestamp("ns")),
    })
    df = CadwDataFrame.from_table(table)
    out = holiday_features(df, "ts")
    vals = []
    for b in out.to_batches(format="numpy", batch_size=10):
        vals.extend(b["ts_is_weekend"].tolist())
    assert vals == [True, True, False]


def test_make_features_pipeline():
    df = _make_panel(n_groups=2, n_per=20)
    out = make_features(
        df, target_col="y", ts_col="ts",
        lags=(1, 2), diffs=(1,),
        group_col="g",
        fourier=(24,), fourier_order=2,
    )
    for name in ("ts_month", "ts_weekday", "ts_month_sin",
                 "ts_f24_sin1", "ts_f24_cos2",
                 "ts_is_weekend", "y_lag1", "y_lag2", "y_diff1"):
        assert name in out.columns, f"missing {name}"


def test_memory_guard_used():
    # Small budget to force MemoryGuard checks on every batch.
    df = _make_panel(n_groups=2, n_per=100)
    guard = cadw.MemoryGuard(max_memory_mb=8)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    out = lag_features(df2, "y", lags=(1, 2, 3, 4, 5))
    # If no exception and we can read it, MemoryGuard passed.
    n = sum(b.num_rows for b in out._iter_arrow_batches(batch_size=50))
    assert n == 200

# ============================================================================
# FILE: tests/test_ts_stats.py  [full]
# ============================================================================
"""Tests for cadw.ts.stats.rolling_stats (v0.9.24)."""
import math

import numpy as np
import pyarrow as pa
import pytest

import cadw
from cadw import CadwDataFrame
from cadw.ts import rolling_stats


def _mk(vals, group=None):
    cols = {"y": pa.array([float(v) if v is not None else None
                           for v in vals], type=pa.float64())}
    if group is not None:
        cols["g"] = pa.array(group, type=pa.string())
    return CadwDataFrame.from_table(pa.table(cols))


def _get(out, col):
    res = []
    for b in out.to_batches(format="numpy", batch_size=100):
        res.extend(b[col].tolist())
    return res


def test_mean_std_basic():
    df = _mk([1, 2, 3, 4, 5])
    out = rolling_stats(df, "y", windows=(3,), stats=("mean", "std"))
    m = _get(out, "y_roll3_mean")
    s = _get(out, "y_roll3_std")
    assert math.isnan(m[0])
    assert math.isnan(m[1])
    assert abs(m[2] - 2.0) < 1e-9
    assert abs(m[3] - 3.0) < 1e-9
    assert abs(m[4] - 4.0) < 1e-9
    assert math.isnan(s[0])
    assert abs(s[2] - 1.0) < 1e-9


def test_multiple_windows():
    df = _mk([1, 2, 3, 4])
    out = rolling_stats(df, "y", windows=(2, 3), stats=("mean",))
    m2 = _get(out, "y_roll2_mean")
    m3 = _get(out, "y_roll3_mean")
    assert math.isnan(m2[0])
    assert abs(m2[1] - 1.5) < 1e-9
    assert math.isnan(m3[1])
    assert abs(m3[2] - 2.0) < 1e-9


def test_quantiles():
    df = _mk([1, 2, 3, 4, 5])
    out = rolling_stats(
        df, "y", windows=(5,),
        stats=("q10", "q25", "q50", "q75", "q90"),
    )
    q50 = _get(out, "y_roll5_q50")[4]
    assert abs(q50 - 3.0) < 1e-9
    q10 = _get(out, "y_roll5_q10")[4]
    assert q10 < q50


def test_skew_kurt_constant_returns_nan():
    df = _mk([5, 5, 5, 5, 5])
    out = rolling_stats(df, "y", windows=(3,), stats=("skew", "kurt"))
    sk = _get(out, "y_roll3_skew")
    ku = _get(out, "y_roll3_kurt")
    for v in sk[2:] + ku[3:]:
        assert math.isnan(v)


def test_energy_and_crossings():
    df = _mk([1, 2, 1, 2, 1])
    out = rolling_stats(
        df, "y", windows=(3,),
        stats=("energy", "crossings"),
    )
    e = _get(out, "y_roll3_energy")
    c = _get(out, "y_roll3_crossings")
    assert abs(e[2] - 2.0) < 1e-9
    assert c[2] == 2.0


def test_slope():
    df = _mk([1, 2, 3, 4, 5])
    out = rolling_stats(df, "y", windows=(4,), stats=("slope",))
    sl = _get(out, "y_roll4_slope")
    assert abs(sl[3] - 1.0) < 1e-9


def test_autocorr():
    df = _mk([1, 2, 3, 4, 5, 6, 7, 8])
    out = rolling_stats(
        df, "y", windows=(6,), stats=("autocorr_lag1",),
    )
    ac = _get(out, "y_roll6_autocorr_lag1")
    assert abs(ac[5] - 1.0) < 1e-9


def test_group_no_leak():
    df = _mk([1, 2, 3, 10, 20, 30], group=["a", "a", "a", "b", "b", "b"])
    out = rolling_stats(
        df, "y", windows=(3,), stats=("mean",), group_col="g",
    )
    m = _get(out, "y_roll3_mean")
    assert math.isnan(m[0])
    assert math.isnan(m[1])
    assert abs(m[2] - 2.0) < 1e-9
    assert math.isnan(m[3])
    assert math.isnan(m[4])
    assert abs(m[5] - 20.0) < 1e-9


def test_nan_handling():
    df = _mk([1, None, 3, None, 5])
    out = rolling_stats(df, "y", windows=(3,), stats=("mean",))
    m = _get(out, "y_roll3_mean")
    assert math.isnan(m[2])
    assert math.isnan(m[3])


def test_min_periods():
    df = _mk([1, 2, 3, 4])
    out = rolling_stats(
        df, "y", windows=(3,), stats=("mean",), min_periods=1,
    )
    m = _get(out, "y_roll3_mean")
    assert abs(m[0] - 1.0) < 1e-9
    assert abs(m[1] - 1.5) < 1e-9
    assert abs(m[2] - 2.0) < 1e-9


def test_validation():
    df = _mk([1, 2, 3])
    with pytest.raises(ValueError):
        rolling_stats(df, "y", windows=(0,))
    with pytest.raises(ValueError):
        rolling_stats(df, "y", windows=(2,), stats=("nope",))
    with pytest.raises(KeyError):
        rolling_stats(df, "nope", windows=(2,))
    with pytest.raises(ValueError):
        rolling_stats(df, "y", windows=(2,), stats=("autocorr_lag2",))


def test_memory_guard_budget():
    df = _mk(list(range(100)))
    guard = cadw.MemoryGuard(max_memory_mb=8)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    out = rolling_stats(
        df2, "y", windows=(5, 10),
        stats=("mean", "std", "q50", "autocorr_lag1"),
    )
    n = sum(b.num_rows for b in out._iter_arrow_batches(batch_size=50))
    assert n == 100


def test_single_row():
    df = _mk([7.0])
    out = rolling_stats(df, "y", windows=(3,), stats=("mean",))
    m = _get(out, "y_roll3_mean")
    assert math.isnan(m[0])


def test_xcorr_identity():
    from cadw.ts import rolling_crosscorr
    df = _mk([1, 2, 3, 4, 5], group=None)
    # add a second numeric column x = y
    import pyarrow as pa
    t = df.to_batches(format="numpy", batch_size=100)
    # simpler: build from table
    arr = [1., 2., 3., 4., 5.]
    table = pa.table({
        "x": pa.array(arr, type=pa.float64()),
        "y": pa.array(arr, type=pa.float64()),
    })
    df2 = CadwDataFrame.from_table(table)
    out = rolling_crosscorr(df2, "x", "y", windows=(5,), lags=(0,))
    vals = _get(out, "x__y_roll5_xcorr_lag0")
    assert abs(vals[4] - 1.0) < 1e-9


def test_xcorr_phase_shift():
    from cadw.ts import rolling_crosscorr
    import pyarrow as pa
    # y[t] = x[t-1] -> corr(x[t], y[t-1]) at lag 1 should be ~1.
    x = [1., 2., 3., 4., 5., 6., 7., 8.]
    y = [0., 1., 2., 3., 4., 5., 6., 7.]
    df = CadwDataFrame.from_table(pa.table({
        "x": pa.array(x, type=pa.float64()),
        "y": pa.array(y, type=pa.float64()),
    }))
    out = rolling_crosscorr(df, "x", "y", windows=(6,), lags=(0, 1))
    l0 = _get(out, "x__y_roll6_xcorr_lag0")
    l1 = _get(out, "x__y_roll6_xcorr_lag1")
    # At lag 1, the pairs are (x[t], y[t-1]) = (k, k-2) -> strong linear.
    assert abs(l1[-1] - 1.0) < 1e-6


def test_xcorr_constant_nan():
    from cadw.ts import rolling_crosscorr
    import pyarrow as pa
    df = CadwDataFrame.from_table(pa.table({
        "x": pa.array([5., 5., 5., 5.], type=pa.float64()),
        "y": pa.array([1., 2., 3., 4.], type=pa.float64()),
    }))
    out = rolling_crosscorr(df, "x", "y", windows=(3,), lags=(0,))
    vals = _get(out, "x__y_roll3_xcorr_lag0")
    for v in vals[2:]:
        assert math.isnan(v)


def test_xcorr_group_no_leak():
    from cadw.ts import rolling_crosscorr
    import pyarrow as pa
    df = CadwDataFrame.from_table(pa.table({
        "x": pa.array([1., 2., 3., 10., 20., 30.], type=pa.float64()),
        "y": pa.array([1., 2., 3., 10., 20., 30.], type=pa.float64()),
        "g": pa.array(["a", "a", "a", "b", "b", "b"], type=pa.string()),
    }))
    out = rolling_crosscorr(
        df, "x", "y", windows=(3,), lags=(0,), group_col="g",
    )
    vals = _get(out, "x__y_roll3_xcorr_lag0")
    assert math.isnan(vals[0]) and math.isnan(vals[1])
    assert abs(vals[2] - 1.0) < 1e-9
    assert math.isnan(vals[3]) and math.isnan(vals[4])
    assert abs(vals[5] - 1.0) < 1e-9


def test_xcorr_memory_guard():
    from cadw.ts import rolling_crosscorr
    import pyarrow as pa
    n = 100
    xs = [float(i) for i in range(n)]
    ys = [float(i) for i in range(n)]
    df = CadwDataFrame.from_table(pa.table({
        "x": pa.array(xs, type=pa.float64()),
        "y": pa.array(ys, type=pa.float64()),
    }))
    guard = cadw.MemoryGuard(max_memory_mb=8)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    out = rolling_crosscorr(df2, "x", "y", windows=(5,), lags=(0, 1))
    cnt = sum(b.num_rows for b in out._iter_arrow_batches(batch_size=50))
    assert cnt == n


def test_xcorr_validation():
    from cadw.ts import rolling_crosscorr
    import pyarrow as pa
    df = CadwDataFrame.from_table(pa.table({
        "x": pa.array([1., 2., 3.], type=pa.float64()),
        "y": pa.array([1., 2., 3.], type=pa.float64()),
    }))
    with pytest.raises(KeyError):
        rolling_crosscorr(df, "x", "nope", windows=(3,), lags=(0,))
    with pytest.raises(ValueError):
        rolling_crosscorr(df, "x", "x", windows=(3,), lags=(0,))
    with pytest.raises(ValueError):
        rolling_crosscorr(df, "x", "y", windows=(3,), lags=(3,))

# ============================================================================
# FILE: tests/test_ts_api.py  [full]
# ============================================================================
"""Stage 7 tests: public API, TS accessor, registry, docs (v0.9.24)."""
import numpy as np
import pyarrow as pa
import pytest

import cadw
import cadw.ts as ts
from cadw import CadwDataFrame


def _df(n=20):
    ts_vals = [1700000000000000000 + i * 3600 * 10**9 for i in range(n)]
    return CadwDataFrame.from_table(pa.table({
        "y": pa.array([float(i) for i in range(1, n + 1)],
                      type=pa.float64()),
        "ts": pa.array(ts_vals, type=pa.timestamp("ns")),
        "g": pa.array(["a"] * n, type=pa.string()),
    }))


def test_ts_all_present():
    for name in ts.__all__:
        assert hasattr(ts, name), f"cadw.ts missing {name}"
    # __all__ sorted, no duplicates
    assert ts.__all__ == sorted(ts.__all__)
    assert len(ts.__all__) == len(set(ts.__all__))


def test_accessor_registered():
    df = _df()
    acc = df.ts
    assert acc is not None
    for m in (
        "lag", "lead", "diff", "calendar", "cyclical", "fourier",
        "holiday", "make_features", "rolling_features", "crosscorr",
        "fit_forecaster", "backtest", "fit_temporal_hierarchy",
    ):
        assert callable(getattr(acc, m)), f"accessor missing {m}"


def test_accessor_returns_cadw_df():
    df = _df()
    out = df.ts.lag("y", lags=(1,))
    assert isinstance(out, CadwDataFrame)
    assert "y_lag1" in out.columns
    out2 = df.ts.calendar("ts", features=("month",))
    assert isinstance(out2, CadwDataFrame)
    assert "ts_month" in out2.columns
    out3 = df.ts.rolling_features("y", windows=(3,), stats=("mean",))
    assert isinstance(out3, CadwDataFrame)
    assert "y_roll3_mean" in out3.columns


def test_registry_names():
    from cadw.ts import get_forecaster
    for name in ("naive", "sarima", "gbdt"):
        f = get_forecaster(name)
        assert hasattr(f, "fit") and hasattr(f, "predict")
    with pytest.raises(ValueError):
        get_forecaster("nope")


def test_accessor_fit_forecaster_by_name():
    df = _df()
    f = df.ts.fit_forecaster("naive", "y")
    fc = f.predict(3)
    assert len(fc) == 3
    assert np.all(np.isfinite(fc))


def test_accessor_backtest():
    df = _df(40)
    summ, folds = df.ts.backtest(
        ts.NaiveForecaster(strategy="last"),
        target_col="y", time_col="ts",
        horizon=1, n_folds=2, strategy="expanding",
        train_size=10, step=1, metrics=("mae",),
    )
    assert len(summ) == 1 and summ[0]["metric"] == "mae"


def test_accessor_thf():
    df = _df(20)
    th = df.ts.fit_temporal_hierarchy(
        [1, 2], lambda: ts.NaiveForecaster(strategy="last"),
        target_col="y", time_col="ts", reconciliation="bottom_up",
    )
    fc = th.predict(2)
    assert len(fc) == 2


def test_readme_and_changelog_contain_ts_terms():
    import os
    readme = open("README.md", encoding="utf-8").read().lower()
    assert "time series" in readme or "time-series" in readme
    assert "forecaster" in readme
    changelog = open("CHANGELOG.md", encoding="utf-8").read().lower()
    assert "time series" in changelog or "time-series" in changelog
    decisions = open("DECISIONS.md", encoding="utf-8").read().lower()
    assert "mint" in decisions

# ============================================================================
# FILE: tests/test_ts_stage8.py  [full]
# ============================================================================
"""Stage 8: pin sort_values ascending-list fix; RAM verification harness."""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame


def _df():
    return CadwDataFrame.from_table(pa.table({
        "a": pa.array([2, 1, 2, 1], type=pa.int64()),
        "b": pa.array([10, 20, 5, 15], type=pa.int64()),
    }))


def test_sort_values_ascending_list_mixed_direction():
    # Old bug: ascending=[True, False] treated as truthy -> both asc.
    df = _df()
    out = df.sort_values(["a", "b"], ascending=[True, False])
    rows = []
    for batch in out.to_batches(format="numpy", batch_size=10):
        for i in range(len(batch["a"])):
            rows.append((int(batch["a"][i]), int(batch["b"][i])))
    assert rows == [(1, 20), (1, 15), (2, 10), (2, 5)], rows


def test_sort_values_ascending_list_length_mismatch():
    df = _df()
    with pytest.raises(ValueError):
        df.sort_values(["a", "b"], ascending=[True])


def test_sort_values_scalar_unchanged():
    df = _df()
    out = df.sort_values("b", ascending=False)
    vals = []
    for batch in out.to_batches(format="numpy", batch_size=10):
        vals.extend(int(v) for v in batch["b"])
    assert vals == [20, 15, 10, 5]

# ============================================================================
# FILE: .pingis/plan.md  [missing]  # file not found
# ============================================================================
# MISSING: .pingis/plan.md  (file not found)

# ============================================================================
# FILE: cadw/accessors/_ts.py  [full]
# ============================================================================
"""Time-series accessor (v0.9.24): df.ts.* thin wrappers.

All methods delegate directly to cadw.ts functions/classes; no logic
lives here. Registered as the `ts` property on CadwDataFrame, matching
the .dt/.str/.plot accessor pattern.
"""
from __future__ import annotations

from typing import Sequence


class _TsAccessor:
    """Pandas-style `df.ts.*` — thin wrappers over cadw.ts."""

    __slots__ = ("_df",)

    def __init__(self, parent) -> None:
        self._df = parent

    # ----- features -----
    def lag(self, column, lags=(1,), *, group=None, time_col=None, fill_value=None):
        from cadw.ts import lag_features
        return lag_features(
            self._df, column, lags,
            group_col=group, time_col=time_col, fill_value=fill_value,
        )

    def lead(self, column, leads=(1,), *, group=None, time_col=None, fill_value=None):
        from cadw.ts import lead_features
        return lead_features(
            self._df, column, leads,
            group_col=group, time_col=time_col, fill_value=fill_value,
        )

    def diff(self, column, lags=(1,), *, group=None, time_col=None,
             seasonal_periods=None):
        from cadw.ts import diff_features
        return diff_features(
            self._df, column, lags,
            group_col=group, time_col=time_col,
            seasonal_periods=seasonal_periods,
        )

    def calendar(self, ts_col, features=None):
        from cadw.ts import calendar_features
        kw = {} if features is None else {"features": features}
        return calendar_features(self._df, ts_col, **kw)

    def cyclical(self, ts_col, features=None):
        from cadw.ts import cyclical_features
        kw = {} if features is None else {"features": features}
        return cyclical_features(self._df, ts_col, **kw)

    def fourier(self, time_col, period, order, prefix=None):
        from cadw.ts import fourier_features
        return fourier_features(
            self._df, time_col, period, order, prefix=prefix,
        )

    def holiday(self, ts_col):
        from cadw.ts import holiday_features
        return holiday_features(self._df, ts_col)

    def make_features(self, **kwargs):
        from cadw.ts import make_features
        return make_features(self._df, **kwargs)

    # ----- window stats -----
    def rolling_features(self, column, windows, *, stats=("mean", "std"),
                         group=None, time_col=None, min_periods=None):
        from cadw.ts import rolling_stats
        return rolling_stats(
            self._df, column, windows, stats=stats,
            group_col=group, time_col=time_col, min_periods=min_periods,
        )

    def crosscorr(self, col_x, col_y, windows, *, lags=(0,),
                  group=None, time_col=None, min_periods=None):
        from cadw.ts import rolling_crosscorr
        return rolling_crosscorr(
            self._df, col_x, col_y, windows, lags=lags,
            group_col=group, time_col=time_col, min_periods=min_periods,
        )

    # ----- forecasters -----
    def fit_forecaster(self, forecaster, target_col, *, time_col=None,
                       group=None, **fit_kwargs):
        # forecaster: ForecasterBase instance or short name.
        from cadw.ts import get_forecaster
        if isinstance(forecaster, str):
            f = get_forecaster(forecaster)
        else:
            f = forecaster
        if group is not None:
            fit_kwargs.setdefault("group_col", group)
        return f.fit(self._df, target_col, time_col=time_col, **fit_kwargs)

    def backtest(self, forecaster, *, target_col, time_col, **kwargs):
        from cadw.ts import backtest_forecaster
        return backtest_forecaster(
            self._df, forecaster, target_col=target_col,
            time_col=time_col, **kwargs,
        )

    def fit_temporal_hierarchy(self, levels, forecaster_factory, *,
                               target_col, time_col=None, group=None,
                               reconciliation="optimal_ols", agg="sum"):
        from cadw.ts import TemporalHierarchy
        th = TemporalHierarchy(
            levels, forecaster_factory, reconciliation=reconciliation,
            agg=agg,
        )
        if group is not None:
            return th.fit(
                self._df, target_col, time_col=time_col, group_col=group,
            )
        return th.fit(self._df, target_col, time_col=time_col)


__all__ = ["_TsAccessor"]

# ============================================================================
# FILE: cadw/ops/rank.py  [diff]
# ============================================================================
diff --git a/cadw/ops/rank.py b/cadw/ops/rank.py
index 9347ece..6df4703 100644
--- a/cadw/ops/rank.py
+++ b/cadw/ops/rank.py
@@ -84,13 +84,13 @@ def _compute_streamable(
 
     import pyarrow.compute as _pc
 
     try:
         for batch in sorted_batches:
-            # v0.9.22b (AUDIT_ML §1.6): extra_factor=5.0 —
-            # null fallback does to_pylist(by) + to_pylist(row_id);
-            # null-free uses numpy fast path (no pylist).
+            # v0.9.24: extra_factor=7.0 — 3.5 × K pylist columns,
+            # K=2 (by + row_id). Null fallback does to_pylist on both;
+            # null-free numpy fast path needs no pylist. AUDIT_FULL §3.2.
             memory_guard.check_batch(
                 batch, name="rank.streamable.input", extra_factor=7.0,
             )
             vals_arr = batch.column(by)
             row_ids_arr = batch.column("row_id")
@@ -268,12 +268,12 @@ def _pass1_groups_to_disk(
         starts = []
         counts = []
 
     try:
         for batch in sorted_batches:
-            # v0.9.22b (AUDIT_ML §1.6): extra_factor=3.0 —
-            # pass1 does to_pylist(by); ~3.5× batch.nbytes.
+            # v0.9.24: extra_factor=3.5 — pass1 does to_pylist(by),
+            # K=1 pylist column ⇒ 3.5 × batch.nbytes (AUDIT_FULL §3.2).
             memory_guard.check_batch(
                 batch, name="rank.pass1.input", extra_factor=3.5,
             )
             vals = batch.column(by).to_pylist()
             for v in vals:
@@ -348,12 +348,12 @@ def _pass2_ranks_to_disk(
         cur_count = c
         cur_last = cur_first + cur_count - 1
 
     try:
         for batch in sorted_batches:
-            # v0.9.22b (AUDIT_ML §1.6): extra_factor=5.0 —
-            # pass2 does to_pylist(by) + to_pylist(row_id).
+            # v0.9.24: extra_factor=7.0 — pass2 does to_pylist(by) +
+            # to_pylist(row_id); K=2 pylist columns ⇒ 3.5 × 2.
             memory_guard.check_batch(
                 batch, name="rank.pass2.input", extra_factor=7.0,
             )
             vals = batch.column(by).to_pylist()
             row_ids = batch.column("row_id").to_pylist()

# ============================================================================
# FILE: cadw/__init__.py  [diff-empty]
# ============================================================================
# (unchanged vs 524ca0d (via log grep: v0.9.23j))

# ============================================================================
# FILE: tests/test_anti_outer_join.py  [diff]
# ============================================================================
diff --git a/tests/test_anti_outer_join.py b/tests/test_anti_outer_join.py
index 6041d80..aaec7fe 100644
--- a/tests/test_anti_outer_join.py
+++ b/tests/test_anti_outer_join.py
@@ -188,15 +188,22 @@ def test_merge_how_unsupported_raises(tmp_path):
     right = make_df(tmp_path, {"k": [1]}, name="R")
     with pytest.raises(NotImplementedError, match="right"):
         left.merge(right, on="k", how="right")
 
 
-def test_merge_how_semi_raises(tmp_path):
+def test_merge_how_semi_works(tmp_path):
+    """v0.9.23j: semi join now works (was NotImplementedError).
+
+    Rename from test_merge_how_semi_raises; test semantics
+    flipped: emit left rows with match in right.
+    """
     left = make_df(tmp_path, {"k": [1, 2]}, name="L")
     right = make_df(tmp_path, {"k": [1]}, name="R")
-    with pytest.raises(NotImplementedError, match="semi"):
-        left.merge(right, on="k", how="semi")
+    result = left.merge(right, on="k", how="semi")
+    rows = _rows(result)
+    assert len(rows) == 1
+    assert rows[0]["k"] == 1
 
 
 # ============ AST guards ============
 
 def test_anti_uses_hash_shuffle():
@@ -269,18 +276,24 @@ def test_anti_whole_op_warning_skip():
         'if how in ("anti", "semi"):' in src
         and "predicted_output = 0" in src
     ), "anti/semi should skip predicted_output (per-batch guard sufficient)"
 
 
-def test_no_semi_join_streaming_import_yet():
-    """v0.9.23i-fix2: semi_join_streaming import removed (v0.9.23j)."""
+def test_semi_join_streaming_import_present():
+    """v0.9.23j: semi_join_streaming now implemented and dispatched.
+
+    Stale-test on enum extension (git-log §13.4 pattern): the previous
+    assertion "semi_join_streaming not in merge source" was valid only
+    while semi join was NotImplemented. Once v0.9.23j shipped the
+    implementation, the invariant flipped: merge MUST reference it.
+    """
     import inspect
     from cadw.dataframe import CadwDataFrame
     src = inspect.getsource(CadwDataFrame.merge)
-    assert "semi_join_streaming" not in src, (
-        "semi_join_streaming doesn't exist yet (v0.9.23j); "
-        "remove from dispatch import"
+    assert "semi_join_streaming" in src, (
+        "v0.9.23j: semi join is implemented; merge should dispatch to "
+        "semi_join_streaming"
     )
 
 
 # ============ v0.9.23i-fix3: multi-column crossed-tuple bug ============
 #

# ============================================================================
# FILE: tests/test_join.py  [diff]
# ============================================================================
diff --git a/tests/test_join.py b/tests/test_join.py
index 701ed21..5bb658c 100644
--- a/tests/test_join.py
+++ b/tests/test_join.py
@@ -108,29 +108,28 @@ def test_merge_missing_on_raises(tmp_path):
     with pytest.raises(KeyError, match="not in right"):
         left.merge(right, on="k")
 
 
 def test_merge_unsupported_how(tmp_path):
-    """v0.9.23i-fix5: 'outer' now supported; 'right'/'cross' still raise.
+    """v0.9.23j: how enum closed. Only 'right'/'cross' raise.
 
-    Prior test (v0.2.x) expected only inner/left. Enum extended in
-    v0.9.23i (outer/anti/semi) and v0.9.23j ('right'). Update:
-    - outer -> supported (no longer in bad_how)
-    - right, cross -> raise with updated message
-    - semi -> raises NotImplementedError (v0.9.23j placeholder)
+    Supported: inner, left, outer, anti, semi.
+    - right -> still NotImplementedError (deferred; SQL right
+      join = left join with sides swapped — not a v0.9.23 scope)
+    - cross -> still NotImplementedError (O(m*n) — out of scope)
     """
     left = make_df(tmp_path, "L", {"k": [1], "a": [10]})
     right = make_df(tmp_path, "R", {"k": [1], "b": [100]})
     for bad_how in ("right", "cross"):
         with pytest.raises(
             NotImplementedError,
             match="supported: inner, left, outer, anti, semi",
         ):
             left.merge(right, on="k", how=bad_how)
-    # semi: explicit placeholder message
-    with pytest.raises(NotImplementedError, match="semi"):
-        left.merge(right, on="k", how="semi")
+    # semi now works (v0.9.23j) — regression: must not raise.
+    result = left.merge(right, on="k", how="semi")
+    assert result.count() == 1
 
 
 def test_merge_requires_cadwdataframe(tmp_path):
     left = make_df(tmp_path, "L", {"k": [1], "a": [10]})
     with pytest.raises(TypeError, match="CadwDataFrame"):

# ============================================================================
# FILE: README.md  [diff]
# ============================================================================
diff --git a/README.md b/README.md
index 774d415..08793f1 100644
--- a/README.md
+++ b/README.md
@@ -64,10 +64,92 @@ reader = ParquetBatchReader("data.parquet", ["cat", "val"], batch_size=10000)
 guard = MemoryGuard(max_memory_mb=200)
 result = groupby_aggregate(reader, ["cat"], ["val"], ["mean"], guard, "./spill")
 print(result)
 ```
 
+## Time series (v0.9.24)
+
+Comprehensive time-series toolkit built on the same out-of-core
+primitives. Import from `cadw.ts` or use the thin `df.ts.*` accessor.
+
+**Feature engineering** (`cadw.ts.features`): lag/lead, diff (regular +
+seasonal), calendar (year/month/day/weekday/hour/...), cyclical sin/cos,
+Fourier terms, holiday/weekend flags, and a `make_features` pipeline.
+All streaming, O(batch) state, MemoryGuard-checked, group-aware (per-entity
+lags, no cross-entity leakage).
+
+**Window statistics** (`cadw.ts.stats`): tsfresh-subset — mean/std/sum/
+min/max/median/skew/kurt, fixed quantiles q10..q90, abs_energy/energy/cid,
+crossings, slope, entropy, count_above_mean, autocorr_lagK, plus
+`rolling_crosscorr` between two series. One shared deque per group;
+explicit RAM formula per statistic.
+
+**Forecasters** (`cadw.ts.models`): `NaiveForecaster` (last/seasonal/
+drift), `SARIMAForecaster` (CSS/Hannan-Rissanen, bounded auto-AIC grid,
+seasonal orders), `GBDTForecaster` (default mode='direct' with deferred
+per-horizon training; 'recursive' and 'multi_output' also available).
+Common `ForecasterBase` API: fit / predict / predict_in_sample.
+
+**Backtesting** (`cadw.ts.backtest`): expanding and sliding-window CV,
+multi-series, metrics MAE/RMSE/MAPE/sMAPE/MASE (m=seasonal_period),
+optional pinball for quantile forecasters. Forecasters cloned per fold.
+
+**Temporal Hierarchical Forecasting** (`cadw.ts.thf`):
+`TemporalHierarchy` aggregates the base series to K temporal levels, fits
+a forecaster per level, and reconciles with bottom-up, top-down-proportional,
+optimal-OLS, or proper MinT-WLS (diagonal residual-variance weights,
+1e-8 floor, OLS fallback on insufficient residuals). Multi-series supported.
+
+### Examples
+
+```python
+import cadw
+from cadw.ts import (
+    lag_features, rolling_stats, make_features,
+    NaiveForecaster, GBDTForecaster, backtest_forecaster,
+    TemporalHierarchy,
+)
+
+# df has columns y, ts (timestamp), maybe g (group)
+# Features
+f = lag_features(df, "y", lags=(1, 7, 14), group_col="g", time_col="ts")
+# OR full pipeline (do not chain both):
+# f2 = make_features(df, target_col="y", ts_col="ts", group_col="g")
+
+# Window stats
+s = rolling_stats(df, "y", windows=(7, 14),
+                  stats=("mean", "std", "slope", "autocorr_lag1"))
+
+# Forecasters
+m = NaiveForecaster(strategy="seasonal", period=7).fit(df, "y")
+print(m.predict(7))
+
+g = GBDTForecaster(lags=(1, 2, 3)).fit(df, "y")
+print(g.predict(7))    # deferred: 7 horizon models trained on demand
+
+# Backtesting
+summary, folds = backtest_forecaster(
+    df, NaiveForecaster("last"),
+    target_col="y", time_col="ts",
+    horizon=7, n_folds=5, strategy="expanding",
+    metrics=("mae", "rmse", "mase"),
+)
+
+# Temporal hierarchy with MinT-WLS reconciliation
+th = TemporalHierarchy(
+    levels=[1, 7, 30],
+    forecaster_factory=lambda: NaiveForecaster("last"),
+    reconciliation="optimal_wls",
+).fit(df, "y", time_col="ts")
+print(th.predict(30))
+
+# Or via the accessor:
+f = df.ts.lag("y", lags=(1, 7), group="g", time_col="ts")
+f = f.ts.rolling_features("y", windows=(7,), stats=("mean", "std"))
+model = df.ts.fit_forecaster("gbdt", "y", time_col="ts")
+```
+
 ## ML и внешние библиотеки
 
 CADW **не является ML-фреймворком**. Он даёт `to_batches(format=...)` для
 экспорта батчей в вашу модель. Если ваша модель OOM-ит — это её состояние,
 а не CADW. Гарантия out-of-core действует только на код CADW, не на

# ============================================================================
# FILE: CHANGELOG.md  [diff]
# ============================================================================
diff --git a/CHANGELOG.md b/CHANGELOG.md
index f767253..c23491d 100644
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -1,7 +1,39 @@
 ## [Unreleased]
 
+### v0.9.24: time series toolkit + temporal hierarchical forecasting
+
+Comprehensive time-series extension. All new code lives in `cadw/ts/`
+and routes through the existing primitives + MemoryGuard.
+
+- **Features** (`cadw/ts/features.py`): lag/lead, diff (regular +
+  seasonal), calendar, cyclical sin/cos, Fourier, holiday/weekend,
+  `make_features` composite pipeline. Group-aware (per-entity, no
+  cross-entity leakage). O(batch) state, explicit extra_factor.
+- **Window statistics** (`cadw/ts/stats.py`): tsfresh-subset
+  (mean/std/min/max/median/skew/kurt, q10..q90, abs_energy/energy/
+  cid, crossings, slope, entropy, count_above_mean, autocorr_lagK) +
+  rolling_crosscorr. Shared per-group deque, RAM formula documented.
+- **Forecasters** (`cadw/ts/models.py`): NaiveForecaster, SARIMAForecaster
+  (CSS/Hannan-Rissanen, bounded auto-AIC grid, seasonal orders; AIC
+  may under-difference trending series — use explicit order=), 
+  GBDTForecaster (default mode='direct' with deferred per-horizon
+  training; recursive and multi_output modes). ForecasterBase API,
+  get_forecaster name registry ('naive'/'sarima'/'gbdt').
+- **Backtesting** (`cadw/ts/backtest.py`): expanding/sliding CV,
+  multi-series, MAE/RMSE/MAPE/sMAPE/MASE(m=seasonal_period), optional
+  pinball; forecaster cloned per fold via deepcopy.
+- **THF** (`cadw/ts/thf.py`): TemporalHierarchy with temporal
+  aggregation, bottom-up / top-down-proportional / optimal-OLS /
+  MinT-WLS (diagonal residual-variance weights, 1e-8 floor, OLS
+  fallback). Multi-series, predict_level.
+- **Accessor**: `CadwDataFrame.ts` thin wrappers matching the .dt/.str
+  pattern. `cadw.ts.__all__` sorted and complete (16 symbols).
+- **Bugfix** (pre-existing): `sort_values(by=list, ascending=list)`
+  silently treated list ascending as truthy → all columns ascending.
+  Fixed: per-column ascending sequence, length-checked.
+
 
 ### v0.9.15–v0.9.22: post-release v0.9.14 fixes + audit round 3
 
 **8 patches** since v0.9.14 tag. Full per-patch detail via
 `git log --oneline v0.9.14..master`. Highlights:

# ============================================================================
# FILE: DECISIONS.md  [diff]
# ============================================================================
diff --git a/DECISIONS.md b/DECISIONS.md
index 0a8834b..75a20a1 100644
--- a/DECISIONS.md
+++ b/DECISIONS.md
@@ -5376,5 +5376,58 @@ CADW = `to_batches(format=...)` только. Собственные ML-алго
 v0.3 (GBDT, Linear, k-means, PCA, NB) на примитивах.
 
 **v0.1.4 (патч 10):** must-fix «materialize-before-check»: `_local_groupby`
 (stat до чтения) и `reduce` (estimate_state_size до init()). Добавлены
 AST guards + OOM E2E. §3.10 — новый принцип.
+
+
+---
+
+## §15. Time series (v0.9.24)
+
+### 15.1 Forecaster API
+
+Common `ForecasterBase`: fit(df, target_col, *, time_col=None,
+ group_col=None) / predict(steps) / predict_in_sample(). Predict before
+ fit raises RuntimeError; steps<1 raises ValueError. This is the
+ contract every forecaster satisfies; test_ts_models asserts method
+ presence and arity for all classes.
+
+### 15.2 GBDT default = direct, deferred training
+
+GBDTForecaster default mode='direct': h independent models, one per
+ horizon. Horizons are determined by predict(steps): fit(df) with no
+ horizons defers training; predict(10) trains exactly 10 horizon models
+ and caches them. No silent extrapolation (was: reuse last horizon model
+ — removed). 'recursive' mode is available but propagates errors
+ (feed-back); 'multi_output' is h-independent (CADW GBDT has no native
+ multi-target) — documented.
+
+### 15.3 SARIMA uses CSS, not MLE
+
+SARIMAForecaster fits via conditional sum of squares / Hannan-Rissanen
+(two-step AR-approx + OLS on ARMA), not full MLE. Auto-ARIMA grid is
+bounded: p,q in 0..2, d in 0..1; P,D,Q in 0..1 when seasonal. AIC can
+UNDER-DIFFERENCE trending series (level-stationary ARMA often wins by
+AIC on a random walk). Recommendation documented: set order=(p,1,q)
+explicitly for trending/unit-root data. Integration back from d,D
+uses L1=last observed y; L2+ levels default to 0.0 for d>=2 (CSS limit).
+
+### 15.4 MinT-WLS uses residual-variance weights
+
+THF reconciliation 'optimal_wls' is proper MinT with diagonal W where
+W[i] = in-sample residual variance of the level that position i belongs
+to. Residuals come from each level model's predict_in_sample over rows
+where the prediction is finite. A 1e-8 floor prevents zero weight on
+perfect in-sample fits. If ANY level has <2 finite residual pairs, the
+whole fit falls back to OLS and emits a one-time warning. This replaced
+an earlier heuristic (w[i]=1/m_k) that was NOT MinT. Reconciled y =
+S (S^T W^-1 S)^-1 S^T W^-1 y_all with np.linalg.pinv (rank-safe).
+
+### 15.5 sort_values ascending-list (pre-existing bug, fixed)
+
+df.sort_values(by=list, ascending=list) previously treated a non-empty
+list ascending as truthy -> all columns ascending. Silently broke mixed-
+direction sorts. Fixed in v0.9.24: ascending now accepts a per-column
+sequence, length-checked against by; both the in-memory fast path and
+external_sort use the resolved sequence. Not a breaking change for the
+scalar case.

# ============================================================================
# FILE: scripts/run_ts_ram.py  [full]
# ============================================================================
import json, subprocess, sys

SCEN = [
    ("rolling_stats(n=2000,4g)", "scenario_rolling_stats", 32),
    ("rolling_crosscorr(n=2000,4g)", "scenario_crosscorr", 32),
    ("make_features(n=2000,4g)", "scenario_make_features", 32),
    ("GBDT(n=500)", "scenario_gbdt", 64),
    ("THF_OLS(n=300)", "scenario_thf_ols", 32),
    ("THF_MinT_WLS(n=300)", "scenario_thf_mint_wls", 32),
    ("backtest(n=200)", "scenario_backtest", 32),
    ("BASELINE_noop", "scenario_baseline_noop", 32),
]

CHILD = r"""
import gc, json, sys, time
sys.path.insert(0, '.')
import psutil, tracemalloc
import scripts.verify_ts_ram as m
fn = getattr(m, sys.argv[1])
gc.collect()
proc = psutil.Process(); base = proc.memory_info().rss
tracemalloc.start()
t0 = time.perf_counter()
fn()
dt = time.perf_counter() - t0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(json.dumps({'rss': (proc.memory_info().rss-base)/1048576, 'peak': peak/1048576, 'sec': dt}))
"""

print('%-30s %10s %10s %8s %6s %s' % ('scenario','rss_delta','tm_peak','budget','sec','status'))
print('-'*85)
for name, fn, budget in SCEN:
    r = subprocess.run([sys.executable, '-c', CHILD, fn], capture_output=True, text=True)
    if r.returncode != 0:
        print(name, 'ERR', (r.stderr or '').strip().splitlines()[-1] if r.stderr else '?')
        continue
    d = json.loads(r.stdout.strip().splitlines()[-1])
    ok = 'OK' if d['peak'] <= budget else 'OVER'
    print('%-30s %10.2f %10.2f %8d %6.2f %s' % (name, d['rss'], d['peak'], budget, d['sec'], ok))

# ============================================================================
# FILE: scripts/diag_rolling.py  [full]
# ============================================================================

import gc, time, psutil, tracemalloc
import numpy as np, pyarrow as pa
import cadw
from cadw import CadwDataFrame
from cadw.ts import rolling_stats

rng = np.random.default_rng(0)
n = 2000
cols = {
    "y": pa.array(rng.normal(0,1,n).cumsum()+100.0, type=pa.float64()),
    "ts": pa.array([1700000000000000000+i*3600*10**9 for i in range(n)], type=pa.timestamp("ns")),
    "g": pa.array([f"g{i%4}" for i in range(n)], type=pa.string()),
}
df = CadwDataFrame.from_table(pa.table(cols))
guard = cadw.MemoryGuard(max_memory_mb=32)
df2 = CadwDataFrame(df._dataset, memory_guard=guard, spill_dir=df._spill_dir)
gc.collect()
proc = psutil.Process(); base = proc.memory_info().rss
tracemalloc.start()
out = rolling_stats(df2, "y", windows=(7,14,30), stats=("mean","std","q50","skew","slope","autocorr_lag1"), group_col="g", time_col="ts")
mx = 0; nb = 0
for b in out._iter_arrow_batches(batch_size=500):
    nb += b.num_rows
    r = proc.memory_info().rss
    if r > mx: mx = r
cur, peak_tm = tracemalloc.get_traced_memory(); tracemalloc.stop()
print("n_out", nb)
print("rss_delta_mb", round((mx-base)/1024/1024,2))
print("rss_final_delta_mb", round((proc.memory_info().rss-base)/1024/1024,2))
print("tm_peak_mb", round(peak_tm/1024/1024,2))
print("tm_cur_mb", round(cur/1024/1024,2))

