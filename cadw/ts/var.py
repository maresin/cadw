"""VAR(p) vector autoregression (D.4.5, v0.9.38).

Reference: Lutkepohl, *New Introduction to Multiple Time Series
Analysis*, Springer 2005, ch. 2-3 (VAR(p) representation, OLS
estimation, stability); Hamilton, *Time Series Analysis*, Princeton
1994, ch. 11.
See DECISION_LOG for algorithm reference.

VAR(p) with intercept, estimated by equation-by-equation OLS on a
stacked lag design matrix. Supports multiple endogenous series (k >= 1),
lag order p >= 1, and provides both in-sample fit and multi-step
forecasts. Stability is checked via the companion matrix spectral radius.
RAM: O(n*k + k^2*p^2); the training panel is materialised once.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.ml._logging import log_fit
from cadw.ts.models import ForecasterBase, _check_df, _extract_series


def _stack_lags(Y, p):
    """Build the lag design matrix [1, Y[t-1], ..., Y[t-p]] and target Y[t]."""
    n, k = Y.shape
    if n <= p:
        raise ValueError(f"var: need n > p; n={n}, p={p}")
    rows = n - p
    X = np.empty((rows, 1 + k * p), dtype=np.float64)
    X[:, 0] = 1.0
    for lag in range(1, p + 1):
        X[:, 1 + (lag - 1) * k:1 + lag * k] = Y[p - lag:n - lag]
    return X, Y[p:]


def _companion(A_list):
    """Companion matrix from lag coefficient matrices [A1..Ap]."""
    p = len(A_list)
    k = A_list[0].shape[0]
    C = np.zeros((k * p, k * p))
    C[:k, :k * p] = np.hstack(A_list)
    if p > 1:
        C[k:, :(p - 1) * k] = np.eye((p - 1) * k)
    return C


class VARModel(ForecasterBase):
    """Fitted VAR(p) with intercept, OLS coefficients, stability check."""

    def __init__(self, p=1, columns: Optional[Sequence[str]] = None):
        super().__init__()
        if p < 1:
            raise ValueError("VARModel: p must be >= 1")
        self.p = int(p)
        self.columns = list(columns) if columns is not None else None
        self.Y_ = None
        self.coef_ = None
        self.intercept_ = None
        self.sigma2_ = None
        self.stable_ = None
        self.fitted_ = None

    def fit(self, df, target_col, *, time_col=None, group_col=None):
        # target_col: str -> univariate case; list of names -> multivariate
        cols = [target_col] if isinstance(target_col, str) else list(target_col)
        if not cols:
            raise ValueError("VARModel: need at least one endogenous column")
        for c in cols:
            if c not in df._columns:
                raise KeyError(f"VARModel: column {c!r} not in {df._columns}")
        series = []
        for c in cols:
            series.append(_extract_series(df, c, time_col=time_col))
        n = series[0].size
        if any(s.size != n for s in series):
            raise ValueError("VARModel: all series must have the same length")
        if n <= self.p + 1:
            raise ValueError(f"VARModel: need n > p+1; n={n}, p={self.p}")
        # v0.9.90-S5: guard before O(n*k) + O(n*k*p) allocations.
        # Peak = Y (n*k*8) + X (n-p)*(1+k*p)*8 + Yt ((n-p)*k*8).
        mg = getattr(df, "_memory_guard", None)
        k = len(cols)
        if mg is not None:
            est = (n * k
                   + (n - self.p) * (1 + k * self.p)
                   + (n - self.p) * k) * 8
            mg.check_state(est, name="var.fit")
        Y = np.column_stack(series)
        if np.any(~np.isfinite(Y)):
            raise ValueError("VARModel: series contains non-finite values")
        X, Yt = _stack_lags(Y, self.p)
        # equation-by-equation OLS: B is (1 + k*p) x k
        B, *_ = np.linalg.lstsq(X, Yt, rcond=None)
        k = Y.shape[1]
        self.columns = cols
        self.Y_ = Y
        self.intercept_ = B[0].copy()
        self.coef_ = [B[1 + j * k:1 + (j + 1) * k].copy() for j in range(self.p)]
        resid = Yt - X @ B
        self.sigma2_ = np.sum(resid * resid, axis=0) / max(1, X.shape[0])
        self.fitted_ = (X @ B).copy()
        comp = _companion(self.coef_)
        self.stable_ = bool(np.max(np.abs(np.linalg.eigvals(comp))) < 1.0)
        self._is_fitted = True
        log_fit("var", loss=float(np.sum(resid * resid)),
                state_bytes=self.state_nbytes())
        return self

    def state_nbytes(self):
        if self.Y_ is None:
            return 0
        k = self.Y_.shape[1]
        return int(self.Y_.nbytes + k * k * self.p * 8 + k * 8 + 256)

    def _predict(self, steps):
        # `ForecasterBase.predict(steps)` predicts a single column; here we
        # produce forecasts of the FIRST endogenous column, matching the
        # univariate ForecasterBase contract. `forecast_vector` gives all.
        return self.forecast_vector(steps)[:, 0]

    def forecast_vector(self, steps):
        """Multi-step forecast matrix (steps x k) for all endogenous series."""
        k = self.Y_.shape[1]
        hist = list(self.Y_[-self.p:]) if self.p > 0 else []
        out = np.empty((steps, k), dtype=np.float64)
        buf = [h.copy() for h in hist]
        for h in range(steps):
            yhat = self.intercept_.copy()
            for lag in range(self.p):
                # coef_[lag][j, m] = effect of variable j on equation m
                yhat = yhat + self.coef_[lag].T @ buf[-(lag + 1)]
            out[h] = yhat
            buf.append(yhat)
        return out

    def _predict_in_sample(self):
        return self.fitted_[:, 0].copy()
