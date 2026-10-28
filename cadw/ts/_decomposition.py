"""STL (Seasonal-Trend decomposition using LOESS) — v0.9.77a.

Cleveland, Cleveland, McRae, Terpenning (1990), "STL: A Seasonal-Trend
Decomposition Procedure Based on LOESS", Journal of Official Statistics
6(1): 3-73.

Implementation notes:
  - Local linear LOESS smoother (tricube weight) on the trend.
  - Seasonal component estimated per phase (position within cycle) via
    low-pass filter over cycles.
  - Alternating (trend, seasonal) passes for n_iter rounds.

Deviations from R's stats::stl / statsmodels STL:
  - No robust (bisquare) outer loop: one non-robust pass by default.
    Robust variant deferred (backlog).
  - No post-smoothing of seasonal with LOESS; uses 3x3 moving-average
    low-pass over cycles (standard). Matches the original paper's
    "low-pass filter" step.
  - LOESS window is a hard width (in points), not span fraction. R's
    stl uses same convention. Default trend_window follows R's formula:
    nextodd(ceil(1.5 * period / (1 - 1.5/seasonal_window))).

Use cases:
  - Theta STM: trend component feeds Theta0, remainder feeds Theta2.
  - GBDT pipeline: seasonal/trend removed before fit, added back after.
  - MSTL: multiple seasonal periods applied iteratively.

Pure numpy (no sklearn/statsmodels/scipy/torch imports).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class STLResult:
    """Decomposition result.

    Attributes
    ----------
    trend : np.ndarray, shape (n,)
    seasonal : np.ndarray, shape (n,)
    remainder : np.ndarray, shape (n,)
    period : int
        Seasonal period used.
    trend_window : int
        LOESS window (odd, points) for trend.
    seasonal_window : int
        Window (cycles) for seasonal low-pass.
    n_iter : int
        Number of (trend, seasonal) iterations.
    """
    trend: np.ndarray
    seasonal: np.ndarray
    remainder: np.ndarray
    period: int
    trend_window: int
    seasonal_window: int
    n_iter: int


# ── Helpers ───────────────────────────────────────────────────────

def _nextodd(x: int) -> int:
    """Smallest odd integer >= x, minimum 3."""
    if x < 3:
        return 3
    return x if x % 2 == 1 else x + 1


def _tricube(u: np.ndarray) -> np.ndarray:
    """Tricube weight w(u) = (1 - |u|^3)^3 for |u| < 1, else 0."""
    a = np.abs(u)
    out = np.zeros_like(a)
    mask = a < 1.0
    uu = a[mask]
    out[mask] = (1.0 - uu ** 3) ** 3
    return out


def _loess_local_linear(y: np.ndarray, window: int) -> np.ndarray:
    """Local linear LOESS smoother of width `window` (odd).

    For each i, fit y[t] ≈ a + b*(t - i) with tricube weights over
    t in [i - h, i + h] where h = (window - 1) // 2. Returns the fitted
    value at i (the intercept a). At edges the window is truncated
    (same as R's stl; not reflected).
    """
    n = y.size
    if n < 3 or window < 3:
        return y.copy()
    h = (window - 1) // 2
    out = np.empty(n, dtype=np.float64)
    t_full = np.arange(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - h)
        hi = min(n - 1, i + h)
        t = t_full[lo:hi + 1]
        yy = y[lo:hi + 1]
        u = (t - i) / (h + 1.0)
        w = _tricube(u)
        # weighted local linear regression
        wsum = w.sum()
        if wsum < 1e-12:
            out[i] = y[i]
            continue
        xt = (t - i)
        # solve [S0 S1; S1 S2] [a; b] = [Sy; Sxy]
        S0 = wsum
        S1 = float(np.sum(w * xt))
        S2 = float(np.sum(w * xt * xt))
        Sy = float(np.sum(w * yy))
        Sxy = float(np.sum(w * xt * yy))
        det = S0 * S2 - S1 * S1
        if abs(det) < 1e-12:
            out[i] = Sy / S0
        else:
            a = (S2 * Sy - S1 * Sxy) / det
            out[i] = a
    return out


def _low_pass_seasonal_cycles(detrended: np.ndarray, period: int,
                              window_cycles: int) -> np.ndarray:
    """Per-phase smoothing over cycles with moving average.

    detrended length n, phase j in 0..period-1. Extract detrended[j::period],
    apply centered moving average of width `window_cycles`, tile back to
    length n.
    """
    n = detrended.size
    n_cycles = (n + period - 1) // period
    out = np.zeros(n, dtype=np.float64)
    wc = max(1, int(window_cycles))
    half = (wc - 1) // 2
    for j in range(period):
        idx = np.arange(j, n, period)
        vals = detrended[idx]
        m = vals.size
        smoothed = np.empty(m, dtype=np.float64)
        for k in range(m):
            lo = max(0, k - half)
            hi = min(m, k + half + 1)
            smoothed[k] = vals[lo:hi].mean()
        out[idx] = smoothed
    # Center seasonal to mean 0 (additive decomposition)
    out -= out.mean()
    return out


# ── Main STL ──────────────────────────────────────────────────────

def stl_decompose(
    y: np.ndarray,
    *,
    period: int,
    seasonal_window: int = 7,
    trend_window: int | None = None,
    n_iter: int = 2,
    memory_guard=None,
) -> STLResult:
    """STL decomposition: y = trend + seasonal + remainder.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
    period : int
        Seasonal period. Must be >= 2 and n >= 2 * period.
    seasonal_window : int, default=7
        Window (odd, in cycles) for low-pass seasonal smoothing. R
        default is 7.
    trend_window : int | None, default=None
        LOESS window (odd, in points) for trend. None -> R-formula
        nextodd(ceil(1.5 * period / (1 - 1.5/seasonal_window))).
    n_iter : int, default=2
        Number of (trend, seasonal) alternating passes. R default
        inner = 2.

    Returns
    -------
    STLResult
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    if n < 2 * period:
        raise ValueError(
            f"n={n} < 2*period={2*period}; need at least 2 cycles")
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")

    # v0.9.87-ts-transforms-guard: STL allocates ~6 n-sized float64
    # arrays (trend, seasonal, remainder + LOESS/intermediate work
    # columns). Guard before alloc so large n raises
    # MemoryLimitExceeded, not silent OOM. Lazy import preserves
    # module purity (no cadw.* at top of file).
    if memory_guard is not None:
        from cadw.core.memory_guard import MemoryLimitExceeded
        est_state = n * 8 * 8 + 4096  # ~8n float64 conservative bound
        try:
            memory_guard.check_state(
                est_state, name="stl_decompose.state")
        except MemoryLimitExceeded:
            raise MemoryLimitExceeded(
                f"stl_decompose: estimated peak {est_state} B "
                f"(n={n}, ~8n float64) exceeds state_limit. "
                f"STL requires full series in RAM; reduce n or use "
                f"chunked / streaming decomposition (backlog). "
                f"§13.4 v0.9.87")

    if trend_window is None:
        # R's default: nextodd(ceil(1.5*period / (1 - 1.5/seasonal_window)))
        sw = max(3, seasonal_window)
        raw = int(np.ceil(1.5 * period / (1.0 - 1.5 / sw)))
        trend_window = _nextodd(raw)
    trend_window = _nextodd(int(trend_window))
    seasonal_window = max(3, int(seasonal_window))
    if seasonal_window % 2 == 0:
        seasonal_window += 1

    trend = np.zeros(n, dtype=np.float64)
    seasonal = np.zeros(n, dtype=np.float64)
    for _ in range(n_iter):
        deseason = y - seasonal
        trend = _loess_local_linear(deseason, trend_window)
        detrended = y - trend
        seasonal = _low_pass_seasonal_cycles(
            detrended, period, seasonal_window)
    remainder = y - trend - seasonal

    return STLResult(
        trend=trend,
        seasonal=seasonal,
        remainder=remainder,
        period=period,
        trend_window=trend_window,
        seasonal_window=seasonal_window,
        n_iter=n_iter,
    )

