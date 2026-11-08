"""Numeric primitives — pure math, no domain (v0.9.132 DEC-054).

Reusable helpers extracted from per-module copies in cadw/ts/*.py.
No pandas / no Arrow / no I/O; numpy + math only.

Prior to v0.9.132, ols_fit and erfinv each had 2 identical copies
(otm.py + theta.py). AES / prediction intervals across all Theta-
family forecasters use erfinv; OLS trend line is shared by OTM and
STM theta.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def ols_fit(y: np.ndarray) -> Tuple[float, float]:
    """OLS (intercept, slope) on y[t] vs t.

    Returns (a, b) such that y_hat[t] = a + b*t. Handles n < 2 (returns
    mean, 0) and degenerate x variance (den < 1e-12 -> mean, 0).

    Pure math; input must be 1D np.ndarray of floats.
    """
    n = y.size
    if n < 2:
        return float(y.mean()) if n else 0.0, 0.0
    x = np.arange(n, dtype=np.float64)
    xm = float(x.mean())
    ym = float(y.mean())
    num = float(np.sum((x - xm) * (y - ym)))
    den = float(np.sum((x - xm) ** 2))
    if den < 1e-12:
        return ym, 0.0
    b = num / den
    return ym - b * xm, b


def erfinv(x: float) -> float:
    """Inverse error function via Newton iteration on math.erf.

    Domain: x in (-1, 1); raises ValueError outside.
    3 Newton steps; used for prediction intervals (z = sqrt(2) * erfinv(...)).
    """
    if x <= -1.0 or x >= 1.0:
        raise ValueError("erfinv: x must be in (-1, 1)")
    a = 0.147
    ln = math.log(1.0 - x * x)
    t = 2.0 / (math.pi * a) + ln / 2.0
    g = math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)
    for _ in range(3):
        err = math.erf(g) - x
        d = 2.0 / math.sqrt(math.pi) * math.exp(-g * g)
        g -= err / d
    return g
