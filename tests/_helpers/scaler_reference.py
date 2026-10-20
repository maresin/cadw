"""Reference scalers for tests (§13.4/164, /171).

Derived from sklearn definitions. Test bodies must use these — not
manual arithmetic.

Pattern (§13.4/171): for EXACT-property tests (e.g., "does transform
use FROZEN train stats?"), use deterministic data + reference helper.
RNG + statistical assertion only for CONVERGENCE tests ("does the
model learn?"), not "which stats were used?" — different distributions
are needed to distinguish frozen-vs-refit.
"""
from __future__ import annotations

import numpy as np


def _to_f64(xs) -> np.ndarray:
    return np.asarray(xs, dtype=np.float64)


def standard_scaler_transform(
    x_train, x_test, *, with_mean: bool = True, with_std: bool = True,
) -> np.ndarray:
    """(x - mean_train) / std_train (ddof=0). Degenerate std=0 → scale=1."""
    xtr = _to_f64(x_train)
    xte = _to_f64(x_test)
    mean = xtr.mean()
    var = xtr.var(ddof=0)
    std = float(np.sqrt(var)) if var > 0 else 1.0
    out = xte.copy()
    if with_mean:
        out = out - mean
    if with_std:
        out = out / std
    return out


def minmax_scaler_transform(
    x_train, x_test, *, feature_range=(0.0, 1.0),
) -> np.ndarray:
    """(x - min_train) * (high-low)/(max-min) + low. Constant → low."""
    xtr = _to_f64(x_train)
    xte = _to_f64(x_test)
    mn = xtr.min()
    mx = xtr.max()
    low, high = feature_range
    span = high - low
    data_range = mx - mn
    if data_range > 0:
        scale = span / data_range
        out = (xte - mn) * scale + low
    else:
        out = np.full_like(xte, low)
    return out


def maxabs_scaler_transform(x_train, x_test) -> np.ndarray:
    """x / max(|min_train|, |max_train|). All-zero → scale=1."""
    xtr = _to_f64(x_train)
    xte = _to_f64(x_test)
    max_abs = float(max(abs(xtr.min()), abs(xtr.max())))
    if max_abs == 0.0:
        max_abs = 1.0
    return xte / max_abs


def robust_scaler_transform(x_train, x_test) -> np.ndarray:
    """(x - median_train) / IQR_train. IQR=0 → scale=1."""
    xtr = _to_f64(x_train)
    xte = _to_f64(x_test)
    q25, q50, q75 = np.percentile(xtr, [25, 50, 75])
    iqr = float(q75 - q25)
    if iqr == 0.0:
        iqr = 1.0
    return (xte - q50) / iqr
