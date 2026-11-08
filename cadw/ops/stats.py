"""Statistical primitives (v0.9.134 DEC-054).

Per-row normalization + autocorrelation. No Arrow / no I/O; numpy only.

History:
- v0.9.134: created. Moved znormalize_rows from cadw.ops.numeric;
  extracted acf from cadw.ts.transforms.AutoStationarizer._acf (forward
  extraction, AP10 exception per DEC-056: 5+ prospective consumers —
  binseg, pelt, bocpd, matrix_profile, stray).
"""
from __future__ import annotations

import numpy as np


def znormalize_rows(X: np.ndarray) -> np.ndarray:
    """Per-row z-normalization with zero-std guard.

    Each row standardized independently: mu and sd per row.
    Rows with sd < 1e-12 -> sd forced to 1 (output = X - mu, i.e. zeros).
    """
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def acf(y: np.ndarray, lag: int) -> float:
    """Autocorrelation at given lag (Pearson-style, normalized by var).

    Returns 0.0 if n <= lag or if variance < 1e-12 (degenerate series).
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n <= lag:
        return 0.0
    mean = float(y.mean())
    c0 = float(((y - mean) ** 2).mean())
    if c0 < 1e-12:
        return 0.0
    c = float(((y[:-lag] - mean) * (y[lag:] - mean)).mean())
    return c / c0
