"""Numeric primitives — pure math, no domain (v0.9.134 DEC-054).

Elementwise numeric functions (erfinv, sigmoid). Statistics
(znormalize_rows) and linear algebra (ols_fit, sq_dists) have their
own modules per AI_CONSTITUTION §5.1 (one home per concept).

History:
- v0.9.132: created with ols_fit + erfinv (dedup from otm/theta).
- v0.9.133: added znormalize_rows (from autoencoder/patch_memory_ad).
- v0.9.134: home-correction — ols_fit -> linalg, znormalize -> stats,
  sigmoid <- optim. Now: erfinv, sigmoid only.

No pandas / no Arrow / no I/O; numpy + math only.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np




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


def sigmoid(x):
    """Numerically stable sigmoid: 1 / (1 + exp(-x)).

    Avoids overflow for large |x| by handling positive / negative
    branches separately. Returns same dtype/shape as input.
    Prior to v0.9.134 in cadw.ops.optim (moved here: elementwise
    numeric, not SGD).
    """
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def norm_cdf(x):
    """Standard normal CDF Φ(x) = 0.5 * (1 + erf(x / √2)).

    Uses stdlib math.erf (no external dependency). Accepts scalar or
    array (via numpy vectorize when array).
    """
    import math
    import numpy as _np
    if isinstance(x, _np.ndarray):
        return _np.vectorize(lambda v: 0.5 * (1.0 + math.erf(
            float(v) / math.sqrt(2.0))), otypes=[_np.float64])(x)
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


