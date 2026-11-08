"""Neural-network weight initialization (v0.9.134 DEC-054).

Isolated from cadw.ml._common (which mixes Arrow + NN utilities).

History:
- v0.9.134: created. Moved _xavier_init from cadw.ml._common
  (semantic home: NN-specific).
"""
from __future__ import annotations

import numpy as np


def _xavier_init(
    fan_in: int, fan_out: int, rng: np.random.Generator,
) -> np.ndarray:
    """Xavier/Glorot uniform weight init: U(-sqrt(6/(fan_in+fan_out)),
    sqrt(6/(fan_in+fan_out)))."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))
