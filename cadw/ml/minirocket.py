"""MiniRocket-style random convolution features (D.4.1, v0.9.37).

Reference: Dempster, Schmidt, Webb, *MINIROCKET: A Very Fast (Almost
Deterministic) Transform for Time Series Classification*, KDD 2021.
See DECISION_LOG for algorithm reference.

Design notes (deliberate, documented):
- Kernels are length 9 with weights in {-1, +1}; exactly three +1 weights
  are placed, one in each third of the kernel, drawn from a seeded RNG so
  the transform is deterministic across runs (paper: fixed kernel set).
- Dilation grows geometrically per dilation group; num_kernels kernels are
  spread over the dilation groups.
- Features are PPV (proportion of positive values) per kernel per series.
- Memory is O(max series length + num_kernels); series are processed one
  at a time and only the accumulated PPV sums are retained.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.dataframe import CadwDataFrame
from cadw.ml._logging import log_fit

KERNEL_LEN = 9


def make_kernels(num_kernels=84, seed=0, max_dilation=32):
    """Deterministic MiniRocket kernel set: (weights, dilation) pairs."""
    if num_kernels < 1:
        raise ValueError("make_kernels: num_kernels must be >= 1")
    if max_dilation < 1:
        raise ValueError("make_kernels: max_dilation must be >= 1")
    rng = np.random.RandomState(seed)
    dilations = []
    d = 1
    while len(dilations) < num_kernels:
        dilations.append(int(d))
        if d >= max_dilation:
            d = 1
        else:
            d = min(d * 2, max_dilation)
    kernels = []
    for d in dilations:
        w = -np.ones(KERNEL_LEN, dtype=np.float64)
        for third in range(3):
            pos = third * 3 + int(rng.randint(0, 3))
            w[pos] = 1.0
        kernels.append((w, int(d)))
    return kernels


def _ppv_series(x, weights, dilation):
    """PPV of the dilated convolution of one series with one kernel."""
    x = np.asarray(x, dtype=np.float64)
    n_out = x.size - (KERNEL_LEN - 1) * dilation
    if n_out <= 0:
        return 0.0
    acc = np.zeros(n_out, dtype=np.float64)
    for j in range(KERNEL_LEN):
        w = weights[j]
        if w == 0.0:
            continue
        acc += w * x[j * dilation: j * dilation + n_out]
    return float(np.mean(acc > 0.0))


# v0.9.89-D4-refactor: _iter_series extracted to cadw.ml._series
from cadw.ml._series import iter_series as _iter_series

class MiniRocketModel:
    """Fitted MiniRocket feature transform."""

    def __init__(self, kernels, value_col, id_col, params):
        self.kernels_ = list(kernels)
        self.value_col_ = value_col
        self.id_col_ = id_col
        self.params_ = dict(params)
        self.fitted_ = True

    def state_nbytes(self):
        return int(len(self.kernels_) * (KERNEL_LEN * 8 + 8) + 1024)

    def to_dict(self):
        return {"params": self.params_, "value_col": self.value_col_,
                "id_col": self.id_col_, "num_kernels": len(self.kernels_)}

    def transform(self, df, memory_guard=None) -> np.ndarray:
        """Return an (n_series, num_kernels) PPV feature matrix."""
        mg = memory_guard or df._memory_guard
        rows = []
        for _sid, series in _iter_series(df, self.value_col_, self.id_col_,
                                         mg, "minirocket.transform"):
            feats = np.array([_ppv_series(series, w, d)
                              for (w, d) in self.kernels_], dtype=np.float64)
            # v0.9.90-S3 (G2): pre-check BEFORE append.
            mg.check_state((len(rows) + 1) * len(self.kernels_) * 8,
                           name="minirocket.features")
            rows.append(feats)
        if not rows:
            return np.zeros((0, len(self.kernels_)), dtype=np.float64)
        return np.vstack(rows)


def fit_minirocket(df, value_col, *, id_col=None, num_kernels=84,
                   seed=0, max_dilation=32,
                   memory_guard: Optional[MemoryGuard] = None):
    """Fit (construct) the deterministic MiniRocket kernel set.

    The transform itself is deterministic given the kernel set, so fitting
    amounts to validating the input and building the kernels. Feature
    extraction streams one series at a time.
    """
    if value_col not in df._columns:
        raise KeyError(f"fit_minirocket: column {value_col!r} not in {df._columns}")
    if id_col is not None and id_col not in df._columns:
        raise KeyError(f"fit_minirocket: column {id_col!r} not in {df._columns}")
    mg = memory_guard or df._memory_guard
    kernels = make_kernels(num_kernels=num_kernels, seed=seed,
                           max_dilation=max_dilation)
    model = MiniRocketModel(kernels, value_col, id_col,
                            {"num_kernels": num_kernels, "seed": seed,
                             "max_dilation": max_dilation})
    log_fit("minirocket", state_bytes=model.state_nbytes())
    mg.check_state(model.state_nbytes(), name="minirocket.model")
    return model
