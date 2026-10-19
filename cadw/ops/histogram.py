"""Histogram reduction ops — 1D/2D (v0.7.4a).

`Histogram2DOp`: 2D binning via Reduce. State O(bins_x × bins_y).
Used by cadw.viz._aggregate.hist2d for aggregate-first scatter plots.

For 1D, use `cadw.ops.reduce.HistogramOp` (already exists).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyarrow as pa

from cadw.ops.reduce import ReduceOp


class Histogram2DOp(ReduceOp):
    """2D histogram: counts[bins_x, bins_y].

    Requires fixed x_min/max, y_min/max (two-pass or sample-based).
    Nulls in either column → row skipped.
    """

    def __init__(
        self,
        x_col: str,
        y_col: str,
        *,
        bins_x: int,
        bins_y: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        if bins_x < 1 or bins_y < 1:
            raise ValueError("Histogram2DOp: bins must be >= 1")
        if x_max <= x_min:
            raise ValueError("Histogram2DOp: x_max must be > x_min")
        if y_max <= y_min:
            raise ValueError("Histogram2DOp: y_max must be > y_min")
        self.x_col = x_col
        self.y_col = y_col
        self.bins_x = int(bins_x)
        self.bins_y = int(bins_y)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

    def init(self):
        return np.zeros((self.bins_x, self.bins_y), dtype=np.int64)

    def update(self, state, batch):
        x = _to_f64(batch.column(self.x_col))
        y = _to_f64(batch.column(self.y_col))
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        if x.shape[0] == 0:
            return state
        x_idx = np.clip(
            ((x - self.x_min) / (self.x_max - self.x_min)
             * self.bins_x).astype(np.int64),
            0, self.bins_x - 1,
        )
        y_idx = np.clip(
            ((y - self.y_min) / (self.y_max - self.y_min)
             * self.bins_y).astype(np.int64),
            0, self.bins_y - 1,
        )
        np.add.at(state, (x_idx, y_idx), 1)
        return state

    def merge(self, a, b):
        return a + b

    def state_nbytes(self, state):
        return int(state.nbytes)

    def estimate_state_nbytes(self):
        return self.bins_x * self.bins_y * 8

    def finalize(self, state):
        return {
            "counts": state,
            "x_edges": np.linspace(self.x_min, self.x_max, self.bins_x + 1),
            "y_edges": np.linspace(self.y_min, self.y_max, self.bins_y + 1),
        }


def _to_f64(arr: pa.Array) -> np.ndarray:
    np_arr = arr.to_numpy(zero_copy_only=False)
    if np_arr.dtype == object:
        out = np.empty(len(np_arr), dtype=np.float64)
        for i, v in enumerate(np_arr):
            out[i] = np.nan if v is None else float(v)
        return out
    return np_arr.astype(np.float64, copy=False)
