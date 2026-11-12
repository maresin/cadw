"""Binning: map continuous features to uint8 codes.

Three strategies:
- quantile: equal-frequency (default, LightGBM style)
- uniform: equal-width (simpler, more sensitive to outliers)
- weighted: equal Hessian mass (XGBoost weighted sketch)

All return (codes (n,d) uint8, thresholds list[np.ndarray]).
`thresholds[j]` = array of bin edges for feature j; codes = searchsorted.

Missing values (NaN): if `missing_bin=True`, NaN gets its own bin (last).
Otherwise: NaN → 0 (goes left, or explicit default direction handled
by tree builder via `default_left`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BinningResult:
    codes: np.ndarray               # (n, d) uint8
    thresholds: list                # list[np.ndarray] — per feature
    n_bins_per_feature: np.ndarray  # (d,) uint16
    has_missing_per_feature: np.ndarray  # (d,) bool


def _quantile_thresholds(col: np.ndarray, max_bins: int) -> np.ndarray:
    """Equal-frequency bin edges (midpoints between consecutive unique vals)."""
    finite = col[~np.isnan(col)]
    if len(finite) == 0:
        return np.array([], dtype=np.float64)
    unique_vals = np.unique(finite)
    if len(unique_vals) <= 1:
        return np.array([], dtype=np.float64)
    n_bins = min(max_bins, len(unique_vals))
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    targets = np.quantile(finite, qs)
    boundaries = []
    for t in targets:
        pos = np.searchsorted(unique_vals, t, side="right")
        if 0 < pos < len(unique_vals):
            boundaries.append(
                (unique_vals[pos - 1] + unique_vals[pos]) / 2.0)
    if not boundaries:
        return np.array([], dtype=np.float64)
    return np.unique(np.array(boundaries, dtype=np.float64))


def _uniform_thresholds(col: np.ndarray, max_bins: int) -> np.ndarray:
    """Equal-width bin edges, snapped to midpoints between unique vals."""
    finite = col[~np.isnan(col)]
    if len(finite) == 0:
        return np.array([], dtype=np.float64)
    unique_vals = np.unique(finite)
    if len(unique_vals) <= 1:
        return np.array([], dtype=np.float64)
    lo, hi = float(unique_vals[0]), float(unique_vals[-1])
    n_bins = min(max_bins, len(unique_vals))
    raw_edges = np.linspace(lo, hi, n_bins + 1)[1:-1]
    boundaries = []
    for e in raw_edges:
        pos = np.searchsorted(unique_vals, e, side="right")
        if 0 < pos < len(unique_vals):
            boundaries.append(
                (unique_vals[pos - 1] + unique_vals[pos]) / 2.0)
    if not boundaries:
        return np.array([], dtype=np.float64)
    return np.unique(np.array(boundaries, dtype=np.float64))


def _weighted_thresholds(
    col: np.ndarray, hess: np.ndarray, max_bins: int,
) -> np.ndarray:
    """Equal-Hessian-mass edges (XGBoost weighted sketch).

    Sort by col, accumulate hess, split at equal-mass boundaries.
    """
    finite = ~np.isnan(col)
    if not finite.any():
        return np.array([], dtype=np.float64)
    idx = np.where(finite)[0]
    order = idx[np.argsort(col[idx], kind="stable")]
    h_sorted = hess[order]
    c_sorted = col[order]
    total_h = float(h_sorted.sum())
    if total_h <= 0:
        return _quantile_thresholds(col, max_bins)
    cum_h = np.cumsum(h_sorted)
    # Targets: k/total for k=1..n_bins-1
    n_bins = min(max_bins, len(np.unique(c_sorted)))
    if n_bins <= 1:
        return np.array([], dtype=np.float64)
    targets = total_h * np.arange(1, n_bins) / n_bins
    unique_vals = np.unique(c_sorted)
    boundaries = []
    for t in targets:
        pos = int(np.searchsorted(cum_h, t, side="left"))
        pos = min(pos, len(c_sorted) - 1)
        val = float(c_sorted[pos])
        idx = np.searchsorted(unique_vals, val, side="right")
        if 0 < idx < len(unique_vals):
            boundaries.append(
                (unique_vals[idx - 1] + unique_vals[idx]) / 2.0)
    if not boundaries:
        return np.array([], dtype=np.float64)
    return np.unique(np.array(boundaries, dtype=np.float64))


def compute_binning(
    X: np.ndarray,
    max_bins: int = 255,
    mode: str = "quantile",
    *,
    hess: Optional[np.ndarray] = None,
    missing_bin: bool = False,
) -> BinningResult:
    """Fit binning on (n, d) X.

    mode ∈ {"quantile", "uniform", "weighted"}.
    weighted requires `hess` (per-row hessian from loss).
    """
    if mode not in ("quantile", "uniform", "weighted"):
        raise ValueError(f"binning mode must be quantile|uniform|weighted; "
                         f"got {mode!r}")
    if mode == "weighted" and hess is None:
        raise ValueError("weighted binning requires hess")

    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    codes = np.zeros((n, d), dtype=np.uint8)
    thresholds = []
    n_bins_per = np.zeros(d, dtype=np.uint16)
    has_miss = np.zeros(d, dtype=bool)

    for j in range(d):
        col = X[:, j]
        has_nan = bool(np.isnan(col).any())
        has_miss[j] = has_nan

        if mode == "quantile":
            thresh = _quantile_thresholds(col, max_bins)
        elif mode == "uniform":
            thresh = _uniform_thresholds(col, max_bins)
        else:
            thresh = _weighted_thresholds(col, hess, max_bins)

        codes_j = np.searchsorted(thresh, col, side="left").astype(np.uint8)

        if has_nan:
            if missing_bin:
                # NaN gets highest bin, shift others down if needed
                # For simplicity: keep NaN at n_bins (cap at 255)
                n_bins_eff = len(thresh) + 1
                if n_bins_eff > 255:
                    n_bins_eff = 255
                codes_j[np.isnan(col)] = min(255, n_bins_eff)
            else:
                # NaN -> 0 (goes left); tree builder handles default dir
                codes_j[np.isnan(col)] = 0

        codes[:, j] = codes_j
        thresholds.append(thresh)
        n_bins_per[j] = max(1, int(codes_j.max()) + 1)

    return BinningResult(codes, thresholds, n_bins_per, has_miss)


def apply_binning(
    X: np.ndarray, thresholds: list, missing_bin: bool = False,
) -> np.ndarray:
    """Apply pre-fitted binning to new data (predict path)."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    codes = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        col = X[:, j]
        thresh = thresholds[j]
        codes_j = np.searchsorted(thresh, col, side="left").astype(np.uint8)
        if np.isnan(col).any():
            if missing_bin:
                codes_j[np.isnan(col)] = min(255, len(thresh) + 1)
            else:
                codes_j[np.isnan(col)] = 0
        codes[:, j] = codes_j
    return codes
