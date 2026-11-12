"""Dense per-level histogram for symmetric trees."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Histogram:
    """Per-level (or per-node) gradient/hessian histogram.

    G, H: (n_leaves, d, n_bins_max) float64. Only first n_bins_per_feature[j]
    columns are valid for feature j; rest are zero.
    """
    G: np.ndarray
    H: np.ndarray
    n_bins_per_feature: np.ndarray
    n_leaves: int


def build_histogram(
    codes: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    leaf_idx: np.ndarray,
    n_leaves: int,
    n_bins_per_feature: np.ndarray,
) -> Histogram:
    """Fast histogram: single bincount over flattened (n*d) entries.

    Vectorized: one bincount per (grad, hess) instead of d.
    """
    n, d = codes.shape
    n_bins_max = max(1, int(n_bins_per_feature.max()))

    # Combined flat index: leaf * (d*n_bins_max) + feature * n_bins_max + code
    codes_i64 = codes.astype(np.int64)
    # Clip codes to per-feature n_bins (defensive)
    for j in range(d):
        nb = int(n_bins_per_feature[j])
        if nb > 0 and not (codes_i64[:, j] < nb).all():
            codes_i64[:, j] = np.clip(codes_i64[:, j], 0, nb - 1)
    feat_offsets = (np.arange(d, dtype=np.int64) * n_bins_max)[None, :]
    leaf_offsets = (leaf_idx.astype(np.int64) * (d * n_bins_max))[:, None]
    flat = (leaf_offsets + feat_offsets + codes_i64).ravel()   # (n*d,)
    grad_flat = np.repeat(grad, d)
    hess_flat = np.repeat(hess, d)
    size = n_leaves * d * n_bins_max

    G_flat = np.bincount(flat, weights=grad_flat, minlength=size)
    H_flat = np.bincount(flat, weights=hess_flat, minlength=size)
    G = G_flat.reshape(n_leaves, d, n_bins_max)
    H = H_flat.reshape(n_leaves, d, n_bins_max)

    return Histogram(G, H, n_bins_per_feature, n_leaves)
