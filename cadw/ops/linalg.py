"""Linear algebra primitives (v0.9.134 DEC-054).

Closed-form regressions + pairwise distances. No Arrow / no I/O;
numpy only.

History:
- v0.9.134: created. Moved ols_fit from cadw.ops.numeric; extracted
  sq_dists from cadw.ml.patch_memory_ad (forward extraction, AP10
  exception per DEC-056: 12+ prospective consumers — kmeans, kmeans_ad,
  kshape, tspq, iforest, eif, DBSCAN, HDBSCAN, LOF, bisecting,
  minibatch, spectral).
- v0.9.143: added nmf_mu (Lee & Seung 1999 multiplicative update).
  Promoted from .pingis/impl/J/nmf.py; consumers NMF x2 + Top2Vec
  + LDA_embedding (3 -> P1 per PRIMITIVES_BACKLOG L7).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def ols_fit(y: np.ndarray) -> Tuple[float, float]:
    """OLS (intercept, slope) on y[t] vs t.

    Returns (a, b) such that y_hat[t] = a + b*t. Handles n < 2
    (returns mean, 0) and degenerate x variance (den < 1e-12 -> mean, 0).
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


def sq_dists(z: np.ndarray, mem: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of z and rows of mem.

    Returns (n_z, n_mem). Identity:
      ||z_i - mem_k||^2 = z_i . z_i - 2 z_i . mem_k + mem_k . mem_k
    """
    z_sq = (z * z).sum(axis=1)
    m_sq = (mem * mem).sum(axis=1)
    return z_sq[:, None] - 2.0 * (z @ mem.T) + m_sq[None, :]


def nmf_mu(X_b: np.ndarray, H: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    """One mini-batch Lee & Seung mu update of H.

    W_b = NNLS projection of batch X_b onto row space of H (closed-form,
    clipped to >= 0). Then H <- H * (W_b^T X_b) / (W_b^T W_b H + eps).

    Inputs:
      X_b: dense non-negative batch, shape (b, V)
      H:   non-negative components, shape (k, V)
    Returns:
      H_new, same shape as H, non-negative.

    Reference: Lee & Seung (1999/2001) multiplicative updates; per-batch
    closed-form W is the "NNLS W step" adaptation for streaming NMF.
    """
    k = H.shape[0]
    HtH = H @ H.T + eps * np.eye(k, dtype=H.dtype)
    W_b = np.maximum(X_b @ H.T @ np.linalg.inv(HtH), 0.0)
    WtX = W_b.T @ X_b
    WtW = W_b.T @ W_b
    denom = WtW @ H + eps
    return H * (WtX / denom)
