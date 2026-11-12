"""Learning-to-Rank primitives (v0.9.154, P1 L13, DEC-056).

Gradient functions for pairwise / listwise LTR losses:
- ranknet_grad: pairwise logistic (Burges 2005)
- listnet_grad: listwise Plackett-Luce (Cao et al. 2007)
- lambdarank_grad: pairwise with NDCG@k delta (Burges 2010)

Consumers: PairwiseLogistic, ListNet (both described_only).
LambdaMART has its own inline `_lambda_gradients` (verified, not
touched) — this module serves new consumers.

No Arrow / no I/O; numpy only.
"""
from __future__ import annotations

import numpy as np


def _softmax_stable(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (subtract max)."""
    z = np.asarray(z, dtype=np.float64)
    zm = z - z.max()
    e = np.exp(zm)
    return e / e.sum()


def _ndcg_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    """NDCG@k with 2^rel - 1 gains (standard trec_eval convention)."""
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    n = y.size
    if n == 0:
        return 0.0
    k = min(k, n)
    gains = np.power(2.0, y) - 1.0
    positions = np.arange(1, n + 1, dtype=np.float64)
    log_denom = np.log2(positions + 1.0)

    # Ideal DCG: sort gains descending
    ideal_sorted = np.sort(gains)[::-1][:k]
    idcg = float((ideal_sorted / log_denom[:k]).sum())
    if idcg <= 0:
        return 0.0

    order = np.argsort(-s, kind="stable")[:k]
    dcg = float((gains[order] / log_denom[:k]).sum())
    return dcg / idcg


def ranknet_grad(
    s: np.ndarray,
    y: np.ndarray,
    *,
    sigma: float = 1.0,
) -> np.ndarray:
    """RankNet pairwise logistic gradient (Burges 2005).

    For each pair (i,j) with y_i > y_j:
      P_ij     = sigmoid(sigma * (s_i - s_j))
      lambda_ij = sigma * (1 - P_ij)
    Accumulated: grad_i += lambda_ij, grad_j -= lambda_ij.

    Args:
      s: (n,) scores.
      y: (n,) relevance labels (higher = more relevant).
      sigma: sigmoid scale (>0).

    Returns:
      (n,) float64 gradient with per-item pair aggregation.

    Complexity: O(n^2) pairs (intrinsic to RankNet); for n > ~5k
    callers should subsample per query. Loop-free via broadcasting.
    """
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if s.shape != y.shape:
        raise ValueError(f"s.shape={s.shape} != y.shape={y.shape}")
    if s.ndim != 1:
        raise ValueError(f"s, y must be 1-D; got s.ndim={s.ndim}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0; got {sigma}")
    n = s.size
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    # score_diff[i,j] = s_i - s_j; label_diff[i,j] = y_i - y_j
    sd = s[:, None] - s[None, :]           # (n, n)
    yd = y[:, None] - y[None, :]           # (n, n)
    # only i > j pairs (y_i > y_j means y_i - y_j > 0)
    mask = yd > 0.0
    # sigmoid(-sigma * sd) = 1 - sigmoid(sigma * sd)
    p_loss = 1.0 / (1.0 + np.exp(np.clip(sigma * sd, -500.0, 500.0)))
    lam = sigma * p_loss * mask
    # grad_i = sum_j lam[i,j] - sum_j lam[j,i]
    return lam.sum(axis=1) - lam.sum(axis=0)


def listnet_grad(
    s: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """ListNet (Plackett-Luce) top-1 gradient (Cao et al. 2007).

    With P_y = softmax(y) and P_s = softmax(s):
      grad_i = P_s(i) - P_y(i)        (== ∂CE/∂s_i)

    This is the **positive** CE gradient ∂L/∂s where
    L = -Σ_i P_y(i) · log P_s(i). To minimize L via gradient descent:
      coef -= lr · X.T @ grad

    NOTE (v0.9.226-fix1, P31): prior docstring claimed "Negative of
    CE gradient ... direction to move s" — that was WRONG (source is
    softmax(s) - softmax(y), the positive CE gradient). Consumers
    must use `-=` step (unlike ranknet_grad which returns negative
    direction-to-move — different convention).

    Args:
      s: (n,) scores.
      y: (n,) relevance labels (any real; softmax applies over query).

    Returns:
      (n,) float64 gradient; sum(grad) == 0 (since both softmax sum
      to 1, P_s - P_y sums to 0).

    Complexity: O(n).
    """
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if s.shape != y.shape:
        raise ValueError(f"s.shape={s.shape} != y.shape={y.shape}")
    if s.ndim != 1:
        raise ValueError(f"s, y must be 1-D; got s.ndim={s.ndim}")
    if s.size == 0:
        return np.empty(0, dtype=np.float64)
    return _softmax_stable(s) - _softmax_stable(y)


def lambdarank_grad(
    s: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 10,
    sigma: float = 1.0,
) -> np.ndarray:
    """LambdaRank gradient (Burges 2010).

    For each pair (i,j) with y_i > y_j:
      delta_ij     = |NDCG@k after swap(i,j) - NDCG@k before|
      lambda_ij    = sigma * delta_ij / (1 + exp(sigma*(s_i - s_j)))
    Accumulated: grad_i += lambda_ij, grad_j -= lambda_ij.

    delta_ij for a swap simplifies (Burges 2010) to:
      | (2^y_i - 2^y_j) * (1/log2(pos_j+1) - 1/log2(pos_i+1)) |

    Args:
      s: (n,) scores.
      y: (n,) relevance labels (higher = more relevant).
      k: NDCG truncation (only affects pairs at positions < k).
      sigma: sigmoid scale (>0).

    Returns:
      (n,) float64 gradient.

    Complexity: O(n^2 + n log n) (pair loop + initial sort).
    """
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if s.shape != y.shape:
        raise ValueError(f"s.shape={s.shape} != y.shape={y.shape}")
    if s.ndim != 1:
        raise ValueError(f"s, y must be 1-D; got s.ndim={s.ndim}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0; got {sigma}")
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    n = s.size
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    k = min(k, n)
    # current sorted positions (by descending score)
    order = np.argsort(-s, kind="stable")
    rank = np.empty(n, dtype=np.float64)
    rank[order] = np.arange(1, n + 1, dtype=np.float64)
    discount = 1.0 / np.log2(rank + 1.0)     # (n,)

    sd = s[:, None] - s[None, :]             # (n, n)
    yd = y[:, None] - y[None, :]             # (n, n)
    mask = yd > 0.0

    # NDCG@k delta per pair (only if both positions relevant)
    gain_diff = np.abs(np.power(2.0, y[:, None]) - np.power(2.0, y[None, :]))
    disc_diff = np.abs(discount[:, None] - discount[None, :])
    delta = gain_diff * disc_diff
    # mask out pairs at position >= k (both)
    in_k = (rank[:, None] <= k) & (rank[None, :] <= k)
    delta = delta * in_k

    # lambda
    p_loss = 1.0 / (1.0 + np.exp(np.clip(sigma * sd, -500.0, 500.0)))
    lam = sigma * delta * p_loss * mask
    return lam.sum(axis=1) - lam.sum(axis=0)
