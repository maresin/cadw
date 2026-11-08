"""Shared GBDT primitives (v0.9.151, P1 group D).

Losses / gradient stubs consumed by quantile and threshold GBDT variants:
- pinball_loss / pinball_grad (quantile regression; LinearQuantReg,
  GBDT_pinball, CQR)
- adaboost_weight (RUSBoost; exponential loss reweighting)
- cumulative_logit_prob (OrderedProbitBayesian; ordinal GBDT)

No Arrow, no I/O; numpy only.
"""
from __future__ import annotations

import numpy as np


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> np.ndarray:
    """Pinball (quantile) loss L_q(r) = max(q*r, (q-1)*r), r = y - y_pred.

    Args:
      y_true, y_pred: arrays, broadcastable.
      quantile: q in (0, 1).

    Returns:
      per-element loss (same shape as broadcast).

    Reference: Koenker & Bassett 1978.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1); got {quantile}")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    r = y_true - y_pred
    return np.where(r >= 0.0, quantile * r, (quantile - 1.0) * r)


def pinball_grad(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> np.ndarray:
    """Negative gradient of pinball loss (GBDT residual target).

    r_target = q if y_true >= y_pred else q - 1

    Used as GBDT leaf-fit target for the next boosting round.

    Args:
      y_true, y_pred: arrays, broadcastable.
      quantile: q in (0, 1).

    Returns:
      (y_true.shape) float64 — negative gradient.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1); got {quantile}")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.where(y_true >= y_pred, quantile, quantile - 1.0)


def adaboost_weight(
    alpha: float,
    misclassified: np.ndarray,
) -> np.ndarray:
    """AdaBoost sample weight multiplier.

    w_new = w_old * exp(alpha * (2*misclassified - 1))
    -> exp(+alpha) for misclassified samples, exp(-alpha) for correct.

    Args:
      alpha: base-classifier weight (typically alpha > 0).
      misclassified: boolean or 0/1 array (same shape as w_old).

    Returns:
      per-sample multiplier; multiply old weights by this.

    Reference: Freund & Schapire 1997.
    """
    mis = np.asarray(misclassified, dtype=np.float64)
    return np.exp(alpha * (2.0 * mis - 1.0))


def cumulative_logit_prob(
    eta: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Cumulative logit (proportional-odds) probabilities.

    Model: P(Y <= k) = sigmoid(theta_k - eta), k = 0..K-2
    -> P(Y = k) = P(Y <= k) - P(Y <= k-1), P(Y = K-1) = 1 - P(Y <= K-2)

    Args:
      eta: (n,) or scalar — linear predictor.
      theta: (K-1,) strictly increasing thresholds.

    Returns:
      (n, K) float64 — probabilities per class; rows sum to 1.

    Used as the final probability layer for ordinal GBDT (GBDT_threshold,
    OrderedProbitBayesian).
    """
    eta = np.asarray(eta, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 1:
        raise ValueError(f"theta must be 1-D; got {theta.shape}")
    if theta.size < 1:
        raise ValueError("theta must have at least 1 threshold")
    if (np.diff(theta) <= 0).any():
        raise ValueError("theta must be strictly increasing")

    K = theta.size + 1
    eta_flat = eta.reshape(-1)
    # sigmoid(theta_k - eta) for k=0..K-2, shape (n, K-1)
    z = theta[None, :] - eta_flat[:, None]
    cum_p = 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))  # (n, K-1)
    # P(Y=k) = cum_p[k] - cum_p[k-1], last = 1 - cum_p[-1]
    probs = np.empty((eta_flat.size, K), dtype=np.float64)
    probs[:, 0] = cum_p[:, 0]
    if K > 2:
        probs[:, 1:-1] = np.diff(cum_p, axis=1)
    probs[:, -1] = 1.0 - cum_p[:, -1]
    return probs.reshape(eta.shape + (K,))
