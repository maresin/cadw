"""Causal inference primitives (v0.9.152, P1 L11, DEC-056).

- dr_score: doubly-robust (AIPW) pseudo-outcome
- aipw_ate: convenience mean
Consumers: DR_learner, CausalForest (x2), DR_learner_uplift.

No Arrow / no I/O; numpy only.
"""
from __future__ import annotations

import numpy as np


def dr_score(
    y: np.ndarray,
    t: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    e: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Doubly-robust (AIPW) pseudo-outcome per row.

    Formula (Robins, Rotnitzky & Zhao 1994; Chernozhukov et al. 2018):
      AIPW_i = mu1_i - mu0_i
               + (t_i / e_i) * (y_i - mu1_i)
               - ((1 - t_i) / (1 - e_i)) * (y_i - mu0_i)

    `mean(dr_score) = ATE` estimate with double robustness:
      - If mu0,mu1 are correct: second+third terms average to 0
      - If propensity e is correct: mu0,mu1 correction terms vanish
      - If both correct: exact (matches mu1 - mu0 mean)

    Args:
      y: (n,) observed outcome.
      t: (n,) treatment indicator, values in {0, 1}.
      mu0, mu1: (n,) outcome predictions under control / treated
                (typically from a T-learner, fit out-of-fold).
      e: (n,) propensity scores P(t=1 | x); clipped to [eps, 1-eps].
      eps: floor and ceiling for propensity (default 1e-8).

    Returns:
      (n,) float64 pseudo-outcome; mean = ATE estimate.

    Raises:
      ValueError: shape mismatch, t outside {0,1}, eps not in (0, 0.5).

    Memory: O(n) — one temp per arithmetic step. No streaming; caller
    controls n (typical n=1e5-1e6 for causal inference studies, fits RAM).
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    mu0 = np.asarray(mu0, dtype=np.float64)
    mu1 = np.asarray(mu1, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)

    if not (y.shape == t.shape == mu0.shape == mu1.shape == e.shape):
        raise ValueError(
            f"shape mismatch: y={y.shape}, t={t.shape}, mu0={mu0.shape}, "
            f"mu1={mu1.shape}, e={e.shape}")
    if y.ndim != 1:
        raise ValueError(f"all inputs must be 1-D; got y.ndim={y.ndim}")
    if not (0.0 < eps < 0.5):
        raise ValueError(f"eps must be in (0, 0.5); got {eps}")
    if y.size == 0:
        return np.empty(0, dtype=np.float64)

    # Validate t
    uniq = np.unique(t)
    if not np.isin(uniq, (0.0, 1.0)).all():
        raise ValueError(
            f"t must be binary {{0,1}}; got unique={uniq.tolist()}")

    # Validate / clip e
    if not np.isfinite(e).all():
        raise ValueError("e contains non-finite values")
    e_clip = np.clip(e, eps, 1.0 - eps)

    # AIPW
    term_mu = mu1 - mu0
    term_1 = (t / e_clip) * (y - mu1)
    term_0 = ((1.0 - t) / (1.0 - e_clip)) * (y - mu0)
    return term_mu + term_1 - term_0


def aipw_ate(
    y: np.ndarray,
    t: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    e: np.ndarray,
    *,
    eps: float = 1e-8,
) -> float:
    """Convenience: mean(dr_score(y, t, mu0, mu1, e)) = ATE estimate."""
    return float(dr_score(y, t, mu0, mu1, e, eps=eps).mean())
