"""Row sampling strategies: none, random, GOSS."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SampleMask:
    """Active-row mask + optional re-weight factor.

    `weights`: (n,) float — multiply grad/hess per row before histogram.
    For GOSS: sampled small-grad rows get factor (1-a)/b to compensate.
    """
    mask: np.ndarray       # (n,) bool — rows used in this tree
    weights: np.ndarray    # (n,) float — grad/hess multiplier (1.0 default)


def sample_none(n: int) -> SampleMask:
    return SampleMask(
        mask=np.ones(n, dtype=bool),
        weights=np.ones(n, dtype=np.float64),
    )


def sample_random(n: int, rate: float, rng) -> SampleMask:
    if rate >= 1.0:
        return sample_none(n)
    if rate <= 0.0:
        raise ValueError("subsample must be > 0")
    mask = rng.rand(n) < rate
    # Ensure at least 2 rows so histograms have data
    if mask.sum() < 2:
        idx = rng.choice(n, size=2, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
    weights = np.ones(n, dtype=np.float64)
    return SampleMask(mask, weights)


def sample_goss(
    n: int, grad: np.ndarray,
    top_rate: float = 0.2, other_rate: float = 0.1, rng=None,
) -> SampleMask:
    """GOSS (LightGBM): top `top_rate` by |grad| + random `other_rate` of rest.

    Sample weights: top rows weight=1, other rows weight=(1-a)/b.
    This keeps expected grad/hess sums unbiased.
    """
    if rng is None:
        rng = np.random
    if not (0 < top_rate < 1) or not (0 < other_rate < 1):
        raise ValueError("top_rate, other_rate must be in (0, 1)")

    abs_grad = np.abs(grad)
    n_top = max(1, int(n * top_rate))
    n_other = max(1, int(n * other_rate))

    # Top indices by |grad|
    top_idx = np.argpartition(-abs_grad, n_top - 1)[:n_top]
    top_mask = np.zeros(n, dtype=bool)
    top_mask[top_idx] = True

    # Random from the rest
    rest_idx = np.where(~top_mask)[0]
    n_other = min(n_other, len(rest_idx))
    other_idx = rng.choice(rest_idx, size=n_other, replace=False)

    mask = np.zeros(n, dtype=bool)
    mask[top_idx] = True
    mask[other_idx] = True

    weights = np.ones(n, dtype=np.float64)
    # Compensate small-grad sampling: weight = (1-a)/b for others
    weights[other_idx] = (1.0 - top_rate) / max(other_rate, 1e-9)

    return SampleMask(mask, weights)
