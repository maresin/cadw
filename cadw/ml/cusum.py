"""CUSUM: cumulative sum control chart for changepoint detection.

Two-sided CUSUM for detecting mean shifts:
  target = mean(x)
  s_pos[t] = max(0, s_pos[t-1] + x[t] - target - k)
  s_neg[t] = max(0, s_neg[t-1] - (x[t] - target) - k)
  score[t] = max(s_pos[t], s_neg[t])
  anomaly if score > h

k (slack) typically 0.5 * shift_size, h (decision) typically 3-5 * std.
Reference: Kats (facebookresearch/Kats) cusum_detection (Gaussian CUSUM
with iterative mean estimation).

Pure numpy — no sklearn/scipy/torch/kats in cadw/ (AST-guard).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CUSUMResult:
    decision_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    k: float
    h: float
    target_mean: float
    target_std: float
    contamination: float | None


def _probe_memory_limit_mb(guard) -> float | None:
    def _is_pos(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
    MB = ("max_memory_mb", "_max_memory_mb", "limit_mb", "budget_mb",
          "_budget_mb", "max_mb", "_max_mb", "memory_limit_mb",
          "_memory_limit_mb")
    for a in MB:
        v = getattr(guard, a, None)
        if _is_pos(v):
            return float(v)
    BYTES = ("max_memory_bytes", "_max_memory_bytes", "max_bytes",
             "_max_bytes", "limit_bytes", "_limit_bytes", "budget_bytes",
             "_budget_bytes")
    for a in BYTES:
        v = getattr(guard, a, None)
        if _is_pos(v):
            return float(v) / (1024.0 * 1024.0)
    try:
        d = guard.__dict__
    except AttributeError:
        return None
    for k, v in d.items():
        kl = k.lower()
        if not _is_pos(v):
            continue
        if "mb" in kl and any(t in kl for t in ("mem", "budget", "limit", "max")):
            return float(v)
    return None


def fit_cusum(
    series: np.ndarray,
    *,
    k: float = 0.5,
    h: float = 5.0,
    contamination: float | None = None,
    guard=None,
) -> CUSUMResult:
    """Two-sided CUSUM changepoint detection.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
        1D univariate time series.
    k : float, default=0.5
        Slack (allowance) parameter in std units. k=0.5 * shift_size / std
        is the standard choice.
    h : float, default=5.0
        Decision threshold in std units. h=4-5 gives ARL0 ~ 168-465
        for in-control detection.
    contamination : float | None, default=None
        If set, threshold = percentile(scores, 100*(1-c)) — overrides h.
    guard : MemoryGuard | None

    Returns
    -------
    CUSUMResult
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < 3:
        raise ValueError(f"series too short for CUSUM (min 3), got {n}")

    if guard is not None:
        est_bytes = 8 * n * 3
        limit_mb = _probe_memory_limit_mb(guard)
        if limit_mb is not None and limit_mb > 0:
            if est_bytes > limit_mb * 1024 * 1024:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"CUSUM estimated {est_bytes} bytes > "
                    f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    target = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-12:
        sigma = 1e-12

    # Standardise: k and h are interpreted in std units
    z = (x - target) / sigma
    k_scaled = float(k)
    h_scaled = float(h)

    s_pos = np.zeros(n, dtype=np.float64)
    s_neg = np.zeros(n, dtype=np.float64)
    s_pos_cur = 0.0
    s_neg_cur = 0.0
    for t in range(n):
        if t == 0:
            s_pos_cur = max(0.0, z[t] - k_scaled)
            s_neg_cur = max(0.0, -z[t] - k_scaled)
        else:
            s_pos_cur = max(0.0, s_pos_cur + z[t] - k_scaled)
            s_neg_cur = max(0.0, s_neg_cur - z[t] - k_scaled)
        # Record PRE-reset score (used for labels & output)
        s_pos[t] = s_pos_cur
        s_neg[t] = s_neg_cur
        # Standard CUSUM reset: alarm at score > h -> return to 0
        if s_pos_cur > h_scaled:
            s_pos_cur = 0.0
        if s_neg_cur > h_scaled:
            s_neg_cur = 0.0

    scores = np.maximum(s_pos, s_neg)

    if contamination is not None:
        threshold = float(np.percentile(
            scores, 100.0 * (1.0 - contamination)))
        if threshold <= 0:
            threshold = h_scaled
    else:
        threshold = h_scaled

    labels = (scores > threshold).astype(int)

    return CUSUMResult(
        decision_scores=scores,
        labels=labels,
        threshold=threshold,
        k=k,
        h=h,
        target_mean=target,
        target_std=sigma,
        contamination=contamination,
    )

