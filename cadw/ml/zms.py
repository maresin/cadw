"""ZMS: rolling z-score anomaly detector.

For each timestamp t with trailing window [t-w+1, t]:
  mu = mean(window)
  sigma = std(window)
  score[t] = |x[t] - mu| / sigma   (sigma floored to eps)
  anomaly if score > threshold

Simplest univariate point-anomaly detector after Hampel. Reference:
Merlion.models.anomaly.zms.ZMS (with TemporalResample + LagTransform
preprocessing stripped for native CADW use).

Pure numpy — no sklearn/scipy/merlion/torch in cadw/ (AST-guard).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ZMSResult:
    decision_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    window_size: int
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


def fit_zms(
    series: np.ndarray,
    *,
    window_size: int = 50,
    threshold: float = 3.0,
    contamination: float | None = None,
    guard=None,
) -> ZMSResult:
    """Rolling z-score anomaly detection.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
    window_size : int, default=50
        Trailing window length.
    threshold : float, default=3.0
        Z-score threshold. If `contamination` is set, used only as
        fallback when percentile-derived threshold is 0.
    contamination : float | None, default=None
        If set, threshold = percentile(scores, 100*(1-c)).
    guard : MemoryGuard | None

    Returns
    -------
    ZMSResult
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < window_size:
        raise ValueError(
            f"series length {n} < window_size {window_size}")

    if guard is not None:
        est_bytes = 8 * n * window_size
        limit_mb = _probe_memory_limit_mb(guard)
        if limit_mb is not None and limit_mb > 0:
            if est_bytes > limit_mb * 1024 * 1024:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"ZMS estimated {est_bytes} bytes > "
                    f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    # Trailing windows: pad window_size-1 with edge value
    padded = np.concatenate([np.full(window_size - 1, x[0]), x])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_size)
    mu = windows.mean(axis=1)
    sigma = windows.std(axis=1)
    sigma = np.where(sigma < 1e-12, 1e-12, sigma)

    scores = np.abs(x - mu) / sigma

    if contamination is not None:
        thr = float(np.percentile(
            scores, 100.0 * (1.0 - contamination)))
        if thr <= 0:
            thr = float(threshold)
    else:
        thr = float(threshold)

    labels = (scores > thr).astype(int)

    return ZMSResult(
        decision_scores=scores,
        labels=labels,
        threshold=thr,
        window_size=window_size,
        contamination=contamination,
    )

