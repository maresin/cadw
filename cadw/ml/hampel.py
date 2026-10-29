"""Hampel filter: rolling median + MAD anomaly detector.

For each timestamp t with window [t-half, t+half]:
  med = median(window)
  mad = median(|window - med|)
  sigma = 1.4826 * mad   (MAD -> std for Gaussian)
  score[t] = |x[t] - med| / sigma
  anomaly if score > n_sigma (default 3.0)

Simplest univariate point-anomaly baseline. ~O(n * w * log w) with
sliding_window_view + np.median.

Reference: sktime.transformations.series.outlier_detection.HampelDetector.
Pure numpy — no sklearn/scipy/aeon/torch in cadw/ (AST-guard).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HampelResult:
    decision_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    window_size: int
    n_sigma: float
    contamination: float | None


def _resolve_chunk_nw_hampel(w: int, guard=None) -> int:
    """v0.9.91-S6a: chunk size so peak RAM stays O(chunk*w).

    Target: 8 MB default per chunk; if guard exposes state_limit_bytes,
    cap to state_limit // 8. Bytes per window: window view (w*8) +
    abs(window) copy (w*8) = 2 * w * 8.
    """
    default_target = 8 * 1024 * 1024
    target_bytes = default_target
    if guard is not None:
        sl = getattr(guard, "state_limit_bytes", None)
        if isinstance(sl, (int, float)) and sl > 0:
            target_bytes = min(int(target_bytes), max(1, int(sl) // 8))
    bytes_per_window = 8 * max(int(w), 1) * 2
    n = max(100, target_bytes // bytes_per_window)
    return min(10_000, n)


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


def fit_hampel(
    series: np.ndarray,
    *,
    window_size: int = 7,
    n_sigma: float = 3.0,
    contamination: float | None = None,
    guard=None,
) -> HampelResult:
    """Hampel filter anomaly detection.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
    window_size : int, default=7
        Odd window length (auto-incremented if even).
    n_sigma : float, default=3.0
        Threshold in MAD units. If `contamination` is set, used only as
        fallback when percentile-derived threshold is 0.
    contamination : float | None, default=None
        If set (0 < c < 1), threshold = percentile(scores, 100*(1-c));
        otherwise threshold = n_sigma.
    guard : MemoryGuard | None

    Returns
    -------
    HampelResult
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < 3:
        raise ValueError(f"series too short for Hampel (min 3), got {n}")
    if window_size < 3:
        raise ValueError(
            f"window_size must be >= 3, got {window_size}")
    if window_size % 2 == 0:
        window_size += 1
    if window_size > n:
        window_size = n if n % 2 == 1 else n - 1

    # v0.9.91-S6a: legacy guard path (fallback — no check_state).
    # Kept for duck-typed guards exposing only max_memory_mb.
    # For real MemoryGuard, chunked loop below uses check_state per chunk.
    has_check_state = (
        guard is not None and hasattr(guard, "check_state")
        and callable(guard.check_state)
    )
    if guard is not None and not has_check_state:
        # Peak = window view + abs copy = 2 * n * w * 8.
        est_bytes = int(n * window_size * 8 * 2)
        limit_mb = _probe_memory_limit_mb(guard)
        if limit_mb is not None and limit_mb > 0:
            if est_bytes > limit_mb * 1024 * 1024:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"Hampel estimated {est_bytes} bytes > "
                    f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    half = window_size // 2
    padded = np.pad(x, half, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window_size)
    n_w = view.shape[0]
    chunk = _resolve_chunk_nw_hampel(window_size, guard)

    med = np.empty(n_w, dtype=np.float64)
    mad = np.empty(n_w, dtype=np.float64)
    for i in range(0, n_w, chunk):
        end = min(i + chunk, n_w)
        nb = end - i
        if has_check_state:
            guard.check_state(
                nb * window_size * 8 * 2, name="hampel.chunk")
        blk = np.ascontiguousarray(view[i:end])
        m_blk = np.median(blk, axis=1)
        med[i:end] = m_blk
        mad[i:end] = np.median(np.abs(blk - m_blk[:, None]), axis=1)

    sigma = 1.4826 * mad
    sigma = np.where(sigma < 1e-12, 1e-12, sigma)

    scores = np.abs(x - med) / sigma

    if contamination is not None:
        threshold = float(np.percentile(
            scores, 100.0 * (1.0 - contamination)))
        if threshold <= 0:
            threshold = float(n_sigma)
    else:
        threshold = float(n_sigma)

    labels = (scores > threshold).astype(int)

    return HampelResult(
        decision_scores=scores,
        labels=labels,
        threshold=threshold,
        window_size=window_size,
        n_sigma=n_sigma,
        contamination=contamination,
    )

