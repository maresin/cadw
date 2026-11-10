"""BinarySegmentation: greedy top-down change-point detection (L2 cost).

Recursively split the series at the point that maximizes L2 cost
reduction, until n_bkps change-points detected. Reference:
ruptures Binseg (Truong, Oudre, Vayatis 2020).

Algorithm (paper faithful):
  cost(segment) = sum((x - mean)^2)
  parent_cost = cost(full)
  for split i in [min_size, n-min_size]:
      gain = parent_cost - cost(left) - cost(right)
  best_i = argmax gain
  Repeat on best segment until n_bkps boundaries found.

Score: 1.0 at each detected changepoint timestamp, 0.0 elsewhere.
Detections are timestamps, not intervals; the source of
detection_delay / timing error evaluation in D3.2.

Pure numpy — no sklearn/scipy/ruptures/torch in cadw/ (AST-guard
test_no_bench_deps_in_core.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BinSegResult:
    decision_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    changepoints: np.ndarray
    segments: list
    n_bkps: int
    min_size: int
    contamination: float | None


# v0.9.130 (DEC-054): _probe_memory_limit_mb -> 
# cadw.core.memory_guard.probe_limit_mb (single source of truth).
from cadw.core.memory_guard import probe_limit_mb as _probe_memory_limit_mb


def _segment_cost(x: np.ndarray) -> float:
    """L2 cost (SSE): sum((x - mean)^2)."""
    if len(x) == 0:
        return 0.0
    mu = float(x.mean())
    return float(((x - mu) ** 2).sum())


def _best_split_from_cs(
    cs: np.ndarray, cs2: np.ndarray,
    lo: int, hi: int, min_size: int,
) -> tuple[int, float]:
    """Return (split_index_global, gain) using global prefix arrays.

    v0.9.181: replaces per-call np.cumsum(x[lo:hi]) (O(hi-lo) alloc each
    call, cumulatively O(n*n_bkps) temporaries) with O(1)-per-query access
    on precomputed global prefix arrays.

    Parameters
    ----------
    cs : np.ndarray, shape (n,)
        Global prefix sums: cs[i] = sum(x[0:i+1]).
    cs2 : np.ndarray, shape (n,)
        Global prefix sums of squares: cs2[i] = sum(x[0:i+1]^2).
    lo, hi : int
        Segment boundaries: segment = x[lo:hi], 0 <= lo < hi <= n.
    min_size : int
        Minimum segment length.

    Returns
    -------
    (split_index_global, gain) or (-1, 0.0) if no valid split.
    """
    n_seg = hi - lo
    if n_seg < 2 * min_size:
        return -1, 0.0

    cs_lo_minus_1 = float(cs[lo - 1]) if lo > 0 else 0.0
    cs2_lo_minus_1 = float(cs2[lo - 1]) if lo > 0 else 0.0

    s_parent = float(cs[hi - 1]) - cs_lo_minus_1
    s2_parent = float(cs2[hi - 1]) - cs2_lo_minus_1
    parent_cost = s2_parent - s_parent * s_parent / n_seg

    best_i, best_gain = -1, 0.0
    for i in range(lo + min_size, hi - min_size + 1):
        n_l = i - lo
        n_r = hi - i
        s_l = float(cs[i - 1]) - cs_lo_minus_1
        s2_l = float(cs2[i - 1]) - cs2_lo_minus_1
        cost_l = s2_l - s_l * s_l / n_l
        s_r = s_parent - s_l
        s2_r = s2_parent - s2_l
        cost_r = s2_r - s_r * s_r / n_r
        gain = parent_cost - cost_l - cost_r
        if gain > best_gain:
            best_gain, best_i = gain, i
    return best_i, best_gain


def _best_split(x: np.ndarray, min_size: int) -> tuple[int, float]:
    """Legacy wrapper (v0.9.181): returns local split index for x[0:n].

    Kept for backward-compat with tests that call _best_split directly.
    Internally uses _best_split_from_cs (single cumsum build per call —
    acceptable for one-off utility calls, not in the fit loop).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 2 * min_size:
        return -1, 0.0
    cs = np.cumsum(x)
    cs2 = np.cumsum(x * x)
    gi, gain = _best_split_from_cs(cs, cs2, 0, n, min_size)
    if gi < 0:
        return -1, 0.0
    return gi, gain  # gi is local index because lo=0


def fit_binseg(
    series: np.ndarray,
    *,
    n_bkps: int = 3,
    min_size: int = 20,
    contamination: float | None = None,
    guard=None,
) -> BinSegResult:
    """Binary segmentation change-point detection.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
    n_bkps : int, default=3
        Number of change-points to detect.
    min_size : int, default=20
        Minimum segment length.
    contamination : float | None, default=None
        If set, threshold = percentile(1-contamination) on scores,
        otherwise threshold = 0.5 (score > 0.5 = flagged cp).
    guard : MemoryGuard | None

    Returns
    -------
    BinSegResult
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < 2 * min_size:
        raise ValueError(
            f"series length {n} < 2*min_size={2*min_size}")
    if n_bkps < 1:
        raise ValueError(f"n_bkps must be >= 1, got {n_bkps}")
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}")

    # v0.9.181: explicit two-path guard. Peak = x + cs + cs2 + scores + labels
    # = 5 * n * f64 (+ 512 header); decision_scores + labels materialized
    # O(n) in RAM by design (streaming refactor v0.9.185+).
    est_bytes = 5 * n * 8 + 512
    if guard is not None:
        has_check_state = (
            hasattr(guard, "check_state") and callable(guard.check_state))
        if has_check_state:
            guard.check_state(est_bytes, name="binseg.state")
        else:
            limit_mb = _probe_memory_limit_mb(guard)
            if limit_mb is not None and limit_mb > 0:
                if est_bytes > limit_mb * 1024 * 1024:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"BinSeg estimated {est_bytes} bytes > "
                        f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    # v0.9.181: global prefix arrays computed ONCE (previously _best_split
    # recomputed np.cumsum(x[lo:hi]) per segment per iteration).
    cs = np.cumsum(x)
    cs2 = np.cumsum(x * x)

    segments: list[tuple[int, int]] = [(0, n)]
    while len(segments) < n_bkps + 1:
        best_seg_idx = -1
        best_split_global = -1
        best_gain = 0.0
        for k, (lo, hi) in enumerate(segments):
            gi, g = _best_split_from_cs(cs, cs2, lo, hi, min_size)
            if gi < 0:
                continue
            if g > best_gain:
                best_gain = g
                best_seg_idx = k
                best_split_global = gi
        if best_seg_idx < 0 or best_gain <= 1e-12:
            break
        lo, hi = segments[best_seg_idx]
        mid = best_split_global
        segments[best_seg_idx] = (lo, mid)
        segments.insert(best_seg_idx + 1, (mid, hi))

    changepoints = np.array([hi for (_, hi) in segments[:-1]], dtype=np.int64)
    scores = np.zeros(n, dtype=np.float64)
    for cp in changepoints:
        if 0 <= int(cp) < n:
            scores[int(cp)] = 1.0

    if contamination is not None:
        threshold = float(np.percentile(
            scores, 100.0 * (1.0 - contamination)))
        if threshold <= 0:
            threshold = 0.5
    else:
        threshold = 0.5

    labels = (scores > threshold).astype(int)

    return BinSegResult(
        decision_scores=scores,
        labels=labels,
        threshold=threshold,
        changepoints=changepoints,
        segments=segments,
        n_bkps=n_bkps,
        min_size=min_size,
        contamination=contamination,
    )

