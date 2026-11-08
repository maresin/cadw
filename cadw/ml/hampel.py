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
import pyarrow as pa


@dataclass
class HampelResult:
    decision_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    window_size: int
    n_sigma: float
    contamination: float | None

    def state_nbytes(self) -> int:
        """Persistent state: decision_scores(n) + labels(n) + header.

        v0.9.115: honest O(n) (CoxPH-precedent, bocpd/eif/kmeans_ad v0.9.114).
        Formula: 2*n*f64 + 264 (threshold 8 + dataclass header 256).
        bounded_by_domain n<=1e6 accepts this (16 MB at n=1e6).
        """
        return int(
            self.decision_scores.nbytes
            + self.labels.nbytes
            + 8    # threshold (Python float)
            + 256  # header
        )


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


# v0.9.130 (DEC-054): _probe_memory_limit_mb -> 
# cadw.core.memory_guard.probe_limit_mb (single source of truth).
from cadw.core.memory_guard import probe_limit_mb as _probe_memory_limit_mb



def fit_hampel(
    df_or_series,
    value_col=None,
    *,
    window_size: int = 7,
    n_sigma: float = 3.0,
    contamination: float | None = None,
    guard=None,
    output: str = "memory",
    batch_size: "int | None" = None,
) -> HampelResult:
    """Hampel filter anomaly detection (v0.9.115 dual dispatch).

    df path (new, v0.9.115):
        fit_hampel(df, value_col)  # collects to ndarray, calls series

    Legacy np.ndarray path (back-compat):
        fit_hampel(series)  # in-memory, equivalent to output="memory"

    output="memory"|"auto" -> in-memory. "disk" -> NotImplementedError
    (SpillArray for O(n) decision_scores is v0.9.116+, DEC-013).
    """
    if output not in ("memory", "auto"):
        raise NotImplementedError(
            f"fit_hampel: output={output!r} not supported "
            f"(decision_scores is O(n) ndarray; SpillArray for hampel "
            f"is v0.9.116+, DEC-013)")
    is_df = (hasattr(df_or_series, "_iter_filtered_batches")
             and hasattr(df_or_series, "_columns"))
    if is_df:
        if value_col is None:
            raise ValueError(
                "fit_hampel: value_col required when df is passed")
        if value_col not in df_or_series._columns:
            raise KeyError(
                f"fit_hampel: column {value_col!r} not in "
                f"{df_or_series._columns}")
        return _fit_hampel_df(
            df_or_series, value_col,
            window_size=window_size, n_sigma=n_sigma,
            contamination=contamination, guard=guard,
            batch_size=batch_size,
        )
    if value_col is not None:
        raise ValueError(
            "fit_hampel: value_col only used with df input; "
            "omit for np.ndarray")
    return _fit_hampel_series(
        df_or_series,
        window_size=window_size, n_sigma=n_sigma,
        contamination=contamination, guard=guard,
    )


def _fit_hampel_series(series, *, window_size, n_sigma, contamination, guard):
    """np.ndarray path (output= handled by public fit_hampel)."""
    return _fit_hampel_series_impl(
        series,
        window_size=window_size, n_sigma=n_sigma,
        contamination=contamination, guard=guard,
    )


def _fit_hampel_df(df, value_col, *, window_size, n_sigma, contamination,
                   guard, batch_size=None):
    """df path; collects series to ndarray then calls series path."""
    mg = guard if guard is not None else getattr(df, "_memory_guard", None)
    parts = []
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="hampel.batch", extra_factor=2.0)
        x_b = b.column(value_col).cast(
            pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        if x_b.size > 0:
            parts.append(x_b)
    if not parts:
        raise ValueError(f"fit_hampel: column {value_col!r} is empty")
    series = np.concatenate(parts) if len(parts) > 1 else parts[0]
    return _fit_hampel_series(
        series,
        window_size=window_size, n_sigma=n_sigma,
        contamination=contamination, guard=mg,
    )


def _fit_hampel_series_impl(
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

    # v0.9.116-fix (G2): check_state BEFORE output O(n) allocations.
    # Peak = med(n_w) + mad(n_w) + sigma(n_w) + scores(n) + labels(n)
    #       ~ 5*n_w*8 B (n_w ~ n). state_nbytes excludes this transient;
    # state is persistent post-fit. Probe: dRSS=41.4 MB at n=1e6.
    if has_check_state:
        guard.check_state(
            int(5 * n_w * 8 + 256), name="hampel.output")

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

