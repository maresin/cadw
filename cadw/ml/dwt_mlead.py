"""DWT-MLEAD: Multilevel DWT-based anomaly detection (Thill, Konen, Baeck 2017).

Non-learning density-based detector for univariate time series.
Multilevel Haar DWT → sliding windows → Gaussian likelihood (Mahalanobis)
→ threshold → aggregate across levels.

Pure numpy — no sklearn/aeon/torch imports in cadw/ (AST-guard
test_no_bench_deps_in_core.py).

Reference:
  Thill, M., Konen, W., Baeck, T. (2017). "Time Series Anomaly Detection
  with Discrete Wavelet Transformation". ESANN 2017.
  aeon: aeon.anomaly_detection.DWT_MLEAD
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa


@dataclass
class DwtMleadResult:
    """Result of fit_dwt_mlead.

    Attributes
    ----------
    anomaly_scores : np.ndarray, shape (n,)
        Per-point anomaly count (number of DWT levels flagging the point).
    is_anomaly : np.ndarray, shape (n,), bool
        True where anomaly_scores >= min_anomaly_consistency.
    level_counts : np.ndarray, shape (max_level,)
        Number of flagged windows at each DWT level (index i = level i+1).
    start_level, max_level : int
        DWT level range used.
    quantile_epsilon : float
        Threshold quantile parameter.
    min_anomaly_consistency : int
        Minimum number of levels flagging a point.
    """
    anomaly_scores: np.ndarray
    is_anomaly: np.ndarray
    level_counts: np.ndarray
    start_level: int
    max_level: int
    quantile_epsilon: float
    min_anomaly_consistency: int

    def state_nbytes(self) -> int:
        """Persistent state: anomaly_scores(n f64) + is_anomaly(n bool)
        + level_counts(max_level i64) + header.

        v0.9.115: honest O(n) (CoxPH-precedent, bocpd/eif/kmeans_ad v0.9.114).
        Formula: n*f64 + n*bool + max_level*i64 + 264.
        max_level = log2(next_pow2(n)) <= 20 for n<=1e6.
        bounded_by_domain n<=1e6 accepts this (~9 MB at n=1e6).
        """
        return int(
            self.anomaly_scores.nbytes
            + self.is_anomaly.nbytes
            + self.level_counts.nbytes
            + 8    # representative scalar (4 fields collapsed per precedent)
            + 256  # header
        )


def _pad_to_next_power_of_two(x: np.ndarray) -> np.ndarray:
    """Pad series (wrap mode) to next power of 2 >= len(x)."""
    n = len(x)
    if n <= 1:
        return x.copy()
    target = 1 << (n - 1).bit_length()
    if target == n:
        return x.copy()
    pad = target - n
    return np.concatenate([x, x[:pad]])


def _haar_dwt_level(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-level Haar DWT. Requires even-length input.

    Returns (approx, detail), both length len(x)/2.
    """
    n = len(x)
    if n % 2 != 0:
        raise ValueError(f"Haar DWT requires even length, got {n}")
    x0 = x[0::2]
    x1 = x[1::2]
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    return (x0 + x1) * inv_sqrt2, (x0 - x1) * inv_sqrt2


def _multilevel_haar_dwt(x: np.ndarray) -> list[np.ndarray]:
    """Return list of detail coefficient arrays (index 0 = level 1, finest)."""
    details = []
    current = x
    while len(current) >= 2:
        current, detail = _haar_dwt_level(current)
        details.append(detail)
    return details


# v0.9.87-P4b: _sliding_windows removed (dead helper after streaming
# refactor; was the only G3 violation in this module).


# v0.9.87-P4b (audit HIGH-1 full fix): streaming chunk helpers for
# per-level mahalanobis distances. Peak RAM O(chunk_nw*w + w^2)
# regardless of n. See METHODOLOGY §5 (windowed state_type).


def _resolve_chunk_nw(d: int, guard=None) -> int:
    """Chunk size so per-chunk peak RAM stays within budget.

    Default target: 8 MB per chunk. If guard exposes
    state_limit_bytes, cap target to state_limit // 8.
    """
    default_target = 8 * 1024 * 1024
    target_bytes = default_target
    if guard is not None:
        state_limit = getattr(guard, "state_limit_bytes", None)
        if (isinstance(state_limit, (int, float))
                and state_limit > 0):
            target_bytes = min(int(target_bytes),
                               max(1, int(state_limit) // 8))
    bytes_per_window = 8 * max(int(d), 1) * 6
    n = max(100, target_bytes // bytes_per_window)
    return min(10_000, n)


def _iter_detail_window_chunks(detail_1d: np.ndarray, w: int,
                               chunk_nw: int):
    """Yield (chunk, start) chunks of shape (nb, w) from 1D detail.

    Only `chunk_nw` windows materialized at a time (nb <= chunk_nw).
    """
    n = len(detail_1d)
    if w < 1 or w > n:
        return
    n_w = n - w + 1
    offsets = np.arange(w, dtype=np.int64)
    for start in range(0, n_w, chunk_nw):
        end = min(start + chunk_nw, n_w)
        nb = end - start
        abs_starts = np.arange(start, end, dtype=np.int64)
        idx = abs_starts[:, None] + offsets[None, :]
        yield detail_1d[idx], start


def _stream_mahalanobis(detail_1d: np.ndarray, w: int,
                        chunk_nw: int) -> np.ndarray:
    """Streaming mahalanobis distances; 3 passes, O(chunk_nw*w + w^2) peak."""
    n = len(detail_1d)
    if w < 1 or w > n:
        return np.zeros(0, dtype=np.float64)
    n_w = n - w + 1
    if n_w < 2:
        return np.zeros(n_w, dtype=np.float64)

    total = np.zeros(w, dtype=np.float64)
    for chunk, _ in _iter_detail_window_chunks(detail_1d, w, chunk_nw):
        total += chunk.sum(axis=0)
    mean = total / n_w

    M2 = np.zeros((w, w), dtype=np.float64)
    for chunk, _ in _iter_detail_window_chunks(detail_1d, w, chunk_nw):
        centered = chunk - mean
        M2 += centered.T @ centered
    cov = M2 / max(n_w - 1, 1) + np.eye(w) * 1e-8
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    scores_chunks = []
    for chunk, _ in _iter_detail_window_chunks(detail_1d, w, chunk_nw):
        centered = chunk - mean
        md_sq = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
        scores_chunks.append(np.sqrt(np.maximum(md_sq, 0.0)))
    if not scores_chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(scores_chunks)


# NOTE: _mahalanobis_distances below is kept as a numerical reference
# for test_streaming_mahalanobis_matches_batch; production code uses
# _stream_mahalanobis. Do not call from non-test code.
def _mahalanobis_distances(windows: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of each row from the mean row.

    Uses empirical covariance with a tiny ridge (1e-8) for numerical
    stability; falls back to pseudo-inverse on singular cov.
    """
    n_w, w = windows.shape
    if n_w < 2:
        return np.zeros(n_w, dtype=np.float64)
    mean = windows.mean(axis=0)
    centered = windows - mean
    cov = (centered.T @ centered) / max(n_w - 1, 1)
    cov = cov + np.eye(w) * 1e-8
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
    md_sq = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
    return np.sqrt(np.maximum(md_sq, 0.0))


def _reverse_window_boolean(flags: np.ndarray, w: int) -> np.ndarray:
    """Map window-level booleans to point-level counts.

    For each True at window position i, points [i, i+w) get +1.
    Returns array of length len(flags) + w - 1.
    """
    n = len(flags) + w - 1
    out = np.zeros(n, dtype=np.int64)
    for i in np.flatnonzero(flags):
        out[i:i + w] += 1
    return out


# v0.9.130 (DEC-054): _probe_memory_limit_mb -> 
# cadw.core.memory_guard.probe_limit_mb (single source of truth).
from cadw.core.memory_guard import probe_limit_mb as _probe_memory_limit_mb



def fit_dwt_mlead(
    df_or_series,
    value_col=None,
    *,
    start_level: int = 3,
    quantile_epsilon: float = 0.01,
    min_anomaly_consistency: int = 2,
    guard=None,
    output: str = "memory",
    batch_size: "int | None" = None,
) -> DwtMleadResult:
    """DWT-MLEAD anomaly detection (v0.9.115 dual dispatch).

    df path (new, v0.9.115):
        fit_dwt_mlead(df, value_col)  # collects to ndarray, calls series

    Legacy np.ndarray path (back-compat):
        fit_dwt_mlead(series)  # in-memory, equivalent to output="memory"

    output="memory"|"auto" -> in-memory. "disk" -> NotImplementedError
    (SpillArray for O(n) anomaly_scores is v0.9.116+, DEC-013).
    """
    if output not in ("memory", "auto"):
        raise NotImplementedError(
            f"fit_dwt_mlead: output={output!r} not supported "
            f"(anomaly_scores is O(n) ndarray; SpillArray for dwt_mlead "
            f"is v0.9.116+, DEC-013)")
    is_df = (hasattr(df_or_series, "_iter_filtered_batches")
             and hasattr(df_or_series, "_columns"))
    if is_df:
        if value_col is None:
            raise ValueError(
                "fit_dwt_mlead: value_col required when df is passed")
        if value_col not in df_or_series._columns:
            raise KeyError(
                f"fit_dwt_mlead: column {value_col!r} not in "
                f"{df_or_series._columns}")
        return _fit_dwt_mlead_df(
            df_or_series, value_col,
            start_level=start_level,
            quantile_epsilon=quantile_epsilon,
            min_anomaly_consistency=min_anomaly_consistency,
            guard=guard, batch_size=batch_size,
        )
    if value_col is not None:
        raise ValueError(
            "fit_dwt_mlead: value_col only used with df input; "
            "omit for np.ndarray")
    return _fit_dwt_mlead_series(
        df_or_series,
        start_level=start_level,
        quantile_epsilon=quantile_epsilon,
        min_anomaly_consistency=min_anomaly_consistency,
        guard=guard,
    )


def _fit_dwt_mlead_series(series, *, start_level, quantile_epsilon,
                          min_anomaly_consistency, guard):
    """np.ndarray path (output= handled by public fit_dwt_mlead)."""
    return _fit_dwt_mlead_series_impl(
        series,
        start_level=start_level,
        quantile_epsilon=quantile_epsilon,
        min_anomaly_consistency=min_anomaly_consistency,
        guard=guard,
    )


def _fit_dwt_mlead_df(df, value_col, *, start_level, quantile_epsilon,
                      min_anomaly_consistency, guard, batch_size=None):
    """df path; collects series to ndarray then calls series path."""
    mg = guard if guard is not None else getattr(df, "_memory_guard", None)
    parts = []
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="dwt_mlead.batch", extra_factor=2.0)
        x_b = b.column(value_col).cast(
            pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        if x_b.size > 0:
            parts.append(x_b)
    if not parts:
        raise ValueError(f"fit_dwt_mlead: column {value_col!r} is empty")
    series = np.concatenate(parts) if len(parts) > 1 else parts[0]
    return _fit_dwt_mlead_series(
        series,
        start_level=start_level,
        quantile_epsilon=quantile_epsilon,
        min_anomaly_consistency=min_anomaly_consistency,
        guard=mg,
    )


def _fit_dwt_mlead_series_impl(
    series: np.ndarray,
    *,
    start_level: int = 3,
    quantile_epsilon: float = 0.01,
    min_anomaly_consistency: int = 2,
    guard=None,
) -> DwtMleadResult:
    """DWT-MLEAD anomaly detection.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
        1D univariate time series.
    start_level : int, default=3
        First DWT level to analyze (levels 1..start_level-1 are ignored;
        they capture high-frequency noise).
    quantile_epsilon : float, default=0.01
        At each level, points above (1 - quantile_epsilon) percentile
        of Mahalanobis distances are flagged as anomalies.
    min_anomaly_consistency : int, default=2
        Minimum number of levels flagging a point for it to be reported.
    guard : MemoryGuard | None
        Optional. Estimated peak memory ~ 8 * n * max_level bytes.

    Returns
    -------
    DwtMleadResult
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < 4:
        raise ValueError(f"series too short for DWT-MLEAD (min 4), got {n}")

    # v0.9.90-S3 (G2): estimate peak BEFORE pad+DWT allocations.
    # n_padded = next power of 2 ≥ n; details total ≈ n_padded values.
    n_padded_est = (1 << (n - 1).bit_length()) if n > 1 else 1
    max_level_est = ((n_padded_est.bit_length() - 1)
                     if n_padded_est > 1 else 0)
    if guard is not None:
        _w_peak_pre = max(2, 1 << ((max_level_est - start_level) // 2))
        _chunk_pre = _resolve_chunk_nw(_w_peak_pre, guard)
        est_details_pre = int(n_padded_est * 8 * 2)
        est_chunk_pre = (_chunk_pre * _w_peak_pre * 8 * 4
                         + _w_peak_pre * _w_peak_pre * 8 * 2)
        est_bytes_pre = int(est_details_pre + est_chunk_pre)
        if hasattr(guard, "check_state") and callable(guard.check_state):
            guard.check_state(est_bytes_pre, name="dwt_mlead.state")
        else:
            limit_mb = _probe_memory_limit_mb(guard)
            if limit_mb is not None and limit_mb > 0:
                max_bytes = limit_mb * 1024 * 1024
                if est_bytes_pre > max_bytes:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"DWT-MLEAD estimated {est_bytes_pre} bytes > "
                        f"guard {max_bytes} bytes (limit_mb={limit_mb})"
                    )

    x_padded = _pad_to_next_power_of_two(x)
    details = _multilevel_haar_dwt(x_padded)
    max_level = len(details)

    if start_level >= max_level:
        return DwtMleadResult(
            anomaly_scores=np.zeros(n, dtype=np.float64),
            is_anomaly=np.zeros(n, dtype=bool),
            level_counts=np.zeros(max_level, dtype=np.int64),
            start_level=start_level,
            max_level=max_level,
            quantile_epsilon=quantile_epsilon,
            min_anomaly_consistency=min_anomaly_consistency,
        )

    # v0.9.90-S3 (G2): peak already checked BEFORE pad+DWT allocations.
    # Here: chunk size for streaming mahalanobis (uses actual max_level,
    # should match pre-guard estimate).
    _w_peak = max(2, 1 << ((max_level - start_level) // 2))
    _chunk_nw = _resolve_chunk_nw(_w_peak, guard)

    anomaly_counts = np.zeros(n, dtype=np.int64)
    level_counts = np.zeros(max_level, dtype=np.int64)

    for level in range(start_level, max_level):
        detail = details[level - 1]
        if len(detail) < 4:
            continue
        # Window size: coarser levels (smaller index diff) get larger windows.
        # Formula: w = 2 ^ ((max_level - level) // 2), floored at 2.
        w = max(2, 2 ** ((max_level - level) // 2))
        if w > len(detail):
            w = 2
        if len(detail) < w:
            continue

        # v0.9.87-P4b: streaming (no full (n_w, w) materialization)
        md = _stream_mahalanobis(detail, w, _chunk_nw)
        if len(md) == 0:
            continue

        threshold = float(np.quantile(md, 1.0 - quantile_epsilon))
        flags = md > threshold
        if not flags.any():
            continue

        detail_counts = _reverse_window_boolean(flags, w)
        # Each level-l detail coef maps to 2^level original points.
        scale = 1 << level
        upsampled = np.repeat(detail_counts, scale)
        if len(upsampled) < n:
            upsampled = np.concatenate(
                [upsampled, np.zeros(n - len(upsampled), dtype=np.int64)]
            )
        else:
            upsampled = upsampled[:n]

        anomaly_counts += upsampled
        level_counts[level - 1] = int(flags.sum())

    is_anomaly = anomaly_counts >= min_anomaly_consistency

    return DwtMleadResult(
        anomaly_scores=anomaly_counts.astype(np.float64),
        is_anomaly=is_anomaly,
        level_counts=level_counts,
        start_level=start_level,
        max_level=max_level,
        quantile_epsilon=quantile_epsilon,
        min_anomaly_consistency=min_anomaly_consistency,
    )

