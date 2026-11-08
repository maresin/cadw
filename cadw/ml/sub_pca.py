"""Sub-PCA: windowed PCA reconstruction-error anomaly detection.

Standard PCA-based TS anomaly detector: sliding windows -> center ->
PCA (eigendecomposition of covariance) -> project / reconstruct ->
per-window L2 residual -> per-timestamp aggregation -> threshold.

Pure numpy — no sklearn/scipy/torch in cadw/ (AST-guard
test_no_bench_deps_in_core.py).

Reference: sklearn.decomposition.PCA (same subspace projection,
reconstruction-invariant to sign ambiguity).
TSB-AD leaderboard: PCA = VUS-PR 0.42 (univariate), 0.31 (multivariate).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa


@dataclass
class SubPCAResult:
    """Result of fit_sub_pca.

    Attributes
    ----------
    decision_scores : np.ndarray, shape (n,)
        Per-timestamp anomaly score (aggregated window residuals).
    labels : np.ndarray, shape (n,), int
        0/1 binary labels (score > threshold).
    threshold : float
        percentile(100*(1-contamination)) of decision_scores.
    components : np.ndarray, shape (k, window_size)
        Top-k principal components (rows).
    mean : np.ndarray, shape (window_size,)
        Per-position mean of windows; subtracted before PCA.
    window_size, n_components : int
    contamination : float
    aggregate : str
        "max" or "mean".
    """
    decision_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    components: np.ndarray
    mean: np.ndarray
    window_size: int
    n_components: int
    contamination: float
    aggregate: str

    def state_nbytes(self) -> int:
        """Persistent state: decision_scores(n f64) + labels(n i64)
        + components(n_components*window_size f64) + mean(window_size f64)
        + header.

        v0.9.115: honest O(n). Formula (1D case):
        2*n*f64 + n_components*window_size*f64 + window_size*f64 + 264.
        For multichannel (c>1): d_pre=window_size*c, components/mean
        scale by c; formula documents 1D case, .nbytes captures actual.
        bounded_by_domain n<=1e6.
        """
        return int(
            self.decision_scores.nbytes
            + self.labels.nbytes
            + self.components.nbytes
            + self.mean.nbytes
            + 8    # threshold
            + 256  # dataclass header
        )


# v0.9.130 (DEC-054): _probe_memory_limit_mb -> 
# cadw.core.memory_guard.probe_limit_mb (single source of truth).
from cadw.core.memory_guard import probe_limit_mb as _probe_memory_limit_mb


# v0.9.87-P4a: _sliding_windows removed (dead helper; no call sites
# in cadw/ or tests/). Streaming uses _iter_window_chunks below.


def _fit_pca_eigh(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA via eigendecomposition of covariance. Returns (eigvals, eigvecs).

    eigvals descending; eigvecs columns correspond (sign arbitrary).
    O(w^3) instead of O(n*w^2) for SVD; w is window_size (~50).
    """
    n_w = len(centered)
    cov = (centered.T @ centered) / max(n_w - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def _resolve_k(n_components, eigvals: np.ndarray, w: int) -> int:
    """Resolve k from None / int / float ∈ (0,1)."""
    m = len(eigvals)
    if n_components is None:
        return max(1, w // 2)
    if isinstance(n_components, float) and 0.0 < n_components < 1.0:
        total = float(eigvals.sum())
        if total < 1e-12:
            return 1
        cumsum = np.cumsum(eigvals / total)
        return int(np.searchsorted(cumsum, n_components) + 1)
    k = int(n_components)
    return max(1, min(k, m))


def _map_max(window_scores: np.ndarray, w: int, n: int,
             stride: int = 1) -> np.ndarray:
    """Per-timestamp max over covering windows.

    stride=1 (default): fast path via sliding_window_view + vectorized max.
    stride>1: loop path (window i covers [i*stride, i*stride+w)).
    """
    n_w = len(window_scores)
    if stride == 1:
        padded = np.concatenate([
            np.full(w - 1, -np.inf), window_scores, np.full(w - 1, -np.inf),
        ])
        sw = np.lib.stride_tricks.sliding_window_view(padded, w)
        result = sw.max(axis=1)
        return np.where(np.isfinite(result), result, 0.0)[:n]
    scores = np.full(n, -np.inf, dtype=np.float64)
    for i in range(n_w):
        start = i * stride
        end = min(start + w, n)
        if start < n:
            np.maximum(scores[start:end], window_scores[i],
                       out=scores[start:end])
    scores[~np.isfinite(scores)] = 0.0
    return scores


def _map_mean(window_scores: np.ndarray, w: int, n: int,
              stride: int = 1) -> np.ndarray:
    """Per-timestamp mean over covering windows.

    stride=1 (default): cumsum delta trick. stride>1: loop path.
    """
    n_w = len(window_scores)
    if stride == 1:
        sums_delta = np.zeros(n + w + 1, dtype=np.float64)
        sums_delta[:n_w] += window_scores
        sums_delta[w:n_w + w] -= window_scores
        sums = np.cumsum(sums_delta)[:n]

        counts_delta = np.zeros(n + w + 1, dtype=np.float64)
        counts_delta[:n_w] += 1.0
        counts_delta[w:n_w + w] -= 1.0
        counts = np.cumsum(counts_delta)[:n]
        counts = np.maximum(counts, 1.0)
        return sums / counts
    sums = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    for i in range(n_w):
        start = i * stride
        end = min(start + w, n)
        if start < n:
            sums[start:end] += window_scores[i]
            counts[start:end] += 1.0
    counts = np.maximum(counts, 1.0)
    return sums / counts


# v0.9.87-P4a (audit HIGH-1 full fix): streaming chunk helpers.
# Peak RAM O(chunk_nw * d) regardless of n. Output bit-identical
# within float precision (only cov/eig accumulation order differs
# vs pre-P4a; residual math unchanged).
#
# See METHODOLOGY_ALGORITHM_SELECTION.md §5 (windowed state_type);
# PRIMITIVES_AND_PRINCIPLES.md §6 anti-patterns (G3).


def _resolve_chunk_nw(d: int, guard=None) -> int:
    """Choose chunk size so per-chunk peak RAM stays within budget.

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


def _iter_window_chunks(X: np.ndarray, window_size: int,
                        stride: int, chunk_nw: int):
    """Yield (chunk, abs_start) chunks of shape (nb, w*c) from X.

    Only `chunk_nw` windows are materialized at a time (nb <= chunk_nw).
    """
    if X.ndim == 1:
        n = len(X); c = 1
    elif X.ndim == 2:
        n, c = X.shape
    else:
        raise ValueError(
            f"series must be 1D or 2D, got ndim={X.ndim}")
    n_w = (n - window_size) // stride + 1
    offsets = np.arange(window_size, dtype=np.int64)
    for start in range(0, n_w, chunk_nw):
        end = min(start + chunk_nw, n_w)
        nb = end - start
        abs_starts = np.arange(start, end, dtype=np.int64) * stride
        idx = abs_starts[:, None] + offsets[None, :]
        if c == 1:
            chunk = X[idx]
        else:
            chunk = X[idx].reshape(nb, window_size * c)
        yield chunk, start


def _stream_mean_cov(X: np.ndarray, window_size: int, stride: int,
                     chunk_nw: int, d: int):
    """2-pass streaming mean + covariance; peak O(chunk_nw*d + d^2)."""
    total = np.zeros(d, dtype=np.float64)
    n_w = 0
    for chunk, _ in _iter_window_chunks(
            X, window_size, stride, chunk_nw):
        total += chunk.sum(axis=0)
        n_w += chunk.shape[0]
    if n_w < 2:
        raise ValueError(
            f"too few windows for covariance: n_w={n_w} < 2")
    mean = total / n_w

    M2 = np.zeros((d, d), dtype=np.float64)
    for chunk, _ in _iter_window_chunks(
            X, window_size, stride, chunk_nw):
        centered = chunk - mean
        M2 += centered.T @ centered
    cov = M2 / (n_w - 1)
    return mean, cov


def _stream_residual_scores(X: np.ndarray, window_size: int,
                            stride: int, chunk_nw: int,
                            mean: np.ndarray,
                            components: np.ndarray) -> np.ndarray:
    """Streaming residual norms per window; peak O(chunk_nw*d)."""
    scores_chunks = []
    for chunk, _ in _iter_window_chunks(
            X, window_size, stride, chunk_nw):
        centered = chunk - mean
        proj = centered @ components.T
        recon = proj @ components
        residual = centered - recon
        scores_chunks.append(np.linalg.norm(residual, axis=1))
    if not scores_chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(scores_chunks)


def _prepare_windows(series: np.ndarray, window_size: int,
                     stride: int = 1) -> tuple:
    """Return (windows, n, n_channels).

    Univariate (n,): sliding windows -> (n_w, w).
    Multivariate (n, c): multi-channel windows flattened per window
    (n_w, w*c); cross-channel structure preserved in the flat vector.
    v0.9.73: replaces np.ravel() which silently broke multi-channel.
    """
    X = np.asarray(series, dtype=np.float64)
    if X.ndim == 1:
        n = len(X)
        c = 1
    elif X.ndim == 2:
        n, c = X.shape
    else:
        raise ValueError(
            f"series must be 1D or 2D, got ndim={X.ndim}")
    if n < window_size:
        raise ValueError(
            f"series length {n} < window_size {window_size}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    ends = np.arange(0, n - window_size + 1, stride)
    idx = ends[:, None] + np.arange(window_size)[None, :]
    if X.ndim == 1:
        windows = X[idx]
    else:
        windows = X[idx].reshape(-1, window_size * c)
    return windows, n, c


def fit_sub_pca(
    df_or_series, value_col=None, *,
    window_size: int = 50,
    n_components: "int | float | None" = None,
    contamination: float = 0.1,
    aggregate: str = "max",
    guard=None,
    output: str = "memory",
    batch_size: "int | None" = None,
) -> SubPCAResult:
    """Sub-PCA anomaly detection (v0.9.115 dual dispatch).

    df path (new, v0.9.115):
        fit_sub_pca(df, value_col)                # univariate
        fit_sub_pca(df, [c1, c2, ...])            # multichannel (c>1)
        batch_size controls df-iterator (no SGD in sub_pca).

    Legacy np.ndarray path (back-compat):
        fit_sub_pca(series)  # 1D or 2D ndarray, equivalent to output="memory"

    output="memory"|"auto" -> in-memory; "disk" -> NotImplementedError
    (SpillArray for O(n) decision_scores is v0.9.116+, DEC-013).
    """
    if output not in ("memory", "auto"):
        raise NotImplementedError(
            f"fit_sub_pca: output={output!r} not supported "
            f"(decision_scores is O(n) ndarray; SpillArray for sub_pca "
            f"is v0.9.116+, DEC-013)")
    is_df = (hasattr(df_or_series, "_iter_filtered_batches")
             and hasattr(df_or_series, "_columns"))
    if is_df:
        if value_col is None:
            raise ValueError("fit_sub_pca: value_col required when df is passed")
        cols = [value_col] if isinstance(value_col, str) else list(value_col)
        if not cols:
            raise ValueError("fit_sub_pca: value_col empty")
        for c in cols:
            if c not in df_or_series._columns:
                raise KeyError(
                    f"fit_sub_pca: column {c!r} not in {df_or_series._columns}")
        return _fit_sub_pca_df(
            df_or_series, cols,
            window_size=window_size, n_components=n_components,
            contamination=contamination, aggregate=aggregate,
            guard=guard, batch_size=batch_size)
    if value_col is not None:
        raise ValueError("fit_sub_pca: value_col only used with df input")
    return _fit_sub_pca_series(
        df_or_series,
        window_size=window_size, n_components=n_components,
        contamination=contamination, aggregate=aggregate, guard=guard)


def _fit_sub_pca_series(series, *, window_size, n_components,
                         contamination, aggregate, guard):
    return _fit_sub_pca_series_impl(
        series,
        window_size=window_size, n_components=n_components,
        contamination=contamination, aggregate=aggregate, guard=guard)


def _fit_sub_pca_df(df, cols, *, window_size, n_components,
                     contamination, aggregate, guard, batch_size=None):
    """df path; supports str (univariate) or list[str] (multichannel)."""
    mg = guard if guard is not None else getattr(df, "_memory_guard", None)
    is_multi = len(cols) > 1
    parts = []
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="sub_pca.batch", extra_factor=2.0)
        arrays = [
            b.column(c).cast(pa.float64(), safe=False).to_numpy(
                zero_copy_only=False)
            for c in cols
        ]
        block = np.stack(arrays, axis=1) if is_multi else arrays[0]
        if block.shape[0] > 0:
            parts.append(block)
    if not parts:
        raise ValueError(f"fit_sub_pca: columns {cols!r} empty")
    series = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    return _fit_sub_pca_series(
        series,
        window_size=window_size, n_components=n_components,
        contamination=contamination, aggregate=aggregate, guard=mg)


def _fit_sub_pca_series_impl(
    series: np.ndarray,
    *,
    window_size: int = 50,
    n_components: int | float | None = None,
    contamination: float = 0.1,
    aggregate: str = "max",
    guard=None,
) -> SubPCAResult:
    """Sub-PCA anomaly detection.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
        1D univariate time series.
    window_size : int, default=50
        Sliding window length.
    n_components : int | float | None
        None -> window_size // 2.
        float in (0,1) -> smallest k with cumulative variance ratio >= value.
        int -> exact k, clipped to [1, min(n_w-1, window_size)].
    contamination : float, default=0.1
        Fraction flagged; used only for threshold.
    aggregate : str, default="max"
        "max" (point anomalies) or "mean" (segment anomalies).
    guard : MemoryGuard | None

    Returns
    -------
    SubPCAResult
    """
    # v0.9.87-P4a (audit HIGH-1 full fix): streaming chunked
    # pipeline. Peak RAM O(chunk_nw*d) regardless of n. Output
    # bit-identical within float precision (only cov/eig accumulation
    # order differs vs pre-P4a; residual math unchanged).
    _arr = np.asarray(series, dtype=np.float64)
    if _arr.ndim == 1:
        n_pre = len(_arr)
        c_pre = 1
    elif _arr.ndim == 2:
        n_pre, c_pre = _arr.shape
    else:
        raise ValueError(
            f"series must be 1D or 2D, got ndim={_arr.ndim}")
    if n_pre < window_size:
        raise ValueError(
            f"series length {n_pre} < window_size {window_size}")
    n_w_pre = n_pre - window_size + 1
    if n_w_pre < 2:
        raise ValueError(
            f"too few windows: n_windows={n_w_pre} < 2 "
            f"(n={n_pre}, window_size={window_size})")
    d_pre = window_size * c_pre

    chunk_nw = _resolve_chunk_nw(d_pre, guard)

    if guard is not None:
        est_bytes = chunk_nw * d_pre * 8 * 6
        if hasattr(guard, "check_state") and callable(guard.check_state):
            guard.check_state(est_bytes, name="sub_pca.chunk")
        else:
            limit_mb = _probe_memory_limit_mb(guard)
            if limit_mb is not None and limit_mb > 0:
                if est_bytes > limit_mb * 1024 * 1024:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"Sub-PCA per-chunk estimate {est_bytes} "
                        f"bytes > guard "
                        f"{limit_mb * 1024 * 1024:.0f} bytes")

    # Pass 1+2: mean + cov (streaming, O(chunk_nw*d))
    mean_w, cov = _stream_mean_cov(
        _arr, window_size, 1, chunk_nw, d_pre)

    # Eigh on (d, d) covariance — same as pre-P4a
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    k = _resolve_k(n_components, eigvals, d_pre)
    components = eigvecs[:, :k].T  # (k, d_pre)

    # v0.9.116-fix (G2): check_state BEFORE output O(n) allocations.
    # Peak = window_scores(n_w) + scores(n) + labels(n) ~ 3*n_w*8 B
    # (n_w ~ n). state_nbytes excludes this transient; state is
    # persistent post-fit. Probe: dRSS=25.9 MB at n=1e6.
    if (guard is not None and hasattr(guard, "check_state")
            and callable(guard.check_state)):
        _n_out = n_pre
        _n_w_out = n_pre - window_size + 1
        guard.check_state(
            int(3 * _n_w_out * 8 + 2 * _n_out * 8 + 256),
            name="sub_pca.output")

    # Pass 3: streaming residual norms (O(chunk_nw*d))
    window_scores = _stream_residual_scores(
        _arr, window_size, 1, chunk_nw, mean_w, components)

    n = n_pre

    if aggregate == "max":
        scores = _map_max(window_scores, window_size, n)
    elif aggregate == "mean":
        scores = _map_mean(window_scores, window_size, n)
    else:
        raise ValueError(
            f"aggregate must be 'max' or 'mean', got {aggregate!r}")

    threshold = float(np.percentile(
        scores, 100.0 * (1.0 - contamination)))
    labels = (scores > threshold).astype(int)

    return SubPCAResult(
        decision_scores=scores,
        labels=labels,
        threshold=threshold,
        components=components,
        mean=mean_w,
        window_size=window_size,
        n_components=k,
        contamination=contamination,
        aggregate=aggregate,
    )

