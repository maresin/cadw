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


def _probe_memory_limit_mb(guard) -> float | None:
    def _is_pos(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
    MB = ("max_memory_mb", "_max_memory_mb", "limit_mb", "budget_mb",
          "_budget_mb", "max_mb", "_max_mb", "memory_limit_mb",
          "_memory_limit_mb")
    BYTES = ("max_memory_bytes", "_max_memory_bytes", "max_bytes",
             "_max_bytes", "limit_bytes", "_limit_bytes", "budget_bytes",
             "_budget_bytes", "memory_limit_bytes", "_memory_limit_bytes")
    for a in MB:
        v = getattr(guard, a, None)
        if _is_pos(v):
            return float(v)
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
    for k, v in d.items():
        kl = k.lower()
        if not _is_pos(v):
            continue
        if any(t in kl for t in ("mem", "budget", "limit", "max")) and (
                "byte" in kl or kl.endswith("_b")):
            return float(v) / (1024.0 * 1024.0)
    return None


def _sliding_windows(x: np.ndarray, w: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(x, w).copy()


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
    # v0.9.87-fix1a (audit HIGH-1): guard check BEFORE
    # _prepare_windows allocation. Analytic shape only — no window
    # array built yet.
    _arr = np.asarray(series)
    n_pre = int(_arr.shape[0]) if _arr.ndim >= 1 else 0
    c_pre = 1 if _arr.ndim == 1 else int(np.prod(_arr.shape[1:]))
    if n_pre < window_size:
        raise ValueError(
            f"series length {n_pre} < window_size {window_size}")
    n_w_pre = n_pre - window_size + 1
    if n_w_pre < 2:
        raise ValueError(
            f"too few windows: n_windows={n_w_pre} < 2 "
            f"(n={n_pre}, window_size={window_size})")
    d_pre = window_size * c_pre

    if guard is not None:
        est_bytes = 8 * n_w_pre * d_pre
        # Prefer canonical MemoryGuard API (real guard).
        # Fall back to legacy probe for duck-typed guards (test
        # doubles with only `max_memory_mb` attribute).
        if hasattr(guard, "check_state") and callable(guard.check_state):
            guard.check_state(est_bytes, name="sub_pca.windows")
        else:
            limit_mb = _probe_memory_limit_mb(guard)
            if limit_mb is not None and limit_mb > 0:
                if est_bytes > limit_mb * 1024 * 1024:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"Sub-PCA estimated {est_bytes} bytes > "
                        f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    # Allocation happens ONLY after the guard passes.
    windows, n, c = _prepare_windows(series, window_size, stride=1)
    n_w = len(windows)
    d = windows.shape[1]

    mean_w = windows.mean(axis=0)
    centered = windows - mean_w

    eigvals, eigvecs = _fit_pca_eigh(centered)
    k = _resolve_k(n_components, eigvals, d)
    components = eigvecs[:, :k].T  # (k, d)

    proj = centered @ components.T  # (n_w, k)
    recon = proj @ components  # (n_w, w)
    residual = centered - recon
    window_scores = np.linalg.norm(residual, axis=1)

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

