"""T-squared + Q (Hotelling + Squared Prediction Error) anomaly detection.

Multivariate statistical process control based on PCA:
  - fit PCA on baseline / all samples
  - T^2 = ||proj_retained / sqrt(lambda)||^2  (score-space Hotelling)
  - Q   = ||x - recon_retained||^2           (residual SPE)
  - flag anomaly if T^2 > T2_limit OR Q > Q_limit
  - combined score = max(T^2 / T2_limit, Q / Q_limit)

SKAB (v0.9) leaderboard: T-squared+Q = F1 0.76 (best non-learning CPU-friendly).
When only T^2 is used (no residual), F1 = 0.66 — Q residual is what pushes
it into SOTA range for gradual changepoints.

Reference: SKAB ControlCharts (YKatser/ControlCharts) and PyOD PCA.

Pure numpy — no sklearn/pyod/scipy/torch in cadw/ (AST-guard
test_no_bench_deps_in_core.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sub_pca import _map_max, _map_mean


@dataclass
class T2QResult:
    """Result of fit_t2q.

    Attributes
    ----------
    decision_scores : np.ndarray, shape (n,)
        Combined anomaly score = max(T2 / T2_limit, Q / Q_limit).
        > 1.0 means flagged.
    labels : np.ndarray, shape (n,), int
        0/1 per-timestamp labels (T2 > T2_limit OR Q > Q_limit).
    threshold : float
        Combined threshold = 1.0 (scores normalised by their limits).
    t2_scores : np.ndarray, shape (n,)
        Per-timestamp T^2 (Hotelling) score.
    q_scores : np.ndarray, shape (n,)
        Per-timestamp Q (SPE) score.
    t2_limit : float
        Percentile-based T^2 threshold.
    q_limit : float
        Percentile-based Q threshold.
    n_components : int
        Number of retained PCA components.
    components : np.ndarray, shape (n_components, n_features)
        Retained principal axes (rows).
    eigvals : np.ndarray, shape (n_components,)
        Retained eigenvalues.
    mean : np.ndarray, shape (n_features,)
        Training mean.
    window_size, stride : int
        Windowing parameters.
    contamination : float
    aggregate : str
    n_channels : int
    """
    decision_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    t2_scores: np.ndarray
    q_scores: np.ndarray
    t2_limit: float
    q_limit: float
    n_components: int
    components: np.ndarray
    eigvals: np.ndarray
    mean: np.ndarray
    window_size: int
    stride: int
    contamination: float
    aggregate: str
    n_channels: int


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
        if _is_pos(v) and "mb" in kl and any(
                t in kl for t in ("mem", "budget", "limit", "max")):
            return float(v)
    for k, v in d.items():
        kl = k.lower()
        if _is_pos(v) and any(
                t in kl for t in ("mem", "budget", "limit", "max")) and (
                "byte" in kl or kl.endswith("_b")):
            return float(v) / (1024.0 * 1024.0)
    return None


def _make_windows(X: np.ndarray, w: int, stride: int) -> np.ndarray:
    if X.ndim == 1:
        X = X[:, None]
    n, c = X.shape
    if w == 1:
        return X[stride - 1::stride] if stride > 1 else X.copy()
    n_w = (n - w) // stride + 1
    if n_w < 1:
        raise ValueError(
            f"series length {n} < window_size {w}")
    idx = np.arange(n_w)[:, None] * stride + np.arange(w)[None, :]
    return X[idx].reshape(n_w, w * c)


def _resolve_k(n_components, eigvals: np.ndarray) -> int:
    m = len(eigvals)
    if n_components is None:
        # auto: cumulative variance >= 0.9, at least 1, at most m-1
        # (leave at least 1 component in Q residual for meaningful SPE)
        total = float(eigvals.sum())
        if total < 1e-12:
            return 1
        cumsum = np.cumsum(eigvals / total)
        k = int(np.searchsorted(cumsum, 0.9) + 1)
        return max(1, min(k, max(1, m - 1)))
    if isinstance(n_components, float) and 0.0 < n_components < 1.0:
        total = float(eigvals.sum())
        if total < 1e-12:
            return 1
        cumsum = np.cumsum(eigvals / total)
        k = int(np.searchsorted(cumsum, n_components) + 1)
        return max(1, min(k, m))
    k = int(n_components)
    return max(1, min(k, m))


def _fit_pca_eigh(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_w = len(centered)
    cov = (centered.T @ centered) / max(n_w - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def fit_t2q(
    series: np.ndarray,
    *,
    window_size: int = 1,
    stride: int = 1,
    n_components: int | float | None = None,
    contamination: float = 0.1,
    aggregate: str = "max",
    guard=None,
) -> T2QResult:
    """T^2 + Q (Hotelling + SPE) anomaly detection.

    Parameters
    ----------
    series : np.ndarray, shape (n,) or (n, c)
        Univariate or multivariate time series.
    window_size : int, default=1
        1 = per-sample SPC (SKAB convention for (n,c) multivariate).
        >1 = windowing (needed for univariate (n,) so PCA is meaningful).
    stride : int, default=1
        Step between windows (only used when window_size > 1).
    n_components : int | float | None
        None -> auto (90% cumulative variance, at least 1, at most m-1).
        float in (0,1) -> cumulative variance threshold.
        int -> exact, clipped to [1, m].
    contamination : float, default=0.1
        Expected anomaly fraction (used only for threshold).
    aggregate : str, default="max"
        For window_size > 1: "max" or "mean".
    guard : MemoryGuard | None

    Returns
    -------
    T2QResult
    """
    X = np.asarray(series, dtype=np.float64)
    if X.ndim == 1:
        n = len(X)
        c = 1
    elif X.ndim == 2:
        n, c = X.shape
    else:
        raise ValueError(f"series must be 1D or 2D, got ndim={X.ndim}")

    if window_size == 1 and c == 1:
        # Univariate per-sample is degenerate (T^2 = (x-mu)^2/var, Q=0);
        # force windowing.
        window_size = min(50, max(2, n // 20))
        if window_size < 2:
            raise ValueError(
                f"univariate per-sample T2Q is degenerate; "
                f"need n >= 40 (got n={n})")

    # v0.9.87-fix1b(2/2) (audit HIGH-1): guard check BEFORE
    # _make_windows allocation. Validation order matches original
    # _make_windows semantics: '< window_size' first, then
    # 'too few samples'.
    if n < window_size:
        raise ValueError(
            f"series length {n} < window_size {window_size}")
    n_s_pre = (n - window_size) // stride + 1
    if n_s_pre < 2:
        raise ValueError(
            f"too few samples ({n_s_pre}) for PCA (need >= 2)")
    d_pre = window_size * c

    if guard is not None:
        est_bytes = 8 * n_s_pre * d_pre
        if hasattr(guard, "check_state") and callable(guard.check_state):
            guard.check_state(est_bytes, name="tspq.windows")
        else:
            limit_mb = _probe_memory_limit_mb(guard)
            if limit_mb is not None and limit_mb > 0:
                if est_bytes > limit_mb * 1024 * 1024:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"T2Q estimated {est_bytes} bytes > "
                        f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    # Allocation happens ONLY after guard passes.
    samples = _make_windows(X, window_size, stride)
    n_s, d = samples.shape

    mean = samples.mean(axis=0)
    centered = samples - mean
    eigvals, eigvecs = _fit_pca_eigh(centered)
    k = _resolve_k(n_components, eigvals)
    components = eigvecs[:, :k].T
    kept_eigvals = eigvals[:k]

    scores_proj = centered @ components.T  # (n_s, k)
    recon = scores_proj @ components       # (n_s, d)
    residual = centered - recon

    # Hotelling T^2 (avoid /0 for degenerate eigenvalues)
    safe_eig = np.where(kept_eigvals < 1e-12, 1.0, kept_eigvals)
    t2 = ((scores_proj ** 2) / safe_eig).sum(axis=1)  # (n_s,)
    # Squared prediction error (Q)
    q = (residual ** 2).sum(axis=1)  # (n_s,)

    # Percentile-based limits (SKAB convention: separate, OR threshold)
    t2_limit = float(np.percentile(t2, 100.0 * (1.0 - contamination)))
    q_limit = float(np.percentile(q, 100.0 * (1.0 - contamination)))
    if t2_limit <= 0:
        t2_limit = 1e-12
    if q_limit <= 0:
        q_limit = 1e-12

    # Combined per-sample score
    combined = np.maximum(t2 / t2_limit, q / q_limit)

    # Aggregate window scores to timestamps
    if window_size == 1:
        t2_ts = t2
        q_ts = q
        dec_ts = combined
    elif aggregate == "max":
        t2_ts = _map_max(t2, window_size, n, stride=stride)
        q_ts = _map_max(q, window_size, n, stride=stride)
        dec_ts = _map_max(combined, window_size, n, stride=stride)
    elif aggregate == "mean":
        t2_ts = _map_mean(t2, window_size, n, stride=stride)
        q_ts = _map_mean(q, window_size, n, stride=stride)
        dec_ts = _map_mean(combined, window_size, n, stride=stride)
    else:
        raise ValueError(
            f"aggregate must be 'max' or 'mean', got {aggregate!r}")

    labels = (dec_ts > 1.0).astype(int)

    return T2QResult(
        decision_scores=dec_ts,
        labels=labels,
        threshold=1.0,
        t2_scores=t2_ts,
        q_scores=q_ts,
        t2_limit=t2_limit,
        q_limit=q_limit,
        n_components=k,
        components=components,
        eigvals=kept_eigvals,
        mean=mean,
        window_size=window_size,
        stride=stride,
        contamination=contamination,
        aggregate=aggregate,
        n_channels=c,
    )

