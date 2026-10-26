"""KMeansAD: k-means clustering anomaly detection on sliding windows.

Classic ML baseline: cluster windowed subsequences via k-means++
initialization + Lloyd iterations; anomaly score per window = distance
to nearest centroid; aggregate to timestamps.

TSB-AD leaderboard: KMeansAD = VUS-PR 0.37 (U), 0.29 (M).
Reference: PyOD KMeans variant; TSB-AD models/KMeansAD.

Simplest k-means shape-clustering after KShapeAD (SBD-based); plain
Euclidean windows this time.

Pure numpy — no sklearn/scipy/torch in cadw/ (AST-guard
test_no_bench_deps_in_core.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sub_pca import _map_max, _map_mean, _prepare_windows


@dataclass
class KMeansADResult:
    decision_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    window_scores: np.ndarray
    centroids: np.ndarray
    labels_per_window: np.ndarray
    window_size: int
    n_clusters: int
    max_iter: int
    n_iter_run: int
    contamination: float
    aggregate: str
    inertia: float


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
        if _is_pos(v) and "mb" in kl and any(
                t in kl for t in ("mem", "budget", "limit", "max")):
            return float(v)
    return None


def _kmeans_plus_plus_init(X: np.ndarray, k: int,
                           rng: np.random.Generator) -> np.ndarray:
    """k-means++ initialization: centroid_j sampled with prob ∝ D(x)^2."""
    n, d = X.shape
    if n < k:
        raise ValueError(f"n_samples ({n}) < n_clusters ({k})")
    centroids = np.empty((k, d), dtype=np.float64)
    idx0 = int(rng.integers(n))
    centroids[0] = X[idx0]
    d_sq = np.sum((X - centroids[0]) ** 2, axis=1)
    for j in range(1, k):
        total = float(d_sq.sum())
        if total < 1e-20:
            centroids[j] = X[int(rng.integers(n))]
        else:
            probs = d_sq / total
            idx = int(rng.choice(n, p=probs))
            centroids[j] = X[idx]
        new_d_sq = np.sum((X - centroids[j]) ** 2, axis=1)
        d_sq = np.minimum(d_sq, new_d_sq)
    return centroids


def _distances_sq(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances ||x-c||^2, shape (n, k)."""
    x_sq = (X * X).sum(axis=1)
    c_sq = (C * C).sum(axis=1)
    return x_sq[:, None] - 2.0 * (X @ C.T) + c_sq[None, :]


def _lloyd(X: np.ndarray, centroids: np.ndarray, max_iter: int,
           tol: float, rng: np.random.Generator) -> tuple:
    """Lloyd iterations. Returns (centroids, labels, dists_sq, n_iter, inertia)."""
    n, _ = X.shape
    k = len(centroids)
    prev_inertia = np.inf
    labels = np.zeros(n, dtype=int)
    dists_sq = np.zeros((n, k), dtype=np.float64)
    inertia = np.inf
    n_iter_run = 0
    for it in range(max_iter):
        dists_sq = _distances_sq(X, centroids)
        labels = np.argmin(dists_sq, axis=1)
        inertia = float(dists_sq[np.arange(n), labels].sum())
        # Update
        new_centroids = centroids.copy()
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centroids[j] = X[mask].mean(axis=0)
            else:
                # Empty cluster: reinit to farthest point (k-means++ style)
                farthest = int(np.argmax(dists_sq[np.arange(n), labels]))
                new_centroids[j] = X[farthest]
        shift = float(np.sum((new_centroids - centroids) ** 2))
        centroids = new_centroids
        n_iter_run = it + 1
        if shift < tol * max(1.0, float((centroids ** 2).sum())):
            break
        if abs(prev_inertia - inertia) < tol * max(prev_inertia, 1.0):
            break
        prev_inertia = inertia
    # Final assignment
    dists_sq = _distances_sq(X, centroids)
    labels = np.argmin(dists_sq, axis=1)
    inertia = float(dists_sq[np.arange(n), labels].sum())
    return centroids, labels, dists_sq, n_iter_run, inertia


def fit_kmeans_ad(
    series: np.ndarray,
    *,
    window_size: int = 50,
    n_clusters: int = 8,
    max_iter: int = 100,
    tol: float = 1e-4,
    contamination: float = 0.1,
    aggregate: str = "max",
    random_state: int | None = 42,
    guard=None,
) -> KMeansADResult:
    """KMeansAD anomaly detection.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
    window_size : int, default=50
    n_clusters : int, default=8
    max_iter : int, default=100
    tol : float, default=1e-4
        Convergence tolerance on centroid shift / inertia.
    contamination : float, default=0.1
    aggregate : str, default="max"
        "max" or "mean".
    random_state : int or None, default=42
    guard : MemoryGuard | None

    Returns
    -------
    KMeansADResult
    """
    windows, n, c = _prepare_windows(series, window_size, stride=1)
    n_w = len(windows)
    d = windows.shape[1]
    if n < window_size:
        raise ValueError(
            f"series length {n} < window_size {window_size}")
    if n_w < n_clusters:
        raise ValueError(
            f"n_windows ({n_w}) < n_clusters ({n_clusters})")
    if aggregate not in ("max", "mean"):
        raise ValueError(
            f"aggregate must be 'max' or 'mean', got {aggregate!r}")

    if guard is not None:
        est_bytes = 8 * n_w * d * 2
        limit_mb = _probe_memory_limit_mb(guard)
        if limit_mb is not None and limit_mb > 0:
            if est_bytes > limit_mb * 1024 * 1024:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"KMeansAD estimated {est_bytes} bytes > "
                    f"guard {limit_mb * 1024 * 1024:.0f} bytes")


    rng = np.random.default_rng(random_state)
    centroids = _kmeans_plus_plus_init(windows, n_clusters, rng)
    centroids, labels_per_window, dists_sq, n_iter_run, inertia = _lloyd(
        windows, centroids, max_iter, tol, rng)

    window_scores = np.sqrt(
        dists_sq[np.arange(n_w), labels_per_window])

    if aggregate == "max":
        scores = _map_max(window_scores, window_size, n, stride=1)
    else:
        scores = _map_mean(window_scores, window_size, n, stride=1)

    threshold = float(np.percentile(
        scores, 100.0 * (1.0 - contamination)))
    out_labels = (scores > threshold).astype(int)

    return KMeansADResult(
        decision_scores=scores,
        labels=out_labels,
        threshold=threshold,
        window_scores=window_scores,
        centroids=centroids,
        labels_per_window=labels_per_window,
        window_size=window_size,
        n_clusters=n_clusters,
        max_iter=max_iter,
        n_iter_run=n_iter_run,
        contamination=contamination,
        aggregate=aggregate,
        inertia=inertia,
    )

