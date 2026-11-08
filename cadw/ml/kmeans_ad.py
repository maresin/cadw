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
import pyarrow as pa

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

    def state_nbytes(self):
        """Persistent state: 2 n-sized output arrays + 2 n_w-sized aux +
        centroids (k,d) + 256 header.

        v0.9.114: honest O(n) — decision_scores + labels + window_scores
        + labels_per_window are all O(n_w~n). bounded_by_domain n<=1e6
        accepts this (CoxPH precedent).
        """
        return int(
            self.decision_scores.nbytes
            + self.labels.nbytes
            + self.window_scores.nbytes
            + self.labels_per_window.nbytes
            + self.centroids.nbytes
            + 256
        )


# v0.9.130 (DEC-054): _probe_memory_limit_mb -> 
# cadw.core.memory_guard.probe_limit_mb (single source of truth).
from cadw.core.memory_guard import probe_limit_mb as _probe_memory_limit_mb


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


def _resolve_chunk_nw_km(d: int, guard=None) -> int:
    """v0.9.91-S6c: chunk size, 8 MB target (or state_limit//8).

    Per-window: chunk buffer (d*8) + dists_sq (k*8) + argmin/argmax
    temps (~2*k*8) ~ 3*d*8 + 3*k*8.
    """
    default_target = 8 * 1024 * 1024
    target_bytes = default_target
    if guard is not None:
        sl = getattr(guard, "state_limit_bytes", None)
        if isinstance(sl, (int, float)) and sl > 0:
            target_bytes = min(int(target_bytes), max(1, int(sl) // 8))
    bytes_per_window = 8 * max(int(d), 1) * 3 + 24
    n = max(100, target_bytes // bytes_per_window)
    return min(10_000, n)


def _iter_window_chunks_km(x: np.ndarray, w: int, chunk_nw: int):
    """Yield (chunk(nb, w*c), start) from 1D or 2D x. stride=1."""
    if x.ndim == 1:
        X = x[:, None]
    else:
        X = x
    n, c = X.shape
    n_w = n - w + 1
    if n_w < 1:
        return
    offsets = np.arange(w, dtype=np.int64)
    for start in range(0, n_w, chunk_nw):
        end = min(start + chunk_nw, n_w)
        abs_starts = np.arange(start, end, dtype=np.int64)
        idx = abs_starts[:, None] + offsets[None, :]
        yield X[idx].reshape(end - start, w * c), start


def fit_kmeans_ad(
    df_or_series,
    value_col=None,
    *,
    window_size: int = 50,
    n_clusters: int = 8,
    max_iter: int = 100,
    tol: float = 1e-4,
    contamination: float = 0.1,
    aggregate: str = "max",
    random_state: int | None = 42,
    guard=None,
    batch_size: "int | None" = None,
) -> KMeansADResult:
    """Dual dispatch (v0.9.116): df + value_col | series (legacy).

    df path: fit_kmeans_ad(df, value_col)
    Legacy:  fit_kmeans_ad(series)  # equivalent to previous behaviour
    """
    is_df = (hasattr(df_or_series, "_iter_filtered_batches")
             and hasattr(df_or_series, "_columns"))
    if is_df:
        if value_col is None:
            raise ValueError(
                "fit_kmeans_ad: value_col required when df is passed")
        if value_col not in df_or_series._columns:
            raise KeyError(
                f"fit_kmeans_ad: column {value_col!r} not in {df_or_series._columns}")
        return _fit_kmeans_ad_df(
            df_or_series, value_col,
            window_size=window_size,
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            contamination=contamination,
            aggregate=aggregate,
            random_state=random_state,
            guard=guard, batch_size=batch_size)
    if value_col is not None:
        raise ValueError(
            "fit_kmeans_ad: value_col only used with df input; omit for np.ndarray")
    return _fit_kmeans_ad_series(
        df_or_series,
        window_size=window_size,
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        contamination=contamination,
        aggregate=aggregate,
        random_state=random_state,
        guard=guard)


def _fit_kmeans_ad_series(series, *,
    window_size: int = 50,
    n_clusters: int = 8,
    max_iter: int = 100,
    tol: float = 1e-4,
    contamination: float = 0.1,
    aggregate: str = "max",
    random_state: int | None = 42,
    guard=None,
):
    return _fit_kmeans_ad_series_impl(
        series,
        window_size=window_size,
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        contamination=contamination,
        aggregate=aggregate,
        random_state=random_state,
        guard=guard)


def _fit_kmeans_ad_df(df, value_col, *,
    window_size: int = 50,
    n_clusters: int = 8,
    max_iter: int = 100,
    tol: float = 1e-4,
    contamination: float = 0.1,
    aggregate: str = "max",
    random_state: int | None = 42,
    guard=None,
    batch_size=None):
    mg = guard if guard is not None else getattr(df, "_memory_guard", None)
    parts = []
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="fit_kmeans_ad.batch", extra_factor=2.0)
        x_b = b.column(value_col).cast(
            pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        if x_b.size > 0:
            parts.append(x_b)
    if not parts:
        raise ValueError(f"fit_kmeans_ad: column {value_col!r} is empty")
    series = np.concatenate(parts) if len(parts) > 1 else parts[0]
    return _fit_kmeans_ad_series(
        series,
        window_size=window_size,
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        contamination=contamination,
        aggregate=aggregate,
        random_state=random_state,
        guard=mg)


def _fit_kmeans_ad_series_impl(
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

    v0.9.91-S6c: two paths.
      - Duck-typed guard (max_memory_mb only, no check_state):
        old path (full (n_w, d) via _prepare_windows). Bit-identical
        to pre-S6c. Tight budget → MemoryLimitExceeded.
      - Real MemoryGuard (has check_state): chunked OOC path.
        Reservoir sample S windows for kmeans++ init; chunked Lloyd
        iterations (assign + accumulate per chunk); chunked final
        score. Peak = S*d*8 + k*d*8 + chunk*d*8 + 3*n_w*8.
        NOT bit-identical to old path (init sample differs); tests
        with recall/contamination thresholds green.
    """
    _arr = np.asarray(series)
    n_pre = int(_arr.shape[0]) if _arr.ndim >= 1 else 0
    c_pre = 1 if _arr.ndim == 1 else int(np.prod(_arr.shape[1:]))
    if n_pre < window_size:
        raise ValueError(
            f"series length {n_pre} < window_size {window_size}")
    n_w_pre = n_pre - window_size + 1
    if n_w_pre < n_clusters:
        raise ValueError(
            f"n_windows ({n_w_pre}) < n_clusters ({n_clusters})")
    if aggregate not in ("max", "mean"):
        raise ValueError(
            f"aggregate must be 'max' or 'mean', got {aggregate!r}")
    d_pre = window_size * c_pre

    has_check_state = (
        guard is not None and hasattr(guard, "check_state")
        and callable(guard.check_state)
    )

    if not has_check_state:
        # ── Old path (back-compat, bit-identical) ─────────────
        if guard is not None:
            est_bytes = 8 * n_w_pre * d_pre * 2
            if hasattr(guard, "check_state"):
                pass  # unreachable (has_check_state is False)
            limit_mb = _probe_memory_limit_mb(guard)
            if limit_mb is not None and limit_mb > 0:
                if est_bytes > limit_mb * 1024 * 1024:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"KMeansAD estimated {est_bytes} bytes > "
                        f"guard {limit_mb * 1024 * 1024:.0f} bytes")

        windows, n, c = _prepare_windows(series, window_size, stride=1)
        n_w = len(windows)
        d = windows.shape[1]
        rng = np.random.default_rng(random_state)
        centroids = _kmeans_plus_plus_init(windows, n_clusters, rng)
        centroids, labels_per_window, dists_sq, n_iter_run, inertia = (
            _lloyd(windows, centroids, max_iter, tol, rng))
        window_scores = np.sqrt(
            dists_sq[np.arange(n_w), labels_per_window])
    else:
        # ── Chunked OOC path (real MemoryGuard) ───────────────
        n = n_pre
        c = c_pre
        d = d_pre
        n_w = n_w_pre
        chunk_nw = _resolve_chunk_nw_km(d, guard)

        # 1. Reservoir sample S windows for kmeans++ init
        #    (bounded by max(n_clusters*10, min(10000, n_w))).
        S = max(n_clusters * 10, min(10000, n_w))
        S = min(S, n_w)
        rng = np.random.default_rng(random_state)
        sample = np.empty((S, d), dtype=np.float64)
        n_seen = 0
        for chunk, _ in _iter_window_chunks_km(
                series, window_size, chunk_nw):
            for row in chunk:
                if n_seen < S:
                    sample[n_seen] = row
                else:
                    j = int(rng.integers(0, n_seen + 1))
                    if j < S:
                        sample[j] = row
                n_seen += 1

        centroids = _kmeans_plus_plus_init(sample[:n_seen], n_clusters, rng)

        # 2. Chunked Lloyd iterations (assign + accumulate)
        n_iter_run = 0
        prev_inertia = np.inf
        for it in range(int(max_iter)):
            sums = np.zeros((n_clusters, d), dtype=np.float64)
            counts = np.zeros(n_clusters, dtype=np.int64)
            inertia_acc = 0.0
            for chunk, _ in _iter_window_chunks_km(
                    series, window_size, chunk_nw):
                nb = chunk.shape[0]
                d_sq = _distances_sq(chunk, centroids)
                labs = np.argmin(d_sq, axis=1)
                inertia_acc += float(
                    d_sq[np.arange(nb), labs].sum())
                for j in range(n_clusters):
                    m = labs == j
                    if m.any():
                        sums[j] += chunk[m].sum(axis=0)
                        counts[j] += int(m.sum())
            new_centroids = centroids.copy()
            for j in range(n_clusters):
                if counts[j] > 0:
                    new_centroids[j] = sums[j] / counts[j]
                # empty: keep old (fallback reinit deferred to S6c-fix)
            shift = float(np.sum((new_centroids - centroids) ** 2))
            centroids = new_centroids
            n_iter_run = it + 1
            if shift < tol * max(1.0, float((centroids ** 2).sum())):
                break
            if abs(prev_inertia - inertia_acc) < tol * max(prev_inertia, 1.0):
                break
            prev_inertia = inertia_acc

        # 3. Chunked final score
        window_scores = np.empty(n_w, dtype=np.float64)
        labels_per_window = np.empty(n_w, dtype=np.int64)
        for chunk, start in _iter_window_chunks_km(
                series, window_size, chunk_nw):
            nb = chunk.shape[0]
            d_sq = _distances_sq(chunk, centroids)
            labs = np.argmin(d_sq, axis=1)
            labels_per_window[start:start + nb] = labs
            window_scores[start:start + nb] = np.sqrt(
                d_sq[np.arange(nb), labs])
        inertia = float(np.sum(window_scores ** 2))

    # ── Common: aggregate + threshold ─────────────────────────
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
