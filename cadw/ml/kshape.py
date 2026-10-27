"""KShapeAD: k-Shape clustering-based TS anomaly detection.

Paparrizos & Gravano, SIGMOD 2015. Non-learning? No — it clusters
sliding-window subsequences (like k-means), then scores each window by
its Shape-Based Distance (SBD) to the nearest centroid. Subsequences far
from all centroids are anomalous.

Pure numpy — no sklearn/PyOD/torch imports in cadw/ (AST-guard
test_no_bench_deps_in_core.py).

Reference: pyod.models.ts_kshape.KShape (#2 in TSB-AD per PyOD docs).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KShapeResult:
    """Result of fit_kshape.

    Attributes
    ----------
    decision_scores : np.ndarray, shape (n,)
        Per-timestamp anomaly score (max SBD to nearest centroid over
        covering windows). Higher = more anomalous.
    labels : np.ndarray, shape (n,), int
        0/1 binary labels (score > threshold).
    threshold : float
        Percentile(1-contamination) of decision_scores.
    centroids : np.ndarray, shape (n_clusters, window_size)
        Learned z-normalized centroids.
    window_size, n_clusters, max_iter : int
        Hyperparameters used.
    n_iter_run : int
        Actual number of Lloyd iterations run (<= max_iter).
    contamination : float
        Contamination rate used for threshold.
    """
    decision_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    centroids: np.ndarray
    window_size: int
    n_clusters: int
    max_iter: int
    n_iter_run: int
    contamination: float


# ── MemoryGuard probe (same defensive logic as dwt_mlead) ─────────

from cadw.core.spill import SpillDir


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


# ── Z-normalization ───────────────────────────────────────────────

def _znormalize(x: np.ndarray) -> np.ndarray:
    s = float(np.std(x))
    if s < 1e-10:
        return np.zeros_like(x, dtype=np.float64)
    return (x - float(np.mean(x))) / s


# ── FFT size helper ───────────────────────────────────────────────

def _fft_size(m: int) -> int:
    L = 1
    while L < 2 * m - 1:
        L *= 2
    return L


# ── SBD (scalar) ──────────────────────────────────────────────────

def _sbd(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Shape-based distance between two z-normalized sequences.

    SBD(x, y) = 1 - max_w CC_w(x, y) / (||x|| * ||y||)

    Returns (dist in [0, 2], optimal shift in [-(m-1), m-1]).
    """
    m = len(x)
    L = _fft_size(m)
    fx = np.fft.fft(x, L)
    fy = np.fft.fft(y, L)
    cc = np.real(np.fft.ifft(fx * np.conj(fy)))
    cc_full = np.concatenate([cc[-(m - 1):], cc[:m]])
    norm = float(np.linalg.norm(x)) * float(np.linalg.norm(y))
    if norm < 1e-10:
        return 2.0, 0
    ncc = cc_full / norm
    idx = int(np.argmax(ncc))
    dist = float(np.clip(1.0 - ncc[idx], 0.0, 2.0))
    shift = idx - (m - 1)
    return dist, shift


# ── SBD batch: one window vs all centroids (vectorized) ───────────

def _sbd_batch(window: np.ndarray, centroids: np.ndarray,
               centroid_ffts: np.ndarray | None = None,
               L: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """SBD from one window to all centroids. Returns (dists (k,), shifts (k,))."""
    m = len(window)
    k = len(centroids)
    if L is None:
        L = _fft_size(m)
    fx = np.fft.fft(window, L)
    if centroid_ffts is None:
        centroid_ffts = np.fft.fft(centroids, L, axis=1)
    cc = np.real(np.fft.ifft(fx[None, :] * np.conj(centroid_ffts), axis=1))  # (k, L)
    cc_full = np.concatenate([cc[:, -(m - 1):], cc[:, :m]], axis=1)  # (k, 2m-1)
    norms = float(np.linalg.norm(window)) * np.linalg.norm(centroids, axis=1)  # (k,)
    wn = float(np.linalg.norm(window))
    if wn < 1e-10:
        return np.full(k, 2.0), np.zeros(k, dtype=int)
    safe_norms = np.where(norms < 1e-10, 1.0, norms)
    ncc = cc_full / safe_norms[:, None]
    idx = np.argmax(ncc, axis=1)
    max_ncc = ncc[np.arange(k), idx]
    dist = np.clip(1.0 - max_ncc, 0.0, 2.0)
    shift = idx - (m - 1)
    # Zero-norm centroid -> distance 2
    zero_centroid = norms < 1e-10
    dist = np.where(zero_centroid, 2.0, dist)
    return dist, shift


# ── Shift alignment ───────────────────────────────────────────────

def _shift_align(x: np.ndarray, shift: int, m: int) -> np.ndarray:
    aligned = np.zeros(m, dtype=np.float64)
    if shift >= 0:
        end = min(m, m - shift)
        if end > 0:
            aligned[shift:shift + end] = x[:end]
    else:
        start = -shift
        end = min(m, m + shift)
        if end > 0:
            aligned[:end] = x[start:start + end]
    return aligned


# ── Shape extraction (centroid) ───────────────────────────────────

def _compute_centroid(members: np.ndarray) -> np.ndarray:
    """Centroid = smallest eigenvector of M = I - Q/n, Q = X^T X.

    Equivalent to largest eigenvector of Q (maximizes Rayleigh quotient
    for SBD similarity). Sign oriented by dot with members mean.
    """
    n, m = members.shape
    if n == 0:
        return np.zeros(m, dtype=np.float64)
    if n == 1:
        return _znormalize(members[0])
    Q = members.T @ members
    M = np.eye(m) - Q / n
    _, eigvecs = np.linalg.eigh(M)  # ascending -> [:, 0] = smallest
    centroid = eigvecs[:, 0]
    if float(np.dot(members.mean(axis=0), centroid)) < 0:
        centroid = -centroid
    return _znormalize(centroid)


# ── Sliding windows ───────────────────────────────────────────────

# v0.9.87-P3 (audit HIGH-1 full fix): streaming kshape.
# windows + window_ffts stored in memmap (SpillDir); assignment
# chunked; centroid update via streaming Q = sum(chunk.T @ chunk).
# Peak Python-side RAM O(chunk_nw*(m+L) + k*L + m²).


def _resolve_chunk_nw_ks(m: int, L: int, k: int,
                         guard=None) -> int:
    """Chunk size so per-chunk peak RAM stays within budget."""
    default_target = 8 * 1024 * 1024
    target_bytes = default_target
    if guard is not None:
        state_limit = getattr(guard, "state_limit_bytes", None)
        if (isinstance(state_limit, (int, float))
                and state_limit > 0):
            target_bytes = min(int(target_bytes),
                               max(1, int(state_limit) // 8))
    # per-window bytes in _assign_all_chunked:
    #   windows(chunk)      : 8m
    #   ffts(chunk)         : 16L
    #   cc                  : 16kL   (complex128)
    #   cc_full / ncc       : 8k(2m-1) each ~= 16k(2m-1)/2 = 8k(2m-1)
    #   norms/shifts/etc    : ~24k
    bytes_per_window = (m * 8 + L * 16 + k * L * 16
                        + k * (2 * m - 1) * 8 + k * 8)
    n = max(100, target_bytes // bytes_per_window)
    return min(10_000, n)


def _iter_x_window_chunks_ks(x: np.ndarray, w: int, chunk_nw: int):
    """Yield (chunk(nb, w), start) chunks from 1D x."""
    n = len(x)
    n_w = n - w + 1
    if n_w < 1:
        return
    offsets = np.arange(w, dtype=np.int64)
    for start in range(0, n_w, chunk_nw):
        end = min(start + chunk_nw, n_w)
        abs_starts = np.arange(start, end, dtype=np.int64)
        idx = abs_starts[:, None] + offsets[None, :]
        yield x[idx], start


def _compute_centroid_from_stats(Q: np.ndarray, mean_member: np.ndarray,
                                  n_members: int,
                                  m: int) -> np.ndarray:
    """Centroid from pre-accumulated Q = sum(member.T @ member) and
    mean_member. Same math as _compute_centroid, streaming input.
    """
    if n_members == 0:
        return np.zeros(m, dtype=np.float64)
    if n_members == 1:
        return _znormalize(mean_member)
    M = np.eye(m) - Q / n_members
    _, eigvecs = np.linalg.eigh(M)
    centroid = eigvecs[:, 0]
    if float(np.dot(mean_member, centroid)) < 0:
        centroid = -centroid
    return _znormalize(centroid)




# ── Map window scores to timestamps (max aggregation) ─────────────

def _map_scores_to_timestamps(window_scores: np.ndarray, w: int,
                              n: int) -> np.ndarray:
    scores = np.full(n, -np.inf)
    for i in range(len(window_scores)):
        end = min(i + w, n)
        np.maximum(scores[i:end], window_scores[i], out=scores[i:end])
    scores[~np.isfinite(scores)] = 0.0
    return scores


# ── Core k-Shape loop ─────────────────────────────────────────────

def _assign_all_chunked(windows_mm, centroids: np.ndarray,
                        ffts_mm, L: int, m: int,
                        chunk_nw: int, n_w: int) -> tuple:
    """Chunked assignment (was _assign_all). Same math, bounded RAM.

    Returns (labels, distances, shifts) — each (n_w,) numpy array.
    """
    k = len(centroids)
    centroid_ffts = np.fft.fft(centroids, L, axis=1)  # (k, L)
    norms_c = np.linalg.norm(centroids, axis=1)  # (k,)
    labels = np.empty(n_w, dtype=np.int64)
    distances = np.empty(n_w, dtype=np.float64)
    shifts = np.empty(n_w, dtype=np.int64)

    for start in range(0, n_w, chunk_nw):
        end = min(start + chunk_nw, n_w)
        w_chunk = np.asarray(windows_mm[start:end])
        f_chunk = np.asarray(ffts_mm[start:end])
        nb = end - start
        cc = np.real(np.fft.ifft(
            f_chunk[:, None, :] * np.conj(centroid_ffts[None, :, :]),
            axis=2))
        cc_full = np.concatenate(
            [cc[:, :, -(m - 1):], cc[:, :, :m]], axis=2)
        norms_w = np.linalg.norm(w_chunk, axis=1)
        norms = norms_w[:, None] * norms_c[None, :]
        safe = np.where(norms < 1e-10, 1.0, norms)
        ncc = cc_full / safe[:, :, None]
        ncc = np.where(norms[:, :, None] < 1e-10, -np.inf, ncc)
        idx = np.argmax(ncc, axis=2)
        max_ncc = np.take_along_axis(
            ncc, idx[:, :, None], axis=2)[:, :, 0]
        dists_per = np.clip(1.0 - max_ncc, 0.0, 2.0)
        dists_per = np.where(norms < 1e-10, 2.0, dists_per)
        best = np.argmin(dists_per, axis=1)
        labels[start:end] = best
        distances[start:end] = dists_per[np.arange(nb), best]
        shifts[start:end] = idx[np.arange(nb), best] - (m - 1)

    return labels, distances, shifts


def _kshape_chunked(windows_mm, ffts_mm, n_w: int, m: int, L: int,
                    n_clusters: int, max_iter: int,
                    rng: np.random.RandomState,
                    chunk_nw: int) -> tuple:
    """Chunked k-Shape. Same math as v0.9.86, bounded RAM.

    windows_mm, ffts_mm: memmap-backed (n_w, m) and (n_w, L).
    windows_mm must contain z-normalized windows (fit_kshape does
    that when filling).
    """
    indices = rng.choice(n_w, size=n_clusters, replace=False)
    centroids = np.array([
        _znormalize(np.asarray(windows_mm[int(i)])) for i in indices
    ])
    labels = np.full(n_w, -1, dtype=np.int64)
    distances = np.full(n_w, np.inf, dtype=np.float64)
    n_iter_run = 0

    for iteration in range(max_iter):
        old_labels = labels.copy()
        labels, distances, shifts = _assign_all_chunked(
            windows_mm, centroids, ffts_mm, L, m, chunk_nw, n_w)
        if np.array_equal(labels, old_labels):
            n_iter_run = iteration + 1
            break
        for k in range(n_clusters):
            Q = np.zeros((m, m), dtype=np.float64)
            mean_sum = np.zeros(m, dtype=np.float64)
            n_members = 0
            for start in range(0, n_w, chunk_nw):
                end = min(start + chunk_nw, n_w)
                chunk_labels = labels[start:end]
                sel = np.where(chunk_labels == k)[0]
                if len(sel) == 0:
                    continue
                w_chunk = np.asarray(windows_mm[start:end])
                chunk_shifts = shifts[start:end]
                aligned = np.empty((len(sel), m), dtype=np.float64)
                for j, li in enumerate(sel):
                    aligned[j] = _znormalize(
                        _shift_align(w_chunk[li],
                                     int(chunk_shifts[li]), m))
                Q += aligned.T @ aligned
                mean_sum += aligned.sum(axis=0)
                n_members += len(sel)
            if n_members == 0:
                idx = int(rng.randint(n_w))
                centroids[k] = _znormalize(
                    np.asarray(windows_mm[idx]))
                continue
            mean_member = mean_sum / n_members
            centroids[k] = _compute_centroid_from_stats(
                Q, mean_member, n_members, m)
        n_iter_run = iteration + 1

    _, distances, _ = _assign_all_chunked(
        windows_mm, centroids, ffts_mm, L, m, chunk_nw, n_w)
    return centroids, labels, distances, n_iter_run


# ── Public API ────────────────────────────────────────────────────

def fit_kshape(
    series: np.ndarray,
    *,
    n_clusters: int = 3,
    window_size: int = 50,
    max_iter: int = 100,
    contamination: float = 0.1,
    random_state: int = 42,
    guard=None,
) -> KShapeResult:
    """KShapeAD: k-Shape clustering-based anomaly detection.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
        1D univariate time series.
    n_clusters : int, default=3
        Number of k-Shape clusters.
    window_size : int, default=50
        Sliding window length.
    max_iter : int, default=100
        Maximum Lloyd iterations.
    contamination : float, default=0.1
        Fraction expected anomalous; used for threshold only.
    random_state : int, default=42
        Seed for centroid init / empty-cluster reinit.
    guard : MemoryGuard | None
        Optional. Estimated memory ~ 8 * n_windows * window_size bytes.

    Returns
    -------
    KShapeResult
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < window_size:
        raise ValueError(
            f"series length {n} < window_size {window_size}")
    n_windows = n - window_size + 1
    if n_windows < n_clusters:
        raise ValueError(
            f"n_windows {n_windows} < n_clusters {n_clusters}")

    # v0.9.87-fix1b (audit HIGH-1): est ×2 to account for the
    # second allocation `znorm_windows = np.array([...])` right
    # after `_sliding_windows`. Both are (n_w, w) float64 arrays.
    if guard is not None:
        # v0.9.87-P3: per-chunk buffers + output + model params.
        _m_pre = window_size
        _L_pre = _fft_size(_m_pre)
        _chunk_pre = _resolve_chunk_nw_ks(
            _m_pre, _L_pre, n_clusters, guard)
        _per_chunk = _chunk_pre * (
            _m_pre * 8 + _L_pre * 16 + n_clusters * _L_pre * 16
            + n_clusters * (2 * _m_pre - 1) * 8 + n_clusters * 8)
        _output = (n - _m_pre + 1) * 24
        _model = _m_pre * _m_pre * 8 + n_clusters * _m_pre * 8 + 16
        est_bytes = int(_per_chunk + _output + _model)
        if hasattr(guard, "check_state") and callable(guard.check_state):
            guard.check_state(est_bytes, name="kshape.windows+znorm")
        else:
            limit_mb = _probe_memory_limit_mb(guard)
            if limit_mb is not None and limit_mb > 0:
                max_bytes = limit_mb * 1024 * 1024
                if est_bytes > max_bytes:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"KShape estimated {est_bytes} bytes > "
                        f"guard {max_bytes} bytes (limit_mb={limit_mb})")

    # v0.9.87-P3: memmap-backed windows + precomputed ffts; chunked
    # assignment + streaming centroid. Peak Python-side RAM
    # O(chunk_nw*(m+L) + k*L + m²). Bit-identical within float.
    m = window_size
    n_w_actual = n - m + 1
    L = _fft_size(m)
    _chunk_nw = _resolve_chunk_nw_ks(m, L, n_clusters, guard)

    rng = np.random.RandomState(random_state)
    with SpillDir(guard=guard, name="kshape") as _sdir:
        win_path = _sdir.path("windows.bin")
        ffts_path = _sdir.path("ffts.bin")
        windows_mm = np.memmap(
            str(win_path), dtype=np.float64, mode="w+",
            shape=(n_w_actual, m))
        ffts_mm = np.memmap(
            str(ffts_path), dtype=np.complex128, mode="w+",
            shape=(n_w_actual, L))

        for chunk, _start in _iter_x_window_chunks_ks(
                x, m, _chunk_nw):
            nb = chunk.shape[0]
            zn = np.empty_like(chunk)
            for j in range(nb):
                zn[j] = _znormalize(chunk[j])
            windows_mm[_start:_start + nb] = zn
        windows_mm.flush()

        for _start in range(0, n_w_actual, _chunk_nw):
            _end = min(_start + _chunk_nw, n_w_actual)
            w_chunk = np.asarray(windows_mm[_start:_end])
            ffts_mm[_start:_end] = np.fft.fft(w_chunk, L, axis=1)
        ffts_mm.flush()

        centroids, _labels, window_distances, n_iter_run = (
            _kshape_chunked(
                windows_mm, ffts_mm, n_w_actual, m, L,
                n_clusters, max_iter, rng, _chunk_nw))

        del windows_mm, ffts_mm

    decision_scores = _map_scores_to_timestamps(
        window_distances, window_size, n)
    threshold = float(np.percentile(
        decision_scores, 100.0 * (1.0 - contamination)))
    labels = (decision_scores > threshold).astype(int)

    return KShapeResult(
        decision_scores=decision_scores,
        labels=labels,
        threshold=threshold,
        centroids=centroids,
        window_size=window_size,
        n_clusters=n_clusters,
        max_iter=max_iter,
        n_iter_run=n_iter_run,
        contamination=contamination,
    )

