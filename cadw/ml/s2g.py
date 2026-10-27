"""S2G (Series2Graph): graph-based subsequence anomaly detection.

Boniol & Palpanas (2020), PVLDB 13(11). "Series2Graph: Graph-based
Subsequence Anomaly Detection in Time Series."

Algorithm (paper, with one documented deviation):
  1. Local convolution (Section 4.1): for each subsequence S of length
     ℓ, compute z = mean of λ = ℓ/3 consecutive points. Result is a
     vector of length ℓ−λ+1 — a shape-invariant representation without
     explicit shift alignment (no SBD in original S2G).
  2. PCA to 3 components, then rotate space so the (max − min) direction
     becomes the mean axis. Keep components 2 and 3 (shape axes) →
     2D point per subsequence. Angle = atan2(comp3, comp2).
  3. Nodes: quantize angles on [−π, π] into r bins, one node per bin.
     Original paper uses KDE-density peaks along r rays; documented
     deviation — uniform bins give a deterministic node count and are
     comparable on synthetic benchmarks. For n >> r the bins correspond
     closely to KDE peaks.
  4. Edge construction: directed edge (N_i, N_j) with i ≠ j counted as
     w(e) = # occurrences of consecutive node pair in the trajectory.
  5. Degrees: deg(N) = number of distinct incident edges (in + out,
     self-loops excluded from W).
  6. Edge normalcy (Def. 3): Norm(e) = w(e) · (deg(src(e)) − 1), only
     for src ≠ dst.
  7. Query scoring (Section 4.4): query length ℓ_q ≥ ℓ. Slide pattern
     window of length ℓ across the query → path of nodes of length
     ℓ_q − ℓ + 1. Path normalcy = mean normalcy over non-self edges in
     the path. Query anomaly score = −Path normalcy (higher = more
     anomalous).
  8. Timestamp score = max over covering queries (worst-case).
  9. Threshold θ = percentile(1 − contamination) on timestamp scores.

Reference: TimeEval/TimeEval-algorithms/series2graph; s2gpp (Rust).

Pure numpy — no sklearn/scipy/torch/aeon in cadw/ (AST-guard
test_no_bench_deps_in_core.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class S2GResult:
    decision_scores: np.ndarray
    labels: np.ndarray
    threshold: float
    node_ids: np.ndarray
    transition_matrix: np.ndarray
    degrees: np.ndarray
    embedding: np.ndarray
    window_size: int
    query_window_size: int
    convolution_size: int
    n_nodes: int
    contamination: float


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


# ── Local convolution (Section 4.1, step 1) ───────────────────────

def _local_convolution(windows: np.ndarray, lam: int) -> np.ndarray:
    """Sliding-mean (uniform convolution) of length lam inside each window.

    windows: (n_w, l). Returns (n_w, l - lam + 1) — shape-invariant
    subsequence representation (no shift alignment needed).
    """
    n_w, w = windows.shape
    if lam < 1 or lam > w:
        raise ValueError(
            f"convolution_size must be in [1, window_size={w}], got {lam}")
    out_len = w - lam + 1
    if out_len < 1:
        raise ValueError(
            f"convolution_size={lam} leaves no output (window_size={w})")
    cumsum = np.cumsum(windows, axis=1)
    cs_padded = np.concatenate(
        [np.zeros((n_w, 1), dtype=windows.dtype), cumsum], axis=1)
    return (cs_padded[:, lam:] - cs_padded[:, :-lam]) / float(lam)


# ── 3D PCA + rotation (Section 4.1, steps 2–3) ────────────────────

def _embed_shape(z_matrix: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA on z_matrix, keep top-3 components, rotate so (max−min) direction
    becomes the mean axis. Keep shape components 2 & 3 → 2D points.

    Returns (points_2d (n, 2), components (d, 3), mean (d,)).
    """
    n, d = z_matrix.shape
    if n < 2:
        raise ValueError(f"need >= 2 rows for PCA, got {n}")
    mean = z_matrix.mean(axis=0)
    centered = z_matrix - mean
    cov = (centered.T @ centered) / max(n - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:3]
    components = eigvecs[:, order]  # (d, 3)
    proj3 = centered @ components    # (n, 3)

    # v1 = direction from min to max in 3D projection space
    v1 = proj3.max(axis=0) - proj3.min(axis=0)
    norm1 = float(np.linalg.norm(v1))
    if norm1 < 1e-12:
        v1 = np.array([1.0, 0.0, 0.0])
    else:
        v1 = v1 / norm1
    # Orthonormal basis with v1 as first axis
    v2_seed = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(v1, v2_seed))) > 0.9:
        v2_seed = np.array([1.0, 0.0, 0.0])
    e1 = v1
    e2 = v2_seed - np.dot(v2_seed, e1) * e1
    n2 = float(np.linalg.norm(e2))
    if n2 < 1e-12:
        e2 = np.array([0.0, 0.0, 1.0])
        e2 = e2 - np.dot(e2, e1) * e1
        n2 = float(np.linalg.norm(e2))
    e2 = e2 / max(n2, 1e-12)
    e3 = np.cross(e1, e2)

    # Rotate projections so v1 → x-axis
    R = np.stack([e1, e2, e3], axis=0)  # (3, 3)
    proj_rot = proj3 @ R.T              # (n, 3)
    points_2d = proj_rot[:, 1:3]        # shape components (axes 2, 3)
    return points_2d, components, mean


# ── Node quantization (paper uses KDE peaks; uniform bins here) ──

def _angles_from_points(points_2d: np.ndarray) -> np.ndarray:
    return np.arctan2(points_2d[:, 1], points_2d[:, 0])


def _quantize_to_nodes(angles: np.ndarray, n_nodes: int) -> np.ndarray:
    """Uniform bin on [−π, π] into n_nodes bins → node id per angle."""
    edges = np.linspace(-np.pi, np.pi, n_nodes + 1)
    ids = np.digitize(angles, edges) - 1
    return np.clip(ids, 0, n_nodes - 1)


# ── Graph construction (Section 4.2) ──────────────────────────────

def _build_graph(node_ids: np.ndarray, n_nodes: int) -> np.ndarray:
    """W[src, dst] = # transitions (src ≠ dst) between consecutive nodes."""
    W = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for t in range(1, len(node_ids)):
        a, b = int(node_ids[t - 1]), int(node_ids[t])
        if a != b:
            W[a, b] += 1.0
    return W


def _compute_degrees(W: np.ndarray) -> np.ndarray:
    """deg(N) = # distinct incident edges (in + out, self-loops excluded)."""
    in_deg = (W > 0).sum(axis=0).astype(np.float64)
    out_deg = (W > 0).sum(axis=1).astype(np.float64)
    return in_deg + out_deg


# ── Path normalcy (Definition 3 + Section 4.4) ────────────────────

def _path_normalcy(path_nodes: np.ndarray, W: np.ndarray,
                   deg: np.ndarray) -> float:
    """Mean edge normalcy over non-self transitions in path.

    Norm(e) = w(e) · (deg(src(e)) − 1) for src ≠ dst.
    Path with no non-self edges → 0.0 (conservative: no evidence of
    normality).
    """
    if len(path_nodes) < 2:
        return 0.0
    total = 0.0
    count = 0
    for k in range(len(path_nodes) - 1):
        a, b = int(path_nodes[k]), int(path_nodes[k + 1])
        if a == b:
            continue
        w_ab = float(W[a, b])
        d = float(deg[a])
        # (d - 1) can be 0 if the node has exactly 1 out-edge and 0 in-edges.
        # That edge alone is "non-normal" by Definition 3 — include it as
        # zero contribution (worst case for anomaly).
        normalcy = w_ab * max(d - 1.0, 0.0)
        total += normalcy
        count += 1
    if count == 0:
        return 0.0
    return total / count


# ── Query scoring (Section 4.4) ───────────────────────────────────

def _score_queries(x: np.ndarray, *, window_size: int,
                   query_window_size: int, convolution_size: int,
                   n_nodes: int, W: np.ndarray, deg: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (all_pattern_nodes, query_scores, query_starts).

    Trajectory of pattern nodes over the whole series is computed once;
    each query re-uses a slice of length (ℓ_q − ℓ + 1).
    """
    n = len(x)
    n_patterns = n - window_size + 1
    if n_patterns < 1:
        raise ValueError(
            f"series length {n} < window_size {window_size}")
    windows = np.lib.stride_tricks.sliding_window_view(x, window_size).copy()
    z = _local_convolution(windows, convolution_size)
    points_2d, _, _ = _embed_shape(z)
    angles = _angles_from_points(points_2d)
    pattern_nodes = _quantize_to_nodes(angles, n_nodes)

    n_q = n - query_window_size + 1
    if n_q < 1:
        # query_window_size longer than series: use one window covering all
        n_q = 1
    query_starts = np.arange(n_q)
    path_len = query_window_size - window_size + 1
    if path_len < 1:
        raise ValueError(
            f"query_window_size ({query_window_size}) < window_size "
            f"({window_size})")
    scores = np.empty(n_q, dtype=np.float64)
    for i, start in enumerate(query_starts):
        path = pattern_nodes[start:start + path_len]
        normalcy = _path_normalcy(path, W, deg)
        scores[i] = -normalcy  # higher = more anomalous
    return pattern_nodes, scores, query_starts


# ── Aggregation: max over covering queries ────────────────────────

def _aggregate_to_timestamps(query_scores: np.ndarray,
                             query_starts: np.ndarray,
                             query_window_size: int,
                             n: int) -> np.ndarray:
    """Per-timestamp score = max over queries that cover the timestamp."""
    ts_scores = np.full(n, -np.inf, dtype=np.float64)
    for score, start in zip(query_scores, query_starts):
        end = min(start + query_window_size, n)
        np.maximum(ts_scores[start:end], score, out=ts_scores[start:end])
    ts_scores[~np.isfinite(ts_scores)] = 0.0
    return ts_scores


# ── Public API ────────────────────────────────────────────────────

def fit_s2g(
    series: np.ndarray,
    *,
    window_size: int = 50,
    query_window_size: int = 75,
    n_nodes: int = 50,
    contamination: float = 0.1,
    guard=None,
) -> S2GResult:
    """S2G anomaly detection.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
    window_size : int, default=50
        Pattern length ℓ (TimeEval default).
    query_window_size : int, default=75
        Query length ℓ_q ≥ ℓ (TimeEval default).
    n_nodes : int, default=50
        Number of angle bins (paper uses r=50 rays by default).
    contamination : float, default=0.1
        Expected anomaly fraction (threshold = percentile(1-contamination)).
    guard : MemoryGuard | None

    Returns
    -------
    S2GResult
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < window_size:
        raise ValueError(
            f"series length {n} < window_size {window_size}")
    if query_window_size < window_size:
        raise ValueError(
            f"query_window_size ({query_window_size}) must be >= "
            f"window_size ({window_size})")
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be >= 2, got {n_nodes}")

    lam = max(1, window_size // 3)  # TimeEval: convolution_size = ℓ/3

    # v0.9.87-fix1b (audit HIGH-1): est_bytes must cover BOTH
    # (n_w, w) window arrays — one in fit_s2g (pattern_windows) and
    # one inside _score_queries. Plus z = _local_convolution output
    # (n_w, w-lam+1). Previous formula
    # 8*(n_w)*(w-lam+1)*2 underestimates by ~1.5x.
    if guard is not None:
        _n_w = n - window_size + 1
        # window (fit_s2g) + conv output (z) + window (_score_queries)
        # conservative: 4 * (n_w, w)
        est_bytes = 8 * _n_w * window_size * 4
        if hasattr(guard, "check_state") and callable(guard.check_state):
            guard.check_state(est_bytes, name="s2g.windows")
        else:
            limit_mb = _probe_memory_limit_mb(guard)
            if limit_mb is not None and limit_mb > 0:
                if est_bytes > limit_mb * 1024 * 1024:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"S2G estimated {est_bytes} bytes > "
                        f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    # Compute pattern node trajectory (needed for graph)
    pattern_windows = np.lib.stride_tricks.sliding_window_view(
        x, window_size).copy()
    z = _local_convolution(pattern_windows, lam)
    points_2d, components, mean = _embed_shape(z)
    angles = _angles_from_points(points_2d)
    pattern_nodes = _quantize_to_nodes(angles, n_nodes)

    # Graph
    W = _build_graph(pattern_nodes, n_nodes)
    deg = _compute_degrees(W)

    # Query scoring (re-uses pattern trajectory; no recompute)
    _, query_scores, query_starts = _score_queries(
        x, window_size=window_size, query_window_size=query_window_size,
        convolution_size=lam, n_nodes=n_nodes, W=W, deg=deg)

    # Aggregate to timestamps
    ts_scores = _aggregate_to_timestamps(
        query_scores, query_starts, query_window_size, n)

    threshold = float(np.percentile(
        ts_scores, 100.0 * (1.0 - contamination)))
    labels = (ts_scores > threshold).astype(int)

    return S2GResult(
        decision_scores=ts_scores,
        labels=labels,
        threshold=threshold,
        node_ids=pattern_nodes,
        transition_matrix=W,
        degrees=deg,
        embedding=points_2d,
        window_size=window_size,
        query_window_size=query_window_size,
        convolution_size=lam,
        n_nodes=n_nodes,
        contamination=contamination,
    )

