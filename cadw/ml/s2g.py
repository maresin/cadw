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
import pyarrow as pa


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

    def state_nbytes(self) -> int:
        """Persistent state: decision_scores(n f64) + labels(n i64)
        + node_ids(n_w i64) + transition_matrix(n_nodes² f64)
        + degrees(n_nodes f64) + embedding(n_w,2 f64) + header.

        v0.9.115: honest O(n) (CoxPH-precedent). Formula:
        40*n*f64 + n_nodes*n_nodes*f64 + n_nodes*f64 + 264.
        bounded_by_domain n<=1e6 accepts this (~40 MB at n=1e6).
        """
        return int(
            self.decision_scores.nbytes
            + self.labels.nbytes
            + self.node_ids.nbytes
            + self.transition_matrix.nbytes
            + self.degrees.nbytes
            + self.embedding.nbytes
            + 8    # threshold
            + 256  # dataclass header
        )


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


# v0.9.87-P4c (audit HIGH-1 full fix): streaming s2g pipeline.
# Peak RAM O(chunk_nw * (w + 3*d_z)) regardless of n. Output
# pattern_nodes(n_w,) + points_2d(n_w,2) kept (public S2GResult
# fields). See METHODOLOGY §5 (windowed state_type).


def _resolve_chunk_nw_s2g(w: int, d_z: int, guard=None) -> int:
    """Chunk size so per-chunk peak RAM stays within budget."""
    default_target = 8 * 1024 * 1024
    target_bytes = default_target
    if guard is not None:
        state_limit = getattr(guard, "state_limit_bytes", None)
        if (isinstance(state_limit, (int, float))
                and state_limit > 0):
            target_bytes = min(int(target_bytes),
                               max(1, int(state_limit) // 8))
    bytes_per_window = (w + 3 * d_z + 6) * 8
    n = max(100, target_bytes // bytes_per_window)
    return min(10_000, n)


def _iter_x_window_chunks(x: np.ndarray, window_size: int,
                          chunk_nw: int):
    """Yield (chunk(nb, w), start) chunks from 1D x.

    Strided read + bounded materialization: only chunk_nw windows
    resident at a time.
    """
    n = len(x)
    n_w = n - window_size + 1
    if n_w < 1:
        return
    offsets = np.arange(window_size, dtype=np.int64)
    for start in range(0, n_w, chunk_nw):
        end = min(start + chunk_nw, n_w)
        abs_starts = np.arange(start, end, dtype=np.int64)
        idx = abs_starts[:, None] + offsets[None, :]
        yield x[idx], start


def _basis_from_v1(v1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (e1, e2, e3) with e1 = v1 (unit)."""
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
    return e1, e2, e3


def _stream_embed_to_nodes(x: np.ndarray, window_size: int,
                           convolution_size: int, n_nodes: int,
                           chunk_nw: int):
    """Streaming (n,) -> (pattern_nodes(n_w,), points_2d(n_w,2)).

    4 passes:
      1. mean_z = sum(z)/n_w
      2. cov_z = sum((z-mean)^T (z-mean)) / (n_w-1)
      3. proj3 min/max for v1 rotation axis
      4. final: proj3 -> rotate -> points_2d -> angles -> nodes
    Peak O(chunk_nw * (w + 3*d_z) + d_z^2).
    """
    n = len(x)
    n_w = n - window_size + 1
    if n_w < 2:
        raise ValueError(f"too few windows: n_w={n_w} < 2")
    d_z = window_size - convolution_size + 1

    # Pass 1: mean_z
    total = np.zeros(d_z, dtype=np.float64)
    for cw, _ in _iter_x_window_chunks(x, window_size, chunk_nw):
        z = _local_convolution(cw, convolution_size)
        total += z.sum(axis=0)
    mean_z = total / n_w

    # Pass 2: cov_z
    M2 = np.zeros((d_z, d_z), dtype=np.float64)
    for cw, _ in _iter_x_window_chunks(x, window_size, chunk_nw):
        z = _local_convolution(cw, convolution_size)
        centered = z - mean_z
        M2 += centered.T @ centered
    cov_z = M2 / max(n_w - 1, 1)

    # Eigh -> top-3 components (deterministic order)
    eigvals, eigvecs = np.linalg.eigh(cov_z)
    order = np.argsort(eigvals)[::-1][:3]
    components = eigvecs[:, order]  # (d_z, 3)

    # Pass 3: proj3 min/max (for v1 axis)
    proj_min = np.full(3, np.inf, dtype=np.float64)
    proj_max = np.full(3, -np.inf, dtype=np.float64)
    for cw, _ in _iter_x_window_chunks(x, window_size, chunk_nw):
        z = _local_convolution(cw, convolution_size)
        centered = z - mean_z
        proj3 = centered @ components
        proj_min = np.minimum(proj_min, proj3.min(axis=0))
        proj_max = np.maximum(proj_max, proj3.max(axis=0))

    v1 = proj_max - proj_min
    norm1 = float(np.linalg.norm(v1))
    if norm1 < 1e-12:
        v1 = np.array([1.0, 0.0, 0.0])
    else:
        v1 = v1 / norm1
    e1, e2, e3 = _basis_from_v1(v1)
    R = np.stack([e1, e2, e3], axis=0)

    # Pass 4: final embed + quantize
    edges = np.linspace(-np.pi, np.pi, n_nodes + 1)
    points_chunks = []
    nodes_chunks = []
    for cw, _ in _iter_x_window_chunks(x, window_size, chunk_nw):
        z = _local_convolution(cw, convolution_size)
        centered = z - mean_z
        proj3 = centered @ components
        proj_rot = proj3 @ R.T
        p2 = proj_rot[:, 1:3]
        angles = np.arctan2(p2[:, 1], p2[:, 0])
        ids = np.digitize(angles, edges) - 1
        nodes = np.clip(ids, 0, n_nodes - 1)
        points_chunks.append(p2)
        nodes_chunks.append(nodes)

    return (np.concatenate(nodes_chunks),
            np.concatenate(points_chunks))


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
    e1, e2, e3 = _basis_from_v1(v1)

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

def _score_queries(pattern_nodes: np.ndarray, n: int, *,
                   window_size: int, query_window_size: int,
                   W: np.ndarray, deg: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return (query_scores, query_starts).

    v0.9.87-P4c: accepts pre-computed pattern_nodes (was: recomputed
    windows from x internally — duplicate work + second G3 violation).
    Caller (fit_s2g) passes pattern_nodes from _stream_embed_to_nodes.

    pattern_nodes: (n_patterns,) int64 pattern trajectory.
    n: original series length (needed for n_q = n - query_window + 1).
    """
    n_patterns = len(pattern_nodes)
    if n_patterns < 1:
        raise ValueError(
            f"series length {n} < window_size {window_size}")
    n_q = n - query_window_size + 1
    if n_q < 1:
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
        scores[i] = -normalcy
    return scores, query_starts


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
    df_or_series, value_col=None, *,
    window_size: int = 50, query_window_size: int = 75,
    n_nodes: int = 50, contamination: float = 0.1,
    guard=None, output: str = "memory", io_batch_size: "int | None" = None,
) -> S2GResult:
    """S2G anomaly detection (v0.9.115 dual dispatch).

    df path: fit_s2g(df, value_col, io_batch_size=None)
    (io_batch_size controls df-iterator batch size).

    output="memory"|"auto" -> in-memory; "disk" -> NotImplementedError
    (SpillArray v0.9.116+, DEC-013).
    """
    if output not in ("memory", "auto"):
        raise NotImplementedError(
            f"fit_s2g: output={output!r} not supported "
            f"(decision_scores is O(n) ndarray; SpillArray v0.9.116+, DEC-013)")
    is_df = (hasattr(df_or_series, "_iter_filtered_batches")
             and hasattr(df_or_series, "_columns"))
    if is_df:
        if value_col is None:
            raise ValueError("fit_s2g: value_col required when df is passed")
        if value_col not in df_or_series._columns:
            raise KeyError(
                f"fit_s2g: column {value_col!r} not in {df_or_series._columns}")
        return _fit_s2g_df(
            df_or_series, value_col,
            window_size=window_size, query_window_size=query_window_size,
            n_nodes=n_nodes, contamination=contamination,
            guard=guard, io_batch_size=io_batch_size)
    if value_col is not None:
        raise ValueError("fit_s2g: value_col only used with df input")
    return _fit_s2g_series(
        df_or_series,
        window_size=window_size, query_window_size=query_window_size,
        n_nodes=n_nodes, contamination=contamination, guard=guard)


def _fit_s2g_series(series, *, window_size, query_window_size, n_nodes,
                     contamination, guard):
    return _fit_s2g_series_impl(
        series,
        window_size=window_size, query_window_size=query_window_size,
        n_nodes=n_nodes, contamination=contamination, guard=guard)


def _fit_s2g_df(df, value_col, *, window_size, query_window_size, n_nodes,
                 contamination, guard, io_batch_size=None):
    mg = guard if guard is not None else getattr(df, "_memory_guard", None)
    parts = []
    for b in df._iter_filtered_batches(batch_size=io_batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="s2g.batch", extra_factor=2.0)
        x_b = b.column(value_col).cast(
            pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        if x_b.size > 0:
            parts.append(x_b)
    if not parts:
        raise ValueError(f"fit_s2g: column {value_col!r} is empty")
    series = np.concatenate(parts) if len(parts) > 1 else parts[0]
    return _fit_s2g_series(
        series,
        window_size=window_size, query_window_size=query_window_size,
        n_nodes=n_nodes, contamination=contamination, guard=mg)


def _fit_s2g_series_impl(
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

    # v0.9.87-P4c (audit HIGH-1 full fix): streaming embed + nodes;
    # dedupe _score_queries (was: recomputed pattern_nodes from x).
    # Peak RAM = O(chunk_nw*(w+3*d_z)) + output pattern_nodes(n_w,)
    # + points_2d(n_w,2) + W(n_nodes^2) — none of which is O(n*w).
    _n_w = n - window_size + 1
    _d_z = window_size - lam + 1
    _chunk_nw = _resolve_chunk_nw_s2g(window_size, _d_z, guard)

    if guard is not None:
        # input x (n*8) + per-chunk buffers + output (nodes 8*n_w +
        # points 16*n_w) + W/deg (n_nodes^2*8)
        per_chunk = _chunk_nw * (window_size + 3 * _d_z + 6) * 8
        output = _n_w * 24
        W_bounded = n_nodes * n_nodes * 8
        est_bytes = int(n * 8 + per_chunk + output + W_bounded)
        if hasattr(guard, "check_state") and callable(guard.check_state):
            guard.check_state(est_bytes, name="s2g.state")
        else:
            limit_mb = _probe_memory_limit_mb(guard)
            if limit_mb is not None and limit_mb > 0:
                if est_bytes > limit_mb * 1024 * 1024:
                    from cadw.core.memory_guard import MemoryLimitExceeded
                    raise MemoryLimitExceeded(
                        f"S2G estimated {est_bytes} bytes > "
                        f"guard {limit_mb * 1024 * 1024:.0f} bytes")

    # Streaming embed: (n,) -> pattern_nodes(n_w,), points_2d(n_w,2)
    pattern_nodes, points_2d = _stream_embed_to_nodes(
        x, window_size, lam, n_nodes, _chunk_nw)

    # Graph (bounded by n_nodes^2)
    W = _build_graph(pattern_nodes, n_nodes)
    deg = _compute_degrees(W)

    # Query scoring (uses pre-computed pattern_nodes — no recompute)
    query_scores, query_starts = _score_queries(
        pattern_nodes, n,
        window_size=window_size, query_window_size=query_window_size,
        W=W, deg=deg)

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

