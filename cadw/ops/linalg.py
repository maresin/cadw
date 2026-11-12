"""Linear algebra primitives (v0.9.134 DEC-054).

Closed-form regressions + pairwise distances. No Arrow / no I/O;
numpy only.

History:
- v0.9.134: created. Moved ols_fit from cadw.ops.numeric; extracted
  sq_dists from cadw.ml.patch_memory_ad (forward extraction, AP10
  exception per DEC-056: 12+ prospective consumers — kmeans, kmeans_ad,
  kshape, tspq, iforest, eif, DBSCAN, HDBSCAN, LOF, bisecting,
  minibatch, spectral).
- v0.9.143: added nmf_mu (Lee & Seung 1999 multiplicative update).
  Promoted from internal audit prototype; consumers NMF x2 + Top2Vec
  + LDA_embedding (3 -> P1 per PRIMITIVES_BACKLOG L7).
- v0.9.144: added knn_query (L1) + core_distances (L2).
  Chunked query against streamed X; state O(n_q * k).
  knn_query: 8 consumers, core_distances: 3 consumers (P1).
- v0.9.145: added pairwise_kernel (L3: rbf/poly/laplacian,
  parameterized) + chol_solve (L4). L3: 10 cards, L4: 5 cards.
- v0.9.146: added eigh_topk (L19). 8 consumers (GPR, Spectral x5,
  KernelPCA, GPLVM). n<=2000 dense (numpy.eigh), else scipy eigsh.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def ols_fit(y: np.ndarray) -> Tuple[float, float]:
    """OLS (intercept, slope) on y[t] vs t.

    Returns (a, b) such that y_hat[t] = a + b*t. Handles n < 2
    (returns mean, 0) and degenerate x variance (den < 1e-12 -> mean, 0).
    """
    n = y.size
    if n < 2:
        return float(y.mean()) if n else 0.0, 0.0
    x = np.arange(n, dtype=np.float64)
    xm = float(x.mean())
    ym = float(y.mean())
    num = float(np.sum((x - xm) * (y - ym)))
    den = float(np.sum((x - xm) ** 2))
    if den < 1e-12:
        return ym, 0.0
    b = num / den
    return ym - b * xm, b


def ols_fit_streaming(y_iter):
    """v0.9.188: streaming OLS (intercept, slope) on y[t] vs t.

    Bit-identical (atol 1e-9) to ``ols_fit(y)`` for full y.
    O(1) RAM: accumulate sum_y, sum_iy, n. Closed-form:
      xm = (n-1)/2, ym = sum_y/n
      num = sum_iy - xm*sum_y               (since sum(i*y_i) - xm*sum_y)
      den = n*(n*n - 1)/12                  (sum((i - xm)^2))
      b = num/den, a = ym - b*xm

    Parameters
    ----------
    y_iter : iterable of 1-D np.ndarray (batches, ordered)

    Returns
    -------
    (a, b, n) — intercept, slope, count.
    """
    sum_y = 0.0
    sum_iy = 0.0
    n_total = 0
    for b in y_iter:
        if b.size == 0:
            continue
        # sum(y)
        sum_y += float(np.sum(b))
        # sum(i * y_i) for global index i = n_total + local_j
        # = n_total * sum(b) + sum(j * b_j) for j in [0, b.size)
        local_sum = float(np.sum(b))
        j = np.arange(b.size, dtype=np.float64)
        local_iy = float(np.sum(j * b))
        sum_iy += n_total * local_sum + local_iy
        n_total += b.size

    if n_total < 2:
        # Degenerate: n=0 -> (0,0,0); n=1 -> (mean, 0, 1)
        return (sum_y / max(1, n_total) if n_total else 0.0, 0.0, n_total)

    xm = (n_total - 1) / 2.0
    ym = sum_y / n_total
    den = n_total * (n_total * n_total - 1) / 12.0
    if den < 1e-12:
        return ym, 0.0, n_total
    num = sum_iy - xm * sum_y
    b = num / den
    a = ym - b * xm
    return float(a), float(b), n_total


def sq_dists(z: np.ndarray, mem: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of z and rows of mem.

    Returns (n_z, n_mem). Identity:
      ||z_i - mem_k||^2 = z_i . z_i - 2 z_i . mem_k + mem_k . mem_k
    """
    z_sq = (z * z).sum(axis=1)
    m_sq = (mem * mem).sum(axis=1)
    return z_sq[:, None] - 2.0 * (z @ mem.T) + m_sq[None, :]


def nan_sq_dists(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Squared nan-euclidean distances (partial-distance kNN base).

    v0.9.199. Matches sklearn.metrics.pairwise.nan_euclidean_distances(
    squared=True) bit-for-bit (P8b verified max diff 0.0).

    Formula per pair (q, x):
      sum_common((q_i - x_i)^2) * d / n_common
    where `common` = dims where BOTH q and x non-NaN.
    Returns NaN when n_common == 0 (no donor — pair skipped).

    Args:
      Q: (n_q, d) float64 (may contain NaN).
      X: (n_x, d) float64 (may contain NaN); caller chunks n_x for OOC.

    Returns (n_q, n_x) float64.

    Note: computes full (n_q, n_x, d) broadcast internally — caller must
    chunk X to bound peak to O(n_q * chunk * d * 16) bytes.
    """
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    if Q.ndim != 2 or X.ndim != 2:
        raise ValueError(
            f"nan_sq_dists: Q, X must be 2-D; got {Q.shape}, {X.shape}")
    if Q.shape[1] != X.shape[1]:
        raise ValueError(
            f"nan_sq_dists: Q.d={Q.shape[1]} != X.d={X.shape[1]}")
    n_q, d = Q.shape
    n_x = X.shape[0]
    if n_q == 0 or n_x == 0:
        return np.empty((n_q, n_x), dtype=np.float64)

    # Broadcasted squared diff with NaN mask per pair
    q_nan = np.isnan(Q)                            # (n_q, d)
    x_nan = np.isnan(X)                            # (n_x, d)
    diff = Q[:, None, :] - X[None, :, :]           # (n_q, n_x, d)
    common = ~(q_nan[:, None, :] | x_nan[None, :, :])   # (n_q, n_x, d)
    sum_sq = np.where(common, diff * diff, 0.0).sum(axis=-1)  # (n_q, n_x)
    n_common = common.sum(axis=-1)                 # (n_q, n_x)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(n_common > 0,
                         float(d) / np.maximum(n_common, 1),
                         np.nan)
        return sum_sq * scale



def svd_topk(
    df,
    X_cols,
    k: int,
    *,
    compute_u: bool = False,
    batch_size=None,
    memory_guard=None,
):
    """Top-k SVD via chunked Gram Reduce (batch-layer, OOC in n_rows).

    v0.9.205 (ROADMAP §OOC by primitive pattern #1):
      Pass 1: G += X_batch.T @ X_batch (Reduce, state O(d^2))
      eigh_topk(G, k, which="LA") -> V (d, k), sigma = sqrt(clip(lambda, 0))
      svd_flip(V.T, u_based_decision=False) for sklearn TruncatedSVD parity
      Pass 2 (if compute_u): U_batch = X_batch @ V / sigma -> SpillArray.

    Args:
      df: CadwDataFrame with X_cols numeric (no NaN allowed).
      X_cols: list[str] (d columns).
      k: number of components (1 <= k <= min(n, d)).
      compute_u: if True, compute U (n, k) as AutoArrayResult (SpillArray,
        OOC in n); if False (default), U=None (fit_transform computes
        X @ V / sigma or X @ V directly as needed).
      batch_size: rows per batch; auto from guard.
      memory_guard: optional MemoryGuard.

    Returns:
      (U, sigma, V):
        U: (n_rows, k) AutoArrayResult OR None
        sigma: (k,) f64 — descending, clipped to >= 0
        V: (d, k) f64 — descending columns (columns_ = V.T)

    Sklearn TruncatedSVD(algorithm="arpack", tol=0) parity:
      singular_values_ == sigma (atol 1e-9)
      components_ == V.T after svd_flip (atol 1e-9)

    OOC in n_rows: state = d*d*f64*2 (G + eigh working copy) + k*f64*2 + 256;
    independent of n. bounded_by_domain {d: 1e4} (d^2*16 = 1.6 GB at d=1e4);
    guard refuses above. For d>1e4 use sparse path (not this primitive).
    """
    import pyarrow as pa

    if not X_cols:
        raise ValueError("svd_topk: X_cols is empty")
    if k < 1:
        raise ValueError(f"svd_topk: k must be >= 1; got {k}")
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"svd_topk: columns missing: {missing}; available: {df.columns}")

    mg = memory_guard or getattr(df, "_memory_guard", None)
    n_total = int(df.count())
    d = len(X_cols)
    if n_total < 1:
        raise ValueError(f"svd_topk: n must be >= 1; got {n_total}")
    if k > min(n_total, d):
        raise ValueError(
            f"svd_topk: k={k} > min(n={n_total}, d={d})")

    if batch_size is None:
        batch_size = max(1, min(n_total, 1000))
    batch_size = max(1, min(int(batch_size), n_total))

    # State guard: G (d, d) + eigh copy + sigma + V
    state_bytes = d * d * 8 * 2 + k * 8 * 2 + 256
    if mg is not None and hasattr(mg, "check_state"):
        mg.check_state(state_bytes, name="svd_topk.state")

    # ---- Pass 1: G = X^T X (chunked Reduce) ----
    G = np.zeros((d, d), dtype=np.float64)
    n_processed = 0
    for b in df._iter_filtered_batches(batch_size=batch_size):
        if mg is not None and hasattr(mg, "check_batch"):
            mg.check_batch(b, name="svd_topk.X", extra_factor=2.0)
        X_b = _batch_to_X_svd(b, X_cols)
        if X_b.shape[0] == 0:
            continue
        if np.isnan(X_b).any():
            raise ValueError(
                "svd_topk: X contains NaN; impute first (fit_knn_imputer)")
        G += X_b.T @ X_b
        n_processed += X_b.shape[0]
    if n_processed != n_total:
        raise RuntimeError(
            f"svd_topk: X rows mismatch ({n_processed} != {n_total})")

    # ---- eigh_topk on G ----
    # LA = largest algebraic; eigenvalues of X^T X are sigma^2 (>=0)
    lam, V = eigh_topk(G, k, which="LA", memory_guard=mg)
    # Clip tiny negatives (numerical)
    lam = np.maximum(lam, 0.0)
    sigma = np.sqrt(lam)

    # svd_flip parity with sklearn (u_based_decision=False -> flip based on VT)
    # Vt = V.T (k, d); per-row (component) max abs; sign-flip
    Vt = V.T.copy()
    max_abs_rows = np.argmax(np.abs(Vt), axis=1)
    signs = np.sign(Vt[np.arange(Vt.shape[0]), max_abs_rows])
    signs[signs == 0] = 1.0
    Vt = Vt * signs[:, np.newaxis]
    V = Vt.T  # back to (d, k)

    # ---- Pass 2 (optional): U = X V / sigma ----
    U = None
    if compute_u:
        from cadw.core.spill import SpillArray
        U_sp = SpillArray(
            guard=mg, width=k, dtype=np.float64,
            name="svd_topk.U", spill_ratio=0.5)
        sigma_safe = np.maximum(sigma, 1e-12)
        for b in df._iter_filtered_batches(batch_size=batch_size):
            if mg is not None and hasattr(mg, "check_batch"):
                mg.check_batch(b, name="svd_topk.U", extra_factor=3.0)
            X_b = _batch_to_X_svd(b, X_cols)
            if X_b.shape[0] == 0:
                continue
            U_b = (X_b @ V) / sigma_safe[None, :]
            U_sp.write_array(U_b)
        U = U_sp.finalize()

    return U, sigma, V


def _batch_to_X_svd(b, X_cols):
    """Extract (b, d) f64 from pa.RecordBatch (NaN preserved)."""
    import pyarrow as pa
    arrs = []
    for c in X_cols:
        col = b.column(c)
        if not pa.types.is_floating(col.type):
            col = col.cast(pa.float64())
        arrs.append(col.to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False))
    return np.column_stack(arrs) if arrs else np.empty((b.num_rows, 0))



def nmf_mu(X_b: np.ndarray, H: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    """One mini-batch Lee & Seung mu update of H.

    W_b = NNLS projection of batch X_b onto row space of H (closed-form,
    clipped to >= 0). Then H <- H * (W_b^T X_b) / (W_b^T W_b H + eps).

    Inputs:
      X_b: dense non-negative batch, shape (b, V)
      H:   non-negative components, shape (k, V)
    Returns:
      H_new, same shape as H, non-negative.

    Reference: Lee & Seung (1999/2001) multiplicative updates; per-batch
    closed-form W is the "NNLS W step" adaptation for streaming NMF.
    """
    k = H.shape[0]
    HtH = H @ H.T + eps * np.eye(k, dtype=H.dtype)
    W_b = np.maximum(X_b @ H.T @ np.linalg.inv(HtH), 0.0)
    WtX = W_b.T @ X_b
    WtW = W_b.T @ W_b
    denom = WtW @ H + eps
    return H * (WtX / denom)


# ============================================================================
# kNN primitives (v0.9.144, P1 L1 + L2 per PRIMITIVES_BACKLOG)
# ============================================================================
#
# Patterns:
#   knn_query      — chunked X streaming + top-k merge. State O(n_q * k).
#   core_distances — k-th NN distance per row (self-excluded).
#                    Delegates to knn_query(X, X, k+1)[:, k].
#
# Both OOC in n_X (number of candidate rows); n_q (query rows) must fit
# state limit (bounded_state_dims: {n_q: 1, k: 1}).
# ============================================================================

_KNN_STATE_HEADER = 256


def _resolve_knn_chunk_size(n_q: int, d: int, memory_guard=None) -> int:
    """Pick chunk_size so that n_q * chunk * f64 fits comfortable budget."""
    target_bytes = 8 * 1024 * 1024  # 8 MB default per chunk
    if memory_guard is not None:
        try:
            avail = memory_guard.available_bytes
            target_bytes = max(64 * 1024, min(target_bytes, avail // 4))
        except Exception:
            pass
    per_row = 8 * d  # one X_c row
    return max(1, int(target_bytes // max(per_row, 1)))


def knn_query(
    X: np.ndarray,
    Q: np.ndarray,
    k: int,
    *,
    return_dist: bool = False,
    metric: str = "euclidean",
    chunk_size: int | None = None,
    memory_guard=None,
):
    """k nearest neighbours of Q in X (streaming over X chunks).

    Args:
      X: (n_x, d) float64 candidates.
      Q: (n_q, d) float64 queries.
      k: number of neighbours (1 <= k <= n_x).
      return_dist: if True, also return distances (n_q, k).
      metric: "euclidean" (default) | "cosine".
      chunk_size: candidate rows per chunk; auto-derived from guard.
      memory_guard: optional MemoryGuard (state + chunk checks).

    Returns:
      idx: (n_q, k) int64 — indices into X (0-based).
      dist: (n_q, k) float64 — distance (euclidean: L2; cosine: 1-cos).
            Only returned if return_dist=True.

    State: O(n_q * k) for top-k heap; chunk transient O(n_q * chunk_size).
    OOC in n_x: peak independent of n_x. Refuses at guard if state or
    chunk buffer exceeds budget.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    if X.ndim != 2 or Q.ndim != 2:
        raise ValueError(f"X, Q must be 2-D; got {X.shape}, {Q.shape}")
    if X.shape[1] != Q.shape[1]:
        raise ValueError(
            f"X.shape[1]={X.shape[1]} != Q.shape[1]={Q.shape[1]}")
    if metric not in ("euclidean", "cosine"):
        raise ValueError(
            f"metric must be 'euclidean'|'cosine'; got {metric!r}")
    n_x, d = X.shape
    n_q = Q.shape[0]
    if n_x == 0:
        raise ValueError("knn_query: X is empty")
    if k < 1 or k > n_x:
        raise ValueError(f"k must be in [1, n_x={n_x}]; got {k}")

    # state guard
    state_bytes = n_q * k * 8 * 2 + _KNN_STATE_HEADER
    if memory_guard is not None:
        memory_guard.check_state(state_bytes, name="knn_query.state")

    if chunk_size is None:
        chunk_size = _resolve_knn_chunk_size(n_q, d, memory_guard)
    chunk_size = max(1, int(chunk_size))

    # metric preprocessing (norms for cosine)
    if metric == "cosine":
        Q_norm = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)

    top_d = np.full((n_q, k), np.inf, dtype=np.float64)
    top_i = np.full((n_q, k), -1, dtype=np.int64)

    for start in range(0, n_x, chunk_size):
        end = min(start + chunk_size, n_x)
        Xc = X[start:end]
        chunk_bytes = n_q * (end - start) * 8
        if memory_guard is not None:
            memory_guard.check_state(
                chunk_bytes, name="knn_query.chunk")

        if metric == "euclidean":
            D = sq_dists(Q, Xc)  # (n_q, chunk)
        else:  # cosine
            Xc_norm = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
            D = 1.0 - (Q_norm @ Xc_norm.T)  # (n_q, chunk) in [0, 2]

        # merge chunk with current top-k
        merged_d = np.concatenate([top_d, D], axis=1)
        chunk_ids = np.broadcast_to(
            np.arange(start, end, dtype=np.int64),
            (n_q, end - start))
        merged_i = np.concatenate([top_i, chunk_ids], axis=1)

        # argpartition + stable sort of the k survivors
        part = np.argpartition(merged_d, k - 1, axis=1)[:, :k]
        part_d = np.take_along_axis(merged_d, part, axis=1)
        part_i = np.take_along_axis(merged_i, part, axis=1)
        order = np.argsort(part_d, axis=1, kind="stable")
        top_d = np.take_along_axis(part_d, order, axis=1)
        top_i = np.take_along_axis(part_i, order, axis=1)

    if not return_dist:
        return top_i
    if metric == "euclidean":
        return top_i, np.sqrt(np.maximum(top_d, 0.0))
    return top_i, top_d


def knn_query_stream(
    X_iter,
    Q: np.ndarray,
    k: int,
    *,
    return_dist: bool = False,
    metric: str = "euclidean",
    chunk_size: int | None = None,
    memory_guard=None,
):
    """k nearest neighbours of Q in X (X streamed from an iterator).

    v0.9.197: streaming X variant of knn_query. Additive — knn_query
    stays (X in ndarray for caller-controlled batch). Use this when X
    does not fit in RAM: X from memmap/SpillArray iterator.

    Args:
      X_iter: re-iterable iterator over batches. Each batch is
              pa.RecordBatch (all columns float64) OR (b, d) np.ndarray.
              Re-iterable required (each call iterates from start).
      Q: (n_q, d) float64 queries; must fit in RAM (typically n_q=bat).
      k: number of neighbours (1 <= k <= n_x; n_x checked at end).
      return_dist: also return distances (n_q, k).
      metric: "euclidean" (default) | "cosine".
      chunk_size: candidate rows per X batch (if RecordBatch yields
                  small batches, first batch's rows hint). Auto from guard.
      memory_guard: optional MemoryGuard.

    Returns:
      idx: (n_q, k) int64 — indices into X (global, 0-based, in the
           order X_iter yields rows).
      dist: (n_q, k) float64 — only if return_dist=True.

    State: O(n_q * k) top-k heap (same as knn_query).
    OOC in n_x: X is not materialized; per-batch transient
    O(n_q * chunk) same as knn_query.
    """
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    if Q.ndim != 2:
        raise ValueError(f"knn_query_stream: Q must be 2-D; got {Q.shape}")
    if metric not in ("euclidean", "cosine"):
        raise ValueError(
            f"knn_query_stream: metric must be 'euclidean'|'cosine'; "
            f"got {metric!r}")
    if k < 1:
        raise ValueError(f"knn_query_stream: k must be >= 1; got {k}")

    n_q, d = Q.shape

    state_bytes = n_q * k * 8 * 2 + _KNN_STATE_HEADER
    if memory_guard is not None:
        memory_guard.check_state(
            state_bytes, name="knn_query_stream.state")

    if chunk_size is None:
        chunk_size = _resolve_knn_chunk_size(n_q, d, memory_guard)
    chunk_size = max(1, int(chunk_size))

    if metric == "cosine":
        Q_norm = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)

    top_d = np.full((n_q, k), np.inf, dtype=np.float64)
    top_i = np.full((n_q, k), -1, dtype=np.int64)

    global_offset = 0
    for batch in X_iter:
        # Accept pa.RecordBatch OR np.ndarray
        if hasattr(batch, "num_columns"):  # pa.RecordBatch
            cols = [np.asarray(batch.column(i), dtype=np.float64)
                    for i in range(batch.num_columns)]
            Xc = np.column_stack(cols) if cols else np.empty((0, d))
        else:
            Xc = np.ascontiguousarray(batch, dtype=np.float64)
        if Xc.size == 0:
            continue
        if Xc.shape[1] != d:
            raise ValueError(
                f"knn_query_stream: X batch has d={Xc.shape[1]}, "
                f"Q has d={d}")
        end = global_offset + Xc.shape[0]

        chunk_bytes = n_q * Xc.shape[0] * 8
        if memory_guard is not None:
            memory_guard.check_state(
                chunk_bytes, name="knn_query_stream.chunk")

        if metric == "euclidean":
            D = sq_dists(Q, Xc)  # (n_q, b)
        else:
            Xc_norm = Xc / (
                np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
            D = 1.0 - (Q_norm @ Xc_norm.T)  # (n_q, b) in [0, 2]

        merged_d = np.concatenate([top_d, D], axis=1)
        chunk_ids = np.broadcast_to(
            np.arange(global_offset, end, dtype=np.int64),
            (n_q, Xc.shape[0]))
        merged_i = np.concatenate([top_i, chunk_ids], axis=1)

        part = np.argpartition(merged_d, k - 1, axis=1)[:, :k]
        part_d = np.take_along_axis(merged_d, part, axis=1)
        part_i = np.take_along_axis(merged_i, part, axis=1)
        order = np.argsort(part_d, axis=1, kind="stable")
        top_d = np.take_along_axis(part_d, order, axis=1)
        top_i = np.take_along_axis(part_i, order, axis=1)

        global_offset = end

    n_x = global_offset
    if n_x == 0:
        raise ValueError("knn_query_stream: X_iter yielded 0 rows")
    if k > n_x:
        raise ValueError(f"k must be in [1, n_x={n_x}]; got {k}")

    if not return_dist:
        return top_i
    if metric == "euclidean":
        return top_i, np.sqrt(np.maximum(top_d, 0.0))
    return top_i, top_d


def core_distances(
    X: np.ndarray,
    k: int,
    *,
    chunk_size: int | None = None,
    memory_guard=None,
) -> np.ndarray:
    """k-th nearest-neighbour distance per row of X (self-excluded).

    Args:
      X: (n, d) float64.
      k: 1 <= k < n.
      chunk_size, memory_guard: forwarded to knn_query.

    Returns:
      (n,) float64 — ||x_i - x_(i,k+1)||, where x_(i,j) is j-th nearest
      neighbour of x_i (x_(i,1) = self with distance 0).

    OOC in n for RAM: state = n * (k+1) * f64 * 2 + 256.
    Compute is O(n^2) — intrinsic to kNN self-query; callers should
    subsample (HDBSCAN typically n<=10^4 per batch) or bound n.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got {X.shape}")
    n, d = X.shape
    if k < 1 or k >= n:
        raise ValueError(f"k must be in [1, n-1]; got k={k}, n={n}")
    # k+1 nearest: self is at position 0, target at index k
    _, dist = knn_query(
        X, X, k + 1, return_dist=True, metric="euclidean",
        chunk_size=chunk_size, memory_guard=memory_guard,
    )
    return dist[:, k].copy()


# ============================================================================
# Kernel + Cholesky primitives (v0.9.145, P1 L3 + L4)
# ============================================================================
#
# pairwise_kernel — parameterized over {"rbf", "poly", "laplacian"}.
#   One function, not 3 wrappers (AP22: no per-variant duplication).
#   Chunked Y streaming; peak O(n_x * chunk_y), independent of n_y.
#
# chol_solve — Cholesky + triangular solve; falls back to np.linalg.solve
#   when Cholesky fails (kernel matrix not strictly PSD due to noise).
# ============================================================================

_SUPPORTED_KERNELS = ("linear", "rbf", "poly", "laplacian", "sigmoid", "cosine")
_KERNEL_STATE_HEADER = 256


def _resolve_kernel_chunk_size(
    n_x: int, d: int, memory_guard=None,
) -> int:
    """Pick chunk_size in Y so n_x * chunk * f64 fits budget."""
    target_bytes = 8 * 1024 * 1024
    if memory_guard is not None:
        try:
            avail = memory_guard.available_bytes
            target_bytes = max(64 * 1024, min(target_bytes, avail // 4))
        except Exception:
            pass
    per_row = 8 * max(n_x, 1)
    return max(1, int(target_bytes // per_row))


def pairwise_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "rbf",
    *,
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
    chunk_size: int | None = None,
    memory_guard=None,
) -> np.ndarray:
    """Pairwise kernel matrix K(X, Y) of shape (n_x, n_y).

    Args:
      X: (n_x, d) float64.
      Y: (n_y, d) float64.
      kernel: "linear" | "rbf" | "poly" | "laplacian" | "sigmoid" | "cosine".
        - linear:    K = <x, y>
        - rbf:       K = exp(-gamma * ||x-y||^2)
        - poly:      K = (gamma * <x,y> + coef0) ** degree
        - laplacian: K = exp(-gamma * ||x-y||_1)
        - sigmoid:   K = tanh(gamma * <x,y> + coef0)
        - cosine:    K = <x,y> / (||x|| * ||y||)
      gamma, degree, coef0: kernel hyperparameters.
      chunk_size: Y rows per chunk (auto from guard if None).
      memory_guard: optional MemoryGuard.

    Returns:
      K: (n_x, n_y) float64.

    OOC in n_y: peak = O(n_x * chunk_y), independent of n_y.
    K matrix itself is (n_x, n_y) — caller responsible for that size.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"X, Y must be 2-D; got {X.shape}, {Y.shape}")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X.shape[1]={X.shape[1]} != Y.shape[1]={Y.shape[1]}")
    if kernel not in _SUPPORTED_KERNELS:
        raise ValueError(
            f"kernel must be one of {_SUPPORTED_KERNELS}; got {kernel!r}")
    if kernel not in ("linear", "cosine") and gamma <= 0:
        raise ValueError(f"gamma must be > 0; got {gamma}")
    if kernel == "poly":
        if degree < 1:
            raise ValueError(f"degree must be >= 1; got {degree}")

    n_x, d = X.shape
    n_y = Y.shape[0]
    if memory_guard is not None:
        memory_guard.check_state(
            n_x * n_y * 8 + _KERNEL_STATE_HEADER,
            name="pairwise_kernel.output")

    if chunk_size is None:
        chunk_size = _resolve_kernel_chunk_size(n_x, d, memory_guard)
    chunk_size = max(1, int(chunk_size))

    # cosine: L2-normalize X once (Yc normalized per chunk)
    X_norm = None
    if kernel == "cosine":
        Xn = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / np.maximum(Xn, 1e-12)

    K = np.empty((n_x, n_y), dtype=np.float64)
    for start in range(0, n_y, chunk_size):
        end = min(start + chunk_size, n_y)
        Yc = Y[start:end]
        if memory_guard is not None:
            memory_guard.check_state(
                n_x * (end - start) * 8, name="pairwise_kernel.chunk")

        if kernel == "linear":
            Kc = X @ Yc.T
        elif kernel == "rbf":
            D2 = sq_dists(X, Yc)
            Kc = np.exp(-gamma * np.maximum(D2, 0.0))
        elif kernel == "poly":
            Kc = (gamma * (X @ Yc.T) + coef0) ** degree
        elif kernel == "sigmoid":
            Kc = np.tanh(gamma * (X @ Yc.T) + coef0)
        elif kernel == "cosine":
            Yn = np.linalg.norm(Yc, axis=1, keepdims=True)
            Ycn = Yc / np.maximum(Yn, 1e-12)
            Kc = X_norm @ Ycn.T
        else:  # laplacian
            diff = X[:, None, :] - Yc[None, :, :]
            Kc = np.exp(-gamma * np.abs(diff).sum(axis=2))
        K[:, start:end] = Kc
    return K


def rbf_kernel(X, Y, gamma=1.0, **kw):
    """Convenience wrapper: pairwise_kernel(X, Y, 'rbf', gamma=gamma)."""
    return pairwise_kernel(X, Y, "rbf", gamma=gamma, **kw)


def poly_kernel(X, Y, gamma=1.0, degree=3, coef0=1.0, **kw):
    """Convenience wrapper: pairwise_kernel(X, Y, 'poly', ...)."""
    return pairwise_kernel(
        X, Y, "poly", gamma=gamma, degree=degree, coef0=coef0, **kw)


def laplacian_kernel(X, Y, gamma=1.0, **kw):
    """Convenience wrapper: pairwise_kernel(X, Y, 'laplacian', gamma=gamma)."""
    return pairwise_kernel(X, Y, "laplacian", gamma=gamma, **kw)


def chol_solve(
    K: np.ndarray,
    b: np.ndarray,
    *,
    jitter: float = 1e-8,
    max_attempts: int = 3,
) -> np.ndarray:
    """Solve Kx = b via Cholesky with progressive jitter fallback.

    Args:
      K: (n, n) symmetric matrix (kernel Gram, typically PSD).
      b: (n,) or (n, m) right-hand side.
      jitter: initial jitter added to diagonal (default 1e-8).
      max_attempts: number of Cholesky retries with jitter *= 10.
                    After max_attempts, falls back to np.linalg.solve
                    (no jitter; works for any non-singular K).

    Returns:
      x: (n,) or (n, m) solution.

    Raises:
      np.linalg.LinAlgError: if both Cholesky (all attempts) and solve fail.

    Memory: O(n^2) intermediate (L factor). No OOC — caller manages K size.
    """
    K = np.ascontiguousarray(K, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square 2-D; got {K.shape}")
    if b.shape[0] != K.shape[0]:
        raise ValueError(
            f"b.shape[0]={b.shape[0]} != K.shape[0]={K.shape[0]}")
    n = K.shape[0]

    current_jitter = float(jitter)
    for attempt in range(max_attempts + 1):
        try:
            K_jit = K if attempt == 0 else K + current_jitter * np.eye(n)
            L = np.linalg.cholesky(K_jit)
            # Solve L L^T x = b via forward + back substitution
            y = np.linalg.solve(L, b)
            x = np.linalg.solve(L.T, y)
            return x
        except np.linalg.LinAlgError:
            current_jitter *= 10.0
            continue

    # Cholesky exhausted; fall back to general solve
    return np.linalg.solve(K, b)


# ============================================================================
# Eigen decomposition (v0.9.146, P1 L19)
# ============================================================================

_EIGH_DENSE_THRESHOLD = 2000  # n <= 2000 -> dense numpy; else scipy eigsh
_EIGH_STATE_HEADER = 256


def eigh_topk(
    M: np.ndarray,
    k: int,
    *,
    which: str = "LA",
    sigma: "float | None" = None,
    memory_guard=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k eigenvectors of symmetric M.

    Args:
      M: (n, n) symmetric float64 (or convertible).
      k: number of eigenvectors.
      which:
        "LA" — largest algebraic (PCA, consensus; eigvals descending)
        "SA" — smallest algebraic (spectral on L=D-W; ascending)
        "LM" — largest magnitude
        "SM" — smallest magnitude (spectral on L_sym; |λ| ascending)
      memory_guard: optional; checks state BEFORE dense alloc.

    Returns:
      (eigvals (k,) float64, eigvecs (n, k) float64)

    Method:
      n <= 2000: numpy.linalg.eigh (full, O(n^3), ~n^2 RAM)
      n >  2000: scipy.sparse.linalg.eigsh (Lanczos, O(n*k) RAM);
                 "SM" uses shift-invert (sigma=0) for reliability.

    OOC in n: NO for dense path (n^2); partial for scipy (n*k).
    Refuses at guard if dense state > budget. For large n, callers
    should subsample or use scipy path directly with their own adjacency.
    """
    M = np.ascontiguousarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be square 2-D; got {M.shape}")
    n = M.shape[0]
    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, n={n}]; got {k}")
    if which not in ("LA", "SA", "LM", "SM"):
        raise ValueError(
            f"which must be in {{LA,SA,LM,SM}}; got {which!r}")

    if n <= _EIGH_DENSE_THRESHOLD:
        state_bytes = n * n * 8 + _EIGH_STATE_HEADER
    else:
        state_bytes = n * k * 8 * 2 + _EIGH_STATE_HEADER
    if memory_guard is not None:
        memory_guard.check_state(state_bytes, name="eigh_topk.state")

    if n <= _EIGH_DENSE_THRESHOLD:
        eigvals, eigvecs = np.linalg.eigh(M)  # ascending by value
        if which == "LA":
            order = np.argsort(eigvals)[-k:][::-1]
        elif which == "SA":
            order = np.argsort(eigvals)[:k]
        elif which == "LM":
            order = np.argsort(np.abs(eigvals))[-k:][::-1]
        else:  # SM
            order = np.argsort(np.abs(eigvals))[:k]
        return eigvals[order].copy(), eigvecs[:, order].copy()

    # Large n: scipy.sparse.linalg.eigsh
    try:
        from scipy.sparse.linalg import eigsh
    except ImportError as e:
        raise ImportError(
            f"eigh_topk: n={n} > {_EIGH_DENSE_THRESHOLD} requires scipy "
            f"({e}); install scipy or subsample"
        ) from e

    if which in ("LA", "LM"):
        vals, vecs = eigsh(M, k=k, which=which)
    elif which == "SA":
        vals, vecs = eigsh(M, k=k, which="SA")
    else:  # SM: shift-invert for spectral clustering reliability
        # v0.9.203: sigma param (None -> 0.0 legacy; sklearn uses -1e-5)
        _sig = 0.0 if sigma is None else float(sigma)
        vals, vecs = eigsh(M, k=k, sigma=_sig, which="LM")

    if which == "LA":
        order = np.argsort(vals)[-k:][::-1]
    elif which == "SA":
        order = np.argsort(vals)[:k]
    elif which == "LM":
        order = np.argsort(np.abs(vals))[-k:][::-1]
    else:  # SM
        order = np.argsort(np.abs(vals))[:k]
    return vals[order].copy(), vecs[:, order].copy()
