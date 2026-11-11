"""Graph primitives (v0.9.147 → v0.9.203).

- graph_laplacian:        dense L = D-W | I - D^-1/2 W D^-1/2 | I - D^-1 W
- graph_laplacian_sparse: sparse variant (v0.9.203, sklearn
                          SpectralClustering affinity is scipy.sparse)
- eigsh_topk:             top-k eigenvectors; accepts dense or sparse L
                          (sparse dispatch -> scipy.sparse.linalg.eigsh)
- mst:                    minimum spanning tree via scipy.sparse.csgraph
"""
from __future__ import annotations

import warnings

import numpy as np
from typing import Optional


_GRAPH_HEADER = 256


def graph_laplacian(
    W: np.ndarray,
    *,
    normalized: str = "none",
    memory_guard=None,
) -> np.ndarray:
    """Compute graph Laplacian L from adjacency W.

    Args:
      W: (n, n) symmetric non-negative adjacency (or similarity) matrix.
      normalized:
        "none" — L = D - W (unnormalized; eigenvalues in [0, 2*d_max])
        "sym"  — L_sym = I - D^-1/2 W D^-1/2 (symmetric normalized)
        "rw"   — L_rw = I - D^-1 W (random-walk normalized; non-symmetric)

    Returns:
      L: (n, n) float64.

    OOC: no. Peak = O(n^2) for the output plus temporaries.
    Caller must subsample for large n (sparse support v0.9.148+).
    """
    W = np.ascontiguousarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square 2-D; got {W.shape}")
    if normalized not in ("none", "sym", "rw"):
        raise ValueError(
            f"normalized must be 'none'|'sym'|'rw'; got {normalized!r}")
    n = W.shape[0]

    if memory_guard is not None:
        # output + one temporary for the normalization diagonal
        memory_guard.check_state(
            n * n * 8 * 2 + _GRAPH_HEADER, name="graph_laplacian.state")

    d = W.sum(axis=1)
    d_safe = np.maximum(d, 1e-12)

    if normalized == "none":
        # D - W: keep memory = n^2 + n (diag only added to diagonal)
        L = -W.copy()
        L[np.diag_indices(n)] += d
        return L
    elif normalized == "sym":
        d_inv_sqrt = 1.0 / np.sqrt(d_safe)
        L = np.eye(n) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
        return L
    else:  # rw
        d_inv = 1.0 / d_safe
        L = np.eye(n) - (d_inv[:, None] * W)
        return L


def graph_laplacian_sparse(
    W,
    *,
    normalized: str = "sym",
    memory_guard=None,
):
    """Sparse graph Laplacian L from sparse adjacency W.

    v0.9.203. Returns scipy.sparse — for sklearn SpectralClustering
    affinity (sparse, nnz=n*k for kNN). Bit-identical to
    graph_laplacian(W.toarray()) for dense-sparse-representable W
    (P7 verified: L_sym max diff 2.78e-17).

    Args:
      W: scipy.sparse (n, n) — symmetric non-negative adjacency.
      normalized: "none" | "sym" | "rw".
      memory_guard: optional.

    Returns:
      L: scipy.sparse (csr_matrix) same shape (n, n).
    """
    import scipy.sparse as sp
    if not sp.issparse(W):
        raise TypeError(
            f"graph_laplacian_sparse: W must be scipy.sparse; "
            f"got {type(W).__name__} (use graph_laplacian for dense)")
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square; got {W.shape}")
    if normalized not in ("none", "sym", "rw"):
        raise ValueError(
            f"normalized must be 'none'|'sym'|'rw'; got {normalized!r}")

    W = W.tocsr().astype(np.float64)
    n = W.shape[0]
    d = np.asarray(W.sum(axis=1)).ravel()
    d_safe = np.maximum(d, 1e-12)

    if normalized == "none":
        # L = D - W
        D = sp.diags(d)
        return (D - W).tocsr()
    elif normalized == "sym":
        # L_sym = I - D^-1/2 W D^-1/2
        d_inv_sqrt = sp.diags(1.0 / np.sqrt(d_safe))
        return (sp.eye(n, format="csr")
                - d_inv_sqrt @ W @ d_inv_sqrt).tocsr()
    else:  # rw
        # L_rw = I - D^-1 W
        d_inv = sp.diags(1.0 / d_safe)
        return (sp.eye(n, format="csr") - d_inv @ W).tocsr()



def eigsh_topk(
    L,
    k: int,
    *,
    which: str = "SM",
    sigma: Optional[float] = None,
    memory_guard=None,
):
    """Top-k eigenvectors of a graph Laplacian (dense or sparse L).

    v0.9.203: accepts scipy.sparse; dispatch -> scipy.sparse.linalg.eigsh.
    Dense path delegates to cadw.ops.linalg.eigh_topk.

    Args:
      L: (n, n) symmetric Laplacian — np.ndarray OR scipy.sparse.
      k: number of eigenvectors.
      which: "SM" | "SA" | "LA" | "LM".
      sigma: shift-invert parameter for `which="SM"`. None (default) ->
        0.0 (CADW legacy). sklearn.manifold._spectral_embedding uses
        sigma=-1e-5 (Kniazev). Pass -1e-5 for sklearn parity.
      memory_guard: forwarded.

    Returns:
      (eigvals, eigvecs) — (k,) f64 and (n, k) f64.

    Sparse path: scipy.sparse.linalg.eigsh, O(n*k) RAM.
    Dense path (n<=2000): numpy.linalg.eigh; n>2000 scipy eigsh.
    """
    import scipy.sparse as sp
    if sp.issparse(L):
        from scipy.sparse.linalg import eigsh
        L = L.tocsr().astype(np.float64)
        n = L.shape[0]
        if k < 1 or k > n:
            raise ValueError(f"k must be in [1, n={n}]; got {k}")
        if which not in ("SM", "SA", "LM", "LA"):
            raise ValueError(
                f"which must be in {{SM,SA,LM,LA}}; got {which!r}")
        if memory_guard is not None:
            memory_guard.check_state(
                n * k * 8 * 2 + _GRAPH_HEADER,
                name="eigsh_topk.sparse_state")
        sigma_val = 0.0 if sigma is None else float(sigma)
        if which == "SM":
            vals, vecs = eigsh(L, k=k, sigma=sigma_val, which="LM")
        else:
            vals, vecs = eigsh(L, k=k, which=which)
        if which in ("SM", "SA"):
            order = np.argsort(vals)[:k]
        else:
            order = np.argsort(np.abs(vals))[-k:][::-1]
        return vals[order].copy(), vecs[:, order].copy()

    # Dense path: delegate
    from cadw.ops.linalg import eigh_topk
    return eigh_topk(L, k, which=which, sigma=sigma,
                      memory_guard=memory_guard)


def mst(
    n: int,
    edges: np.ndarray,
    *,
    memory_guard=None,
) -> np.ndarray:
    """Minimum spanning tree via Kruskal (scipy.sparse.csgraph).

    Args:
      n: number of vertices (0..n-1).
      edges: (m, 3) array-like — rows (u, v, weight), 0-indexed vertices.
             For undirected graph, provide each edge once (u, v);
             internally symmetrized via maximum weight (w_uv = w_vu).
             Use finite positive weights (zeros interpreted as "no edge"
             by scipy — apply epsilon if minimal weights can be 0).
      memory_guard: optional.

    Returns:
      (n_connected-1, 3) float64 — MST edges (u, v, w), sorted by u.
      For connected graph n_connected == n -> (n-1, 3).
      For disconnected graph, emits UserWarning and returns (n_comp-1, 3)
      — the forest; caller should handle components (e.g. HDBSCAN splits
      disconnected components into separate clusters).

    OOC: no. Peak = O(m + n) for the sparse adjacency build.
    For n=10K, m=n^2=100M edges — ~2.4 GB (float64 x 3, scipy COO).
    Caller may subsample or use dense W directly via scipy.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if memory_guard is not None:
        memory_guard.check_state(
            n * 8 + _GRAPH_HEADER, name="mst.vertices")

    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 2 or edges.shape[1] != 3:
        raise ValueError(
            f"edges must be (m, 3); got {edges.shape}")
    m = edges.shape[0]
    if m == 0:
        if n > 1:
            warnings.warn(
                f"mst: empty edges for n={n}; returning empty forest",
                stacklevel=2)
        return np.empty((0, 3), dtype=np.float64)

    if memory_guard is not None:
        memory_guard.check_state(
            m * 3 * 8 + _GRAPH_HEADER, name="mst.edges")

    # Build symmetric sparse adjacency (COO -> CSR)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
    except ImportError as e:
        raise ImportError(
            f"mst: requires scipy ({e}); install scipy"
        ) from e

    u = edges[:, 0].astype(np.int64)
    v = edges[:, 1].astype(np.int64)
    w = edges[:, 2]
    if (u < 0).any() or (u >= n).any() or (v < 0).any() or (v >= n).any():
        raise ValueError(f"mst: edge endpoint out of [0, {n})")
    if (w <= 0).any():
        # scipy treats zeros as no-edge; small negatives would be errors.
        w = np.maximum(w, 1e-15)
        warnings.warn(
            "mst: found non-positive weights; clamped to 1e-15 "
            "(scipy minimum_spanning_tree drops zeros)", stacklevel=2)

    # Undirected: for duplicate (u,v) and (v,u), take max (equiv to having
    # the edge once with that weight). For unique edge only one direction,
    # symmetrize.
    G_uv = coo_matrix((w, (u, v)), shape=(n, n)).tocsr()
    G_vu = coo_matrix((w, (v, u)), shape=(n, n)).tocsr()
    G = G_uv.maximum(G_vu)  # elementwise max -> symmetric

    # Sanity: connectivity
    n_comp, _ = connected_components(G, directed=False)
    T = minimum_spanning_tree(G)
    T = T.tocoo()

    if n_comp > 1:
        warnings.warn(
            f"mst: graph disconnected ({n_comp} components); "
            f"returning forest ({T.nnz} edges, expected {n-1})",
            stacklevel=2)

    out = np.column_stack([
        T.row.astype(np.float64),
        T.col.astype(np.float64),
        T.data.astype(np.float64),
    ])
    # Sort by u for determinism
    order = np.lexsort((out[:, 1], out[:, 0]))
    return out[order]
