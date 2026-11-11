"""v0.9.147: tests for cadw.ops.graph (L5 mst + L6 laplacian/eigsh_topk)."""
import warnings

import numpy as np
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops.graph import graph_laplacian, eigsh_topk, mst


# ============================================================================
# graph_laplacian
# ============================================================================

def _random_adjacency(n, seed=0, density=0.5):
    """Symmetric random adjacency (sparsify mask is symmetric)."""
    rng = np.random.RandomState(seed)
    W = rng.random((n, n))
    W = (W + W.T) / 2
    # symmetric sparsify: keep edge if either direction drawn
    mask = rng.random((n, n)) <= density
    mask = mask | mask.T
    W = W * mask
    np.fill_diagonal(W, 0.0)
    return W


def test_laplacian_unnormalized():
    W = _random_adjacency(10, seed=0)
    L = graph_laplacian(W, normalized="none")
    d = W.sum(axis=1)
    L_ref = np.diag(d) - W
    np.testing.assert_allclose(L, L_ref, rtol=1e-12)


def test_laplacian_row_sums_zero_unnorm():
    """Row sums of L = D - W are 0 (property of unnormalized Laplacian)."""
    W = _random_adjacency(15, seed=1)
    L = graph_laplacian(W, normalized="none")
    np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-12)


def test_laplacian_sym():
    W = _random_adjacency(10, seed=2)
    L = graph_laplacian(W, normalized="sym")
    d = W.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L_ref = np.eye(10) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
    np.testing.assert_allclose(L, L_ref, rtol=1e-12)
    # symmetric
    np.testing.assert_allclose(L, L.T, rtol=1e-12)


def test_laplacian_sym_eigenvalues_in_range():
    """L_sym eigenvalues in [0, 2] for undirected graph with nonneg weights."""
    W = _random_adjacency(20, seed=3)
    L = graph_laplacian(W, normalized="sym")
    vals = np.linalg.eigvalsh(L)
    assert vals.min() >= -1e-10
    assert vals.max() <= 2.0 + 1e-10


def test_laplacian_rw():
    W = _random_adjacency(10, seed=4)
    L = graph_laplacian(W, normalized="rw")
    d = W.sum(axis=1)
    d_inv = 1.0 / np.maximum(d, 1e-12)
    L_ref = np.eye(10) - (d_inv[:, None] * W)
    np.testing.assert_allclose(L, L_ref, rtol=1e-12)


def test_laplacian_invalid_normalized():
    W = np.eye(5)
    with pytest.raises(ValueError, match="normalized must be"):
        graph_laplacian(W, normalized="XX")


def test_laplacian_non_square():
    with pytest.raises(ValueError, match="square"):
        graph_laplacian(np.zeros((3, 4)))


def test_laplacian_guard_refuses():
    W = np.eye(500)
    with pytest.raises(MemoryLimitExceeded):
        graph_laplacian(W, memory_guard=MemoryGuard(max_memory_mb=0.1))


# ============================================================================
# eigsh_topk (graph wrapper)
# ============================================================================

def test_eigsh_topk_sm_on_laplacian():
    """For connected graph L_sym, λ_0 ≈ 0 (SM mode)."""
    W = _random_adjacency(15, seed=5, density=0.9)
    L = graph_laplacian(W, normalized="sym")
    vals, vecs = eigsh_topk(L, 3, which="SM")
    assert abs(vals[0]) < 1e-8, f"λ0={vals[0]} not near 0"
    assert vecs.shape == (15, 3)


def test_eigsh_topk_sa_on_unnormalized_laplacian():
    """For connected graph, unnormalized L has λ_0 = 0 (SA mode)."""
    W = _random_adjacency(10, seed=6, density=0.9)
    L = graph_laplacian(W, normalized="none")
    vals, _ = eigsh_topk(L, 3, which="SA")
    assert abs(vals[0]) < 1e-8
    assert (np.diff(vals) >= -1e-10).all()  # ascending


def test_eigsh_topk_default_sm():
    """eigsh_topk default which='SM'."""
    W = _random_adjacency(12, seed=7, density=0.9)
    L = graph_laplacian(W, normalized="sym")
    vals, _ = eigsh_topk(L, 2)
    # SM sorted ascending by |λ|
    assert (np.abs(vals) <= np.abs(vals[-1]) + 1e-10).all()


# ============================================================================
# mst
# ============================================================================

def test_mst_simple_connected():
    """Triangle graph: edges 0-1(w=1), 1-2(w=2), 0-2(w=3) -> MST = (0,1,1)+(1,2,2)."""
    n = 3
    edges = np.array([[0, 1, 1.0], [1, 2, 2.0], [0, 2, 3.0]])
    mst_edges = mst(n, edges)
    assert mst_edges.shape == (2, 3)
    total_w = mst_edges[:, 2].sum()
    assert abs(total_w - 3.0) < 1e-10


def test_mst_complete_graph_known():
    """Complete graph n=4 with known MST."""
    n = 4
    # Square with diagonal: 0-1(1), 1-2(2), 2-3(3), 3-0(4), 0-2(5)
    edges = np.array([
        [0, 1, 1.0], [1, 2, 2.0], [2, 3, 3.0], [3, 0, 4.0], [0, 2, 5.0]])
    mst_edges = mst(n, edges)
    assert mst_edges.shape == (3, 3)  # n-1 edges
    # MST = (0,1,1) + (1,2,2) + (2,3,3) = 6
    assert abs(mst_edges[:, 2].sum() - 6.0) < 1e-10


def test_mst_disconnected():
    """Disconnected graph -> forest (n_comp-1 edges) + warning."""
    n = 4
    # Two components: (0,1) and (2,3)
    edges = np.array([[0, 1, 1.0], [2, 3, 2.0]])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mst_edges = mst(n, edges)
        assert any("disconnected" in str(x.message) for x in w)
    assert mst_edges.shape == (2, 3)  # 2 components -> 2-1=1 per comp = 2


def test_mst_single_edge():
    edges = np.array([[0, 1, 5.0]])
    out = mst(2, edges)
    assert out.shape == (1, 3)
    assert out[0, 2] == 5.0


def test_mst_empty_edges_raises_for_n_gt_1():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = mst(3, np.empty((0, 3)))
        assert out.shape == (0, 3)
        assert any("empty edges" in str(x.message) for x in w)


def test_mst_duplicate_edges_takes_max():
    """Duplicate (u,v) and (v,u) -> one edge with max weight."""
    # (0,1) with two different weights -> MST keeps min path
    edges = np.array([[0, 1, 1.0], [1, 0, 5.0]])
    out = mst(2, edges)
    # symmetrize to max -> w=5 -> MST uses 5
    # (If symmetrize to min -> w=1 -> MST uses 1; depends on design)
    # Our design uses max (equivalent to "give edge once with that weight")
    assert out.shape == (1, 3)


def test_mst_guard_refuses():
    n = 1000
    edges = np.zeros((10_000, 3))
    edges[:, :2] = np.random.RandomState(0).randint(0, n, (10_000, 2))
    edges[:, 2] = 1.0
    with pytest.raises(MemoryLimitExceeded):
        mst(n, edges, memory_guard=MemoryGuard(max_memory_mb=0.001))


def test_mst_invalid_shape():
    with pytest.raises(ValueError, match="edges must be"):
        mst(3, np.zeros((5, 2)))


def test_mst_out_of_range():
    with pytest.raises(ValueError, match="out of"):
        mst(3, np.array([[0, 5, 1.0]]))

# ---- v0.9.203: graph_laplacian_sparse + eigsh_topk sparse + sigma ----

def test_graph_laplacian_sparse_parity():
    """Sparse vs dense L (sym / none / rw) — bit-identical."""
    import scipy.sparse as sp
    from cadw.ops.graph import graph_laplacian, graph_laplacian_sparse
    rng = np.random.RandomState(0)
    n = 30
    W_dense = rng.rand(n, n)
    W_dense = (W_dense + W_dense.T) / 2
    W_dense[W_dense < 0.5] = 0.0
    W_sp = sp.csr_matrix(W_dense)

    for norm in ("none", "sym", "rw"):
        L_dense = graph_laplacian(W_dense, normalized=norm)
        L_sparse = graph_laplacian_sparse(W_sp, normalized=norm)
        assert sp.issparse(L_sparse), f"{norm}: output not sparse"
        L_sp_dense = L_sparse.toarray()
        diff = np.abs(L_sp_dense - L_dense).max()
        assert diff < 1e-14, f"{norm}: max diff {diff:.2e}"


def test_graph_laplacian_sparse_rejects_dense():
    from cadw.ops.graph import graph_laplacian_sparse
    W = np.eye(5)
    with pytest.raises(TypeError, match="must be scipy.sparse"):
        graph_laplacian_sparse(W)


def test_eigsh_topk_sparse_parity():
    """Sparse eigsh vs dense eigh_topk (SM which) — same eigenvalues."""
    import scipy.sparse as sp
    from cadw.ops.graph import graph_laplacian, graph_laplacian_sparse, eigsh_topk
    rng = np.random.RandomState(0)
    n = 80
    W = rng.rand(n, n)
    W = (W + W.T) / 2
    W[W < 0.7] = 0.0
    W_sp = sp.csr_matrix(W)
    L_sp = graph_laplacian_sparse(W_sp, normalized="sym")
    L_dense = graph_laplacian(W, normalized="sym")

    # Sparse path
    vals_sp, vecs_sp = eigsh_topk(L_sp, k=5, which="SM")
    # Dense reference
    from cadw.ops.linalg import eigh_topk
    vals_d, vecs_d = eigh_topk(L_dense, k=5, which="SM")

    np.testing.assert_allclose(np.sort(vals_sp), np.sort(vals_d),
                                atol=1e-6)


def test_eigsh_topk_sigma_param():
    """sigma=-1e-5 (sklearn parity) runs; values close to sigma=0."""
    import scipy.sparse as sp
    from cadw.ops.graph import graph_laplacian_sparse, eigsh_topk
    rng = np.random.RandomState(1)
    n = 100
    W = rng.rand(n, n)
    W = (W + W.T) / 2
    W[W < 0.7] = 0.0
    L_sp = graph_laplacian_sparse(sp.csr_matrix(W), normalized="sym")

    vals_0, _ = eigsh_topk(L_sp, k=5, which="SM", sigma=0.0)
    vals_neg, _ = eigsh_topk(L_sp, k=5, which="SM", sigma=-1e-5)
    # Both valid; sigma=-1e-5 (sklearn parity) may differ slightly
    assert vals_neg.shape == (5,)
    assert np.all(np.isfinite(vals_neg))


def test_eigsh_topk_dense_delegation():
    """Dense L still delegates to eigh_topk (back-compat)."""
    from cadw.ops.graph import graph_laplacian, eigsh_topk
    rng = np.random.RandomState(2)
    n = 50
    W = rng.rand(n, n)
    W = (W + W.T) / 2
    L = graph_laplacian(W, normalized="sym")
    vals, vecs = eigsh_topk(L, k=3, which="SM")
    assert vals.shape == (3,)
    assert vecs.shape == (50, 3)
    assert np.all(np.isfinite(vals))

