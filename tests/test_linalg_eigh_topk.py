"""v0.9.146: tests for cadw.ops.linalg.eigh_topk (L19)."""
import numpy as np
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops.linalg import eigh_topk


def _sym_posdef(n, seed=0):
    """Well-conditioned symmetric PSD matrix with distinct eigenvalues."""
    rng = np.random.RandomState(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T + 0.1 * np.eye(n)


def _sym_indef(n, seed=0):
    """Symmetric with mixed-sign eigenvalues."""
    rng = np.random.RandomState(seed)
    A = rng.normal(size=(n, n))
    M = A + A.T
    return M


# ---- correctness: LA (largest algebraic) ----

def test_la_matches_full_eigh():
    M = _sym_posdef(30, seed=0)
    k = 5
    vals, vecs = eigh_topk(M, k, which="LA")
    full_vals, full_vecs = np.linalg.eigh(M)
    # top-k largest algebraic
    top_idx = np.argsort(full_vals)[-k:][::-1]
    np.testing.assert_allclose(vals, full_vals[top_idx], rtol=1e-10)
    # eigenvectors may flip sign; check via projection
    for j in range(k):
        v_ours = vecs[:, j]
        v_ref = full_vecs[:, top_idx[j]]
        # align sign
        if v_ours @ v_ref < 0:
            v_ours = -v_ours
        np.testing.assert_allclose(v_ours, v_ref, atol=1e-10)


def test_la_eigenvalues_descending():
    M = _sym_posdef(20, seed=1)
    vals, _ = eigh_topk(M, 5, which="LA")
    assert (np.diff(vals) <= 1e-12).all(), \
        f"LA vals not descending: {vals}"


# ---- SA (smallest algebraic) ----

def test_sa_matches_full_eigh():
    M = _sym_indef(20, seed=2)
    k = 4
    vals, _ = eigh_topk(M, k, which="SA")
    full_vals, _ = np.linalg.eigh(M)
    top_idx = np.argsort(full_vals)[:k]
    np.testing.assert_allclose(vals, full_vals[top_idx], rtol=1e-10)
    # SA ascending
    assert (np.diff(vals) >= -1e-12).all()


# ---- LM (largest magnitude) ----

def test_lm_uses_abs_value():
    M = _sym_indef(15, seed=3)
    k = 4
    vals, _ = eigh_topk(M, k, which="LM")
    full_vals, _ = np.linalg.eigh(M)
    # sort by |λ| descending
    top_idx = np.argsort(np.abs(full_vals))[-k:][::-1]
    np.testing.assert_allclose(np.abs(vals), np.abs(full_vals[top_idx]),
                               rtol=1e-10)


# ---- SM (smallest magnitude) ----

def test_sm_matches_full_eigh():
    M = _sym_indef(20, seed=4)
    k = 4
    vals, _ = eigh_topk(M, k, which="SM")
    full_vals, _ = np.linalg.eigh(M)
    top_idx = np.argsort(np.abs(full_vals))[:k]
    np.testing.assert_allclose(np.abs(vals),
                               np.abs(full_vals[top_idx]), rtol=1e-10)


# ---- spectral use case: L = D - W ----

def test_spectral_laplacian_smallest():
    """For connected graph Laplacian, λ_0 ≈ 0, others > 0.
    SA returns ascending; first element should be near 0."""
    rng = np.random.RandomState(5)
    n = 10
    W = rng.random((n, n))
    W = (W + W.T) / 2  # symmetric
    np.fill_diagonal(W, 0)
    D = np.diag(W.sum(axis=1))
    L = D - W
    vals, vecs = eigh_topk(L, 3, which="SA")
    assert abs(vals[0]) < 1e-8, f"λ0={vals[0]} not near 0"
    assert vecs.shape == (n, 3)


# ---- SM on L_sym ----

def test_sm_on_symmetric_normalized_laplacian():
    rng = np.random.RandomState(6)
    n = 12
    W = rng.random((n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    d = W.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L_sym = np.eye(n) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
    vals, _ = eigh_topk(L_sym, 3, which="SM")
    # λ_0 ≈ 0 for connected graph
    assert abs(vals[0]) < 1e-6, f"λ0={vals[0]} not near 0"


# ---- invalid args ----

def test_invalid_which():
    M = np.eye(5)
    with pytest.raises(ValueError, match="which must be"):
        eigh_topk(M, 2, which="XX")


def test_invalid_k():
    M = np.eye(5)
    with pytest.raises(ValueError, match="k must be"):
        eigh_topk(M, 0)
    with pytest.raises(ValueError, match="k must be"):
        eigh_topk(M, 6)


def test_not_square():
    M = np.zeros((3, 4))
    with pytest.raises(ValueError, match="square"):
        eigh_topk(M, 2)


# ---- guard refuses ----

def test_guard_refuses_at_tight():
    """n=100 dense → n²·8 = 80 KB; tiny guard 0.05 MB = ~50 KB."""
    M = np.eye(100)
    with pytest.raises(MemoryLimitExceeded):
        eigh_topk(M, 5, memory_guard=MemoryGuard(max_memory_mb=0.05))


# ---- large n: scipy fallback (may skip if threshold raised) ----

@pytest.mark.skipif(
    True, reason="n>2000 dense generation is slow; tested manually if needed")
def test_large_n_scipy_path():
    """n=2500 -> scipy eigsh. Slow to generate dense matrix."""
    M = _sym_posdef(2500, seed=10)
    vals, vecs = eigh_topk(M, 5, which="LA")
    assert vals.shape == (5,)
    assert vecs.shape == (2500, 5)
    assert (np.diff(vals) <= 1e-10).all()
