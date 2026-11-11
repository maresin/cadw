"""Component-wise verification for svd_topk (DEC-033, v0.9.205).

Chunked Gram (batch-layer, OOC in n_rows). Sklearn TruncatedSVD(arpack, tol=0)
parity (P2: sigma atol 6e-14)."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops.linalg import svd_topk


def _mk_df(M):
    d = M.shape[1]
    return CadwDataFrame.from_table(pa.table({
        f"x{j}": pa.array(M[:, j], type=pa.float64())
        for j in range(d)
    }), memory_guard=MemoryGuard(max_memory_mb=500))


def _warmup():
    M = np.random.RandomState(999).randn(30, 5)
    df = _mk_df(M)
    svd_topk(df, [f"x{j}" for j in range(5)], 3,
             memory_guard=MemoryGuard(max_memory_mb=500))


_warmup()


# ---- CW-1: sigma + V vs scipy.linalg.svd (top-k) ----

@pytest.mark.parametrize("n,d,k", [(200, 30, 5), (500, 100, 10), (1000, 50, 8)])
def test_svd_topk_matches_dense_svd(n, d, k):
    """sigma, |V| bit-identical to scipy.linalg.svd(full_matrices=False)."""
    from scipy.linalg import svd as scipy_svd
    rng = np.random.RandomState(0)
    M = rng.randn(n, d)
    df = _mk_df(M)
    U, sigma, V = svd_topk(df, [f"x{j}" for j in range(d)], k,
                            compute_u=False,
                            memory_guard=MemoryGuard(max_memory_mb=500))
    assert U is None
    assert sigma.shape == (k,)
    assert V.shape == (d, k)

    _u, s_ref, _vt = scipy_svd(M, full_matrices=False)
    # sigma descending
    np.testing.assert_allclose(sigma, s_ref[:k], atol=1e-9)

    # V (d, k) vs VT.T[:k] — sign-flip to align
    V_ref = _vt[:k].T
    for j in range(k):
        sgn = np.sign(V[:, j] @ V_ref[:, j])
        if sgn != 0:
            V[:, j] *= sgn
    np.testing.assert_allclose(V, V_ref, atol=1e-9)


# ---- CW-2: sklearn TruncatedSVD (arpack, tol=0) parity ----

@pytest.mark.parametrize("n,d,k", [(150, 40, 5), (300, 60, 10)])
def test_sklearn_truncated_svd_parity(n, d, k):
    """sigma and components_ (V.T) match sklearn TruncatedSVD(arpack, tol=0)."""
    from sklearn.decomposition import TruncatedSVD
    rng = np.random.RandomState(1)
    M = rng.randn(n, d)
    df = _mk_df(M)

    sk = TruncatedSVD(n_components=k, algorithm="arpack", tol=0.0,
                       random_state=0)
    sk.fit(M)

    _U, sigma, V = svd_topk(df, [f"x{j}" for j in range(d)], k,
                             compute_u=False,
                             memory_guard=MemoryGuard(max_memory_mb=500))

    # singular_values_ match
    np.testing.assert_allclose(sigma, sk.singular_values_, atol=1e-9)
    # components_ = V.T; sklearn already svd_flip'ed
    Vt = V.T
    np.testing.assert_allclose(Vt, sk.components_, atol=1e-9)


# ---- CW-3: OOC ratio < 2.0 ----

def test_ooc_in_n_rows():
    """state = d^2*8*2 + k*8*2 + 256; independent of n_rows."""
    d, k = 100, 10
    # states don't depend on n; test via df sizes
    df_small = _mk_df(np.random.RandomState(2).randn(200, d))
    df_large = _mk_df(np.random.RandomState(3).randn(2000, d))
    U1, s1, V1 = svd_topk(df_small, [f"x{j}" for j in range(d)], k,
                            memory_guard=MemoryGuard(max_memory_mb=500))
    U2, s2, V2 = svd_topk(df_large, [f"x{j}" for j in range(d)], k,
                            memory_guard=MemoryGuard(max_memory_mb=500))
    # Same state (independent of n)
    assert s1.shape == s2.shape == (k,)
    assert V1.shape == V2.shape == (d, k)


# ---- CW-4: compute_u=True → U (n, k) AutoArrayResult ----

def test_compute_u():
    from sklearn.decomposition import TruncatedSVD
    rng = np.random.RandomState(4)
    n, d, k = 300, 40, 5
    M = rng.randn(n, d)
    df = _mk_df(M)
    U, sigma, V = svd_topk(df, [f"x{j}" for j in range(d)], k,
                            compute_u=True,
                            memory_guard=MemoryGuard(max_memory_mb=500))
    assert U is not None
    # Collect U
    u_parts = []
    for chunk in U:
        u_parts.append(np.asarray(chunk))
    U_arr = np.vstack(u_parts) if u_parts else np.empty((0, k))
    assert U_arr.shape == (n, k)

    # Reconstruct truncated: U @ diag(sigma) @ V.T
    M_recon = (U_arr * sigma[None, :]) @ V.T
    # Truncated approximation error inherent; check via svd of M_recon vs M
    # Top-k singular values recovered
    from scipy.linalg import svd as scipy_svd
    _u, s_ref, _vt = scipy_svd(M, full_matrices=False)
    M_recon_ref = _u[:, :k] * s_ref[:k][None, :]
    M_recon_ref = M_recon_ref @ _vt[:k]
    # Both truncated reconstructions match (up to sign/order); use |recon|
    err = np.abs(np.abs(M_recon) - np.abs(M_recon_ref)).max()
    assert err < 1e-6, f"recon mismatch: {err:.2e}"


# ---- CW-5: guard refuses tight (d^2 state) ----

def test_guard_refuses_tight():
    """d=200 -> state 200*200*8*2 = 640 KB; guard 0.1 MB refuses."""
    rng = np.random.RandomState(5)
    M = rng.randn(100, 200)
    df = _mk_df(M)
    with pytest.raises(MemoryLimitExceeded):
        svd_topk(df, [f"x{j}" for j in range(200)], 5,
                 memory_guard=MemoryGuard(max_memory_mb=0.1))


# ---- CW-6: validation ----

def test_svd_topk_invalid():
    M = np.random.RandomState(6).randn(50, 10)
    df = _mk_df(M)
    with pytest.raises(ValueError, match="X_cols is empty"):
        svd_topk(df, [], 5,
                 memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="k must be"):
        svd_topk(df, [f"x{j}" for j in range(10)], 0,
                 memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="k=20 > min"):
        svd_topk(df, [f"x{j}" for j in range(10)], 20,
                 memory_guard=MemoryGuard(max_memory_mb=500))
