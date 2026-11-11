"""Component-wise verification for fit_truncated_svd (DEC-033, v0.9.205).

Sklearn TruncatedSVD(algorithm="arpack", tol=0) parity (P3: sigma atol
1e-9). Pattern #1 batch-layer OOC in n_rows via svd_topk chunked Gram.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import fit_truncated_svd, TruncatedSVDResult


def _mk(M):
    cols = {f"x{j}": pa.array(M[:, j], type=pa.float64())
            for j in range(M.shape[1])}
    return CadwDataFrame.from_table(pa.table(cols),
                                     memory_guard=MemoryGuard(max_memory_mb=500))


def _extract_z(df_out):
    parts = []
    d = len(df_out.columns)
    for b in df_out._iter_filtered_batches():
        parts.append(np.column_stack([
            b.column(df_out.columns[j]).to_numpy(zero_copy_only=False)
            for j in range(d)
        ]))
    return np.vstack(parts) if parts else np.empty((0, d))


def _warmup():
    M = np.random.RandomState(999).randn(30, 10)
    df = _mk(M)
    r = fit_truncated_svd(df, [f"x{j}" for j in range(10)], 3,
                           memory_guard=MemoryGuard(max_memory_mb=500))
    r.release_train_state()


_warmup()


# ---- CW-1: sklearn parity (4 metrics + transform) ----

@pytest.mark.parametrize("n,d,k", [(150, 40, 5), (400, 60, 8)])
def test_sklearn_parity(n, d, k):
    from sklearn.decomposition import TruncatedSVD
    rng = np.random.RandomState(0)
    M = rng.randn(n, d)
    df = _mk(M)
    sk = TruncatedSVD(n_components=k, algorithm="arpack", tol=0.0,
                       random_state=0)
    sk.fit(M)
    r = fit_truncated_svd(df, [f"x{j}" for j in range(d)], k,
                           algorithm="arpack",
                           memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_allclose(r.singular_values_, sk.singular_values_,
                                atol=1e-9)
    np.testing.assert_allclose(r.components_, sk.components_, atol=1e-9)
    np.testing.assert_allclose(r.explained_variance_,
                                sk.explained_variance_, atol=1e-9)
    np.testing.assert_allclose(r.explained_variance_ratio_,
                                sk.explained_variance_ratio_, atol=1e-9)
    # transform parity
    Q = rng.randn(20, d)
    df_q = _mk(Q)
    out = r.transform(df_q, output="memory")
    Z_cadw = _extract_z(out)
    Z_sk = sk.transform(Q)
    np.testing.assert_allclose(Z_cadw, Z_sk, atol=1e-9)
    r.release_train_state()


# ---- CW-2: fit_transform parity ----

def test_fit_transform_parity():
    from sklearn.decomposition import TruncatedSVD
    rng = np.random.RandomState(1)
    M = rng.randn(200, 50)
    df = _mk(M)
    sk = TruncatedSVD(n_components=8, algorithm="arpack", tol=0.0,
                       random_state=0)
    Z_sk = sk.fit_transform(M)
    r = fit_truncated_svd(df, [f"x{j}" for j in range(50)], 8,
                           algorithm="arpack",
                           memory_guard=MemoryGuard(max_memory_mb=500))
    out = r.fit_transform(df, output="memory")
    Z_cadw = _extract_z(out)
    np.testing.assert_allclose(Z_cadw, Z_sk, atol=1e-9)
    r.release_train_state()


# ---- CW-3: state_nbytes reasonable ----

def test_state_nbytes():
    rng = np.random.RandomState(2)
    M = rng.randn(300, 80)
    df = _mk(M)
    r = fit_truncated_svd(df, [f"x{j}" for j in range(80)], 10,
                           memory_guard=MemoryGuard(max_memory_mb=500))
    # components (10*80*8), sigma (10*8), var (10*8), evr (10*8) + 256
    # ~ 6400 + 80 + 80 + 80 + 256 = 6896
    assert 6000 < r.state_nbytes() < 10000, f"state {r.state_nbytes()}"
    r.release_train_state()


# ---- CW-4: guard refuses tight (d^2 transient) ----

def test_guard_refuses_tight():
    """d=200 -> svd_topk state = 200*200*8*2 + ... = 640 KB; guard 0.1 MB."""
    rng = np.random.RandomState(3)
    M = rng.randn(200, 200)
    df = _mk(M)
    with pytest.raises(MemoryLimitExceeded):
        fit_truncated_svd(df, [f"x{j}" for j in range(200)], 10,
                           memory_guard=MemoryGuard(max_memory_mb=0.1))


# ---- CW-5: release idempotent + transform raises ----

def test_release_idempotent():
    rng = np.random.RandomState(4)
    M = rng.randn(100, 20)
    df = _mk(M)
    r = fit_truncated_svd(df, [f"x{j}" for j in range(20)], 5,
                           memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.components_ is not None
    r.release_train_state()
    assert r.components_ is None
    r.release_train_state()  # idempotent
    with pytest.raises(RuntimeError, match="released"):
        r.transform(df)


# ---- CW-6: reproducibility ----

def test_reproducibility():
    rng = np.random.RandomState(5)
    M = rng.randn(150, 30)
    df = _mk(M)
    r1 = fit_truncated_svd(df, [f"x{j}" for j in range(30)], 5,
                            memory_guard=MemoryGuard(max_memory_mb=500))
    r2 = fit_truncated_svd(df, [f"x{j}" for j in range(30)], 5,
                            memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(r1.components_, r2.components_)
    np.testing.assert_array_equal(r1.singular_values_, r2.singular_values_)
    r1.release_train_state()
    r2.release_train_state()


# ---- CW-7: algorithm='randomized' raises (deferred) ----

def test_randomized_deferred():
    M = np.random.RandomState(6).randn(50, 10)
    df = _mk(M)
    with pytest.raises(ValueError, match="algorithm must be 'arpack'"):
        fit_truncated_svd(df, [f"x{j}" for j in range(10)], 5,
                           algorithm="randomized",
                           memory_guard=MemoryGuard(max_memory_mb=500))


# ---- CW-8: validation ----

def test_invalid():
    M = np.random.RandomState(7).randn(50, 10)
    df = _mk(M)
    with pytest.raises(ValueError, match="X_cols is empty"):
        fit_truncated_svd(df, [], 5,
                           memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="n_components must be"):
        fit_truncated_svd(df, [f"x{j}" for j in range(10)], 0,
                           memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="need 1 <= k"):
        fit_truncated_svd(df, [f"x{j}" for j in range(10)], 20,
                           memory_guard=MemoryGuard(max_memory_mb=500))
