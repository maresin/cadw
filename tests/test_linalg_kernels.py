"""v0.9.145: tests for pairwise_kernel (L3) + chol_solve (L4)."""
import numpy as np
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ops.linalg import (
    pairwise_kernel, rbf_kernel, poly_kernel, laplacian_kernel, chol_solve,
)


# ============================================================================
# pairwise_kernel — correctness
# ============================================================================

def test_rbf_kernel_diag_is_one():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(20, 3))
    K = rbf_kernel(X, X, gamma=0.5)
    np.testing.assert_allclose(np.diag(K), 1.0, rtol=1e-12)


def test_rbf_kernel_symmetric():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(50, 4))
    K = rbf_kernel(X, X, gamma=0.7)
    np.testing.assert_allclose(K, K.T, rtol=1e-12)


def test_rbf_kernel_matches_manual():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(10, 3))
    Y = rng.normal(size=(7, 3))
    K = rbf_kernel(X, Y, gamma=0.3)
    # manual: ||x-y||^2
    D = np.zeros((10, 7))
    for i in range(10):
        for j in range(7):
            D[i, j] = np.sum((X[i] - Y[j]) ** 2)
    K_ref = np.exp(-0.3 * D)
    np.testing.assert_allclose(K, K_ref, rtol=1e-10)


def test_poly_kernel_matches_manual():
    rng = np.random.RandomState(3)
    X = rng.normal(size=(10, 3))
    Y = rng.normal(size=(5, 3))
    K = poly_kernel(X, Y, gamma=0.5, degree=3, coef0=1.0)
    K_ref = (0.5 * (X @ Y.T) + 1.0) ** 3
    np.testing.assert_allclose(K, K_ref, rtol=1e-10)


def test_laplacian_kernel_matches_manual():
    rng = np.random.RandomState(4)
    X = rng.normal(size=(8, 3))
    Y = rng.normal(size=(6, 3))
    K = laplacian_kernel(X, Y, gamma=0.7)
    # manual: L1 dist
    D = np.zeros((8, 6))
    for i in range(8):
        for j in range(6):
            D[i, j] = np.abs(X[i] - Y[j]).sum()
    K_ref = np.exp(-0.7 * D)
    np.testing.assert_allclose(K, K_ref, rtol=1e-10)


def test_chunk_invariant():
    rng = np.random.RandomState(5)
    X = rng.normal(size=(10, 3))
    Y = rng.normal(size=(100, 3))
    K_a = pairwise_kernel(X, Y, "rbf", gamma=0.5, chunk_size=13)
    K_b = pairwise_kernel(X, Y, "rbf", gamma=0.5, chunk_size=1000)
    np.testing.assert_allclose(K_a, K_b, rtol=1e-12)


def test_poly_chunk_invariant():
    rng = np.random.RandomState(6)
    X = rng.normal(size=(10, 3))
    Y = rng.normal(size=(100, 3))
    K_a = pairwise_kernel(X, Y, "poly", degree=2, chunk_size=13)
    K_b = pairwise_kernel(X, Y, "poly", degree=2, chunk_size=1000)
    np.testing.assert_allclose(K_a, K_b, rtol=1e-10)


def test_invalid_kernel():
    X = np.zeros((3, 2))
    with pytest.raises(ValueError, match="kernel must be"):
        pairwise_kernel(X, X, "hamming")


def test_invalid_gamma():
    X = np.zeros((3, 2))
    with pytest.raises(ValueError, match="gamma must be"):
        pairwise_kernel(X, X, "rbf", gamma=0)


def test_shape_mismatch():
    X = np.zeros((3, 2))
    Y = np.zeros((4, 3))
    with pytest.raises(ValueError, match="shape"):
        pairwise_kernel(X, Y, "rbf")


def test_guard_refuses_at_tight():
    X = np.zeros((1000, 3))
    Y = np.zeros((1000, 3))
    with pytest.raises(MemoryLimitExceeded):
        pairwise_kernel(X, Y, "rbf",
                        memory_guard=MemoryGuard(max_memory_mb=0.1))


# ============================================================================
# chol_solve — correctness
# ============================================================================

def test_chol_solve_psd_exact():
    """For well-conditioned PSD K, chol_solve matches np.linalg.solve."""
    rng = np.random.RandomState(10)
    A = rng.normal(size=(20, 20))
    K = A @ A.T + 0.1 * np.eye(20)  # PSD
    b = rng.normal(size=20)
    x = chol_solve(K, b)
    x_ref = np.linalg.solve(K, b)
    np.testing.assert_allclose(x, x_ref, rtol=1e-10)


def test_chol_solve_matrix_rhs():
    rng = np.random.RandomState(11)
    A = rng.normal(size=(15, 15))
    K = A @ A.T + 0.5 * np.eye(15)
    B = rng.normal(size=(15, 3))
    X = chol_solve(K, B)
    X_ref = np.linalg.solve(K, B)
    np.testing.assert_allclose(X, X_ref, rtol=1e-10)


def test_chol_solve_indefinite_falls_back():
    """Non-PSD K -> Cholesky fails -> np.linalg.solve fallback."""
    rng = np.random.RandomState(12)
    K = rng.normal(size=(10, 10))
    K = K + K.T  # symmetric but not PSD
    # ensure non-singular
    K += 5.0 * np.eye(10)
    b = rng.normal(size=10)
    x = chol_solve(K, b)
    x_ref = np.linalg.solve(K, b)
    np.testing.assert_allclose(x, x_ref, rtol=1e-8)


def test_chol_solve_invalid_K():
    K = np.zeros((3, 4))
    b = np.zeros(3)
    with pytest.raises(ValueError, match="K must be square"):
        chol_solve(K, b)


def test_chol_solve_shape_mismatch():
    K = np.eye(3)
    b = np.zeros(4)
    with pytest.raises(ValueError, match="b.shape"):
        chol_solve(K, b)

# ---- v0.9.200: linear kernel ----

def test_linear_kernel_matches_manual():
    """linear: K = X @ Y.T."""
    from cadw.ops.linalg import pairwise_kernel
    rng = np.random.RandomState(0)
    X = rng.randn(10, 4)
    Y = rng.randn(15, 4)
    K_cadw = pairwise_kernel(X, Y, kernel="linear")
    K_ref = X @ Y.T
    np.testing.assert_allclose(K_cadw, K_ref, atol=1e-12)


def test_linear_kernel_chunk_invariant():
    """linear: chunk_size doesn't affect output."""
    from cadw.ops.linalg import pairwise_kernel
    rng = np.random.RandomState(1)
    X = rng.randn(8, 3)
    Y = rng.randn(20, 3)
    K1 = pairwise_kernel(X, Y, kernel="linear", chunk_size=20)
    K2 = pairwise_kernel(X, Y, kernel="linear", chunk_size=5)
    np.testing.assert_allclose(K1, K2, atol=1e-14)


def test_linear_kernel_sklearn_parity():
    """linear: matches sklearn.metrics.pairwise.linear_kernel."""
    from sklearn.metrics.pairwise import linear_kernel
    from cadw.ops.linalg import pairwise_kernel
    rng = np.random.RandomState(2)
    X = rng.randn(12, 5)
    Y = rng.randn(17, 5)
    K_cadw = pairwise_kernel(X, Y, kernel="linear")
    K_sk = linear_kernel(X, Y)
    np.testing.assert_allclose(K_cadw, K_sk, atol=1e-12)


def test_linear_kernel_ignores_gamma():
    """linear: gamma unused; gamma=0.001 or huge both fine."""
    from cadw.ops.linalg import pairwise_kernel
    X = np.random.RandomState(3).randn(5, 3)
    Y = np.random.RandomState(4).randn(6, 3)
    K1 = pairwise_kernel(X, Y, kernel="linear", gamma=0.001)
    K2 = pairwise_kernel(X, Y, kernel="linear", gamma=1000.0)
    np.testing.assert_allclose(K1, K2, atol=1e-14)

