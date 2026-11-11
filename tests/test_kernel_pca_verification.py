"""Component-wise verification for fit_kernel_pca (DEC-033, v0.9.200).

Sklearn parity is primary evidence (P6 formulas + B smoke 1.9e-15).
Intrinsic O(n^2) fit (bounded_by_domain {n: 1e4}, DEC-075).
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import fit_kernel_pca, KernelPCAResult


def _mk_df(X):
    d = X.shape[1]
    cols = {f"x{j}": pa.array(X[:, j], type=pa.float64())
            for j in range(d)}
    return CadwDataFrame.from_table(pa.table(cols),
                                     memory_guard=MemoryGuard(max_memory_mb=500))


def _extract(out_df):
    parts = []
    d = len(out_df.columns)
    for b in out_df._iter_filtered_batches():
        parts.append(np.column_stack([
            b.column(out_df.columns[j]).to_numpy(zero_copy_only=False)
            for j in range(d)
        ]))
    return np.vstack(parts) if parts else np.empty((0, d))


def _fit_smoke():
    X = np.random.RandomState(999).randn(30, 3)
    df = _mk_df(X)
    r = fit_kernel_pca(df, ["x0", "x1", "x2"], n_components=2,
                        kernel="linear",
                        memory_guard=MemoryGuard(max_memory_mb=500))
    r.release_train_state()


_fit_smoke()


# ---- CW-1: state_nbytes exact (bit-exact) ----

@pytest.mark.parametrize("n,k", [(20, 2), (50, 2), (50, 5)])
def test_state_nbytes_exact(n, k):
    """state = 256 + X_fit.nbytes + K_fit_rows.nbytes + 8 + eigvals + eigvecs."""
    rng = np.random.RandomState(0)
    X = rng.randn(n, 3)
    df = _mk_df(X)
    r = fit_kernel_pca(df, ["x0", "x1", "x2"], n_components=k,
                        kernel="linear",
                        memory_guard=MemoryGuard(max_memory_mb=500))
    # k_eff may be less if zeros present (unlikely for random)
    assert r.n_components == k
    expected = (256
                + int(r.X_fit.nbytes)
                + int(r.K_fit_rows.nbytes)
                + 8
                + int(r.eigenvalues.nbytes)
                + int(r.eigenvectors.nbytes))
    assert r.state_nbytes() == expected, (
        f"n={n} k={k}: state={r.state_nbytes()}, expected {expected}")
    r.release_train_state()


# ---- CW-2: state grows with n (bounded_state: false honest) ----

def test_state_grows_with_n():
    """State O(n*d + n*k); n=100 > n=50 state."""
    states = []
    for n in (50, 100):
        X = np.random.RandomState(1).randn(n, 3)
        df = _mk_df(X)
        r = fit_kernel_pca(df, ["x0", "x1", "x2"], n_components=3,
                            kernel="linear",
                            memory_guard=MemoryGuard(max_memory_mb=500))
        states.append(r.state_nbytes())
        r.release_train_state()
    # n=100: 100*3*8 + 100*3*8 + 100*8 + ... > n=50
    assert states[1] > states[0] * 1.5, (
        f"state should grow ~2x: {states}")


# ---- CW-3..5: sklearn parity fit_transform (3 kernels) ----

@pytest.mark.parametrize("kernel,params", [
    ("linear", {}),
    ("rbf", {"gamma": 1.0}),
    ("poly", {"degree": 3, "gamma": 1.0, "coef0": 1.0}),
])
def test_sklearn_parity_fit_transform(kernel, params):
    from sklearn.decomposition import KernelPCA
    rng = np.random.RandomState(2)
    X = rng.randn(60, 4)
    df = _mk_df(X)
    kp = KernelPCA(n_components=2, kernel=kernel, **params)
    X_sk = kp.fit_transform(X)
    r = fit_kernel_pca(
        df, [f"x{j}" for j in range(4)],
        n_components=2, kernel=kernel,
        gamma=params.get("gamma"),
        degree=params.get("degree", 3),
        coef0=params.get("coef0", 1.0),
        memory_guard=MemoryGuard(max_memory_mb=500),
    )
    X_cadw = r.fit_transform()
    max_diff = float(np.abs(X_cadw - X_sk).max())
    assert max_diff < 1e-9, (
        f"kernel={kernel}: fit_transform max diff {max_diff:.2e}")
    r.release_train_state()


# ---- CW-6: sklearn parity transform (new data) ----

@pytest.mark.parametrize("kernel,params", [
    ("linear", {}),
    ("rbf", {"gamma": 1.0}),
    ("poly", {"degree": 2, "gamma": 0.5, "coef0": 0.5}),
])
def test_sklearn_parity_transform(kernel, params):
    from sklearn.decomposition import KernelPCA
    rng = np.random.RandomState(3)
    X_train = rng.randn(50, 3)
    X_new = rng.randn(10, 3)
    df_train = _mk_df(X_train)
    df_new = _mk_df(X_new)

    kp = KernelPCA(n_components=2, kernel=kernel, **params).fit(X_train)
    X_sk = kp.transform(X_new)

    r = fit_kernel_pca(
        df_train, ["x0", "x1", "x2"],
        n_components=2, kernel=kernel,
        gamma=params.get("gamma"),
        degree=params.get("degree", 3),
        coef0=params.get("coef0", 1.0),
        memory_guard=MemoryGuard(max_memory_mb=500),
    )
    out_df = r.transform(df_new)
    X_cadw = _extract(out_df)
    max_diff = float(np.abs(X_cadw - X_sk).max())
    assert max_diff < 1e-9, (
        f"kernel={kernel}: transform max diff {max_diff:.2e}")
    r.release_train_state()


# ---- CW-7: n_components=None drops zero eigvals (sklearn parity) ----

def test_n_components_none_drops_zeros():
    from sklearn.decomposition import KernelPCA
    rng = np.random.RandomState(4)
    X = rng.randn(30, 3)
    df = _mk_df(X)
    kp = KernelPCA(n_components=None, kernel="linear").fit(X)
    X_sk = kp.fit_transform(X)
    r = fit_kernel_pca(df, ["x0", "x1", "x2"], n_components=None,
                        kernel="linear",
                        memory_guard=MemoryGuard(max_memory_mb=500))
    # k_eff equals # nonzero eigenvalues
    k_eff_sk = int((kp.eigenvalues_ > 0).sum())
    assert r.n_components == k_eff_sk, (
        f"n_components={r.n_components}, sklearn k_eff={k_eff_sk}")
    X_cadw = r.fit_transform()
    max_diff = float(np.abs(X_cadw - X_sk).max())
    assert max_diff < 1e-9, f"max diff {max_diff:.2e}"
    r.release_train_state()


# ---- CW-8: remove_zero_eig ----

def test_remove_zero_eig():
    """remove_zero_eig drops zero eigenvalues explicitly."""
    rng = np.random.RandomState(5)
    # Low-rank data: 10 samples, rank-2 (last coord is linear combo)
    base = rng.randn(10, 2)
    X = np.column_stack([base[:, 0], base[:, 1],
                         base[:, 0] + base[:, 1]])
    df = _mk_df(X)
    r = fit_kernel_pca(df, ["x0", "x1", "x2"], n_components=None,
                        kernel="linear", remove_zero_eig=True,
                        memory_guard=MemoryGuard(max_memory_mb=500))
    # Rank-2 -> k_eff = 2
    assert r.n_components == 2, (
        f"remove_zero_eig: k_eff={r.n_components} (expected 2)")
    r.release_train_state()


# ---- CW-9: n > 1e4 raises intrinsic bound ----

def test_n_above_intrinsic_bound_raises():
    """n > 1e4 raises MemoryLimitExceeded (O(n^2) intrinsic)."""
    # Fake df without actually building 1e4 rows in memory
    class _FakeDF:
        columns = ["x0", "x1"]
        _memory_guard = MemoryGuard(max_memory_mb=500)
        _spill_dir = "/tmp"
        def count(self):
            return 10001
        def _iter_filtered_batches(self, batch_size=None):
            return iter([])
    with pytest.raises(MemoryLimitExceeded, match="intrinsic bound"):
        fit_kernel_pca(_FakeDF(), ["x0", "x1"], n_components=2,
                        kernel="linear")


# ---- CW-10: release_train_state + transform raises ----

def test_release_idempotent_and_predict_raises():
    rng = np.random.RandomState(6)
    X = rng.randn(30, 3)
    df = _mk_df(X)
    r = fit_kernel_pca(df, ["x0", "x1", "x2"], n_components=2,
                        kernel="linear",
                        memory_guard=MemoryGuard(max_memory_mb=500))
    r.release_train_state()
    assert r.X_fit is None
    assert r.eigenvectors is None
    r.release_train_state()  # idempotent
    with pytest.raises(RuntimeError, match="released"):
        r.transform(df)
    with pytest.raises(RuntimeError, match="released"):
        r.fit_transform()


# ---- CW-11: gamma=None -> 1/d (sklearn parity) ----

def test_gamma_none_default():
    from sklearn.decomposition import KernelPCA
    rng = np.random.RandomState(7)
    X = rng.randn(40, 5)
    df = _mk_df(X)
    kp = KernelPCA(n_components=2, kernel="rbf").fit(X)
    r = fit_kernel_pca(
        df, [f"x{j}" for j in range(5)], n_components=2,
        kernel="rbf", gamma=None,
        memory_guard=MemoryGuard(max_memory_mb=500))
    assert abs(r.gamma - 1.0 / 5) < 1e-12, (
        f"gamma=None should be 1/d=0.2; got {r.gamma}")
    X_sk = kp.fit_transform(X)
    X_cadw = r.fit_transform()
    max_diff = float(np.abs(X_cadw - X_sk).max())
    assert max_diff < 1e-9
    r.release_train_state()


# ---- CW-12: validation ----

def test_empty_X_cols_raises():
    X = np.random.RandomState(8).randn(10, 2)
    df = _mk_df(X)
    with pytest.raises(ValueError, match="X_cols is empty"):
        fit_kernel_pca(df, [], n_components=2,
                       memory_guard=MemoryGuard(max_memory_mb=500))


def test_bad_kernel_raises():
    X = np.random.RandomState(9).randn(10, 2)
    df = _mk_df(X)
    with pytest.raises(ValueError, match="kernel must be one of"):
        fit_kernel_pca(df, ["x0", "x1"], n_components=2,
                       kernel="sigmoid",
                       memory_guard=MemoryGuard(max_memory_mb=500))


def test_bad_gamma_raises():
    X = np.random.RandomState(10).randn(10, 2)
    df = _mk_df(X)
    with pytest.raises(ValueError, match="gamma must be"):
        fit_kernel_pca(df, ["x0", "x1"], n_components=2,
                       kernel="rbf", gamma=-1.0,
                       memory_guard=MemoryGuard(max_memory_mb=500))


def test_nan_X_raises():
    X = np.random.RandomState(11).randn(10, 2)
    X[3, 0] = np.nan
    df = _mk_df(X)
    with pytest.raises(ValueError, match="X contains NaN"):
        fit_kernel_pca(df, ["x0", "x1"], n_components=2,
                       kernel="linear",
                       memory_guard=MemoryGuard(max_memory_mb=500))
