"""Component-wise verification for fit_nystroem_kernel_map (DEC-033, v0.9.202).

sklearn.kernel_approximation.Nystroem parity (bit-identical for
n <= permutation_limit, same seed). State O(model): O(k*d + k^2).
"""
from __future__ import annotations

import warnings

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard
from cadw.ml import fit_nystroem_kernel_map, NystroemResult


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


def _warmup():
    X = np.random.RandomState(999).randn(50, 3)
    df = _mk_df(X)
    r = fit_nystroem_kernel_map(df, ["x0", "x1", "x2"],
                                 n_components=10, kernel="rbf",
                                 gamma=1.0, random_state=0,
                                 memory_guard=MemoryGuard(max_memory_mb=500))
    r.release_train_state()


_warmup()


# ---- CW-1: state_nbytes bit-exact ----

@pytest.mark.parametrize("k,d", [(10, 3), (30, 5), (50, 4)])
def test_state_nbytes_exact(k, d):
    """state = 256 + components_.nbytes + normalization_.nbytes + idx.nbytes."""
    rng = np.random.RandomState(0)
    X = rng.randn(200, d)
    df = _mk_df(X)
    r = fit_nystroem_kernel_map(df, [f"x{j}" for j in range(d)],
                                 n_components=k, kernel="rbf",
                                 gamma=1.0, random_state=0,
                                 memory_guard=MemoryGuard(max_memory_mb=500))
    expected = (256
                + int(r.components_.nbytes)
                + int(r.normalization_.nbytes)
                + int(r.component_indices_.nbytes))
    assert r.state_nbytes() == expected, (
        f"k={k} d={d}: state={r.state_nbytes()}, expected {expected}")
    r.release_train_state()


# ---- CW-2: sklearn parity fit_transform (3 kernels) ----

@pytest.mark.parametrize("kernel,params", [
    ("linear", {}),
    ("rbf", {"gamma": 0.5}),
    ("poly", {"degree": 3, "gamma": 0.5, "coef0": 1.0}),
])
def test_sklearn_parity_fit_transform(kernel, params):
    from sklearn.kernel_approximation import Nystroem
    rng = np.random.RandomState(1)
    X = rng.randn(200, 3)
    df = _mk_df(X)
    nys = Nystroem(n_components=30, kernel=kernel, random_state=42,
                    **params)
    X_sk = nys.fit_transform(X)
    r = fit_nystroem_kernel_map(
        df, ["x0", "x1", "x2"], n_components=30, kernel=kernel,
        gamma=params.get("gamma"),
        degree=params.get("degree", 3),
        coef0=params.get("coef0", 1.0),
        random_state=42,
        memory_guard=MemoryGuard(max_memory_mb=500))
    X_cadw = r.fit_transform(df)
    diff = float(np.abs(X_cadw - X_sk).max())
    assert diff < 1e-12, (
        f"kernel={kernel}: fit_transform max diff {diff:.2e}")
    r.release_train_state()


# ---- CW-3: sklearn parity transform (3 kernels) ----

@pytest.mark.parametrize("kernel,params", [
    ("linear", {}),
    ("rbf", {"gamma": 0.5}),
    ("poly", {"degree": 2, "gamma": 0.5, "coef0": 0.5}),
])
def test_sklearn_parity_transform(kernel, params):
    from sklearn.kernel_approximation import Nystroem
    rng = np.random.RandomState(2)
    X_train = rng.randn(150, 3)
    X_new = rng.randn(20, 3)
    df_train = _mk_df(X_train)
    df_new = _mk_df(X_new)
    nys = Nystroem(n_components=25, kernel=kernel, random_state=42,
                    **params).fit(X_train)
    X_sk = nys.transform(X_new)
    r = fit_nystroem_kernel_map(
        df_train, ["x0", "x1", "x2"], n_components=25, kernel=kernel,
        gamma=params.get("gamma"),
        degree=params.get("degree", 3),
        coef0=params.get("coef0", 1.0),
        random_state=42,
        memory_guard=MemoryGuard(max_memory_mb=500))
    out_df = r.transform(df_new)
    X_cadw = _extract(out_df)
    diff = float(np.abs(X_cadw - X_sk).max())
    assert diff < 1e-12, f"kernel={kernel}: transform max diff {diff:.2e}"
    r.release_train_state()


# ---- CW-4: reproducibility (same seed = same landmarks) ----

def test_reproducibility_same_seed():
    rng = np.random.RandomState(3)
    X = rng.randn(100, 3)
    df = _mk_df(X)
    r1 = fit_nystroem_kernel_map(df, ["x0", "x1", "x2"],
                                   n_components=20, kernel="rbf",
                                   gamma=1.0, random_state=42,
                                   memory_guard=MemoryGuard(max_memory_mb=500))
    r2 = fit_nystroem_kernel_map(df, ["x0", "x1", "x2"],
                                   n_components=20, kernel="rbf",
                                   gamma=1.0, random_state=42,
                                   memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(r1.component_indices_, r2.component_indices_)
    np.testing.assert_array_equal(r1.components_, r2.components_)
    np.testing.assert_allclose(r1.normalization_, r2.normalization_,
                                atol=1e-14)
    r1.release_train_state()
    r2.release_train_state()

    # Different seed -> different landmarks
    r3 = fit_nystroem_kernel_map(df, ["x0", "x1", "x2"],
                                   n_components=20, kernel="rbf",
                                   gamma=1.0, random_state=99,
                                   memory_guard=MemoryGuard(max_memory_mb=500))
    assert not np.array_equal(r1.component_indices_ if r1.components_ is not None
                              else r2.component_indices_,
                              r3.component_indices_)
    r3.release_train_state()


# ---- CW-5: n_components > n -> warn + k_eff = n ----

def test_n_components_gt_n_warns():
    rng = np.random.RandomState(4)
    X = rng.randn(50, 2)
    df = _mk_df(X)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = fit_nystroem_kernel_map(df, ["x0", "x1"],
                                      n_components=100, kernel="rbf",
                                      gamma=1.0, random_state=0,
                                      memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.n_components == 50, (
        f"k_eff={r.n_components} (expected 50 = min(100, 50))")
    assert any("n_components" in str(x.message) for x in w), (
        f"expected warning; got {[str(x.message) for x in w]}")
    r.release_train_state()


# ---- CW-6: reservoir fallback (approximation disclosed) ----

def test_reservoir_fallback_approximation_disclosed():
    rng = np.random.RandomState(5)
    X = rng.randn(500, 2)
    df = _mk_df(X)
    r = fit_nystroem_kernel_map(df, ["x0", "x1"],
                                  n_components=20, kernel="rbf",
                                  gamma=1.0, random_state=0,
                                  _permutation_limit=100,
                                  memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.approximation is not None
    assert r.approximation[0].type == "subsample"
    assert r.approximation[0].axis == "n"
    assert r.n_components == 20
    # transform still works
    out_df = r.transform(df)
    assert out_df.columns
    r.release_train_state()


# ---- CW-7: release + transform raises ----

def test_release_idempotent_and_transform_raises():
    rng = np.random.RandomState(6)
    X = rng.randn(100, 3)
    df = _mk_df(X)
    r = fit_nystroem_kernel_map(df, ["x0", "x1", "x2"],
                                  n_components=20, kernel="rbf",
                                  gamma=1.0, random_state=0,
                                  memory_guard=MemoryGuard(max_memory_mb=500))
    r.release_train_state()
    assert r.components_ is None
    r.release_train_state()  # idempotent
    with pytest.raises(RuntimeError, match="released"):
        r.transform(df)
    with pytest.raises(RuntimeError, match="released"):
        r.fit_transform(df)


# ---- CW-8: validation ----

def test_empty_X_cols_raises():
    X = np.random.RandomState(7).randn(50, 2)
    df = _mk_df(X)
    with pytest.raises(ValueError, match="X_cols is empty"):
        fit_nystroem_kernel_map(df, [], n_components=10, kernel="rbf",
                                 memory_guard=MemoryGuard(max_memory_mb=500))


def test_bad_kernel_raises():
    X = np.random.RandomState(8).randn(50, 2)
    df = _mk_df(X)
    with pytest.raises(ValueError, match="kernel must be one of"):
        fit_nystroem_kernel_map(df, ["x0", "x1"], n_components=10,
                                 kernel="chi2",
                                 memory_guard=MemoryGuard(max_memory_mb=500))


def test_bad_gamma_raises():
    X = np.random.RandomState(9).randn(50, 2)
    df = _mk_df(X)
    with pytest.raises(ValueError, match="gamma must be"):
        fit_nystroem_kernel_map(df, ["x0", "x1"], n_components=10,
                                 kernel="rbf", gamma=-1.0,
                                 memory_guard=MemoryGuard(max_memory_mb=500))


def test_nan_X_raises():
    X = np.random.RandomState(10).randn(50, 2)
    X[3, 0] = np.nan
    df = _mk_df(X)
    with pytest.raises(ValueError, match="X contains NaN"):
        fit_nystroem_kernel_map(df, ["x0", "x1"], n_components=10,
                                 kernel="rbf", gamma=1.0, random_state=0,
                                 memory_guard=MemoryGuard(max_memory_mb=500))
