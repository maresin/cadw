"""Tests for BVAR (D2, v0.9.89)."""
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.ts.bvar import BVARModel, _build_prior


def _var_data(n=300, seed=0, k=2):
    rng = np.random.RandomState(seed)
    A = np.array([[0.5, 0.1], [-0.2, 0.4]])[:k, :k]
    Y = np.zeros((n, k))
    for t in range(1, n):
        Y[t] = A @ Y[t - 1] + rng.normal(0, 1, k)
    return Y


def _df(Y, names=("a", "b")):
    return CadwDataFrame.from_table(pa.table({
        names[j]: pa.array(Y[:, j]) for j in range(Y.shape[1])}))


def _full_panel_bvar(Y, p, lambda1, lambda2, mu):
    """Manual full-panel BVAR — reference for streaming path."""
    n, k = Y.shape
    X = np.empty((n - p, 1 + k * p))
    X[:, 0] = 1.0
    for lag in range(1, p + 1):
        X[:, 1 + (lag - 1) * k:1 + lag * k] = Y[p - lag:n - lag]
    Yt = Y[p:]
    XTX = X.T @ X
    XTY = X.T @ Yt
    beta_prior, omega_inv = _build_prior(k, p, lambda1, lambda2, mu)
    omega_inv_matrix = np.diag(omega_inv)
    A = XTX + omega_inv_matrix
    b = XTY + omega_inv_matrix @ beta_prior
    beta_post = np.linalg.solve(A, b)
    return beta_post


def test_bvar_shape_and_stable():
    Y = _var_data()
    m = BVARModel(p=2).fit(_df(Y), ["a", "b"])
    assert m.beta_.shape == (5, 2)  # 1 + 2*2 = 5
    assert m.stable_ is True
    assert m.n_obs_ == Y.shape[0] - 2


def test_bvar_streaming_matches_full_panel():
    Y = _var_data(seed=2)
    m = BVARModel(p=2).fit(_df(Y), ["a", "b"])
    ref = _full_panel_bvar(Y, p=2, lambda1=0.2, lambda2=1.0, mu=100.0)
    assert np.allclose(m.beta_, ref, atol=1e-8), (
        f"streaming vs full-panel BVAR mismatch: max diff "
        f"{np.max(np.abs(m.beta_ - ref))}")


def test_bvar_determinism():
    Y = _var_data(seed=3)
    a = BVARModel(p=2).fit(_df(Y), ["a", "b"])
    b = BVARModel(p=2).fit(_df(Y), ["a", "b"])
    assert np.array_equal(a.beta_, b.beta_)


def test_bvar_multi_batch_invariance(tmp_path):
    import pyarrow.parquet as pq
    Y = _var_data(seed=4)
    tbl = pa.table({"a": pa.array(Y[:, 0]), "b": pa.array(Y[:, 1])})
    path = str(tmp_path / "d.parquet")
    pq.write_table(tbl, path, row_group_size=13)
    d1 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=1)
    d2 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100000)
    a = BVARModel(p=2).fit(d1, ["a", "b"])
    b = BVARModel(p=2).fit(d2, ["a", "b"])
    assert np.allclose(a.beta_, b.beta_, atol=1e-8)


def test_bvar_prior_shrinkage():
    """Larger lambda1 → coefs closer to prior (own lag1=1, else 0)."""
    Y = _var_data(seed=5)
    tight = BVARModel(p=2, lambda1=0.01).fit(_df(Y), ["a", "b"])
    loose = BVARModel(p=2, lambda1=10.0).fit(_df(Y), ["a", "b"])
    # Prior (eq 'a'): [intercept=0, own_lag1=1.0, cross_lag1=0, own_lag2=0, cross_lag2=0]
    prior_eq0 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    tight_dist = np.abs(tight.beta_[:, 0] - prior_eq0).sum()
    loose_dist = np.abs(loose.beta_[:, 0] - prior_eq0).sum()
    assert tight_dist < loose_dist, (
        f"tight lambda should shrink closer to prior: "
        f"tight_dist={tight_dist:.3f}, loose_dist={loose_dist:.3f}")


def test_bvar_forecast_finite():
    Y = _var_data(seed=6)
    m = BVARModel(p=2).fit(_df(Y), ["a", "b"])
    fc = m.forecast_vector(10)
    assert fc.shape == (10, 2)
    assert np.all(np.isfinite(fc))
    # 1-D predict matches first column
    f1 = m.predict(10)
    assert np.allclose(f1, fc[:, 0])


def test_bvar_state_bounded():
    Y_small = _var_data(n=200, seed=7)
    Y_large = _var_data(n=5000, seed=7)
    a = BVARModel(p=2).fit(_df(Y_small), ["a", "b"])
    b = BVARModel(p=2).fit(_df(Y_large), ["a", "b"])
    assert a.state_nbytes() == b.state_nbytes()  # independent of n
    assert a.state_nbytes() < 4096


def test_bvar_edge_guards():
    Y = _var_data(seed=8)
    with pytest.raises(ValueError, match="p must be"):
        BVARModel(p=0)
    with pytest.raises(ValueError, match="lambda1"):
        BVARModel(p=2, lambda1=0)
    with pytest.raises(KeyError):
        BVARModel(p=2).fit(_df(Y), ["nope"])
    bad = Y.copy(); bad[5, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        BVARModel(p=2).fit(_df(bad), ["a", "b"])
    with pytest.raises(ValueError, match="insufficient"):
        BVARModel(p=2).fit(_df(Y[:3]), ["a", "b"])


def test_bvar_source_contract_no_full_materialization():
    """Streaming: must NOT call _extract_series (full numpy array)."""
    import inspect
    from cadw.ts import bvar as bvar_mod
    src = inspect.getsource(bvar_mod)
    assert "_extract_series" not in src, (
        "BVAR must stream via _iter_filtered_batches, not materialize")
    assert "_iter_filtered_batches" in src
    assert "check_state" in src and "check_batch" in src
    assert "log_fit" in src
