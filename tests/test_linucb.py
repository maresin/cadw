"""Tests for D.6.7 LinUCB disjoint linear bandit (v0.9.41)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import linucb as lmod
from cadw.ml.linucb import fit_linucb, LinUCBModel


def _data(n=2000, d=4, K=3, seed=0):
    rng = np.random.RandomState(seed)
    true_theta = rng.normal(size=(K, d))
    X = rng.normal(size=(n, d))
    arms = rng.randint(0, K, size=n)
    r = np.array([X[i] @ true_theta[arms[i]] for i in range(n)])
    r += 0.01 * rng.normal(size=n)
    return X, arms, r, true_theta


def _df(X, arms, r):
    d = {f"x{j}": pa.array(X[:, j]) for j in range(X.shape[1])}
    d["arm"] = pa.array(arms.astype(np.int64))
    d["r"] = pa.array(r)
    return CadwDataFrame.from_table(pa.table(d))


def test_analytic_ridge_reference():
    """theta_a must equal the closed-form ridge solution."""
    X, arms, r, _ = _data(n=500, seed=0)
    cols = [f"x{j}" for j in range(X.shape[1])]
    l2 = 1.0
    m = fit_linucb(_df(X, arms, r), cols, arm_col="arm", reward_col="r", l2=l2)
    th = m.theta()
    for a in range(3):
        mask = arms == a
        A = X[mask].T @ X[mask] + l2 * np.eye(X.shape[1])
        b = X[mask].T @ r[mask]
        ref = np.linalg.solve(A, b)
        assert np.allclose(th[a], ref, atol=1e-9)


def test_reward_recovery_high_signal():
    X, arms, r, true_theta = _data(n=4000, seed=1)
    cols = [f"x{j}" for j in range(X.shape[1])]
    m = fit_linucb(_df(X, arms, r), cols, arm_col="arm", reward_col="r")
    th = m.theta()
    err = np.max(np.abs(th - true_theta))
    assert err < 0.1, err


def test_ucb_score_formula():
    X, arms, r, _ = _data(n=600, seed=2)
    cols = [f"x{j}" for j in range(X.shape[1])]
    alpha = 0.7
    m = fit_linucb(_df(X, arms, r), cols, arm_col="arm", reward_col="r",
                   alpha=alpha)
    dfp = CadwDataFrame.from_table(pa.table({c: pa.array(X[:, j])
                                             for j, c in enumerate(cols)}))
    S = m.predict(dfp)
    th = m.theta()
    for a in range(3):
        Ainv = np.linalg.inv(m.A_[a])
        ref = X @ th[a] + alpha * np.sqrt(
            np.einsum("ij,ij->i", X @ Ainv, X))
        assert np.allclose(S[:, a], ref, atol=1e-8)


def test_choose_returns_max_ucb_arm():
    X, arms, r, _ = _data(n=400, seed=3)
    cols = [f"x{j}" for j in range(X.shape[1])]
    m = fit_linucb(_df(X, arms, r), cols, arm_col="arm", reward_col="r")
    dfp = CadwDataFrame.from_table(pa.table({c: pa.array(X[:, j])
                                             for j, c in enumerate(cols)}))
    S = m.predict(dfp)
    assert np.array_equal(m.choose(dfp), np.argmax(S, axis=1))


def test_determinism():
    X, arms, r, _ = _data(n=300, seed=4)
    cols = [f"x{j}" for j in range(X.shape[1])]
    a = fit_linucb(_df(X, arms, r), cols, arm_col="arm", reward_col="r").theta()
    b = fit_linucb(_df(X, arms, r), cols, arm_col="arm", reward_col="r").theta()
    assert np.array_equal(a, b)


def test_edge_guards():
    X, arms, r, _ = _data(n=100, seed=5)
    cols = [f"x{j}" for j in range(X.shape[1])]
    with pytest.raises(ValueError, match="alpha"):
        fit_linucb(_df(X, arms, r), cols, arm_col="arm", reward_col="r", alpha=0)
    with pytest.raises(ValueError, match="l2"):
        fit_linucb(_df(X, arms, r), cols, arm_col="arm", reward_col="r", l2=0)
    with pytest.raises(ValueError, match="non-empty"):
        fit_linucb(_df(X, arms, r), [], arm_col="arm", reward_col="r")
    with pytest.raises(KeyError):
        fit_linucb(_df(X, arms, r), cols, arm_col="nope", reward_col="r")
    neg = arms.copy(); neg[0] = -1
    with pytest.raises(ValueError, match="arm ids"):
        fit_linucb(_df(X, neg, r), cols, arm_col="arm", reward_col="r")


def test_guard_nulls():
    X, arms, r, _ = _data(n=100, seed=6)
    X = X.copy(); X[2, 1] = np.nan
    cols = [f"x{j}" for j in range(X.shape[1])]
    with pytest.raises(ValueError, match="nulls"):
        fit_linucb(_df(X, arms, r), cols, arm_col="arm", reward_col="r")


def test_ram_guard_raises_when_tight():
    X, arms, r, _ = _data(n=500, seed=7)
    cols = [f"x{j}" for j in range(X.shape[1])]
    mg = MemoryGuard(max_memory_bytes=1024)
    with pytest.raises(MemoryLimitExceeded):
        fit_linucb(_df(X, arms, r), cols, arm_col="arm", reward_col="r",
                   memory_guard=mg)


def test_multi_batch_consistency(tmp_path):
    import pyarrow.parquet as pq
    X, arms, r, _ = _data(n=700, seed=8)
    cols = [f"x{j}" for j in range(X.shape[1])]
    df = _df(X, arms, r)
    tbl = pa.Table.from_batches(list(df._iter_filtered_batches()))
    path = str(tmp_path / "l.parquet")
    pq.write_table(tbl, path, row_group_size=53)
    dmb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=64)
    a = fit_linucb(df, cols, arm_col="arm", reward_col="r")
    b = fit_linucb(dmb, cols, arm_col="arm", reward_col="r")
    assert np.allclose(a.A_, b.A_, atol=1e-9)
    assert np.allclose(a.b_, b.b_, atol=1e-9)


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(lmod)
    assert "check_batch" in src and "extra_factor=3.0" in src
    assert "log_fit" in src
    assert "import pandas" not in src
