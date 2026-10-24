"""Tests for D.2.2 implicit ALS (v0.9.36)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import als as als_mod
from cadw.ml.als import fit_als, ALSModel


def _data(U=60, I=40, k=4, seed=0):
    rng = np.random.RandomState(seed)
    Xu = rng.normal(size=(U, k)); Yi = rng.normal(size=(I, k))
    us, its = [], []
    for u in range(U):
        for i in range(I):
            if Xu[u] @ Yi[i] > 0.3:
                us.append(u); its.append(i)
    df = CadwDataFrame.from_table(pa.table({
        "u": pa.array(us, type=pa.int64()),
        "i": pa.array(its, type=pa.int64())}))
    return df, np.array(us), np.array(its)


def test_smoke_shapes_and_state():
    df, _, _ = _data()
    m = fit_als(df, "u", "i", n_factors=4, n_iter=5)
    assert isinstance(m, ALSModel)
    assert m.user_factors_.shape == (60, 4)
    assert m.item_factors_.shape == (40, 4)
    assert m.state_nbytes() == (60 + 40) * 4 * 8


def test_objective_decreases():
    df, us, its = _data()
    losses = []
    import cadw.ml._logging as lg
    orig = lg.log_fit
    def spy(model, **kw):
        if kw.get("loss") is not None:
            losses.append(kw["loss"])
        return orig(model, **kw)
    als_mod.log_fit = spy
    try:
        fit_als(df, "u", "i", n_factors=4, n_iter=8, reg=0.05, alpha=5.0)
    finally:
        als_mod.log_fit = orig
    assert len(losses) == 8
    assert losses[-1] < losses[0]


def test_recommend_hits_observed_items():
    df, us, its = _data()
    m = fit_als(df, "u", "i", n_factors=4, n_iter=12, reg=0.05, alpha=5.0)
    obs = set(i for u, i in zip(us, its) if u == 0)
    rec = [i for i, _ in m.recommend(0, n=5)]
    assert len(set(rec) & obs) >= 2, f"top5 {rec} vs observed {sorted(obs)}"


def test_reference_vs_implicit_closed_form_one_sweep():
    """One user update must solve the same k x k system as implicit's
    closed form. Built with a 1-user matrix so the system is explicit."""
    rng = np.random.RandomState(3)
    Y = rng.normal(size=(5, 3))
    items = np.array([0, 2, 4])
    w = np.array([1.0, 2.0, 1.5])
    alpha, reg = 40.0, 0.1
    got = als_mod._solve_user(Y.T @ Y, Y, items, w, alpha, reg)
    # explicit reference: A = YtY + sum (c-1) yyT + reg I ; b = sum c y
    k = 3
    A = Y.T @ Y + reg * np.eye(k)
    b = np.zeros(k)
    for idx, wi in zip(items, w):
        c = 1.0 + alpha * wi
        A += (c - 1.0) * np.outer(Y[idx], Y[idx])
        b += c * Y[idx]
    exp = np.linalg.solve(A, b)
    assert np.allclose(got, exp, atol=1e-10)


def test_determinism_same_seed():
    df, _, _ = _data()
    a = fit_als(df, "u", "i", n_factors=4, n_iter=4, seed=7)
    b = fit_als(df, "u", "i", n_factors=4, n_iter=4, seed=7)
    assert np.array_equal(a.user_factors_, b.user_factors_)
    assert np.array_equal(a.item_factors_, b.item_factors_)


def test_multi_batch_invariance():
    import tempfile, pyarrow.parquet as pq
    df, us, its = _data()
    tbl = pa.table({"u": pa.array(us, type=pa.int64()),
                    "i": pa.array(its, type=pa.int64())})
    path = tempfile.mktemp(suffix=".parquet")
    pq.write_table(tbl, path, row_group_size=13)
    d1 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=1)
    d2 = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100000)
    a = fit_als(d1, "u", "i", n_factors=4, n_iter=3, seed=1)
    b = fit_als(d2, "u", "i", n_factors=4, n_iter=3, seed=1)
    assert np.allclose(a.user_factors_, b.user_factors_, atol=1e-12)


def test_ram_state_independent_of_interactions():
    df_s, _, _ = _data(U=30, I=20)
    df_l, _, _ = _data(U=60, I=40)
    a = fit_als(df_s, "u", "i", n_factors=4, n_iter=2)
    b = fit_als(df_l, "u", "i", n_factors=4, n_iter=2)
    assert a.state_nbytes() == (30 + 20) * 4 * 8
    assert b.state_nbytes() == (60 + 40) * 4 * 8


def test_guard_large_state():
    # (U+I)*k*8 = 400*256*8 = 819 200 B > state limit of a 1 MB budget
    df, _, _ = _data(U=200, I=200, k=8)
    mg = MemoryGuard(max_memory_mb=1)
    with pytest.raises(MemoryLimitExceeded):
        fit_als(df, "u", "i", n_factors=256, n_iter=2, memory_guard=mg)


def test_errors():
    df, _, _ = _data()
    with pytest.raises(KeyError):
        fit_als(df, "nope", "i")
    with pytest.raises(ValueError):
        fit_als(df, "u", "i", n_factors=0)
    empty = CadwDataFrame.from_table(pa.table({
        "u": pa.array([], type=pa.int64()),
        "i": pa.array([], type=pa.int64())}))
    with pytest.raises(ValueError):
        fit_als(empty, "u", "i")


def test_source_contract():
    src = inspect.getsource(als_mod)
    assert "Hu, Koren, Volinsky" in src
    assert "external_sort" in src
    assert "import pandas" not in src
    assert "check_state" in inspect.getsource(als_mod.fit_als)
    assert "log_fit" in inspect.getsource(als_mod.fit_als)
    for line in src.splitlines():
        if "check_batch(" in line:
            assert "extra_factor=" in line
