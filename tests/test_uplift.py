"""Tests for D.3.3 uplift trees + Qini (v0.9.37)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import uplift as upl
from cadw.ml.uplift import fit_uplift, UpliftModel, qini_curve


def _truth(n=2500, seed=0, d=3):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    e = 1.0 / (1.0 + np.exp(-(0.5 * X[:, 0] - 0.3 * X[:, 1])))
    t = (rng.uniform(size=n) < e).astype(float)
    lp = -0.5 + 0.8 * X[:, 2] + 0.6 * t
    p = 1.0 / (1.0 + np.exp(-lp))
    y = (rng.uniform(size=n) < p).astype(float)
    return X, t, y


def _df(X, t, y):
    d = {f"x{j}": pa.array(X[:, j]) for j in range(X.shape[1])}
    d["t"] = pa.array(t); d["y"] = pa.array(y)
    return CadwDataFrame.from_table(pa.table(d))


def test_synthetic_uplift_follows_treatment_effect():
    X, t, y = _truth(seed=0)
    m = fit_uplift(_df(X, t, y), ["x0", "x1", "x2"], "t", "y", n_estimators=30)
    u = m.uplift(_df(X, t, y))
    assert u.shape == (X.shape[0],)
    assert np.corrcoef(u, X[:, 2])[0, 1] > 0.25


def test_qini_coefficient_positive_on_informative_model():
    X, t, y = _truth(seed=1)
    df = _df(X, t, y)
    m = fit_uplift(df, ["x0", "x1", "x2"], "t", "y", n_estimators=30)
    _, _, coef = m.qini(df, "y")
    assert coef > 0.0


def test_qini_curve_reference_random_targeting():
    rng = np.random.RandomState(2)
    n = 2000
    t = (rng.uniform(size=n) < 0.5).astype(float)
    y = (rng.uniform(size=n) < 0.3).astype(float)
    score = rng.normal(size=n)
    f, q, coef = qini_curve(y, t, score)
    assert len(f) == len(q)
    assert f[0] > 0 and f[-1] <= 1.0
    assert abs(coef) < 0.05


def test_determinism_same_seed():
    X, t, y = _truth(seed=3, n=1200)
    df = _df(X, t, y)
    m1 = fit_uplift(df, ["x0", "x1"], "t", "y", n_estimators=15)
    m2 = fit_uplift(df, ["x0", "x1"], "t", "y", n_estimators=15)
    assert np.allclose(m1.uplift(df), m2.uplift(df))


def test_qini_curve_edge_and_guards():
    with pytest.raises(ValueError, match="equal non-zero"):
        qini_curve(np.zeros(0), np.zeros(0), np.zeros(0))
    y = np.array([1.0, 0.0, 1.0]); t = np.ones(3); s = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="both treated and control"):
        qini_curve(y, t, s)


def test_fit_guards_treatment_and_outcome():
    X, t, y = _truth(n=100, seed=4)
    tb = t.copy(); tb[0] = 2.0
    with pytest.raises(ValueError, match="treatment must be binary"):
        fit_uplift(_df(X, tb, y), ["x0"], "t", "y")
    yb = y.copy(); yb[0] = 2.0
    with pytest.raises(ValueError, match="outcome must be binary"):
        fit_uplift(_df(X, t, yb), ["x0"], "t", "y")


def test_fit_guards_missing_col_and_empty():
    X, t, y = _truth(n=100, seed=5)
    with pytest.raises(KeyError):
        fit_uplift(_df(X, t, y), ["zzz"], "t", "y")
    with pytest.raises(ValueError, match="non-empty"):
        fit_uplift(_df(X, t, y), [], "t", "y")


def test_ram_guard_raises_when_tight():
    X, t, y = _truth(n=1500, seed=6)
    mg = MemoryGuard(max_memory_bytes=1024)
    with pytest.raises(MemoryLimitExceeded):
        fit_uplift(_df(X, t, y), ["x0", "x1"], "t", "y",
                   n_estimators=5, memory_guard=mg)


def test_multi_batch_consistent(tmp_path):
    import pyarrow.parquet as pq
    X, t, y = _truth(n=800, seed=7)
    tbl = pa.table({**{f"x{j}": pa.array(X[:, j]) for j in range(3)},
                    "t": pa.array(t), "y": pa.array(y)})
    path = str(tmp_path / "mb.parquet")
    pq.write_table(tbl, path, row_group_size=200)
    df_mb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100)
    df_one = _df(X, t, y)
    m1 = fit_uplift(df_mb, ["x0", "x1", "x2"], "t", "y", n_estimators=10)
    m2 = fit_uplift(df_one, ["x0", "x1", "x2"], "t", "y", n_estimators=10)
    assert np.allclose(m1.uplift(df_one), m2.uplift(df_one), atol=1e-9)


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(upl)
    assert "check_batch" in src and "extra_factor=2.0" in src
    assert "to_pylist" not in src
    assert "log_fit" in src
    assert "import pandas" not in src
