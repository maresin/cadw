"""Tests for D.3.2 T/S-learner + propensity (v0.9.37)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import meta_learners as mlm
from cadw.ml.meta_learners import fit_meta_learner, MetaLearnerModel, overlap_diagnostics


def _truth(n=3000, seed=0, d=3):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    e = 1.0 / (1.0 + np.exp(-(0.5 * X[:, 0] - 0.3 * X[:, 1])))
    t = (rng.uniform(size=n) < e).astype(float)
    tau = 1.0 + 0.5 * X[:, 2]
    y = 2.0 * X[:, 0] + tau * t + rng.normal(scale=0.1, size=n)
    return X, t, y, tau


def _df(X, t, y):
    d = {f"x{j}": pa.array(X[:, j]) for j in range(X.shape[1])}
    d["t"] = pa.array(t); d["y"] = pa.array(y)
    return CadwDataFrame.from_table(pa.table(d))


def _fit(seed=0, kind="t", n=3000, est=40):
    X, t, y, tau = _truth(n=n, seed=seed)
    m = fit_meta_learner(_df(X, t, y), ["x0", "x1", "x2"], "t", "y",
                         kind=kind, n_estimators=est)
    return m, _df(X, t, y), tau


def test_t_learner_synthetic_ate_and_cate():
    m, df, tau = _fit(seed=0, kind="t")
    ate = m.ate(df)
    assert abs(ate - float(np.mean(tau))) < 0.15
    cate = m.cate(df)
    assert cate.shape == (tau.shape[0],)
    assert np.corrcoef(cate, tau)[0, 1] > 0.80


def test_s_learner_synthetic_ate_and_cate():
    m, df, tau = _fit(seed=0, kind="s")
    ate = m.ate(df)
    assert abs(ate - float(np.mean(tau))) < 0.20
    assert np.corrcoef(m.cate(df), tau)[0, 1] > 0.70


def test_determinism_same_seed():
    m1, df, _ = _fit(seed=1, kind="t", n=1500, est=15)
    m2, _, _ = _fit(seed=1, kind="t", n=1500, est=15)
    assert np.allclose(m1.propensity_, m2.propensity_)
    assert np.allclose(m1.cate(df), m2.cate(df))


def test_ipw_ate_finite_and_close():
    m, df, tau = _fit(seed=2, kind="t", n=2000, est=20)
    ipw = m.ipw_ate(df, "y")
    assert np.isfinite(ipw)
    assert abs(ipw - float(np.mean(tau))) < 0.30


def test_overlap_diagnostics_contract():
    d = overlap_diagnostics(np.array([0.02, 0.5, 0.99, 0.5]))
    assert d["n"] == 4
    assert d["min"] == 0.02 and d["max"] == 0.99
    assert d["share_outside_005_095"] == 0.5
    assert np.isnan(overlap_diagnostics(np.array([]))["min"])


def test_edge_guards_nonbinary_and_small_arms():
    X, t, y, _ = _truth(n=50, seed=3)
    t3 = t.copy(); t3[0] = 2.0
    with pytest.raises(ValueError, match="binary"):
        fit_meta_learner(_df(X, t3, y), ["x0"], "t", "y")
    t1 = np.zeros(50); t1[0] = 1.0
    with pytest.raises(ValueError, match="per arm"):
        fit_meta_learner(_df(X, t1, y), ["x0"], "t", "y")


def test_guard_bad_kind_and_missing_col():
    X, t, y, _ = _truth(n=50, seed=4)
    with pytest.raises(ValueError, match="kind"):
        fit_meta_learner(_df(X, t, y), ["x0"], "t", "y", kind="z")
    with pytest.raises(KeyError):
        fit_meta_learner(_df(X, t, y), ["nope"], "t", "y")


def test_ram_guard_raises_when_tight():
    X, t, y, _ = _truth(n=2000, seed=5)
    mg = MemoryGuard(max_memory_bytes=1024)
    with pytest.raises(MemoryLimitExceeded):
        fit_meta_learner(_df(X, t, y), ["x0", "x1"], "t", "y",
                         n_estimators=5, memory_guard=mg)


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(mlm)
    assert "check_batch" in src and "extra_factor=2.0" in src
    assert "to_pylist" not in src
    assert "log_fit" in src
    assert "import pandas" not in src
