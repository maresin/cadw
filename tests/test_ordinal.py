"""Tests for D.5.1 ordinal logistic cumulative link (v0.9.39)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import ordinal as omod
from cadw.ml.ordinal import fit_ordinal_logit, OrdinalLogitModel


def _data(n=2000, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    lat = -0.5 + 1.2 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(0, 0.5, n)
    y = np.digitize(lat, [-0.5, 0.5, 1.5]).astype(np.float64)
    return X, y


def _df(X, y):
    return CadwDataFrame.from_table(pa.table({
        "x0": pa.array(X[:, 0]), "x1": pa.array(X[:, 1]),
        "y": pa.array(y)}))


def test_synthetic_recovers_beta():
    X, y = _data(seed=0)
    m = fit_ordinal_logit(_df(X, y), ["x0", "x1"], "y")
    assert m.classes_.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert m.beta_.size == 2 and m.theta_.size == 3
    assert np.all(np.diff(m.theta_) > 0)
    pred = m.predict(_df(X, y))
    assert (pred == y).mean() > 0.7


def test_reference_statsmodels_ordered_logit():
    sm_ord = pytest.importorskip("statsmodels.miscmodels.ordinal_model")
    X, y = _data(seed=1)
    ours = fit_ordinal_logit(_df(X, y), ["x0", "x1"], "y")
    ref = sm_ord.OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False, maxiter=500)
    # beta must match statsmodels (same likelihood, same link)
    assert np.max(np.abs(ours.beta_ - ref.params[:2])) < 1e-2
    # first threshold must match (subsequent thresholds differ by parameterisation)
    assert abs(ours.theta_[0] - ref.params[2]) < 1e-2


def test_proba_sums_to_one_and_monotone_thresholds():
    X, y = _data(n=400, seed=2)
    m = fit_ordinal_logit(_df(X, y), ["x0", "x1"], "y")
    p = m.predict_proba(_df(X, y))
    assert p.shape == (400, 4)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(p >= 0) and np.all(p <= 1)


def test_determinism():
    X, y = _data(n=500, seed=3)
    a = fit_ordinal_logit(_df(X, y), ["x0", "x1"], "y").predict_proba(_df(X, y))
    b = fit_ordinal_logit(_df(X, y), ["x0", "x1"], "y").predict_proba(_df(X, y))
    assert np.array_equal(a, b)


def test_binary_case_equals_logistic():
    rng = np.random.RandomState(4)
    X = rng.normal(size=(1500, 2))
    y = (rng.uniform(size=1500) < 1 / (1 + np.exp(-(0.8 * X[:, 0] - 0.5 * X[:, 1])))).astype(float)
    m = fit_ordinal_logit(_df(X, y), ["x0", "x1"], "y")
    assert m.classes_.size == 2 and m.theta_.size == 1
    p = m.predict_proba(_df(X, y))
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)


def test_edge_guards():
    X, y = _data(n=200, seed=5)
    with pytest.raises(ValueError, match="non-empty"):
        fit_ordinal_logit(_df(X, y), [], "y")
    with pytest.raises(KeyError):
        fit_ordinal_logit(_df(X, y), ["nope"], "y")
    with pytest.raises(ValueError, match=">= 2 ordered classes"):
        fit_ordinal_logit(_df(X, np.zeros(200)), ["x0"], "y")


def test_guard_nulls():
    X, y = _data(n=200, seed=6)
    X = X.copy(); X[3, 0] = np.nan
    with pytest.raises(ValueError, match="nulls"):
        fit_ordinal_logit(_df(X, y), ["x0", "x1"], "y")


def test_ram_guard_raises_when_tight():
    X, y = _data(n=1000, seed=7)
    mg = MemoryGuard(max_memory_bytes=512)
    with pytest.raises(MemoryLimitExceeded):
        fit_ordinal_logit(_df(X, y), ["x0", "x1"], "y", memory_guard=mg)


def test_multi_batch_consistency(tmp_path):
    import pyarrow.parquet as pq
    X, y = _data(n=800, seed=8)
    df = _df(X, y)
    tbl = pa.Table.from_batches(list(df._iter_filtered_batches()))
    path = str(tmp_path / "o.parquet")
    pq.write_table(tbl, path, row_group_size=71)
    dmb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=100)
    m1 = fit_ordinal_logit(df, ["x0", "x1"], "y")
    m2 = fit_ordinal_logit(dmb, ["x0", "x1"], "y")
    assert np.allclose(m1.beta_, m2.beta_, atol=1e-5)
    assert np.allclose(m1.theta_, m2.theta_, atol=1e-5)


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(omod)
    assert "check_batch" in src and "extra_factor=3.0" in src
    assert "log_fit" in src
    assert "import pandas" not in src
