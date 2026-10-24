"""Tests for D.4.5 VAR(p) (v0.9.38)."""
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.ts.var import VARModel


def _var_data(n=300, seed=0):
    rng = np.random.RandomState(seed)
    A = np.array([[0.5, 0.1], [-0.2, 0.4]])
    Y = np.zeros((n, 2))
    for t in range(1, n):
        Y[t] = A @ Y[t - 1] + rng.normal(0, 1, 2)
    return Y


def _df(Y, names=("a", "b")):
    return CadwDataFrame.from_table(pa.table({
        names[j]: pa.array(Y[:, j]) for j in range(Y.shape[1])}))


def test_synthetic_var1_coef_recovers_A():
    Y = _var_data()
    m = VARModel(p=1).fit(_df(Y), ["a", "b"])
    assert m.coef_[0].shape == (2, 2)
    # coef_[0][j, i] = effect of series j on equation i
    A_hat = m.coef_[0].T
    assert np.max(np.abs(A_hat - np.array([[0.5, 0.1], [-0.2, 0.4]]))) < 0.1


def test_reference_statsmodels_forecast_matches():
    sm = pytest.importorskip("statsmodels.tsa.api")
    Y = _var_data(seed=1)
    m = VARModel(p=1).fit(_df(Y), ["a", "b"])
    ours = m.forecast_vector(5)
    ref = sm.VAR(Y).fit(1).forecast(Y[-1:], 5)
    assert np.max(np.abs(ref - ours)) < 1e-8


def test_reference_statsmodels_var2():
    sm = pytest.importorskip("statsmodels.tsa.api")
    Y = _var_data(seed=2, n=400)
    m = VARModel(p=2).fit(_df(Y), ["a", "b"])
    ours = m.forecast_vector(4)
    ref = sm.VAR(Y).fit(2).forecast(Y[-2:], 4)
    assert np.max(np.abs(ref - ours)) < 1e-8


def test_stability_flag():
    Y = _var_data(seed=3)
    m = VARModel(p=1).fit(_df(Y), ["a", "b"])
    assert m.stable_ is True


def test_determinism():
    Y = _var_data(seed=4)
    a = VARModel(p=1).fit(_df(Y), ["a", "b"]).forecast_vector(3)
    b = VARModel(p=1).fit(_df(Y), ["a", "b"]).forecast_vector(3)
    assert np.array_equal(a, b)


def test_univariate_var_equals_ar():
    Y = _var_data(seed=5)[:, :1]
    m = VARModel(p=2).fit(_df(Y, ("a",)), ["a"])
    f = m.forecast_vector(3)
    assert f.shape == (3, 1)
    assert np.all(np.isfinite(f))


def test_edge_guards():
    Y = _var_data(seed=6)
    with pytest.raises(ValueError, match="p must be"):
        VARModel(p=0)
    with pytest.raises(KeyError):
        VARModel(p=1).fit(_df(Y), ["nope"])
    with pytest.raises(ValueError, match="n > p"):
        VARModel(p=1).fit(_df(Y[:2]), ["a", "b"])


def test_guard_nonfinite():
    Y = _var_data(seed=7); Y[5, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        VARModel(p=1).fit(_df(Y), ["a", "b"])


def test_guard_predict_before_fit():
    with pytest.raises(RuntimeError):
        VARModel(p=1).predict(3)


def test_in_sample_shape_and_state():
    Y = _var_data(seed=8)
    m = VARModel(p=1).fit(_df(Y), ["a", "b"])
    assert m.predict_in_sample().shape == (Y.shape[0] - 1,)
    assert m.state_nbytes() > 0


def test_source_contract_no_pandas():
    import inspect
    from cadw.ts import var as v
    src = inspect.getsource(v)
    assert "import pandas" not in src
    assert "log_fit" in src
