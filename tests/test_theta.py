"""Tests for D.4.4 Theta-method (v0.9.38)."""
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.ts.theta import ThetaForecaster


def _linear(n=80, seed=0):
    t = np.arange(n)
    return 10 + 0.4 * t + np.random.RandomState(seed).normal(0, 0.5, n)


def _df(y):
    return CadwDataFrame.from_table(pa.table({"y": pa.array(y)}))


def test_synthetic_linear_trend_accuracy():
    y = _linear()
    m = ThetaForecaster().fit(_df(y), "y")
    f = m.predict(8)
    truth = 10 + 0.4 * (80 + np.arange(8))
    mase = np.mean(np.abs(f - truth)) / np.mean(np.abs(np.diff(y)))
    assert mase < 3.0, mase


def test_reference_statsmodels_theta_close():
    sm_theta = pytest.importorskip("statsmodels.tsa.forecasting.theta")
    n, h = 80, 8
    full = 10 + 0.4 * np.arange(n + h) + np.random.RandomState(1).normal(0, 0.5, n + h)
    y, truth = full[:n], full[n:]
    ours = ThetaForecaster().fit(_df(y), "y").predict(h)
    ref = np.asarray(sm_theta.ThetaModel(y, period=1).fit().forecast(h))
    om = np.mean(np.abs(ours - truth))
    rm = np.mean(np.abs(ref - truth))
    assert om < rm * 1.6, (om, rm)


def test_flat_series_forecast_flat():
    y = np.full(50, 7.0)
    f = ThetaForecaster().fit(_df(y), "y").predict(5)
    assert np.allclose(f, 7.0)


def test_determinism():
    y = _linear(seed=2)
    a = ThetaForecaster().fit(_df(y), "y").predict(6)
    b = ThetaForecaster().fit(_df(y), "y").predict(6)
    assert np.array_equal(a, b)


def test_edge_guards():
    with pytest.raises(ValueError, match="at least 3"):
        ThetaForecaster().fit(_df(np.arange(2.0)), "y")
    yy = np.arange(10.0); yy[3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        ThetaForecaster().fit(_df(yy), "y")
    with pytest.raises(ValueError, match="seasonal_period"):
        ThetaForecaster(seasonal=True, seasonal_period=1)


def test_guard_predict_before_fit():
    with pytest.raises(RuntimeError):
        ThetaForecaster().predict(3)


def test_in_sample_shape_and_state():
    y = _linear(seed=3)
    m = ThetaForecaster().fit(_df(y), "y")
    assert m.predict_in_sample().shape == y.shape
    assert m.state_nbytes() > 0


def test_multi_batch_consistency(tmp_path):
    import pyarrow.parquet as pq
    y = _linear(seed=4)
    tbl = pa.table({"y": pa.array(y)})
    path = str(tmp_path / "th.parquet")
    pq.write_table(tbl, path, row_group_size=13)
    dmb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=16)
    f1 = ThetaForecaster().fit(_df(y), "y").predict(6)
    f2 = ThetaForecaster().fit(dmb, "y").predict(6)
    assert np.allclose(f1, f2, atol=1e-9)


def test_source_contract_no_pandas():
    import inspect
    from cadw.ts import theta as th
    src = inspect.getsource(th)
    assert "import pandas" not in src
    assert "log_fit" in src
