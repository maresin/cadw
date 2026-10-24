"""Tests for D.4.3 ETS / Holt-Winters (v0.9.38)."""
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.ts.ets import ETSForecaster


def _seasonal(n=96, s=12, seed=0):
    t = np.arange(n + s)
    return (20 + 0.3 * t + 3 * np.sin(2 * np.pi * t / s)
            + np.random.RandomState(seed).normal(0, 0.5, n + s))


def _df(y):
    return CadwDataFrame.from_table(pa.table({"y": pa.array(y)}))


def test_synthetic_seasonal_forecast_accuracy():
    full = _seasonal()
    y, truth = full[:-12], full[-12:]
    m = ETSForecaster(seasonal_period=12, seasonal="additive").fit(_df(y), "y")
    f = m.predict(12)
    mase = np.mean(np.abs(f - truth)) / np.mean(np.abs(np.diff(y)))
    assert mase < 0.6, mase


def test_reference_statsmodels_mase_comparable():
    sm = pytest.importorskip("statsmodels.api")
    full = _seasonal(seed=1)
    y, truth = full[:-12], full[-12:]
    m = ETSForecaster(seasonal_period=12, seasonal="additive").fit(_df(y), "y")
    f = m.predict(12)
    ours = np.mean(np.abs(f - truth)) / np.mean(np.abs(np.diff(y)))
    ref_m = sm.tsa.ExponentialSmoothing(
        y, trend="add", seasonal="add", seasonal_periods=12,
        initialization_method="estimated").fit()
    ref = np.mean(np.abs(ref_m.forecast(12) - truth)) / np.mean(np.abs(np.diff(y)))
    assert ours < ref * 1.5, (ours, ref)


def test_nonseasonal_holt_linear_trend():
    t = np.arange(80)
    y = 5 + 0.5 * t + np.random.RandomState(2).normal(0, 0.3, 80)
    m = ETSForecaster(seasonal_period=1).fit(_df(y), "y")
    f = m.predict(8)
    expected = 5 + 0.5 * (80 + np.arange(8))
    assert np.max(np.abs(f - expected)) < 3.0


def test_determinism():
    full = _seasonal(seed=3)
    y = full[:-12]
    a = ETSForecaster(seasonal_period=12).fit(_df(y), "y").predict(12)
    b = ETSForecaster(seasonal_period=12).fit(_df(y), "y").predict(12)
    assert np.array_equal(a, b)


def test_edge_short_series_raises():
    with pytest.raises(ValueError, match="at least 4"):
        ETSForecaster(seasonal_period=1).fit(_df(np.arange(3.0)), "y")


def test_edge_seasonal_needs_2s():
    with pytest.raises(ValueError, match=r"2\*seasonal_period"):
        ETSForecaster(seasonal_period=12).fit(_df(np.arange(10.0)), "y")


def test_guard_nonfinite_series():
    y = np.arange(20.0); y[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ETSForecaster(seasonal_period=1).fit(_df(y), "y")


def test_guard_predict_before_fit():
    with pytest.raises(RuntimeError):
        ETSForecaster(seasonal_period=1).predict(3)


def test_prediction_interval_contains_point_forecast():
    full = _seasonal(seed=4)
    y = full[:-12]
    m = ETSForecaster(seasonal_period=12).fit(_df(y), "y")
    f = m.predict(6)
    lo, hi = m.prediction_interval(6)
    assert np.all(lo < f) and np.all(f < hi)


def test_in_sample_reasonable():
    full = _seasonal(seed=5)
    y = full[:-12]
    m = ETSForecaster(seasonal_period=12).fit(_df(y), "y")
    pred = m.predict_in_sample()
    assert pred.shape == y.shape
    sse = np.sum((pred - y) ** 2)
    assert np.sqrt(sse / y.size) < 3.0


def test_state_nbytes_positive_after_fit():
    full = _seasonal(seed=6)
    y = full[:-12]
    m = ETSForecaster(seasonal_period=12).fit(_df(y), "y")
    assert m.state_nbytes() > 0
