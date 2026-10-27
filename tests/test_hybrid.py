"""Tests for cadw.ts.hybrid HybridETSGBDTForecaster (v0.9.86)."""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.ts import (
    ETSForecaster,
    HybridETSGBDTForecaster,
)


def _mk(vals, ts=None):
    cols = {"y": pa.array([float(v) for v in vals], type=pa.float64())}
    if ts is not None:
        cols["ts"] = pa.array(ts, type=pa.timestamp("ns"))
    return CadwDataFrame.from_table(pa.table(cols))


def _ts(n):
    return [1700000000000000000 + i * 3600 * 10**9 for i in range(n)]


def _seasonal_trend(n=150, s=12, seed=0):
    t = np.arange(n)
    return (20 + 0.3 * t
            + 3 * np.sin(2 * np.pi * t / s)
            + np.random.RandomState(seed).normal(0, 0.3, n))


def _mase(y_hat, y_true, y_train):
    scale = float(np.mean(np.abs(np.diff(y_train))))
    if scale == 0 or not np.isfinite(scale):
        return float("nan")
    return float(np.mean(np.abs(y_hat - y_true)) / scale)


# 1. Contract: fit/predict/predict_in_sample/prediction_interval/
#    state_nbytes all wired correctly.
def test_contract_basic():
    y = _seasonal_trend(seed=1)
    df = _mk(y, ts=_ts(len(y)))
    m = HybridETSGBDTForecaster(
        ets_kwargs={'seasonal_period': 12, 'seasonal': 'additive'},
    ).fit(df, 'y')
    fc = m.predict(6)
    assert fc.shape == (6,)
    assert np.all(np.isfinite(fc))
    pin = m.predict_in_sample()
    assert pin.shape == (len(y),)
    lo, hi = m.prediction_interval(6)
    assert lo.shape == (6,) and hi.shape == (6,)
    assert np.all(lo < hi)
    assert m.state_nbytes() > 0


# 2. Not-catastrophic on pure-ETS DGP. Loose 2x tolerance: GBDT may
#    pick up spurious noise patterns on finite samples. Real not-hurt
#    assertion is in bench_hybrid.py across many seeds.
def test_does_not_hurt_on_pure_ets_dgp():
    full = _seasonal_trend(n=150, seed=2)
    y_train, y_test = full[:-12], full[-12:]
    df_tr = _mk(y_train, ts=_ts(len(y_train)))
    ets_kw = {'seasonal_period': 12, 'seasonal': 'additive'}
    ets_fc = ETSForecaster(**ets_kw).fit(df_tr, 'y').predict(12)
    hyb_fc = HybridETSGBDTForecaster(ets_kwargs=ets_kw).fit(
        df_tr, 'y').predict(12)
    ets_mase = _mase(ets_fc, y_test, y_train)
    hyb_mase = _mase(hyb_fc, y_test, y_train)
    assert hyb_mase <= ets_mase * 2.0, (ets_mase, hyb_mase)


# 3. In-sample prediction actually adds the GBDT component when
#    residual model is present.
def test_predict_in_sample_applies_gbdt_component():
    y = _seasonal_trend(seed=3)
    df = _mk(y, ts=_ts(len(y)))
    hyb = HybridETSGBDTForecaster(
        ets_kwargs={'seasonal_period': 12, 'seasonal': 'additive'},
    ).fit(df, 'y')
    assert hyb._gbdt is not None, (
        "GBDT should be fitted on residuals for n=150 series")
    ets_in = hyb._ets.predict_in_sample()
    hyb_in = hyb.predict_in_sample()
    assert not np.allclose(ets_in, hyb_in), (
        "predict_in_sample should differ when GBDT residual present")


# 4. predict / predict_in_sample / prediction_interval before fit
def test_predict_before_fit_raises():
    m = HybridETSGBDTForecaster()
    with pytest.raises(RuntimeError, match="before fit"):
        m.predict(3)
    with pytest.raises(RuntimeError, match="before fit"):
        m.predict_in_sample()
    with pytest.raises(RuntimeError, match="before fit"):
        m.prediction_interval(3)


# 5. steps type/range validation
def test_steps_validation():
    y = _seasonal_trend(seed=4)
    df = _mk(y, ts=_ts(len(y)))
    m = HybridETSGBDTForecaster(
        ets_kwargs={'seasonal_period': 12},
    ).fit(df, 'y')
    with pytest.raises(TypeError, match="steps must be int"):
        m.predict("3")
    with pytest.raises(ValueError, match="steps must be >="):
        m.predict(0)


# 6. Short series
def test_short_series_raises():
    df = _mk([1.0, 2.0, 3.0, 4.0, 5.0], ts=_ts(5))
    with pytest.raises(ValueError, match="at least 10"):
        HybridETSGBDTForecaster().fit(df, 'y')


# 7. Multi-series not supported
def test_group_col_raises():
    y = _seasonal_trend(n=50, seed=5)
    df = _mk(y, ts=_ts(50))
    with pytest.raises(NotImplementedError, match="group_col"):
        HybridETSGBDTForecaster().fit(df, 'y', group_col='g')


# 8. GBDT skipped when residual too small
def test_gbdt_skipped_when_resid_too_small():
    y = _seasonal_trend(n=30, seed=6)
    df = _mk(y, ts=_ts(30))
    hyb = HybridETSGBDTForecaster(
        ets_kwargs={'seasonal_period': 12},
        min_resid_train=100,  # more than n
    ).fit(df, 'y')
    assert hyb._gbdt is None
    assert hyb._gbdt_skipped is True
    ets_fc = hyb._ets.predict(6)
    hyb_fc = hyb.predict(6)
    assert np.allclose(ets_fc, hyb_fc)


# 9. Capability flags (for AutoForecastPipeline idempotence)
def test_capability_flags():
    m = HybridETSGBDTForecaster()
    assert m.handles_trend is True
    assert m.handles_seasonality is True
    assert m.handles_multi_seasonality is False


# 10. PI is ETS PI shifted by GBDT point forecast
def test_prediction_interval_shifted_by_gbdt():
    y = _seasonal_trend(seed=7)
    df = _mk(y, ts=_ts(len(y)))
    hyb = HybridETSGBDTForecaster(
        ets_kwargs={'seasonal_period': 12, 'seasonal': 'additive'},
    ).fit(df, 'y')
    assert hyb._gbdt is not None
    lo, hi = hyb.prediction_interval(6)
    ets_lo, ets_hi = hyb._ets.prediction_interval(6)
    gbdt_fc = hyb._gbdt.predict(6)
    assert np.allclose(lo, ets_lo + gbdt_fc)
    assert np.allclose(hi, ets_hi + gbdt_fc)


# 11. Determinism
def test_determinism():
    y = _seasonal_trend(seed=8)
    df = _mk(y, ts=_ts(len(y)))
    kw = {'ets_kwargs': {'seasonal_period': 12}}
    a = HybridETSGBDTForecaster(**kw).fit(df, 'y').predict(6)
    b = HybridETSGBDTForecaster(**kw).fit(df, 'y').predict(6)
    assert np.allclose(a, b)


# 12. gbdt_mode='direct' works end-to-end (alternative to recursive)
def test_gbdt_mode_direct_works():
    y = _seasonal_trend(seed=9)
    df = _mk(y, ts=_ts(len(y)))
    hyb = HybridETSGBDTForecaster(
        ets_kwargs={'seasonal_period': 12},
        gbdt_kwargs={'mode': 'direct', 'lags': (1, 2, 3)},
    ).fit(df, 'y')
    fc = hyb.predict(6)
    assert fc.shape == (6,)
    assert np.all(np.isfinite(fc))
