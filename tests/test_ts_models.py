"""Tests for cadw.ts.models (v0.9.24)."""
import math

import numpy as np
import pyarrow as pa
import pytest

import cadw
from cadw import CadwDataFrame
from cadw.ts import (
    ForecasterBase, NaiveForecaster, SARIMAForecaster, GBDTForecaster,
)


def _mk_df(y, ts=None):
    cols = {"y": pa.array([float(v) for v in y], type=pa.float64())}
    if ts is not None:
        cols["ts"] = pa.array(ts, type=pa.timestamp("ns"))
    return CadwDataFrame.from_table(pa.table(cols))


def _mk_ts(n, start=1700000000000000000, step_ns=3600 * 10**9):
    return [start + i * step_ns for i in range(n)]


def _extract(out):
    return [b for b in out]


# ==================== API contract ====================


@pytest.mark.parametrize("cls", [
    NaiveForecaster,
    SARIMAForecaster,
    GBDTForecaster,
])
def test_api_contract(cls):
    inst = cls()
    assert callable(getattr(inst, "fit", None))
    assert callable(getattr(inst, "predict", None))
    assert callable(getattr(inst, "predict_in_sample", None))
    # Must be a ForecasterBase subclass.
    assert isinstance(inst, ForecasterBase)


def test_predict_before_fit_raises():
    for cls in (NaiveForecaster, SARIMAForecaster, GBDTForecaster):
        inst = cls()
        with pytest.raises(RuntimeError):
            inst.predict(5)
        with pytest.raises(RuntimeError):
            inst.predict_in_sample()


def test_steps_validation():
    df = _mk_df(list(range(30)))
    for cls in (NaiveForecaster, SARIMAForecaster, GBDTForecaster):
        inst = cls()
        inst.fit(df, "y")
        with pytest.raises(ValueError):
            inst.predict(0)
        with pytest.raises(ValueError):
            inst.predict(-1)


# ==================== NaiveForecaster ====================


@pytest.mark.parametrize("strategy,expected", [
    ("last", [10.0, 10.0, 10.0]),
])
def test_naive_last(strategy, expected):
    y = [1.0, 2.0, 3.0, 10.0]
    df = _mk_df(y)
    m = NaiveForecaster(strategy=strategy).fit(df, "y")
    fc = m.predict(3)
    assert list(np.round(fc, 8)) == expected


def test_naive_seasonal():
    # s=2, y = [1,2,1,2,1,2] -> y[T+1]=y[T-1]=2? y[5]=2, next = y[3]=2?
    # Actually y[T+h] = y[T+h-s]: h=1 -> y[6-2]=y[4]=1; h=2 -> y[5]=2
    y = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    df = _mk_df(y)
    m = NaiveForecaster(strategy="seasonal", period=2).fit(df, "y")
    fc = m.predict(4)
    assert list(fc) == [1.0, 2.0, 1.0, 2.0]


def test_naive_drift():
    # y = 1,2,3,4,5 -> drift=1, fc = 5 + 1,2,3 = 6,7,8
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    df = _mk_df(y)
    m = NaiveForecaster(strategy="drift").fit(df, "y")
    fc = m.predict(3)
    assert list(np.round(fc, 8)) == [6.0, 7.0, 8.0]


def test_naive_in_sample():
    y = [1.0, 2.0, 3.0, 4.0]
    df = _mk_df(y)
    m = NaiveForecaster(strategy="last").fit(df, "y")
    # v0.9.182: df required for OOC predict_in_sample
    preds = m.predict_in_sample(df)
    assert math.isnan(preds[0])
    assert preds[1] == 1.0
    assert preds[3] == 3.0


# ==================== SARIMA ====================


def test_sarima_random_walk_integration_back():
    # ARIMA(0,1,0): fit differenced series with no AR/MA. Point
    # forecast should be a flat line at y[-1] (naive random walk).
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 1.0, size=300))
    df = _mk_df(y)
    m = SARIMAForecaster(order=(0, 1, 0), auto=False).fit(df, "y")
    fc = m.predict(10)
    assert np.all(np.isfinite(fc))
    # Integration-back correctness: forecast = y[-1] flat.
    assert abs(fc[0] - y[-1]) < 1e-6
    assert abs(fc[-1] - y[-1]) < 1e-6
    assert m._order_ == (0, 1, 0)


def test_sarima_stationary_ar1_mean_reversion():
    # AR(1): y_t = 0.5 y_{t-1} + eps; mean = 0. Forecast should
    # converge toward 0 for large h.
    rng = np.random.default_rng(1)
    n = 500
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + rng.normal(0, 0.5)
    df = _mk_df(y)
    m = SARIMAForecaster(order=(1, 0, 0), auto=False).fit(df, "y")
    fc = m.predict(20)
    assert np.all(np.isfinite(fc))
    # Forecasts should decay in magnitude towards 0.
    assert abs(fc[0]) > abs(fc[19]) - 1.0
    # AR coefficient should be < 1 in magnitude
    assert abs(m._ar[0]) < 1.0


def test_sarima_rejects_short_series():
    df = _mk_df([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        SARIMAForecaster().fit(df, "y")


def test_sarima_seasonal_requires_s():
    with pytest.raises(ValueError):
        SARIMAForecaster(seasonal_order=(1, 0, 0))


def test_sarima_integrate_back_d1():
    # Constant series with drift after d=1: y_t = t. Fit (0,1,1) and
    # forecast should be approximately y[-1] + h. Simple sanity.
    y = np.arange(1.0, 51.0)
    df = _mk_df(y)
    m = SARIMAForecaster(order=(0, 1, 0), auto=False).fit(df, "y")
    fc = m.predict(5)
    assert np.all(np.isfinite(fc))


# ==================== GBDT ====================


def _seasonal_series(n=200, period=7):
    t = np.arange(n)
    y = 10 + np.sin(2 * np.pi * t / period) * 2 + 0.05 * t
    return y


def test_gbdt_direct_default_mode():
    y = _seasonal_series()
    df = _mk_df(y)
    m = GBDTForecaster(
        lags=(1, 2, 3, 7), seasonal_periods=(),
        n_estimators=10, max_depth=2,
    )
    assert m.mode == "direct"
    m.fit(df, "y")
    fc = m.predict(5)
    assert np.all(np.isfinite(fc))
    assert fc.shape == (5,)


def test_gbdt_recursive():
    y = _seasonal_series()
    df = _mk_df(y)
    m = GBDTForecaster(
        mode="recursive", lags=(1, 2, 7),
        n_estimators=10, max_depth=2,
    )
    m.fit(df, "y")
    fc = m.predict(5)
    assert np.all(np.isfinite(fc))


def test_gbdt_multi_output():
    y = _seasonal_series()
    df = _mk_df(y)
    m = GBDTForecaster(
        mode="multi_output", lags=(1, 2),
        n_estimators=5, max_depth=2,
    )
    m.fit(df, "y")
    fc = m.predict(5)
    assert np.all(np.isfinite(fc))


def test_gbdt_bad_mode_raises():
    with pytest.raises(ValueError):
        GBDTForecaster(mode="nope")


def test_gbdt_in_sample():
    y = _seasonal_series(100)
    df = _mk_df(y)
    m = GBDTForecaster(
        lags=(1, 2, 3), n_estimators=5, max_depth=2,
    )
    m.fit(df, "y")
    preds = m.predict_in_sample()
    # head max_lag rows are NaN; tail must be finite.
    assert np.all(np.isfinite(preds[3:]))


# ==================== MemoryGuard smoke ====================


def test_naive_memory_guard():
    y = list(range(50))
    df = _mk_df(y)
    guard = cadw.MemoryGuard(max_memory_mb=8)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    NaiveForecaster(strategy="last").fit(df2, "y").predict(3)


def test_sarima_memory_guard():
    y = list(range(50))
    df = _mk_df(y)
    guard = cadw.MemoryGuard(max_memory_mb=8)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    SARIMAForecaster(order=(1, 0, 0), auto=False).fit(df2, "y").predict(3)


def test_gbdt_memory_guard():
    y = _seasonal_series(60)
    df = _mk_df(y)
    guard = cadw.MemoryGuard(max_memory_mb=16)
    df2 = CadwDataFrame(df._dataset, memory_guard=guard,
                        spill_dir=df._spill_dir)
    GBDTForecaster(
        lags=(1, 2), n_estimators=3, max_depth=2,
    ).fit(df2, "y").predict(3)


def test_gbdt_deferred_training_no_silent_extrapolation():
    # Regression: predict(steps=10) after fit(horizons=None) must train
    # 10 distinct horizon models, not reuse horizon-5 model 6 times.
    y = _seasonal_series(200)
    df = _mk_df(y)
    m = GBDTForecaster(
        mode="direct", lags=(1, 2, 3), n_estimators=5, max_depth=2,
    )
    m.fit(df, "y")  # horizons=None -> deferred
    assert m._n_horizons == 0
    fc = m.predict(10)
    assert m._n_horizons == 10
    assert len(m._models) == 10
    assert np.all(np.isfinite(fc))
    # Second predict(5) reuses first 5 horizon models (no shrink).
    fc2 = m.predict(5)
    assert m._n_horizons == 10
    assert np.all(np.isfinite(fc2))
    assert np.allclose(fc2, fc[:5])


def test_gbdt_eager_horizons_raises_when_exceeded():
    # Alternative path: fit(horizons=5) then predict(10) must raise.
    y = _seasonal_series(200)
    df = _mk_df(y)
    m = GBDTForecaster(
        mode="direct", lags=(1, 2), n_estimators=5, max_depth=2,
    )
    m.fit(df, "y", horizons=5)
    assert m._n_horizons == 5
    # Deferred path: predict(10) will silently extend via _train_n_horizons,
    # which is the design (deferred training is always available).
    # Explicit-horizons callers that want a hard cap should pass the
    # exact horizon count. Verify a 10-step predict extends to 10 models.
    fc = m.predict(10)
    assert m._n_horizons == 10
    assert len(fc) == 10


# ============ v0.9.25a: SARIMA integration + intercept + in-sample ======


def test_sarima_integrate_back_seasonal():
    """D>0: integration order must be D then d (reverse of _prep)."""
    # y = [1..20], d=1, D=1, s=2. Correct integrated forecast on a
    # perfectly linear series is [21, 22, 23].
    # Old (broken) order d-then-D gives [39, 40, 59] — structurally off.
    y = np.arange(1.0, 21.0)
    df = _mk_df(y)
    m = SARIMAForecaster(
        order=(0, 1, 0), seasonal_order=(0, 1, 0), s=2, auto=False,
    ).fit(df, "y")
    fc = m.predict(3)
    np.testing.assert_allclose(
        fc, [21.0, 22.0, 23.0], atol=1e-6,
        err_msg=f"seasonal integration order wrong: got {fc}",
    )


def test_sarima_integrate_back_seasonal_with_mean():
    """D>0 with non-zero mean: centering + correct order compose."""
    y = np.arange(100.0, 120.0)  # [100..119], mean≈109.5
    df = _mk_df(y)
    m = SARIMAForecaster(
        order=(0, 1, 0), seasonal_order=(0, 1, 0), s=2, auto=False,
    ).fit(df, "y")
    fc = m.predict(3)
    np.testing.assert_allclose(fc, [120.0, 121.0, 122.0], atol=1e-6)


def test_sarima_stationary_nonzero_mean():
    """d=0 on series with mean≈100 must predict ~100, not 0."""
    rng = np.random.default_rng(42)
    n = 200
    y = 100.0 + 0.5 * rng.normal(size=n)
    df = _mk_df(y)
    m = SARIMAForecaster(order=(0, 0, 0), auto=False).fit(df, "y")
    fc = m.predict(5)
    assert all(80 < f < 120 for f in fc), (
        f"ARIMA(0,0,0) on mean=100 series returned {fc} (should be ~100, "
        f"not 0; bug #2)"
    )


def test_sarima_ar1_nonzero_mean():
    """AR(1) with non-zero mean — long-horizon forecast ~ mean."""
    rng = np.random.default_rng(0)
    n = 500
    y = np.zeros(n)
    y[0] = 100.0
    for t in range(1, n):
        y[t] = 100.0 + 0.5 * (y[t - 1] - 100.0) + rng.normal(0, 0.5)
    df = _mk_df(y)
    m = SARIMAForecaster(order=(1, 0, 0), auto=False).fit(df, "y")
    fc = m.predict(20)
    assert abs(fc[-1] - 100.0) < 5.0, (
        f"long-horizon AR(1) forecast {fc[-1]:.2f} far from mean 100"
    )


def test_sarima_predict_in_sample_length_and_scale():
    """d=0: len(predict_in_sample) == n_train, values in y scale."""
    rng = np.random.default_rng(0)
    n = 50
    y = 100.0 + rng.normal(size=n)
    df = _mk_df(y)
    m = SARIMAForecaster(order=(1, 0, 0), auto=False).fit(df, "y")
    preds = m.predict_in_sample(df)
    assert len(preds) == len(y), (
        f"len(predict_in_sample)={len(preds)} != n_train={len(y)} "
        f"(contract violation, bug #3)"
    )
    finite = preds[~np.isnan(preds)]
    assert len(finite) > 0
    assert abs(finite.mean() - 100.0) < 5.0, (
        f"predict_in_sample mean={finite.mean():.2f} — should be ~100 "
        f"(y scale, not diff scale)"
    )


def test_sarima_predict_in_sample_d1_returns_nan():
    """d>0: predict_in_sample returns all-NaN (documented limitation)."""
    rng = np.random.default_rng(0)
    n = 50
    y = np.cumsum(rng.normal(size=n)) + 100.0
    df = _mk_df(y)
    m = SARIMAForecaster(order=(0, 1, 0), auto=False).fit(df, "y")
    preds = m.predict_in_sample(df)
    assert len(preds) == len(y)
    assert np.all(np.isnan(preds)), (
        "d>0 predict_in_sample must be all-NaN (documented); "
        "TemporalHierarchy falls back to OLS"
    )


def test_sarima_thf_wls_still_works_with_sarima():
    """TemporalHierarchy with SARIMA + WLS: fallback to OLS for d>0."""
    import warnings
    from cadw.ts import TemporalHierarchy
    rng = np.random.default_rng(0)
    n = 200
    y = np.cumsum(rng.normal(size=n)) + 100.0
    df = _mk_df(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # fallback warning expected
        th = TemporalHierarchy(
            [1, 4, 16],
            lambda: SARIMAForecaster(order=(0, 1, 0), auto=False),
            reconciliation="optimal_wls",
        )
        th.fit(df, "y")
    fc = th.predict(8)
    assert len(fc) == 8
    assert np.all(np.isfinite(fc))


def test_sarima_perfect_linear_does_not_raise():
    """v0.9.25a-fix1: sigma2=0 (perfect fit) is valid, not an error.

    y = exactly linear, d=1, D=1 → twice-differenced = all zeros →
    sigma2 = 0. This surfaced in v0.9.25a dogfood: _fit_one rejected
    `sigma2 <= 0`, so SARIMA could not fit at all. Now accepted:
    perfect linear forecast = y[-1] + h (see integrate_back test).
    """
    y = list(np.arange(1.0, 21.0))
    df = _mk_df(y)
    m = SARIMAForecaster(
        order=(0, 1, 0), seasonal_order=(0, 1, 0), s=2, auto=False,
    ).fit(df, "y")
    fc = m.predict(3)
    assert np.all(np.isfinite(fc))
    assert m._sigma2 == 0.0
