"""Tests for target transforms (Differencer, TransformedTargetForecaster)."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.ts.transforms import (
    BaseTransformer, Differencer, TransformedTargetForecaster,
)
from cadw.ts.models import NaiveForecaster


def _df(y):
    return CadwDataFrame.from_table(pa.table({"y": pa.array(y)}))


# ── Differencer: unit ───────────────────────────────────────────

def test_differencer_d1_forward():
    y = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
    d = Differencer(lags=1).fit(y)
    out = d.transform(y)
    np.testing.assert_allclose(out, [2.0, 3.0, 4.0, 5.0])


def test_differencer_d2_forward():
    y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # quadratic
    d = Differencer(lags=2).fit(y)
    out = d.transform(y)
    # 1st diff: [3,5,7,9]; 2nd diff: [2,2,2]
    np.testing.assert_allclose(out, [2.0, 2.0, 2.0])


def test_differencer_d0_is_identity():
    y = np.array([1.0, 3.0, 6.0, 10.0])
    d = Differencer(lags=0).fit(y)
    out = d.transform(y)
    np.testing.assert_array_equal(out, y)


def test_differencer_d1_inverse_recovers_original():
    y = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
    d = Differencer(lags=1).fit(y)
    y_diff = d.transform(y)
    # Predict continuation: assume diff stays 6 -> next = 21
    forecast_diff = np.array([6.0])
    forecast = d.inverse_transform(forecast_diff, 1)
    np.testing.assert_allclose(forecast, [21.0], atol=1e-9)


def test_differencer_d2_inverse_recovers_original():
    y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    d = Differencer(lags=2).fit(y)
    y_diff = d.transform(y)
    # Next diff (2nd) = 2 -> next 1st diff = 9+2=11 -> next y = 25+11=36
    forecast = d.inverse_transform(np.array([2.0]), 1)
    np.testing.assert_allclose(forecast, [36.0], atol=1e-9)


def test_differencer_seasonal_forward():
    y = np.arange(24, dtype=np.float64)  # period 12 -> seasonal diff = 12
    d = Differencer(lags=0, seasonal_period=12, D=1).fit(y)
    out = d.transform(y)
    np.testing.assert_allclose(out, np.full(12, 12.0))


def test_differencer_seasonal_inverse():
    y = np.arange(24, dtype=np.float64)
    d = Differencer(lags=0, seasonal_period=12, D=1).fit(y)
    # Predict next 12 with seasonal-diff = 12 (constant trend)
    forecast = d.inverse_transform(np.full(12, 12.0), 12)
    expected = 24 + np.arange(12, dtype=np.float64)
    np.testing.assert_allclose(forecast, expected, atol=1e-9)


def test_differencer_combined_d1_D1():
    # y[t] = t + 12*sin(2pi t/12); d=1 removes trend, D=1 removes season
    t = np.arange(36, dtype=np.float64)
    y = t + 12 * np.sin(2 * np.pi * t / 12)
    d = Differencer(lags=1, seasonal_period=12, D=1).fit(y)
    out = d.transform(y)
    assert out.size == 36 - 1 - 12
    # After both diffs the signal should be near zero
    assert np.abs(out).max() < 2.0


def test_differencer_inverse_roundtrip():
    """Forward then inverse of identity forecast recovers last value."""
    y = np.cumsum(np.random.RandomState(0).normal(1.0, 0.1, 30))
    d = Differencer(lags=1).fit(y)
    # Forecast of 0 (= continuation with no change in diff)
    forecast = d.inverse_transform(np.zeros(1), 1)
    # Should equal last observed value (diff=0 -> level stays flat)
    np.testing.assert_allclose(forecast[0], y[-1], atol=1e-9)


def test_differencer_bad_params():
    with pytest.raises(ValueError, match="lags must be >= 0"):
        Differencer(lags=-1)
    with pytest.raises(ValueError, match="D must be >= 0"):
        Differencer(D=-1)
    with pytest.raises(ValueError, match="D>0 requires seasonal_period"):
        Differencer(D=1, seasonal_period=None)


def test_differencer_short_series_raises():
    y = np.array([1.0, 2.0])
    d = Differencer(lags=3)
    with pytest.raises(ValueError, match="n="):
        d.fit(y)


# ── TransformedTargetForecaster ─────────────────────────────────

def test_pipeline_basic_flow():
    """Differencer + NaiveForecaster on linear trend: predictions
    must continue the trend (not flat)."""
    y = 10.0 + 0.5 * np.arange(50)
    pipeline = TransformedTargetForecaster(
        transformers=[Differencer(lags=1)],
        forecaster=NaiveForecaster(strategy="last"),
    )
    pipeline.fit(_df(y), "y")
    f = pipeline.predict(5)
    expected = 10.0 + 0.5 * (50 + np.arange(5))
    # Naive last on diff = diff is constant 0.5 -> extrapolates +0.5 each
    np.testing.assert_allclose(f, expected, atol=0.1)


def test_pipeline_naive_without_diff_is_flat():
    """Sanity: same y, plain NaiveForecaster -> flat (baseline)."""
    y = 10.0 + 0.5 * np.arange(50)
    m = NaiveForecaster(strategy="last").fit(_df(y), "y")
    f = m.predict(5)
    np.testing.assert_allclose(f, np.full(5, y[-1]), atol=1e-9)


def test_pipeline_predict_in_sample_shape():
    y = 10.0 + 0.5 * np.arange(50)
    pipeline = TransformedTargetForecaster(
        transformers=[Differencer(lags=1)],
        forecaster=NaiveForecaster(strategy="last"),
    )
    pipeline.fit(_df(y), "y")
    fit = pipeline.predict_in_sample()
    assert fit.size == pipeline.forecaster.predict_in_sample().size


def test_pipeline_capability_flags():
    """With Differencer, pipeline handles trend/seasonality even if base
    forecaster doesn't."""
    pipeline = TransformedTargetForecaster(
        transformers=[Differencer(lags=1, seasonal_period=12, D=1)],
        forecaster=NaiveForecaster(strategy="last"),
    )
    assert pipeline.handles_trend is True
    assert pipeline.handles_seasonality is True


def test_pipeline_no_diff_naive_flags():
    pipeline = TransformedTargetForecaster(
        transformers=[Differencer(lags=0)],
        forecaster=NaiveForecaster(strategy="last"),
    )
    assert pipeline.handles_trend is False
    assert pipeline.handles_seasonality is False


def test_pipeline_bad_params():
    with pytest.raises(ValueError, match="at least one transformer"):
        TransformedTargetForecaster(transformers=[],
                                    forecaster=NaiveForecaster())
    with pytest.raises(ValueError, match="forecaster is required"):
        TransformedTargetForecaster(transformers=[Differencer()],
                                    forecaster=None)


def test_pipeline_get_params():
    pipeline = TransformedTargetForecaster(
        transformers=[Differencer(lags=1)],
        forecaster=NaiveForecaster(strategy="last"),
    )
    params = pipeline.get_params()
    assert params["forecaster"] == "NaiveForecaster"
    assert params["transformers"][0]["lags"] == 1


# ── AST guard ────────────────────────────────────────────────────

def test_transforms_no_bench_deps_in_source():
    import ast
    from pathlib import Path
    src = Path("cadw/ts/transforms.py").read_text()
    tree = ast.parse(src)
    forbidden = ("sklearn", "statsmodels", "torch", "pyod", "aeon",
                 "merlion", "kats", "lifelines", "implicit", "ruptures")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in forbidden, (
                    f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            assert top not in forbidden, (
                f"forbidden import-from: {node.module}")

# ── STLTransformer ──────────────────────────────────────────────

def test_stl_transformer_removes_seasonal():
    from cadw.ts.transforms import STLTransformer
    from cadw.ts._decomposition import stl_decompose
    t = np.arange(120, dtype=np.float64)
    s = 12
    y = 10.0 + 5.0 * np.sin(2 * np.pi * t / s)
    tr = STLTransformer(period=s).fit(y)
    y_ds = tr.transform(y)
    r_orig = stl_decompose(y, period=s)
    r_ds = stl_decompose(y_ds, period=s)
    # Seasonal amplitude after transform should be much smaller
    assert r_ds.seasonal.std() < 0.3 * r_orig.seasonal.std()


def test_stl_transformer_inverse_roundtrip():
    from cadw.ts.transforms import STLTransformer
    t = np.arange(120, dtype=np.float64)
    s = 12
    y = 10.0 + 0.2 * t + 5.0 * np.sin(2 * np.pi * t / s)
    tr = STLTransformer(period=s).fit(y)
    y_ds = tr.transform(y)
    y_back = tr.inverse_transform(y_ds, steps=y.size)
    np.testing.assert_allclose(y_back, y, atol=1e-9)


def test_stl_transformer_forecast_continuation():
    from cadw.ts.transforms import STLTransformer
    t = np.arange(120, dtype=np.float64)
    s = 12
    y = 10.0 + 0.2 * t + 5.0 * np.sin(2 * np.pi * t / s)
    tr = STLTransformer(period=s).fit(y)
    steps = 12
    f = tr.inverse_transform(np.zeros(steps), steps=steps)
    # Non-flat: repeating seasonal should have std > 1
    assert f.std() > 1.0, f"f.std()={f.std():.2f}"


def test_stl_transformer_short_series_raises():
    from cadw.ts.transforms import STLTransformer
    with pytest.raises(ValueError, match="at least 2"):
        STLTransformer(period=12).fit(np.arange(10, dtype=np.float64))


def test_stl_transformer_bad_period():
    from cadw.ts.transforms import STLTransformer
    with pytest.raises(ValueError, match="period"):
        STLTransformer(period=1)


def test_stl_transformer_get_params():
    from cadw.ts.transforms import STLTransformer
    tr = STLTransformer(period=12, n_iter=3)
    p = tr.get_params()
    assert p["period"] == 12 and p["n_iter"] == 3


# ── MSTLTransformer ────────────────────────────────────────────

def test_mstl_removes_both_seasonals():
    from cadw.ts.transforms import MSTLTransformer
    n = 500
    t = np.arange(n, dtype=np.float64)
    y = (10.0
         + 3.0 * np.sin(2 * np.pi * t / 7)
         + 2.0 * np.sin(2 * np.pi * t / 30))
    tr = MSTLTransformer(periods=[7, 30]).fit(y)
    y_ds = tr.transform(y)
    assert y_ds.std() < 0.5 * y.std()


def test_mstl_inverse_roundtrip():
    from cadw.ts.transforms import MSTLTransformer
    n = 300
    t = np.arange(n, dtype=np.float64)
    y = (10.0 + 0.05 * t
         + 2.0 * np.sin(2 * np.pi * t / 7)
         + 1.5 * np.sin(2 * np.pi * t / 30))
    tr = MSTLTransformer(periods=[7, 30]).fit(y)
    y_ds = tr.transform(y)
    y_back = tr.inverse_transform(y_ds, steps=n)
    np.testing.assert_allclose(y_back, y, atol=1e-9)


def test_mstl_bad_periods():
    from cadw.ts.transforms import MSTLTransformer
    with pytest.raises(ValueError, match="at least one"):
        MSTLTransformer(periods=[])
    with pytest.raises(ValueError, match=">= 2"):
        MSTLTransformer(periods=[1])


def test_mstl_short_series_raises():
    from cadw.ts.transforms import MSTLTransformer
    with pytest.raises(ValueError, match="at least 2"):
        MSTLTransformer(periods=[30]).fit(np.arange(40, dtype=np.float64))


def test_mstl_get_params():
    from cadw.ts.transforms import MSTLTransformer
    p = MSTLTransformer(periods=[7, 30], n_iter=3).get_params()
    assert p["periods"] == [7, 30] and p["n_iter"] == 3


# ── AutoStationarizer ──────────────────────────────────────────

def test_auto_stat_random_noise_d0():
    from cadw.ts.transforms import AutoStationarizer
    y = np.random.RandomState(0).normal(5.0, 1.0, 200)
    st = AutoStationarizer().fit(y)
    assert st.chosen_d_ == 0, f"got d={st.chosen_d_}"


def test_auto_stat_random_walk_d1():
    from cadw.ts.transforms import AutoStationarizer
    rng = np.random.RandomState(1)
    y = np.cumsum(rng.normal(0, 1, 200))
    st = AutoStationarizer().fit(y)
    assert st.chosen_d_ == 1, f"got d={st.chosen_d_}"


def test_auto_stat_trending_d1():
    from cadw.ts.transforms import AutoStationarizer
    y = (5.0 + 0.5 * np.arange(200)
         + np.random.RandomState(2).normal(0, 0.3, 200))
    st = AutoStationarizer().fit(y)
    assert st.chosen_d_ >= 1


def test_auto_stat_seasonal_D1():
    from cadw.ts.transforms import AutoStationarizer
    n = 200
    t = np.arange(n, dtype=np.float64)
    y = (10.0 + 5.0 * np.sin(2 * np.pi * t / 12)
         + np.random.RandomState(3).normal(0, 0.3, n))
    st = AutoStationarizer(seasonal_period=12).fit(y)
    assert st.chosen_D_ == 1, (
        f"got D={st.chosen_D_} strength={st.seasonal_strength_}")


def test_auto_stat_is_differencer_subclass():
    from cadw.ts.transforms import AutoStationarizer, Differencer
    st = AutoStationarizer()
    assert isinstance(st, Differencer)


def test_auto_stat_transform_after_fit():
    from cadw.ts.transforms import AutoStationarizer
    rng = np.random.RandomState(4)
    y = np.cumsum(rng.normal(0, 1, 100))
    st = AutoStationarizer().fit(y)
    y_t = st.transform(y)
    assert y_t.size == y.size - st.chosen_d_


def test_auto_stat_inverse_after_fit():
    from cadw.ts.transforms import AutoStationarizer
    rng = np.random.RandomState(5)
    y = np.cumsum(rng.normal(0, 1, 100))
    st = AutoStationarizer().fit(y)
    # Identity diff forecast -> next value = last observed (for d=1)
    if st.chosen_d_ == 1:
        f = st.inverse_transform(np.zeros(1), steps=1)
        np.testing.assert_allclose(f[0], y[-1], atol=1e-9)


def test_auto_stat_bad_params():
    from cadw.ts.transforms import AutoStationarizer
    with pytest.raises(ValueError, match="max_d"):
        AutoStationarizer(max_d=-1)
    with pytest.raises(ValueError, match="max_D"):
        AutoStationarizer(max_D=-1)

# ── AutoForecastPipeline ────────────────────────────────────────

def test_autopipeline_naive_on_trend_applies_differencing():
    """NaiveForecaster has no capability flags -> pipeline must add
    differencing -> trend continues (vs plain naive flat)."""
    from cadw.ts.transforms import AutoForecastPipeline
    y = 10.0 + 0.5 * np.arange(60)
    pipe = AutoForecastPipeline(
        forecaster=NaiveForecaster(strategy="last"),
        seasonal_periods=None,
    ).fit(_df(y), "y")
    # Pipeline has 1 transformer (AutoStationarizer)
    assert len(pipe.transformers) == 1
    f = pipe.predict(5)
    expected = 10.0 + 0.5 * (60 + np.arange(5))
    # Naive-last on diff = constant 0.5 -> cumsum continues trend
    np.testing.assert_allclose(f, expected, atol=0.2)


def test_autopipeline_naive_on_stationary_no_diff():
    """On white noise, AutoStationarizer chooses d=0 -> pipeline
    effectively identity -> matches plain naive."""
    from cadw.ts.transforms import AutoForecastPipeline
    rng = np.random.RandomState(0)
    y = rng.normal(5.0, 1.0, 200)
    pipe = AutoForecastPipeline(
        forecaster=NaiveForecaster(strategy="last"),
    ).fit(_df(y), "y")
    # AutoStationarizer chosen d=0 -> len 1 transformer
    assert len(pipe.transformers) == 1
    f_pipe = pipe.predict(3)
    # Plain naive
    m_naive = NaiveForecaster(strategy="last").fit(_df(y), "y")
    f_naive = m_naive.predict(3)
    np.testing.assert_allclose(f_pipe, f_naive, atol=1e-9)


def test_autopipeline_naive_on_seasonal_applies_stl_and_diff():
    """NaiveForecaster + seasonal data + seasonal_periods -> STL + diff
    -> prediction captures seasonality (vs plain naive flat)."""
    from cadw.ts.transforms import AutoForecastPipeline
    n = 200
    s = 12
    t = np.arange(n, dtype=np.float64)
    y = 10.0 + 3.0 * np.sin(2 * np.pi * t / s)
    pipe = AutoForecastPipeline(
        forecaster=NaiveForecaster(strategy="last"),
        seasonal_periods=[s],
    ).fit(_df(y), "y")
    # STL + AutoStationarizer
    assert len(pipe.transformers) == 2
    f = pipe.predict(s)
    # Forecast should have meaningful variation (seasonal)
    assert f.std() > 1.0, f"forecast flat: std={f.std():.2f}"


def test_autopipeline_sarima_naked():
    """SARIMA handles trend+seasonality -> no transforms."""
    from cadw.ts.transforms import AutoForecastPipeline
    from cadw.ts.models import SARIMAForecaster
    # SARIMA signature: order=(p,d,q), seasonal_order=(P,D,Q), s=period.
    pipe = AutoForecastPipeline(
        forecaster=SARIMAForecaster(
            order=(1, 1, 0), seasonal_order=(0, 0, 0), s=12),
        seasonal_periods=[12],
    )
    assert pipe.transformers == []
    assert pipe._pipeline is None


def test_autopipeline_ets_naked():
    """ETS handles both -> no transforms."""
    from cadw.ts.transforms import AutoForecastPipeline
    from cadw.ts.ets import ETSForecaster
    pipe = AutoForecastPipeline(
        forecaster=ETSForecaster(seasonal_period=12),
        seasonal_periods=[12],
    )
    assert pipe.transformers == []


def test_autopipeline_autotheta_naked():
    """AutoTheta handles both -> no transforms."""
    from cadw.ts.transforms import AutoForecastPipeline
    from cadw.ts.auto_theta import AutoThetaForecaster
    pipe = AutoForecastPipeline(
        forecaster=AutoThetaForecaster(seasonal_period=12),
        seasonal_periods=[12],
    )
    assert pipe.transformers == []


def test_autopipeline_capability_flags_after_wrap():
    """Pipeline with transforms sets handles_trend/seas True."""
    from cadw.ts.transforms import AutoForecastPipeline
    pipe = AutoForecastPipeline(
        forecaster=NaiveForecaster(strategy="last"),
        seasonal_periods=[12],
    )
    assert pipe.handles_trend is True
    assert pipe.handles_seasonality is True


def test_autopipeline_multiperiod_mstl():
    """Multiple periods + base handles single -> MSTL."""
    from cadw.ts.transforms import AutoForecastPipeline, MSTLTransformer
    pipe = AutoForecastPipeline(
        forecaster=NaiveForecaster(strategy="last"),
        seasonal_periods=[7, 30],
    )
    assert any(isinstance(t, MSTLTransformer)
               for t in pipe.transformers)


def test_autopipeline_none_forecaster_raises():
    from cadw.ts.transforms import AutoForecastPipeline
    with pytest.raises(ValueError, match="forecaster is required"):
        AutoForecastPipeline(forecaster=None)


def test_autopipeline_get_params():
    from cadw.ts.transforms import AutoForecastPipeline
    pipe = AutoForecastPipeline(
        forecaster=NaiveForecaster(strategy="last"),
        seasonal_periods=[12],
    )
    p = pipe.get_params()
    assert p["forecaster"] == "NaiveForecaster"
    assert p["seasonal_periods"] == [12]
    assert p["n_transformers"] == 2


def test_autopipeline_predict_in_sample_shape():
    from cadw.ts.transforms import AutoForecastPipeline
    y = 10.0 + 0.5 * np.arange(50)
    pipe = AutoForecastPipeline(
        forecaster=NaiveForecaster(strategy="last"),
    ).fit(_df(y), "y")
    fit = pipe.predict_in_sample()
    assert fit.size > 0


def test_autopipeline_ast_no_bench_deps():
    """transforms.py AST guard — no forbidden imports after pipeline add."""
    import ast
    from pathlib import Path
    src = Path("cadw/ts/transforms.py").read_text()
    tree = ast.parse(src)
    forbidden = ("sklearn", "statsmodels", "torch", "pyod", "aeon",
                 "merlion", "kats", "lifelines", "implicit", "ruptures")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in forbidden, (
                    f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            assert top not in forbidden, (
                f"forbidden import-from: {node.module}")
