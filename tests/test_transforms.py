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

