"""Tests for GBDTForecaster capability-based transforms (v0.9.83)."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.ts.models import GBDTForecaster


def _df(y):
    return CadwDataFrame.from_table(pa.table({"y": pa.array(y)}))


def _linear(n=60, slope=2.0, intercept=10.0):
    return intercept + slope * np.arange(n, dtype=np.float64)


def _seasonal(n=120, s=12, trend=0.5, season_amp=5.0, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float64)
    return (20.0 + trend * t
            + season_amp * np.sin(2 * np.pi * t / s)
            + rng.normal(0, 0.3, n))


# ── Naked mode unchanged ─────────────────────────────────────────

def test_gbdt_default_no_transforms():
    """Default GBDTForecaster: _inner_pipeline None, naked behavior."""
    y = _linear(n=60)
    m = GBDTForecaster(n_estimators=20, max_depth=3).fit(_df(y), "y")
    assert m._inner_pipeline is None
    f = m.predict(3)
    assert f.shape == (3,)


def test_gbdt_naked_does_not_continue_steep_trend():
    """Naked GBDT on steep trend: tree cannot extrapolate."""
    y = _linear(n=60, slope=2.0)  # last=128, expected 130..138
    m = GBDTForecaster(n_estimators=30, max_depth=3,
                       differencing="none").fit(_df(y), "y")
    f = m.predict(5)
    # Naked predictions bounded by training range (tree extrapolation)
    assert f.max() < 129.5, f"naked unexpectedly extrapolated: {f}"


# ── Differencing ─────────────────────────────────────────────────

def test_gbdt_differencing_auto_trend_continues():
    """differencing='auto' on steep linear: predictions continue trend."""
    y = _linear(n=60, slope=2.0)  # next = 130, 132, 134, 136, 138
    m = GBDTForecaster(n_estimators=30, max_depth=3,
                       differencing="auto").fit(_df(y), "y")
    assert m._inner_pipeline is not None
    f = m.predict(5)
    expected = 130.0 + 2.0 * np.arange(5)
    np.testing.assert_allclose(f, expected, atol=1.5)


def test_gbdt_differencing_explicit_int():
    """differencing=1 equivalent to auto on pure linear (acf1 ~ 1)."""
    y = _linear(n=60, slope=1.5)
    m1 = GBDTForecaster(n_estimators=20, max_depth=3,
                        differencing=1).fit(_df(y), "y")
    f1 = m1.predict(3)
    expected = 10.0 + 1.5 * (60 + np.arange(3))
    np.testing.assert_allclose(f1, expected, atol=1.5)


# ── STL / MSTL ───────────────────────────────────────────────────

def test_gbdt_decompose_stl_seasonal_captured():
    """decompose='stl' adds seasonality back to forecast."""
    y = _seasonal(n=180, s=12, trend=0.0, season_amp=5.0)
    m = GBDTForecaster(n_estimators=30, max_depth=3,
                       decompose="stl",
                       decompose_periods=[12]).fit(_df(y), "y")
    f = m.predict(12)
    assert f.std() > 1.5, f"flat forecast: std={f.std():.2f}"


def test_gbdt_decompose_mstl_two_periods():
    """decompose='mstl' with 2 periods runs, forecast non-degenerate."""
    t = np.arange(600, dtype=np.float64)
    y = (10.0 + 3.0 * np.sin(2 * np.pi * t / 7)
         + 2.0 * np.sin(2 * np.pi * t / 30))
    m = GBDTForecaster(n_estimators=30, max_depth=3,
                       decompose="mstl",
                       decompose_periods=[7, 30]).fit(_df(y), "y")
    f = m.predict(14)
    assert f.shape == (14,)
    assert np.all(np.isfinite(f))
    assert f.std() > 0.5


def test_gbdt_combined_diff_and_stl():
    """Both transforms: season + trend removed before GBDT."""
    y = _seasonal(n=180, s=12, trend=0.5, season_amp=5.0)
    m = GBDTForecaster(n_estimators=30, max_depth=3,
                       differencing="auto", decompose="stl",
                       decompose_periods=[12]).fit(_df(y), "y")
    assert m._inner_pipeline is not None
    f = m.predict(12)
    assert np.all(np.isfinite(f))
    assert f.std() > 1.0  # seasonality captured


# ── Validation ───────────────────────────────────────────────────

def test_gbdt_stl_without_periods_raises():
    with pytest.raises(ValueError, match="requires decompose_periods"):
        GBDTForecaster(decompose="stl")


def test_gbdt_mstl_one_period_raises():
    with pytest.raises(ValueError, match=">= 2 decompose_periods"):
        GBDTForecaster(decompose="mstl", decompose_periods=[12])


def test_gbdt_bad_differencing_value_raises():
    with pytest.raises(ValueError, match="differencing="):
        GBDTForecaster(differencing="weird")


def test_gbdt_bad_decompose_value_raises():
    with pytest.raises(ValueError, match="decompose="):
        GBDTForecaster(decompose="magic")


# ── Capability flags ────────────────────────────────────────────

def test_gbdt_capability_flags_instance_level():
    g_default = GBDTForecaster()
    assert getattr(g_default, "handles_trend", False) is False
    assert getattr(g_default, "handles_seasonality", False) is False

    g_auto = GBDTForecaster(differencing="auto")
    assert g_auto.handles_trend is True

    g_stl = GBDTForecaster(decompose="stl", decompose_periods=[12])
    assert g_stl.handles_seasonality is True


def test_gbdt_pipeline_idempotent_with_autoforecastpipeline():
    """AutoForecastPipeline(GBDT(differencing='auto')) must not add
    a second AutoStationarizer (idempotent via instance flags)."""
    from cadw.ts.transforms import AutoForecastPipeline, AutoStationarizer
    g = GBDTForecaster(differencing="auto")
    pipe = AutoForecastPipeline(forecaster=g,
                                 seasonal_periods=None)
    # No extra transformers; GBDT handles trend itself
    auto_stats = [t for t in pipe.transformers
                  if isinstance(t, AutoStationarizer)]
    assert auto_stats == [], (
        f"AutoForecastPipeline double-wrapped: {pipe.transformers}")


# ── AST guard ────────────────────────────────────────────────────

def test_gbdt_transforms_no_bench_deps_at_module_level():
    """models.py must not import transforms at module level (would be
    circular); _build_inner_pipeline uses lazy import."""
    import ast
    from pathlib import Path
    src = Path("cadw/ts/models.py").read_text()
    tree = ast.parse(src)
    # Find module-level imports only (not inside functions)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            mod = (node.module or "")
            assert "transforms" not in mod, (
                f"models.py has module-level import from {mod!r}; "
                f"would cause circular dependency")
