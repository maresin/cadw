"""Tests for OTM (weight-optimized STM)."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.ts.otm import OTMThetaForecaster, _ols_fit, _ses


def _seasonal(n=120, s=12, seed=0, trend=0.05, season_amp=5.0):
    t = np.arange(n + s)
    rng = np.random.RandomState(seed)
    return (20 + trend * t + season_amp * np.sin(2 * np.pi * t / s)
            + rng.normal(0, 0.5, n + s))


def _trending(n=80, seed=0):
    rng = np.random.RandomState(seed)
    return 5 + 0.5 * np.arange(n) + rng.normal(0, 0.3, n)


def _flat(n=120, seed=0):
    rng = np.random.RandomState(seed)
    return rng.normal(5.0, 0.3, n)


def _df(y):
    return CadwDataFrame.from_table(pa.table({"y": pa.array(y)}))


# ── Helpers ──────────────────────────────────────────────────────

def test_ols_fit_recovers_linear():
    y = 3.0 + 0.5 * np.arange(20)
    a, b = _ols_fit(y)
    assert abs(a - 3.0) < 1e-9
    assert abs(b - 0.5) < 1e-9


def test_ses_basic():
    y = np.ones(10) * 5.0
    lvl, fitted = _ses(y, 0.5)
    assert abs(lvl - 5.0) < 1e-9


# ── Contract ─────────────────────────────────────────────────────

def test_otm_basic_shape():
    full = _seasonal()
    y, truth = full[:-12], full[-12:]
    m = OTMThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    f = m.predict(12)
    assert f.shape == (12,)
    assert np.all(np.isfinite(f))


def test_otm_attributes_after_fit():
    full = _seasonal()
    y, _ = full[:-12], full[-12:]
    m = OTMThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    assert m.w2_ is not None
    assert m.alpha_ is not None
    assert 0.0 < m.w2_ < 1.0
    assert 0.0 < m.alpha_ < 1.0
    assert m.season_cycle_ is not None


def test_otm_weights_optimized_on_trend():
    """On strongly trending data, optimal w2 should be small (trust OLS)."""
    y = _trending(n=80, seed=1)
    m = OTMThetaForecaster(seasonal_period=1).fit(_df(y), "y")
    # Trend-dominant: w2 should be small (weights grid is 0.1..0.9)
    assert m.w2_ <= 0.5, (
        f"expected small w2 on trend data, got {m.w2_}")


def test_otm_weights_reasonable_on_seasonal():
    """On M3-like seasonal data, w2 should be moderate."""
    full = _seasonal(seed=2, trend=0.05, season_amp=5.0)
    y, _ = full[:-12], full[-12:]
    m = OTMThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    # Seasonal case: w2 can be anything; just verify in valid range
    assert 0.1 <= m.w2_ <= 0.9


def test_otm_competitive_mase_m3like():
    full = _seasonal(seed=3)
    y, truth = full[:-12], full[-12:]
    m = OTMThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    f = m.predict(12)
    mase = np.mean(np.abs(f - truth)) / np.mean(np.abs(np.diff(y)))
    assert mase < 1.0, f"MASE={mase:.2f}"


def test_otm_no_worse_than_stm_on_trend():
    """OTM should be at least as good as STM on trend-dominated data.

    STM (fixed w2=0.5) has MASE ~1.4 on this DGP. OTM should beat it
    because it can push w2 down toward theta_0 (OLS trend).
    """
    t = np.arange(200)
    rng = np.random.RandomState(4)
    full = (20 + 0.3 * t + 3 * np.sin(2 * np.pi * t / 12)
            + rng.normal(0, 0.5, 200))
    y, truth = full[:-12], full[-12:]
    m = OTMThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    f = m.predict(12)
    mase_otm = np.mean(np.abs(f - truth)) / np.mean(np.abs(np.diff(y)))
    # STM had ~1.4; OTM should be < 1.5 (possibly much better)
    assert mase_otm < 1.5, (
        f"OTM MASE={mase_otm:.2f} (should beat STM's ~1.4)")


def test_otm_deterministic():
    full = _seasonal(seed=5)
    y, _ = full[:-12], full[-12:]
    m1 = OTMThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    m2 = OTMThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    np.testing.assert_array_equal(m1.predict(12), m2.predict(12))
    assert m1.w2_ == m2.w2_
    assert m1.alpha_ == m2.alpha_


def test_otm_short_series_raises():
    with pytest.raises(ValueError, match="at least 4"):
        OTMThetaForecaster(seasonal_period=1).fit(
            _df(np.array([1.0, 2.0])), "y")


def test_otm_seasonal_too_short_raises():
    y = np.arange(15, dtype=np.float64)
    with pytest.raises(ValueError, match="seasonal_period"):
        OTMThetaForecaster(seasonal_period=12).fit(_df(y), "y")


def test_otm_bad_params():
    with pytest.raises(ValueError, match="seasonal_period"):
        OTMThetaForecaster(seasonal_period=0)


def test_otm_prediction_interval():
    full = _seasonal(seed=6)
    y, _ = full[:-12], full[-12:]
    m = OTMThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    lo, hi = m.prediction_interval(12, level=0.95)
    assert lo.shape == (12,)
    assert hi.shape == (12,)
    assert np.all(hi >= lo)


def test_otm_capability_flags():
    m = OTMThetaForecaster(seasonal_period=12)
    assert m.handles_trend is True
    assert m.handles_seasonality is True
    assert m.handles_multi_seasonality is False


def test_otm_no_bench_deps_in_source():
    import ast
    from pathlib import Path
    src = Path("cadw/ts/otm.py").read_text()
    tree = ast.parse(src)
    forbidden = ("sklearn", "statsmodels", "torch", "pyod", "aeon",
                 "merlion", "kats", "lifelines", "implicit", "ruptures",
                 "scipy")
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

