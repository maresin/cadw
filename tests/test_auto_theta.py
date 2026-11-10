"""Tests for AutoTheta (CV selection STM/OTM x additive/multiplicative)."""
from __future__ import annotations

import warnings

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.ts.auto_theta import (
    AutoThetaForecaster, _mase, _naive_seasonal_mae,
)


def _seasonal(n=200, s=12, seed=0, trend=0.05, season_amp=5.0):
    t = np.arange(n + s)
    rng = np.random.RandomState(seed)
    return (20 + trend * t + season_amp * np.sin(2 * np.pi * t / s)
            + rng.normal(0, 0.5, n + s))


def _trending(n=200, seed=0, slope=0.5):
    rng = np.random.RandomState(seed)
    return 5 + slope * np.arange(n) + rng.normal(0, 0.3, n)


def _flat(n=120, seed=0):
    rng = np.random.RandomState(seed)
    return rng.normal(5.0, 0.3, n)


def _df(y):
    return CadwDataFrame.from_table(pa.table({"y": pa.array(y)}))


# ── Helpers ──────────────────────────────────────────────────────

def test_naive_seasonal_mae():
    y = np.arange(24, dtype=np.float64)
    d = _naive_seasonal_mae(y, s=12)
    assert abs(d - 12.0) < 1e-9


def test_mase():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 2.1, 3.1])
    m = _mase(y, p, denom=1.0)
    assert abs(m - 0.1) < 1e-9


# ── Contract ─────────────────────────────────────────────────────

def test_autotheta_basic_shape():
    full = _seasonal()
    y, truth = full[:-12], full[-12:]
    m = AutoThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    f = m.predict(12)
    assert f.shape == (12,)
    assert np.all(np.isfinite(f))


def test_autotheta_attributes_after_fit():
    full = _seasonal()
    y, _ = full[:-12], full[-12:]
    m = AutoThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    assert m.best_model_ is not None
    assert m.selected_config_ is not None
    assert len(m.cv_scores_) >= 2
    # At least 2 candidates with val_mase (additive STM + OTM; multiplicative
    # if y > 0)
    fitted = [c for c in m.cv_scores_ if c.get("val_mase") is not None]
    assert len(fitted) >= 2


def test_autotheta_selects_candidate_with_min_val_mase():
    full = _seasonal(seed=1)
    y, _ = full[:-12], full[-12:]
    m = AutoThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    fitted = [c for c in m.cv_scores_ if c.get("val_mase") is not None]
    best = min(c["val_mase"] for c in fitted)
    assert abs(m.best_mase_ - best) < 1e-6
    assert m.selected_config_["label"] in [c["label"] for c in fitted]


def test_autotheta_m3like_competitive():
    full = _seasonal(seed=2, trend=0.05, season_amp=5.0)
    y, truth = full[:-12], full[-12:]
    m = AutoThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    f = m.predict(12)
    mase = np.mean(np.abs(f - truth)) / np.mean(np.abs(np.diff(y)))
    assert mase < 1.0, f"MASE={mase:.2f}"


def test_autotheta_trend_dominated_competitive():
    """On trend-dominated DGP, AutoTheta should pick OTM and be
    competitive with the best of {STM, OTM}."""
    t = np.arange(220)
    rng = np.random.RandomState(3)
    full = (20 + 0.3 * t + 3 * np.sin(2 * np.pi * t / 12)
            + rng.normal(0, 0.5, 220))
    y, truth = full[:-12], full[-12:]
    m = AutoThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    f = m.predict(12)
    mase = np.mean(np.abs(f - truth)) / np.mean(np.abs(np.diff(y)))
    # STM would be ~0.65; OTM ~0.30. AutoTheta should be in that range.
    assert mase < 0.6, (
        f"AutoTheta on trend-dominated MASE={mase:.2f} "
        f"(selected={m.selected_config_['label']})")


def test_autotheta_skips_multiplicative_when_negative():
    y = _flat(n=120, seed=4) - 10.0  # all negative
    m = AutoThetaForecaster(seasonal_period=1).fit(_df(y), "y")
    labels = [c["label"] for c in m.cv_scores_]
    assert "STM_multiplicative" not in labels
    assert "OTM_multiplicative" not in labels


def test_autotheta_deterministic():
    full = _seasonal(seed=5)
    y, _ = full[:-12], full[-12:]
    m1 = AutoThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    m2 = AutoThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    np.testing.assert_array_equal(m1.predict(12), m2.predict(12))
    assert m1.selected_config_["label"] == m2.selected_config_["label"]


def test_autotheta_val_steps_explicit():
    full = _seasonal(seed=6)
    y, _ = full[:-12], full[-12:]
    m = AutoThetaForecaster(seasonal_period=12,
                            val_steps=20).fit(_df(y), "y")
    # Still produces valid output
    f = m.predict(12)
    assert f.shape == (12,)
    assert np.all(np.isfinite(f))


def test_autotheta_short_series_raises():
    with pytest.raises(ValueError, match="at least 4"):
        AutoThetaForecaster(seasonal_period=1).fit(
            _df(np.array([1.0, 2.0])), "y")


def test_autotheta_seasonal_too_short_raises():
    y = np.arange(15, dtype=np.float64)
    with pytest.raises(ValueError, match="seasonal_period"):
        AutoThetaForecaster(seasonal_period=12).fit(_df(y), "y")


def test_autotheta_bad_params():
    with pytest.raises(ValueError, match="seasonal_period"):
        AutoThetaForecaster(seasonal_period=0)


def test_autotheta_prediction_interval():
    full = _seasonal(seed=7)
    y, _ = full[:-12], full[-12:]
    m = AutoThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    lo, hi = m.prediction_interval(12, level=0.95)
    assert lo.shape == (12,)
    assert hi.shape == (12,)
    assert np.all(hi >= lo)


def test_autotheta_fallback_naive():
    """Very short series may not fit any candidate."""
    y = np.array([1.0, 2.0, 1.5, 2.5, 2.0, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = AutoThetaForecaster(seasonal_period=1).fit(_df(y), "y")
    f = m.predict(2)
    assert f.shape == (2,)
    assert np.all(np.isfinite(f))


def test_autotheta_capability_flags():
    m = AutoThetaForecaster(seasonal_period=12)
    assert m.handles_trend is True
    assert m.handles_seasonality is True
    assert m.handles_multi_seasonality is False


def test_autotheta_no_bench_deps_in_source():
    import ast
    from pathlib import Path
    src = Path("cadw/ts/auto_theta.py").read_text()
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

# ---- v0.9.194: deprecation ----

def test_auto_theta_emits_deprecation():
    """AutoThetaForecaster emits DeprecationWarning in v0.9.194."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        AutoThetaForecaster(seasonal_period=12)
    dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert dep, "expected DeprecationWarning; got: " + repr(w)
    msg = str(dep[0].message)
    assert "params='auto'" in msg or "v1.0.0" in msg

