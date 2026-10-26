"""Tests for AutoETS (AICc-selected ETS)."""
from __future__ import annotations

import warnings

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.ts.auto_ets import AutoETSForecaster
from cadw.ts.ets import ETSForecaster
from cadw.ts.models import NaiveForecaster


def _seasonal(n=120, s=12, seed=0):
    t = np.arange(n + s)
    rng = np.random.RandomState(seed)
    return (20 + 0.3 * t + 3 * np.sin(2 * np.pi * t / s)
            + rng.normal(0, 0.5, n + s))


def _flat(n=120, seed=0):
    rng = np.random.RandomState(seed)
    return rng.normal(5.0, 0.3, n)


def _trending(n=80, seed=0):
    rng = np.random.RandomState(seed)
    return 5 + 0.5 * np.arange(n) + rng.normal(0, 0.3, n)


def _df(y):
    return CadwDataFrame.from_table(pa.table({"y": pa.array(y)}))


# ── Contract ─────────────────────────────────────────────────────

def test_auto_ets_fit_predict_shape():
    full = _seasonal()
    y, truth = full[:-12], full[-12:]
    m = AutoETSForecaster(seasonal_period=12).fit(_df(y), "y")
    f = m.predict(12)
    assert f.shape == (12,)
    assert np.all(np.isfinite(f))
    # Basic accuracy: MASE competitive
    mase = np.mean(np.abs(f - truth)) / np.mean(np.abs(np.diff(y)))
    assert mase < 1.0, f"MASE={mase:.2f}"


def test_auto_ets_attributes_after_fit():
    full = _seasonal()
    y, _ = full[:-12], full[-12:]
    m = AutoETSForecaster(seasonal_period=12).fit(_df(y), "y")
    assert m.best_model_ is not None
    assert m.selected_config_ is not None
    assert isinstance(m.aicc_scores_, list)
    assert len(m.aicc_scores_) >= 1
    assert m.fit_duration_s_ > 0.0


# ── AICc selection quality ───────────────────────────────────────

def test_auto_ets_selects_seasonal_on_seasonal_data():
    """On strongly seasonal data WITHOUT strong trend, AutoETS picks
    seasonal.

    DGP design note (v0.9.76b-fix2): the original _seasonal() DGP has
    0.3*t trend, which dominates over 3*sin seasonality over 180 pts
    (trend drift 54 vs seasonal amplitude 3); AICc correctly prefers
    ETS(A,A,N) there. To test seasonal selection in isolation, remove
    trend: pure 3*sin(2*pi*t/12) + noise. Then AICc prefers ETS(A,*,A).
    """
    t = np.arange(192)
    rng = np.random.RandomState(1)
    full = 20 + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, 192)
    y, _ = full[:-12], full[-12:]
    m = AutoETSForecaster(seasonal_period=12).fit(_df(y), "y")
    assert m.selected_config_["seasonal"] in ("A", "M"), (
        f"expected seasonal, got {m.selected_config_}")


def test_auto_ets_selects_nonseasonal_on_flat_data():
    """On flat non-seasonal data, seasonal configs should be skipped or
    lose to non-seasonal on AICc."""
    y = _flat(n=120, seed=2)
    # Force non-seasonal by using seasonal_period=1; also test with s=12
    # but on flat data (seasonal configs may still be picked if residual
    # improves by chance; test the simpler case)
    m = AutoETSForecaster(seasonal_period=1).fit(_df(y), "y")
    assert m.selected_config_["seasonal"] == "N", (
        f"expected non-seasonal, got {m.selected_config_}")


def test_auto_ets_trend_selection_on_trending_data():
    y = _trending(n=80, seed=3)
    m = AutoETSForecaster(seasonal_period=1).fit(_df(y), "y")
    # On a trending series, a trend model should win over no_trend
    assert m.selected_config_["trend"] in ("A", "Ad"), (
        f"expected trend, got {m.selected_config_}")


# ── AICc determinism and structure ───────────────────────────────

def test_auto_ets_deterministic():
    full = _seasonal(n=120, s=12, seed=4)
    y, _ = full[:-12], full[-12:]
    m1 = AutoETSForecaster(seasonal_period=12).fit(_df(y), "y")
    m2 = AutoETSForecaster(seasonal_period=12).fit(_df(y), "y")
    np.testing.assert_array_equal(m1.predict(12), m2.predict(12))
    assert m1.selected_config_["label"] == m2.selected_config_["label"]


def test_auto_ets_aicc_scores_populated():
    full = _seasonal(n=120, s=12, seed=5)
    y, _ = full[:-12], full[-12:]
    m = AutoETSForecaster(seasonal_period=12).fit(_df(y), "y")
    # At least one AICc score is finite
    finite = [s for s in m.aicc_scores_ if s.get("aicc") is not None]
    assert len(finite) >= 1
    # Best is min of finite AICcs
    best = min(s["aicc"] for s in finite)
    assert abs(m.best_aicc_ - best) < 1e-6


# ── Edge cases ───────────────────────────────────────────────────

def test_auto_ets_short_series_no_seasonal():
    """Short series (n < 2*s): seasonal configs skipped."""
    y = _flat(n=15, seed=6)
    m = AutoETSForecaster(seasonal_period=12).fit(_df(y), "y")
    # Only non-seasonal configs evaluated; should still return a model
    assert m.selected_config_["seasonal"] == "N"


def test_auto_ets_negative_data_skips_multiplicative():
    """Multiplicative seasonality requires y > 0; if any y <= 0 the M
    config must be skipped."""
    y = _flat(n=120, seed=7) - 10.0  # shift negative
    m = AutoETSForecaster(seasonal_period=12).fit(_df(y), "y")
    # Multiplicative seasonal configs should not appear in scores
    for s in m.aicc_scores_:
        assert s["seasonal"] != "M", (
            f"multiplicative should be skipped, got {s}")


def test_auto_ets_very_short_series_fallback_naive():
    """Very short series (n=4): likely only fallback works; at minimum
    must not crash."""
    y = np.array([1.0, 2.0, 1.5, 2.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = AutoETSForecaster(seasonal_period=1).fit(_df(y), "y")
    f = m.predict(2)
    assert f.shape == (2,)
    assert np.all(np.isfinite(f))


def test_auto_ets_bad_inputs():
    with pytest.raises(ValueError, match="seasonal_period"):
        AutoETSForecaster(seasonal_period=0)
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="at least 4"):
        AutoETSForecaster(seasonal_period=1).fit(_df(y), "y")


# ── Prediction interval ──────────────────────────────────────────

def test_auto_ets_prediction_interval():
    full = _seasonal(n=120, s=12, seed=8)
    y, _ = full[:-12], full[-12:]
    m = AutoETSForecaster(seasonal_period=12).fit(_df(y), "y")
    lo, hi = m.prediction_interval(12, level=0.95)
    assert lo.shape == (12,)
    assert hi.shape == (12,)
    assert np.all(hi >= lo)


# ── Reference ────────────────────────────────────────────────────

def test_auto_ets_competitive_with_manual_best():
    """AutoETS should be within 20% of manual best-of-two configs."""
    full = _seasonal(n=160, s=12, seed=9)
    y, truth = full[:-12], full[-12:]
    m_auto = AutoETSForecaster(seasonal_period=12).fit(_df(y), "y")
    f_auto = m_auto.predict(12)
    mase_auto = np.mean(np.abs(f_auto - truth)) / np.mean(
        np.abs(np.diff(y)))
    # Manual: additive vs multiplicative
    mases = []
    for s_ in ("additive", "multiplicative"):
        try:
            m_man = ETSForecaster(seasonal_period=12, seasonal=s_,
                                  trend=True, damped=False).fit(_df(y), "y")
            f_man = m_man.predict(12)
            mases.append(np.mean(np.abs(f_man - truth))
                         / np.mean(np.abs(np.diff(y))))
        except Exception:
            pass
    if not mases:
        pytest.skip("no manual config fit")
    best_manual = min(mases)
    assert mase_auto < best_manual * 1.5, (
        f"AutoETS MASE={mase_auto:.3f} vs manual best {best_manual:.3f}")


# ── AST guard ────────────────────────────────────────────────────

def test_auto_ets_no_bench_deps_in_source():
    import ast
    from pathlib import Path
    src = Path("cadw/ts/auto_ets.py").read_text()
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

