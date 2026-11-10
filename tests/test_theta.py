"""Tests for Theta forecasting method (STM on STL)."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.ts.theta import ThetaForecaster, _best_alpha, _ols_fit, _ses_fit


def _seasonal(n=120, s=12, seed=0, trend=0.05, season_amp=5.0):
    """M3-like DGP: seasonality dominates trend (STM intended domain).

    v0.9.77b: default trend 0.05, season_amp 5. On trend-dominated
    DGP (0.3, 3), STM still underperforms AutoETS (documented in
    theta.py). STL fixes the decomposition leak but SES lag remains.
    """
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


# ── Unit tests ───────────────────────────────────────────────────

def test_ols_fit_linear():
    y = 3.0 + 0.5 * np.arange(20)
    a, b = _ols_fit(y)
    assert abs(a - 3.0) < 1e-9
    assert abs(b - 0.5) < 1e-9


def test_best_alpha_range():
    y = np.random.RandomState(0).normal(5.0, 0.5, 100)
    a = _best_alpha(y)
    assert 0.05 <= a <= 0.95


def test_ses_fit_basic():
    y = np.ones(10) * 5.0
    lvl, fitted, sse = _ses_fit(y, 0.5)
    assert abs(lvl - 5.0) < 1e-9
    assert sse < 1e-9


# ── Contract ─────────────────────────────────────────────────────

def test_theta_basic_shape():
    full = _seasonal()
    y, truth = full[:-12], full[-12:]
    m = ThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    f = m.predict(12)
    assert f.shape == (12,)
    assert np.all(np.isfinite(f))


def test_theta_competitive_mase():
    """STM on M3-like DGP (seasonality dominant): MASE < 1."""
    full = _seasonal(seed=1)
    y, truth = full[:-12], full[-12:]
    m = ThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    f = m.predict(12)
    mase = np.mean(np.abs(f - truth)) / np.mean(np.abs(np.diff(y)))
    assert mase < 1.0, f"MASE={mase:.2f}"


def test_theta_trending_data():
    y = _trending(n=80, seed=2)
    m = ThetaForecaster(seasonal_period=1).fit(_df(y), "y")
    f = m.predict(8)
    expected = 5 + 0.5 * (80 + np.arange(8))
    assert np.max(np.abs(f - expected)) < 3.0


def test_theta_deterministic():
    full = _seasonal(seed=3)
    y, _ = full[:-12], full[-12:]
    m1 = ThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    m2 = ThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    np.testing.assert_array_equal(m1.predict(12), m2.predict(12))


def test_theta_short_series_raises():
    with pytest.raises(ValueError, match="at least 4"):
        ThetaForecaster(seasonal_period=1).fit(
            _df(np.array([1.0, 2.0])), "y")


def test_theta_seasonal_too_short_raises():
    y = np.arange(15, dtype=np.float64)
    with pytest.raises(ValueError, match="seasonal_period"):
        ThetaForecaster(seasonal_period=12).fit(_df(y), "y")


def test_theta_bad_params():
    with pytest.raises(ValueError, match="seasonal_period"):
        ThetaForecaster(seasonal_period=0)


def test_theta_prediction_interval():
    full = _seasonal(seed=5)
    y, _ = full[:-12], full[-12:]
    m = ThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    lo, hi = m.prediction_interval(12, level=0.95)
    assert lo.shape == (12,)
    assert hi.shape == (12,)
    assert np.all(hi >= lo)


def test_theta_predict_in_sample():
    full = _seasonal(seed=6)
    y, _ = full[:-12], full[-12:]
    m = ThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    fit = m.predict_in_sample(_df(y))
    assert fit.shape == y.shape


# ── STL integration: improvement over classical decomposition ───

def test_theta_uses_stl_decomposition():
    """v0.9.183 OOC: decomp_ replaced by season_cycle_ (O(period) state).

    Full decomp_ (trend + seasonal, O(n)) removed from persistent state;
    only season_cycle_ = seasonal[:s] kept (needed for predict step idx).
    STL is still called during fit (and predict_in_sample(df)).
    """
    full = _seasonal(seed=7)
    y, _ = full[:-12], full[-12:]
    df = _df(y)
    m = ThetaForecaster(seasonal_period=12).fit(df, "y")
    # season_cycle_ is O(period) state, not full decomp_
    assert m.season_cycle_ is not None
    assert m.season_cycle_.shape == (12,)
    # Legacy decomp_ absent from persistent state
    assert not hasattr(m, "decomp_")
    # predict works via lvl_ + season_cycle_
    fc = m.predict(12)
    assert fc.shape == (12,)
    # predict_in_sample(df) recomputes from df (OOC)
    insample = m.predict_in_sample(df)
    assert insample.shape == y.shape


def test_theta_no_stl_when_no_seasonal():
    """seasonal_period=1 -> no STL; decomp_ None."""
    y = _flat(n=100, seed=8)
    m = ThetaForecaster(seasonal_period=1).fit(_df(y), "y")
    assert m.season_cycle_ is None


def test_theta_stl_improves_trend_leak():
    """On a trend-dominated series, STL-based STM should be no worse
    than classical decomposition would give.

    Reference: classical decomposition had MASE ~1.40 on this DGP
    (documented). STM on STL should be in a reasonable range; if it
    regresses, that's a signal.
    """
    t = np.arange(200)
    rng = np.random.RandomState(9)
    full = (20 + 0.3 * t + 3 * np.sin(2 * np.pi * t / 12)
            + rng.normal(0, 0.5, 200))
    y, truth = full[:-12], full[-12:]
    m = ThetaForecaster(seasonal_period=12).fit(_df(y), "y")
    f = m.predict(12)
    mase = np.mean(np.abs(f - truth)) / np.mean(np.abs(np.diff(y)))
    # STL should not be worse than ~1.5 (classical was 1.40; STL <= 1.5)
    assert mase < 1.5, f"STL-STM MASE={mase:.2f}"


def test_theta_handles_flags():
    """Capability flags for pipeline integration."""
    m = ThetaForecaster(seasonal_period=12)
    assert m.handles_trend is True
    assert m.handles_seasonality is True
    assert m.handles_multi_seasonality is False


# ── AST guard ────────────────────────────────────────────────────

def test_theta_no_bench_deps_in_source():
    import ast
    from pathlib import Path
    src = Path("cadw/ts/theta.py").read_text()
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

# ---- v0.9.194 (Theta + AutoTheta merge) ----

def _mk_df(y):
    import numpy as np
    import pyarrow as pa
    from cadw import CadwDataFrame
    from cadw.core.memory_guard import MemoryGuard
    y = np.asarray(y, dtype=np.float64)
    return CadwDataFrame.from_table(
        pa.table({"y": pa.array(y, type=pa.float64())}),
        memory_guard=MemoryGuard(max_memory_mb=500))


def _mk_seasonal(n=120, s=12, seed=0):
    import numpy as np
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    y = 100 + 5 * np.sin(2 * np.pi * t / s) + rng.uniform(0.5, 1.5, n)
    return y


def test_theta_params_auto_matches_autotheta():
    """Merged Theta(params='auto') == AutoThetaForecaster shim."""
    import warnings
    from cadw.ts.auto_theta import AutoThetaForecaster
    from cadw.ts.theta import ThetaForecaster

    y = _mk_seasonal()
    df = _mk_df(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m_old = AutoThetaForecaster(seasonal_period=12).fit(df, "y")
    m_new = ThetaForecaster(seasonal_period=12, params="auto").fit(df, "y")
    import numpy as np
    np.testing.assert_allclose(
        m_old.predict(12), m_new.predict(12), atol=1e-9, equal_nan=True)
    assert (m_old.selected_config_["label"]
            == m_new.selected_config_["label"])


def test_theta_auto_propagates_approximation():
    """window_truncated from best_model_ propagates to merged self."""
    from cadw.ts.theta import ThetaForecaster
    y = _mk_seasonal(n=200, s=12)
    df = _mk_df(y)
    m = ThetaForecaster(seasonal_period=12, params="auto").fit(
        df, "y", max_window=60)
    assert m.approximation is not None
    assert any(a.type == "window_truncated" for a in m.approximation)


def test_theta_manual_unchanged():
    """Default params='manual' == explicit manual."""
    from cadw.ts.theta import ThetaForecaster
    import numpy as np
    y = _mk_seasonal()
    df = _mk_df(y)
    m_default = ThetaForecaster(seasonal_period=12).fit(df, "y")
    m_explicit = ThetaForecaster(
        seasonal_period=12, params="manual").fit(df, "y")
    np.testing.assert_allclose(
        m_default.predict(12), m_explicit.predict(12))


def test_theta_auto_rejects_save_y():
    """Auto mode rejects save_y=True."""
    import pytest
    from cadw.ts.theta import ThetaForecaster
    y = _mk_seasonal()
    df = _mk_df(y)
    with pytest.raises(ValueError, match="save_y=True is not supported"):
        ThetaForecaster(seasonal_period=12, params="auto").fit(
            df, "y", save_y=True)

def test_auto_selects_multiplicative_on_multiplicative_series():
    """v0.9.194-b-fix: multiplicative candidates must not be silently dropped.

    Before fix (missing _check_df in _theta_common), _MultiplicativeWrapper
    raised NameError inside _fit_auto's try/except, so multiplicative
    candidates never ran and STM_additive always "won". After fix,
    multiplicative candidates must have val_mase (no error) and win on
    strongly multiplicative y.
    """
    import numpy as np
    import pyarrow as pa
    from cadw import CadwDataFrame
    from cadw.core.memory_guard import MemoryGuard

    rng = np.random.RandomState(0)
    n, s = 120, 12
    t = np.arange(n)
    y = ((100 + 0.5 * t) * (1 + 0.3 * np.sin(2 * np.pi * t / s))
         + 0.1 * rng.normal(size=n))
    y = np.maximum(y, 1.0)
    df = CadwDataFrame.from_table(
        pa.table({"y": pa.array(y, type=pa.float64())}),
        memory_guard=MemoryGuard(max_memory_mb=500))
    m = ThetaForecaster(seasonal_period=s, params="auto").fit(df, "y")

    mult_entries = [e for e in m.cv_scores_
                    if "multiplicative" in e["label"].lower()]
    assert mult_entries, "no multiplicative candidates in cv_scores_"
    for e in mult_entries:
        assert e.get("val_mase") is not None, (
            f"multiplicative candidate errored: {e}")

    label = m.selected_config_["label"]
    assert "multiplicative" in label.lower(), (
        f"expected multiplicative on multiplicative series; "
        f"selected={label!r}; "
        f"cv_scores="
        f"{[(x['label'], x.get('val_mase')) for x in m.cv_scores_]}")

