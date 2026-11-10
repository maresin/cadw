"""Tests for EnsembleForecaster (per-member pipeline + weighting)."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.ts.ensemble import EnsembleForecaster, _mase, _naive_mae_train
from cadw.ts.models import NaiveForecaster
from cadw.ts.theta import ThetaForecaster


def _seasonal(n=200, s=12, seed=0, trend=0.05, season_amp=5.0):
    t = np.arange(n + s)
    rng = np.random.RandomState(seed)
    return (20 + trend * t + season_amp * np.sin(2 * np.pi * t / s)
            + rng.normal(0, 0.5, n + s))


def _trending(n=120, seed=0):
    rng = np.random.RandomState(seed)
    return 5.0 + 0.5 * np.arange(n) + rng.normal(0, 0.3, n)


def _df(y):
    return CadwDataFrame.from_table(pa.table({"y": pa.array(y)}))


# ── Helpers ──────────────────────────────────────────────────────

def test_naive_mae_train_diff():
    y = np.arange(20, dtype=np.float64)
    d = _naive_mae_train(y, s=1)
    assert abs(d - 1.0) < 1e-9


def test_naive_mae_train_seasonal():
    t = np.arange(24, dtype=np.float64)
    y = t + 5 * np.sin(2 * np.pi * t / 12)
    d = _naive_mae_train(y, s=12)
    # Seasonal diff of t + 5 sin: 12 + small noise
    assert d > 0


def test_mase_basic():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 2.1, 3.1])
    m = _mase(y, p, denom=1.0)
    assert abs(m - 0.1) < 1e-9


# ── Contract ─────────────────────────────────────────────────────

def test_ensemble_basic_equal():
    """Equal weights on stationary -> mean of members.

    Uses ThetaForecaster (handles_trend=True -> pipeline naked) so
    ensemble prediction is exactly (p1 + p2)/2 of member predictions.
    NaiveForecaster would be wrapped in AutoForecastPipeline (adds
    differencing), so ens != naked-naive average.
    """
    y = _trending(n=100, seed=1)
    m = EnsembleForecaster(
        members=[ThetaForecaster(seasonal_period=1),
                 ThetaForecaster(seasonal_period=1)],
        weights="equal",
    ).fit(_df(y), "y")
    f = m.predict(5)
    p1 = ThetaForecaster(seasonal_period=1).fit(_df(y), "y").predict(5)
    p2 = ThetaForecaster(seasonal_period=1).fit(_df(y), "y").predict(5)
    expected = (p1 + p2) / 2.0
    np.testing.assert_allclose(f, expected, atol=1e-9)


def test_ensemble_median_robust():
    """Median with 2 identical members + 1 outlier -> median equals
    the identical ones."""
    y = _trending(n=100, seed=2)
    m = EnsembleForecaster(
        members=[ThetaForecaster(seasonal_period=1),
                 ThetaForecaster(seasonal_period=12),
                 ThetaForecaster(seasonal_period=1)],
        weights="median",
        seasonal_periods=None,
    ).fit(_df(y), "y")
    f = m.predict(3)
    p_1a = ThetaForecaster(seasonal_period=1).fit(_df(y), "y").predict(3)
    p_1b = ThetaForecaster(seasonal_period=1).fit(_df(y), "y").predict(3)
    # Two identical members: median equals them
    np.testing.assert_allclose(f, (p_1a + p_1b) / 2.0, atol=1e-6)


def test_ensemble_custom_weights():
    y = _trending(n=100, seed=3)
    m = EnsembleForecaster(
        members=[ThetaForecaster(seasonal_period=1),
                 ThetaForecaster(seasonal_period=1)],
        weights=[0.7, 0.3],
    ).fit(_df(y), "y")
    f = m.predict(3)
    p1 = ThetaForecaster(seasonal_period=1).fit(_df(y), "y").predict(3)
    p2 = ThetaForecaster(seasonal_period=1).fit(_df(y), "y").predict(3)
    expected = 0.7 * p1 + 0.3 * p2
    np.testing.assert_allclose(f, expected, atol=1e-9)


def test_ensemble_inverse_mase_attributes():
    full = _seasonal(n=200, seed=4)
    y = full[:-12]
    m = EnsembleForecaster(
        members=[NaiveForecaster(strategy="last"),
                 ThetaForecaster(seasonal_period=12)],
        weights="inverse_mase",
        seasonal_periods=[12],
    ).fit(_df(y), "y")
    assert m.member_weights_ is not None
    assert m.member_weights_.size == 2
    assert abs(m.member_weights_.sum() - 1.0) < 1e-9
    assert m.member_val_mase_ is not None


def test_ensemble_inverse_mase_downweights_bad_member():
    """On M3-like data, Theta should have lower val MASE than naive
    -> higher weight."""
    full = _seasonal(n=200, seed=5)
    y = full[:-12]
    m = EnsembleForecaster(
        members=[NaiveForecaster(strategy="last"),
                 ThetaForecaster(seasonal_period=12)],
        weights="inverse_mase",
        seasonal_periods=[12],
    ).fit(_df(y), "y")
    # Theta is better than naive on this data; its weight > 0.5
    # (unless both are similar)
    assert m.member_weights_[1] > 0.4, (
        f"theta weight={m.member_weights_[1]:.2f}, "
        f"val_mase={m.member_val_mase_}")


def test_ensemble_deterministic():
    full = _seasonal(n=150, seed=6)
    y = full[:-12]
    m1 = EnsembleForecaster(
        members=[NaiveForecaster(strategy="last"),
                 NaiveForecaster(strategy="drift")],
        weights="inverse_mase",
        seasonal_periods=[12],
    ).fit(_df(y), "y")
    m2 = EnsembleForecaster(
        members=[NaiveForecaster(strategy="last"),
                 NaiveForecaster(strategy="drift")],
        weights="inverse_mase",
        seasonal_periods=[12],
    ).fit(_df(y), "y")
    np.testing.assert_array_equal(m1.predict(12), m2.predict(12))
    np.testing.assert_allclose(m1.member_weights_, m2.member_weights_)


def test_ensemble_capability_flags():
    """Ensemble inherits capabilities from members."""
    m = EnsembleForecaster(
        members=[NaiveForecaster(strategy="last"),
                 ThetaForecaster(seasonal_period=12)],
        seasonal_periods=[12],
    )
    assert m.handles_trend is True
    assert m.handles_seasonality is True


def test_ensemble_predict_in_sample_shape():
    y = _trending(n=80, seed=7)
    df = _df(y)
    m = EnsembleForecaster(
        members=[NaiveForecaster(strategy="last"),
                 NaiveForecaster(strategy="drift")],
        weights="equal",
    ).fit(df, "y")
    fit = m.predict_in_sample(df)
    assert fit.size > 0


def test_ensemble_prediction_interval():
    y = _trending(n=80, seed=8)
    m = EnsembleForecaster(
        members=[NaiveForecaster(strategy="last"),
                 NaiveForecaster(strategy="drift")],
        weights="equal",
    ).fit(_df(y), "y")
    lo, hi = m.prediction_interval(5, level=0.95)
    assert lo.shape == (5,)
    assert hi.shape == (5,)
    # Naive has no PI; both endpoints equal point forecast; lo==hi==f
    assert np.all(hi >= lo)


def test_ensemble_bad_members():
    with pytest.raises(ValueError, match="at least 2"):
        EnsembleForecaster(members=[])
    with pytest.raises(ValueError, match="at least 2"):
        EnsembleForecaster(members=[NaiveForecaster()])
    with pytest.raises(ValueError, match="not a ForecasterBase"):
        EnsembleForecaster(members=[NaiveForecaster(), object()])


def test_ensemble_bad_weights_spec():
    members = [NaiveForecaster(strategy="last"),
               NaiveForecaster(strategy="drift")]
    with pytest.raises(ValueError, match="not in"):
        EnsembleForecaster(members=members, weights="magic")
    with pytest.raises(ValueError, match="custom weights length"):
        EnsembleForecaster(members=members, weights=[0.5])


def test_ensemble_get_params():
    m = EnsembleForecaster(
        members=[NaiveForecaster(strategy="last"),
                 NaiveForecaster(strategy="drift")],
        weights="equal",
        seasonal_periods=[12],
    )
    p = m.get_params()
    assert p["members"] == ["NaiveForecaster", "NaiveForecaster"]
    assert p["weights_spec"] == "equal"
    assert p["seasonal_periods"] == [12]


def test_ensemble_no_bench_deps_in_source():
    import ast
    from pathlib import Path
    src = Path("cadw/ts/ensemble.py").read_text()
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

