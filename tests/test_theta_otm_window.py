"""v0.9.191: Theta/OTM s>1 window-truncation (DEC-072 P4)."""
import gc
import statistics
import warnings

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import (
    MemoryGuard, MemoryLimitExceeded, measure_peak_traced,
)
from cadw.core.warnings import ApproximationWarning
from cadw.ts.otm import OTMThetaForecaster
from cadw.ts.theta import ThetaForecaster


def _mk_df(n, s=12, seed=0, guard_mb=500):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    y = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / s) + rng.normal(0, 0.5, n)
    return CadwDataFrame.from_table(
        pa.table({"y": pa.array(y, type=pa.float64())}),
        memory_guard=MemoryGuard(max_memory_mb=guard_mb))


@pytest.mark.parametrize("cls,name", [
    (ThetaForecaster, "Theta"),
    (OTMThetaForecaster, "OTM"),
])
def test_auto_no_truncation_small_n(cls, name):
    """n=120, guard=500MB -> W_eff huge -> no truncation."""
    df = _mk_df(120)
    m = cls(seasonal_period=12).fit(df, "y")
    assert m.approximation is None
    assert m.n_train_ == 120


@pytest.mark.parametrize("cls,name", [
    (ThetaForecaster, "Theta"),
    (OTMThetaForecaster, "OTM"),
])
def test_auto_bit_identical_to_full(cls, name):
    """auto (W>>n) vs save_y=True produce bit-identical predictions."""
    df = _mk_df(120)
    m_auto = cls(seasonal_period=12).fit(df, "y")
    m_full = cls(seasonal_period=12).fit(df, "y", save_y=True)
    np.testing.assert_allclose(
        m_auto.predict(12), m_full.predict(12), atol=1e-9)
    np.testing.assert_allclose(
        m_auto.predict_in_sample(df), m_full.predict_in_sample(),
        atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("cls,name", [
    (ThetaForecaster, "Theta"),
    (OTMThetaForecaster, "OTM"),
])
def test_explicit_truncation_disclosed(cls, name):
    """max_window=60 -> ApproximationInfo + warning."""
    df = _mk_df(120)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = cls(seasonal_period=12).fit(df, "y", max_window=60)
    assert m.approximation is not None
    assert m.approximation[0].type == "window_truncated"
    assert m.approximation[0].effective == 60
    assert m.approximation[0].lost_range == (0, 60)
    assert any(issubclass(x.category, ApproximationWarning) for x in w)


@pytest.mark.parametrize("cls,name", [
    (ThetaForecaster, "Theta"),
    (OTMThetaForecaster, "OTM"),
])
def test_strict_refuse(cls, name):
    df = _mk_df(120)
    expected_model = cls.__name__
    with pytest.raises(MemoryLimitExceeded) as exc:
        cls(seasonal_period=12).fit(df, "y", max_window=10000, strict=True)
    assert exc.value.model == expected_model


@pytest.mark.parametrize("cls,name", [
    (ThetaForecaster, "Theta"),
    (OTMThetaForecaster, "OTM"),
])
def test_save_y_incompatible_with_int_window(cls, name):
    df = _mk_df(120)
    with pytest.raises(ValueError, match="save_y=True incompatible"):
        cls(seasonal_period=12).fit(df, "y", save_y=True, max_window=60)


@pytest.mark.parametrize("cls,name", [
    (ThetaForecaster, "Theta"),
    (OTMThetaForecaster, "OTM"),
])
def test_ooc_fit_peak(cls, name):
    """Fit peak O(W_eff), not O(n). Truncation regime: n >> W_eff.

    v0.9.191-fix3: tight guard (1 MB) to force W_eff << n in both
    measurements. At 500MB guard (W_eff ~2.7M >> 100k), no truncation
    occurs — peak is O(n), ratio ~n ratio (10x). Correct comparison
    requires truncation-active regime on BOTH sides.

    1MB guard * 0.7 * 0.25 / 32 B = ~5.7k W_eff. n=50k -> trunc, n=500k -> trunc.
    Peak bounded by W_eff (constant across n); ratio < 4 (measurement noise).
    """
    rng = np.random.RandomState(0)

    def _peak(n):
        t = np.arange(n)
        yy = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, n)
        df = CadwDataFrame.from_table(
            pa.table({"y": pa.array(yy, type=pa.float64())}),
            memory_guard=MemoryGuard(max_memory_mb=1))  # tight -> truncation
        peaks = []
        for _ in range(3):
            gc.collect()
            m = cls(seasonal_period=12)
            r = measure_peak_traced(m.fit, df, "y", mg=df._memory_guard)
            peaks.append(int(r["peak_bytes"]))
        return int(statistics.median(peaks))

    p_small = _peak(50_000)
    p_large = _peak(500_000)
    ratio = p_large / max(p_small, 1)
    assert ratio < 4.0, (
        f"{name}: fit peak ratio {ratio:.2f} (truncation regime); "
        f"small={p_small}, large={p_large}")
