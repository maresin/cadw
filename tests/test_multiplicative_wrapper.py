"""v0.9.193: _MultiplicativeWrapper streaming (DEC-070).

Refactor removed self.y_ persistent O(n); log-transform streaming per
batch. State_nbytes honest (was under-reporting self.y_).
"""
import warnings

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard
from cadw.core.warnings import ApproximationWarning
from cadw.ts.auto_theta import _MultiplicativeWrapper
from cadw.ts.otm import OTMThetaForecaster
from cadw.ts.theta import ThetaForecaster


def _mk_df(n=120, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    y = (100 + 5 * np.sin(2 * np.pi * t / 12) * (1 + 0.1 * np.cos(t / 10))
         + rng.uniform(0.5, 1.5, n))
    return CadwDataFrame.from_table(
        pa.table({"y": pa.array(y, type=pa.float64())}),
        memory_guard=MemoryGuard(max_memory_mb=500))


@pytest.mark.parametrize("inner_cls", [ThetaForecaster, OTMThetaForecaster])
def test_bit_identical_stream_vs_saved(inner_cls):
    """predict + in_sample bit-identical streaming vs save_y=True."""
    df = _mk_df()
    factory = lambda s: inner_cls(seasonal_period=s)
    m_stream = _MultiplicativeWrapper(factory, 12, max_window=None).fit(df, "y")
    m_saved = _MultiplicativeWrapper(factory, 12, max_window=None).fit(
        df, "y", save_y=True)
    np.testing.assert_allclose(m_stream.predict(12), m_saved.predict(12),
                               atol=1e-9)
    np.testing.assert_allclose(
        m_stream.predict_in_sample(df),
        m_saved.predict_in_sample(),
        atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("inner_cls", [ThetaForecaster, OTMThetaForecaster])
def test_state_ooc_strictly_smaller(inner_cls):
    """Streaming state < save_y state by ~n*8 (no self.y_)."""
    df = _mk_df()
    factory = lambda s: inner_cls(seasonal_period=s)
    m_stream = _MultiplicativeWrapper(factory, 12, max_window=None).fit(df, "y")
    m_saved = _MultiplicativeWrapper(factory, 12, max_window=None).fit(
        df, "y", save_y=True)
    diff = m_saved.state_nbytes() - m_stream.state_nbytes()
    n_bytes = 120 * 8  # O(n) saved y
    assert abs(diff - n_bytes) < 64, (
        f"expected ~{n_bytes} B diff, got {diff}")


@pytest.mark.parametrize("inner_cls", [ThetaForecaster, OTMThetaForecaster])
def test_type_error_without_df(inner_cls):
    """predict_in_sample() without df + no save_y -> TypeError."""
    df = _mk_df()
    factory = lambda s: inner_cls(seasonal_period=s)
    m = _MultiplicativeWrapper(factory, 12, max_window=None).fit(df, "y")
    with pytest.raises(TypeError, match="df is required"):
        m.predict_in_sample()


@pytest.mark.parametrize("inner_cls", [ThetaForecaster, OTMThetaForecaster])
def test_truncation_propagates(inner_cls):
    """max_window=60 -> approximation propagated from inner to wrapper."""
    df = _mk_df()
    factory = lambda s: inner_cls(seasonal_period=s)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = _MultiplicativeWrapper(factory, 12, max_window=60).fit(df, "y")
    assert m.approximation is not None
    assert m.approximation[0].type == "window_truncated"
    assert m.approximation[0].effective == 60
    assert m.approximation[0].lost_range == (0, 60)
    assert any(issubclass(x.category, ApproximationWarning) for x in w)
    # Bit-identical to inner.predict on same y_used
    assert m.n_total_ == 120


def test_negative_y_raises():
    """Multiplicative: y must be > 0. Streaming check per batch."""
    y = np.array([1.0, 2.0, -1.0, 4.0])
    df = CadwDataFrame.from_table(
        pa.table({"y": pa.array(y, type=pa.float64())}),
        memory_guard=MemoryGuard(max_memory_mb=500))
    factory = lambda s: ThetaForecaster(seasonal_period=s)
    with pytest.raises(ValueError, match="y must be > 0"):
        _MultiplicativeWrapper(factory, 1).fit(df, "y")


def test_auto_no_truncation_small_n():
    """n=120, guard=500MB -> W_eff huge -> no approximation."""
    df = _mk_df(120)
    factory = lambda s: ThetaForecaster(seasonal_period=s)
    m = _MultiplicativeWrapper(factory, 12).fit(df, "y")  # max_window="auto"
    assert m.approximation is None
