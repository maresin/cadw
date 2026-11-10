"""v0.9.189: ETS(s>1) window truncation (DEC-072)."""
import gc
import statistics
import warnings

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.approximation import ApproximationInfo
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded, measure_peak_traced
from cadw.core.warnings import ApproximationWarning
from cadw.ts.ets import ETSForecaster


def _mk_df(n, s=12, seed=0, guard_mb=500):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    y = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / s) + rng.normal(0, 0.5, n)
    return CadwDataFrame.from_table(
        pa.table({"y": pa.array(y, type=pa.float64())}),
        memory_guard=MemoryGuard(max_memory_mb=guard_mb))


def test_ets_auto_no_truncation_when_small():
    """n=120, guard=500MB -> W_eff huge -> no truncation."""
    df = _mk_df(120)
    m = ETSForecaster(seasonal_period=12).fit(df, "y")
    assert m.approximation is None
    assert m.n_train_ == 120


def test_ets_auto_bit_identical_to_full():
    """auto (W>>n) vs None produce bit-identical predictions."""
    df = _mk_df(120)
    m_auto = ETSForecaster(seasonal_period=12).fit(df, "y")
    m_none = ETSForecaster(seasonal_period=12).fit(df, "y", max_window=None)
    np.testing.assert_allclose(
        m_auto.predict(12), m_none.predict(12), atol=1e-9)


def test_ets_explicit_truncation():
    """max_window=60 -> truncation disclosed."""
    df = _mk_df(120)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = ETSForecaster(seasonal_period=12).fit(df, "y", max_window=60)
    assert m.approximation is not None
    assert m.approximation[0].type == "window_truncated"
    assert m.approximation[0].effective == 60
    assert m.approximation[0].lost_range == (0, 60)
    assert any(issubclass(x.category, ApproximationWarning) for x in w)


def test_ets_strict_refuses_large_max_window():
    """max_window > n + strict=True -> MemoryLimitExceeded."""
    df = _mk_df(120)
    with pytest.raises(MemoryLimitExceeded) as exc:
        ETSForecaster(seasonal_period=12).fit(
            df, "y", max_window=10000, strict=True)
    assert exc.value.model == "ETSForecaster"
    assert exc.value.params["max_window"] == 10000


def test_ets_insufficient_w_eff():
    """Tiny guard -> W_eff < 2*s -> MemoryLimitExceeded."""
    df = _mk_df(120, guard_mb=0.001)  # 1KB
    with pytest.raises(MemoryLimitExceeded) as exc:
        ETSForecaster(seasonal_period=12).fit(df, "y")
    assert "2*seasonal_period" in str(exc.value)


def test_ets_s1_fast_path():
    """s=1 -> no window logic (fast path)."""
    df = _mk_df(100, s=12)  # whatever, s=1 in fit
    m = ETSForecaster(seasonal_period=1).fit(df, "y")
    assert m.n_train_ == 100
    assert m.approximation is None


def test_ets_ooc_fit_peak():
    """Fit peak O(W_eff), not O(n). n=10k->100k, ratio<4."""
    rng = np.random.RandomState(0)

    def _peak(n):
        t = np.arange(n)
        yy = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, n)
        df = CadwDataFrame.from_table(
            pa.table({"y": pa.array(yy, type=pa.float64())}),
            memory_guard=MemoryGuard(max_memory_mb=500))
        peaks = []
        for _ in range(3):
            gc.collect()
            m = ETSForecaster(seasonal_period=12)
            r = measure_peak_traced(
                m.fit, df, "y", mg=df._memory_guard)
            peaks.append(int(r["peak_bytes"]))
        return int(statistics.median(peaks))

    p_small = _peak(10_000)
    p_large = _peak(100_000)
    ratio = p_large / max(p_small, 1)
    assert ratio < 4.0, (
        f"fit peak ratio {ratio:.2f} (n=10k->100k); "
        f"peak_small={p_small}, peak_large={p_large}")
