"""v0.9.188: P2 streaming primitives + Theta(s=1) fit OOC.

Covers:
- ols_fit_streaming (bit-identical to ols_fit)
- _ses_fit_streaming (bit-identical to theta._ses_fit)
- ThetaForecaster.fit(s=1) streaming (bit-identical predict/in_sample + OOC peak)

NOTE: ETS(s=1, trend=False) uses ets._ses_fit which initializes
tr = y[1]-y[0] and pred = lvl + tr (NOT pure SES). Streaming variant
requires separate _ses_fit_hw_streaming (DEC-070 P3 backlog).
"""
import gc
import statistics

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ops.linalg import ols_fit, ols_fit_streaming
from cadw.ts.theta import _ses_fit as theta_ses_fit
from cadw.ts.theta import _ses_fit_streaming, ThetaForecaster


def _mk_df(y, guard_mb=500):
    return CadwDataFrame.from_table(
        pa.table({"y": pa.array(np.asarray(y, dtype=np.float64))}),
        memory_guard=MemoryGuard(max_memory_mb=guard_mb))


# === ols_fit_streaming ===

@pytest.mark.parametrize("n", [10, 100, 1000])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_ols_fit_streaming_bit_identical(n, seed):
    rng = np.random.RandomState(seed)
    y = rng.normal(100, 5, n)
    a_ref, b_ref = ols_fit(y)
    a_s, b_s, n_s = ols_fit_streaming([y])
    assert n_s == n
    assert abs(a_ref - a_s) < 1e-9, f"a: {a_ref} vs {a_s}"
    assert abs(b_ref - b_s) < 1e-9, f"b: {b_ref} vs {b_s}"


def test_ols_fit_streaming_batched_matches_single():
    rng = np.random.RandomState(0)
    y = rng.normal(100, 5, 1000)
    a1, b1, _ = ols_fit_streaming([y])
    a2, b2, _ = ols_fit_streaming([y[:333], y[333:666], y[666:]])
    assert abs(a1 - a2) < 1e-9
    assert abs(b1 - b2) < 1e-9


def test_ols_fit_streaming_degenerate_n1():
    a, b, n = ols_fit_streaming([np.array([42.0])])
    assert n == 1
    assert abs(a - 42.0) < 1e-12
    assert b == 0.0


# === theta._ses_fit_streaming ===

@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_ses_fit_streaming_bit_identical(alpha):
    rng = np.random.RandomState(0)
    y = rng.normal(100, 5, 500)
    lvl_ref, _fitted_ref, sse_ref = theta_ses_fit(y, alpha)
    lvl_s, sse_s, n_s = _ses_fit_streaming([y], alpha)
    assert n_s == 500
    assert abs(lvl_ref - lvl_s) < 1e-10
    assert abs(sse_ref - sse_s) < 1e-6 * max(1, sse_ref)


def test_ses_fit_streaming_batched_matches_single():
    rng = np.random.RandomState(0)
    y = rng.normal(100, 5, 500)
    lvl1, sse1, _ = _ses_fit_streaming([y], 0.5)
    lvl2, sse2, _ = _ses_fit_streaming([y[:100], y[100:300], y[300:]], 0.5)
    assert abs(lvl1 - lvl2) < 1e-10
    assert abs(sse1 - sse2) < 1e-6 * max(1, sse1)


# === ThetaForecaster(s=1) streaming ===

def test_theta_s1_streaming_bit_identical():
    rng = np.random.RandomState(0)
    y = 10 + 0.1 * np.arange(200) + rng.normal(0, 1, 200)
    df = _mk_df(y)
    m_stream = ThetaForecaster(seasonal_period=1).fit(df, "y")
    m_saved = ThetaForecaster(seasonal_period=1).fit(df, "y", save_y=True)
    np.testing.assert_allclose(
        m_stream.predict(5), m_saved.predict(5), atol=1e-9)
    np.testing.assert_allclose(
        m_stream.predict_in_sample(df), m_saved.predict_in_sample(),
        atol=1e-9, equal_nan=True)
    # state OOC strictly smaller than save_y=True
    assert m_stream.state_nbytes() < m_saved.state_nbytes()


def test_theta_s1_streaming_ooc_peak():
    """Fit peak O(batch), not O(n). n=10k -> 100k, ratio <4."""
    rng = np.random.RandomState(0)

    def _peak(n):
        tt = np.arange(n)
        yy = 10 + 0.1 * tt + rng.normal(0, 1, n)
        df = _mk_df(yy)
        peaks = []
        for _ in range(3):
            gc.collect()
            m = ThetaForecaster(seasonal_period=1)
            r = measure_peak_traced(
                m.fit, df, "y", mg=df._memory_guard)
            peaks.append(int(r["peak_bytes"]))
        return int(statistics.median(peaks))

    p_small = _peak(10_000)
    p_large = _peak(100_000)
    ratio = p_large / max(p_small, 1)
    assert ratio < 4.0, (
        f"fit peak ratio {ratio:.2f} (n=10k -> 100k); "
        f"peak_small={p_small}, peak_large={p_large}")


def test_theta_s12_non_streaming_still_works():
    """s>1 uses non-streaming path (stl_decompose leaf); verify works."""
    rng = np.random.RandomState(0)
    t = np.arange(200)
    y = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 200)
    df = _mk_df(y)
    m = ThetaForecaster(seasonal_period=12).fit(df, "y")
    assert m.predict(12).shape == (12,)
    insample = m.predict_in_sample(df)
    assert insample.shape == y.shape
