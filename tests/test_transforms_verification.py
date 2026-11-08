"""Component-wise verification for TransformedTargetForecaster (v0.9.129).

State = base forecaster state + sum(transformer.state_nbytes()) + 256.
Each transformer's state:
  Differencer : 256 + tail_regular + tail_seasonal
  STLTransformer : 3n*8 + 256 (seasonal + trend + remainder)
  MSTLTransformer : (1+k)*n*8 + 256 (deseasonalized + k seasonals)
  AutoStationarizer : Differencer.state_nbytes() + 8
bounded_by_domain n_total <= 1e6.
"""
from __future__ import annotations
import ctypes, gc

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import (
    MemoryGuard, MemoryLimitExceeded, measure_peak_traced,
)
from cadw.ts.models import NaiveForecaster
from cadw.ts.transforms import (
    AutoStationarizer, Differencer, MSTLTransformer,
    STLTransformer, TransformedTargetForecaster,
)


def _make_df(n, seed=0):
    rng = np.random.RandomState(seed)
    # Non-negative series for AutoStationarizer/STL friendly
    y = (np.abs(np.sin(np.arange(n) * 0.3))
         + 0.5 * np.abs(np.sin(np.arange(n) * 0.3 * 7))
         + rng.normal(0, 0.05, n))
    return CadwDataFrame.from_table(
        pa.table({'y': pa.array(y, type=pa.float64())}))


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


_warmup_df = _make_df(200)
TransformedTargetForecaster(
    [Differencer(lags=1)], NaiveForecaster('last')
).fit(_warmup_df, 'y')


# ---- CW-1: transformer state exact ----

def test_differencer_state_exact():
    y = np.sin(np.arange(100) * 0.3)
    t = Differencer(lags=1).fit(y)
    # 256 header + tail_regular (1*8)
    assert t.state_nbytes() == 256 + 8

    t2 = Differencer(lags=0).fit(y)
    assert t2.state_nbytes() == 256

    t3 = Differencer(lags=1, seasonal_period=7, D=1).fit(
        np.abs(np.sin(np.arange(200) * 0.3)))
    # 256 + 8 (regular) + 7*8 (seasonal)
    assert t3.state_nbytes() == 256 + 8 + 7 * 8


@pytest.mark.parametrize("n", [200, 500, 1000])
def test_stl_transformer_state_exact(n):
    y = np.abs(np.sin(np.arange(n) * 0.3)) + 1.0
    t = STLTransformer(period=7).fit(y)
    # 3*n*8 + 256 (seasonal + trend + remainder)
    assert t.state_nbytes() == 3 * n * 8 + 256


@pytest.mark.parametrize("n", [200, 500, 1000])
def test_mstl_transformer_state_exact(n):
    y = np.abs(np.sin(np.arange(n) * 0.3)) + 1.0
    t = MSTLTransformer(periods=[7]).fit(y)
    # (1 + 1)*n*8 + 256 = deseasonalized + 1 seasonal
    assert t.state_nbytes() == 2 * n * 8 + 256


def test_autostationarizer_state_exact():
    y = np.abs(np.sin(np.arange(200) * 0.3)) + 1.0
    t = AutoStationarizer(seasonal_period=7).fit(y)
    # Differencer state + 8 (seasonal_strength_)
    expected = super(AutoStationarizer, t).state_nbytes() + 8
    assert t.state_nbytes() == expected


# ---- CW-1b: TTF composition ----

def test_ttf_state_composition():
    # Use OTMThetaForecaster: it implements state_nbytes (NaiveForecaster
    # does not; TTF falls back to 4096 via hasattr, which the composition
    # test cannot verify precisely).
    from cadw.ts.otm import OTMThetaForecaster
    df = _make_df(500)
    base = OTMThetaForecaster()
    ttf = TransformedTargetForecaster(
        [Differencer(lags=1), STLTransformer(period=7)], base,
    ).fit(df, 'y')
    expected = (ttf.forecaster.state_nbytes()
                + ttf.transformers[0].state_nbytes()
                + ttf.transformers[1].state_nbytes()
                + 256)
    assert ttf.state_nbytes() == expected, (
        f"state={ttf.state_nbytes()} expected={expected}")


# ---- CW-2: state_ratio_linear ----

def test_state_ratio_linear():
    s_small = TransformedTargetForecaster(
        [STLTransformer(period=7)], NaiveForecaster('last'),
    ).fit(_make_df(500), 'y').state_nbytes()
    s_large = TransformedTargetForecaster(
        [STLTransformer(period=7)], NaiveForecaster('last'),
    ).fit(_make_df(2000), 'y').state_nbytes()
    ratio = s_large / max(s_small, 1)
    # Both STL (3n*8) and Naive (n*8) are O(n) -> 4x n ~ 4x state
    assert 3.0 <= ratio <= 5.5, f"ratio={ratio:.2f}; s={s_small},{s_large}"


# ---- CW-3: peak_linear ----

def test_peak_grows_linearly_with_n():
    # n=500 yields peak ~4KB (below measure_peak baseline noise ~4KB,
    # SNR < 1). Use n=2000/20000 (peak ~16K/160K, SNR > 4).
    def _peak(n):
        df = _make_df(n)
        df._memory_guard = MemoryGuard(max_memory_mb=2000)
        _malloc_trim(); gc.collect()
        fn = TransformedTargetForecaster(
            [Differencer(lags=1)], NaiveForecaster('last'),
        ).fit
        r = measure_peak_traced(fn, df, 'y')
        return int(r["peak_bytes"])
    p_small = _peak(2000)
    p_large = _peak(20000)
    ratio = p_large / max(p_small, 1)
    # 10x n -> ~10x peak (state + Differencer transient both O(n)).
    assert ratio < 15.0, f"ratio={ratio:.2f}; small={p_small}, large={p_large}"


# ---- CW-4: guard_refuses_at_domain_bound ----

def test_guard_refuses_at_domain_bound():
    df = _make_df(1_000_000)
    df._memory_guard = MemoryGuard(max_memory_mb=10)
    fn = TransformedTargetForecaster(
        [Differencer(lags=1)], NaiveForecaster('last'),
    ).fit
    with pytest.raises(MemoryLimitExceeded):
        fn(df, 'y')


# ---- CW-5: tight_guard_raises ----

def test_tight_guard_raises():
    class _FakeMB:
        max_memory_mb = 0.001
        def check_state(self, nbytes, name="state"):
            raise MemoryLimitExceeded(f"{name}: {nbytes} B > limit 0")
    df = _make_df(1000)
    df._memory_guard = _FakeMB()
    fn = TransformedTargetForecaster(
        [Differencer(lags=1)], NaiveForecaster('last'),
    ).fit
    with pytest.raises(MemoryLimitExceeded):
        fn(df, 'y')
