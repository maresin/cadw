"""Component-wise verification for OTMThetaForecaster (DEC-033, v0.9.129).

State = y_.nbytes + fitted_.nbytes + 256 = 2*n*8 + 256.
Fit peak = 31*n*8 + 256 (grid search: 19 SES fitted arrays +
temporaries). bounded_by_domain n<=1e6: at n=1e6 peak = 248 MB.
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
from cadw.ts.otm import OTMThetaForecaster


def _make_df(n, seed=0):
    rng = np.random.RandomState(seed)
    y = (np.sin(np.arange(n) * 0.3)
         + np.arange(n) * 0.01
         + rng.normal(0, 0.05, n))
    return CadwDataFrame.from_table(
        pa.table({'y': pa.array(y, type=pa.float64())}))


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


_warmup_df = _make_df(50)
OTMThetaForecaster().fit(_warmup_df, 'y')


@pytest.mark.parametrize("n", [100, 500, 2000])
def test_state_nbytes_exact(n):
    m = OTMThetaForecaster().fit(_make_df(n), 'y')
    expected = 2 * n * 8 + 256
    assert m.state_nbytes() == expected, (
        f"n={n}: state={m.state_nbytes()} expected={expected}")


def test_state_ratio_linear():
    s_small = OTMThetaForecaster().fit(_make_df(500), 'y').state_nbytes()
    s_large = OTMThetaForecaster().fit(_make_df(2000), 'y').state_nbytes()
    ratio = s_large / max(s_small, 1)
    # 4x n -> ~4x state (2n*8). Tolerate header overhead.
    assert 3.0 <= ratio <= 5.5, f"ratio={ratio:.2f}; s={s_small},{s_large}"


def test_peak_grows_linearly_with_n():
    def _peak(n):
        df = _make_df(n)
        df._memory_guard = MemoryGuard(max_memory_mb=2000)
        _malloc_trim(); gc.collect()
        r = measure_peak_traced(OTMThetaForecaster().fit, df, 'y')
        return int(r["peak_bytes"])
    p_small = _peak(500)
    p_large = _peak(5000)
    ratio = p_large / max(p_small, 1)
    # 10x n -> ~10x peak (state + transients both O(n)).
    assert ratio < 20.0, f"ratio={ratio:.2f}; small={p_small}, large={p_large}"


def test_guard_refuses_at_domain_bound():
    df = _make_df(1_000_000)
    df._memory_guard = MemoryGuard(max_memory_mb=10)
    with pytest.raises(MemoryLimitExceeded):
        OTMThetaForecaster().fit(df, 'y')


def test_tight_guard_raises():
    class _FakeMB:
        max_memory_mb = 0.001
        def check_state(self, nbytes, name="state"):
            raise MemoryLimitExceeded(
                f"{name}: {nbytes} B > limit 0")
    df = _make_df(1000)
    df._memory_guard = _FakeMB()
    with pytest.raises(MemoryLimitExceeded):
        OTMThetaForecaster().fit(df, 'y')
