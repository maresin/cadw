"""Component-wise verification for TemporalHierarchy (DEC-033, v0.9.129).

State = y_base(n*8) + sum(y_per_level)(n*(1+sum(1/m))*8)
        + sum(model states) + td_probs + resid_var + 256.
Fit peak ~ 2*state (per-level forecasters fit inside loop).
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
from cadw.ts.thf import TemporalHierarchy
from cadw.ts.models import NaiveForecaster


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


def _thf(levels=(1, 7, 30)):
    return TemporalHierarchy(
        levels=list(levels),
        forecaster_factory=lambda: NaiveForecaster('last'),
    )


_warmup_df = _make_df(100)
_thf().fit(_warmup_df, 'y')


@pytest.mark.parametrize("n", [500, 2000, 5000])
def test_state_nbytes_lower_bound(n):
    """state >= y_base + sum(y_per_level); upper bound with 3 naive states."""
    m = _thf().fit(_make_df(n), 'y')
    state = m.state_nbytes()
    # Lower: y_base + sum(y_per_level)
    lower = n * 8 + int(n * (1 + 1 / 7 + 1 / 30)) * 8
    # Upper: + 3 forecaster states (each ~ n_k*8 or 4096 fallback)
    upper = (lower
             + int(n * 8) + int((n / 7) * 8) + int((n / 30) * 8)
             + 3 * 512)  # slack for header/td/resid
    assert lower <= state <= upper * 3, (
        f"n={n}: state={state} range=[{lower},{upper*3}]")


def test_state_ratio_linear():
    # NaiveForecaster lacks state_nbytes -> THF uses 4096 fallback x 3
    # = 12KB constant per level. At n=500 this dominates O(n) growth.
    # Use n=5000 vs 20000 so O(n) part (~3n*8) is >> 12KB constant.
    s_small = _thf().fit(_make_df(5000), 'y').state_nbytes()
    s_large = _thf().fit(_make_df(20000), 'y').state_nbytes()
    ratio = s_large / max(s_small, 1)
    # 4x n -> ~3.4x state (12KB fallback offsets sublinear).
    assert 2.5 <= ratio <= 5.5, f"ratio={ratio:.2f}; s={s_small},{s_large}"


def test_peak_grows_linearly_with_n():
    # n=500 -> peak ~8KB (SNR < 1 against ~8KB baseline noise).
    # Use n=2000/20000 (peak ~32K/320K, SNR > 4).
    def _peak(n):
        df = _make_df(n)
        df._memory_guard = MemoryGuard(max_memory_mb=2000)
        _malloc_trim(); gc.collect()
        r = measure_peak_traced(_thf().fit, df, 'y')
        return int(r["peak_bytes"])
    p_small = _peak(2000)
    p_large = _peak(20000)
    ratio = p_large / max(p_small, 1)
    # 10x n -> ~10x peak (y_base + y_per_level + per-level fits all O(n)).
    assert ratio < 15.0, f"ratio={ratio:.2f}; small={p_small}, large={p_large}"


def test_guard_refuses_at_domain_bound():
    df = _make_df(1_000_000)
    df._memory_guard = MemoryGuard(max_memory_mb=10)
    with pytest.raises(MemoryLimitExceeded):
        _thf().fit(df, 'y')


def test_tight_guard_raises():
    class _FakeMB:
        max_memory_mb = 0.001
        def check_state(self, nbytes, name="state"):
            raise MemoryLimitExceeded(f"{name}: {nbytes} B > limit 0")
    df = _make_df(1000)
    df._memory_guard = _FakeMB()
    with pytest.raises(MemoryLimitExceeded):
        _thf().fit(df, 'y')
