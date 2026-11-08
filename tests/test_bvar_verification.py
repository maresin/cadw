"""Component-wise verification for BVARModel (DEC-033, v0.9.139).

State = beta_post((1+k*p), k) + k*p (last_p buffer) + 256 header
      = (m*k + k*p + 256)*f64 where m = 1+k*p.
Fit peak = state + 2 * batch * (k*8) (per-batch transient; extra_factor=2).
bounded_state: true (state independent of n).
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
from cadw.ts.bvar import BVARModel


def _make_df(n, k=2, seed=0):
    rng = np.random.RandomState(seed)
    Y = np.zeros((n, k))
    for t in range(1, n):
        Y[t] = 0.5 * Y[t - 1] + rng.normal(0, 1, k)
    return CadwDataFrame.from_table(pa.table({
        f"y{j}": pa.array(Y[:, j], type=pa.float64()) for j in range(k)
    }))


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


_warmup_df = _make_df(200, 2)
_m = BVARModel(p=2)
_m.fit(_warmup_df, ["y0", "y1"],
       memory_guard=MemoryGuard(max_memory_mb=500))


@pytest.mark.parametrize("k,p", [(2, 1), (2, 4), (3, 3)])
def test_state_nbytes_exact(k, p):
    """state = ((1+k*p)*k + k*p + 256) * 8."""
    df = _make_df(500, k)
    cols = [f"y{j}" for j in range(k)]
    m = BVARModel(p=p)
    m.fit(df, cols, memory_guard=MemoryGuard(max_memory_mb=500))
    expected = ((1 + k * p) * k + k * p + 256) * 8
    assert m.state_nbytes() == expected, (
        f"k={k} p={p}: state={m.state_nbytes()} expected={expected}")


def test_state_independent_of_n():
    k, p = 2, 2
    cols = [f"y{j}" for j in range(k)]
    s_small = BVARModel(p=p)
    s_small.fit(_make_df(500, k), cols,
                memory_guard=MemoryGuard(max_memory_mb=500))
    s_large = BVARModel(p=p)
    s_large.fit(_make_df(5_000, k), cols,
                memory_guard=MemoryGuard(max_memory_mb=500))
    assert s_small.state_nbytes() == s_large.state_nbytes()


def test_ooc_batch_fixed():
    """Peak ratio n=50k/n=10k < 2.0 at fixed batch_size=1000, median-of-3."""
    import statistics
    k, p = 2, 2
    cols = [f"y{j}" for j in range(k)]
    bs = 1000
    peaks = []
    for n in (10_000, 50_000):
        df = _make_df(n, k)
        mg = MemoryGuard(max_memory_mb=50)
        _malloc_trim(); gc.collect()
        cell = []
        for _ in range(3):
            gc.collect()
            m = BVARModel(p=p)
            r = measure_peak_traced(
                m.fit, df, cols, batch_size=bs,
                memory_guard=mg, mg=mg)
            cell.append(int(r["peak_bytes"]))
        peaks.append(int(statistics.median(cell)))
    ratio = peaks[-1] / max(peaks[0], 1)
    assert ratio < 2.0, f"ratio={ratio:.2f}; peaks={peaks}"


def test_guard_refuses_at_tight():
    """Tight guard + large k*p -> MemoryLimitExceeded from bvar.state."""
    k, p = 20, 10
    df = _make_df(1000, k)
    cols = [f"y{j}" for j in range(k)]
    m = BVARModel(p=p)
    # est_state ~364 KB at k=20, p=10; budget=0.2 MB < state
    with pytest.raises(MemoryLimitExceeded):
        m.fit(df, cols, memory_guard=MemoryGuard(max_memory_mb=0.2))


