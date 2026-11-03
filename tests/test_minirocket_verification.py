"""Component-wise verification for fit_minirocket (DEC-033, v0.9.112).

MiniRocket fit is O(1) per n — validates columns + builds deterministic
kernel list, no data iteration. State = O(num_kernels).

Four checks:
  1. CW-1: state_nbytes() == num_kernels*(KERNEL_LEN*8+8) + 1024
  2. CW-2: same num_kernels -> same state (n-independent)
  3. CW-5: fit peak doesn't grow with n (median-of-3)
  4. Spill: tight guard raises MemoryLimitExceeded
"""
from __future__ import annotations
import statistics

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded, measure_peak_traced
from cadw.ml.minirocket import fit_minirocket, KERNEL_LEN


def _make_df(n, seed=0):
    rng = np.random.RandomState(seed)
    return CadwDataFrame.from_table(pa.table({
        "v": pa.array(rng.normal(size=n), type=pa.float64()),
    }))


def _warmup():
    fit_minirocket(_make_df(100), "v", num_kernels=4,
                   memory_guard=MemoryGuard(max_memory_mb=500))


_warmup()


@pytest.mark.parametrize("num_kernels", [1, 10, 84, 500])
def test_state_nbytes_exact(num_kernels):
    """state_nbytes() == num_kernels*(KERNEL_LEN*8 + 8) + 1024, bit-exact."""
    df = _make_df(100)
    m = fit_minirocket(df, "v", num_kernels=num_kernels,
                       memory_guard=MemoryGuard(max_memory_mb=500))
    expected = num_kernels * (KERNEL_LEN * 8 + 8) + 1024
    assert m.state_nbytes() == expected, (
        f"num_kernels={num_kernels}: state={m.state_nbytes()} expected={expected}"
    )


def test_state_independent_of_n():
    """Same num_kernels, different n -> same state."""
    for n in [100, 10000, 100000]:
        m = fit_minirocket(_make_df(n), "v", num_kernels=20,
                           memory_guard=MemoryGuard(max_memory_mb=500))
        if n == 100:
            ref = m.state_nbytes()
        else:
            assert m.state_nbytes() == ref, (
                f"n={n}: {m.state_nbytes()} != {ref}"
            )


def test_fit_ooc():
    """Fit is O(1): peak must not grow with n (median-of-3)."""
    def _peak(n):
        df = _make_df(n)
        peaks = []
        for _ in range(3):
            mg = MemoryGuard(max_memory_mb=500)
            r = measure_peak_traced(
                fit_minirocket, df, "v", num_kernels=84,
                memory_guard=mg, mg=mg,
            )
            peaks.append(int(r["peak_bytes"]))
        return int(statistics.median(peaks))

    p_small = _peak(1000)
    p_large = _peak(100000)
    ratio = p_large / max(p_small, 1)
    assert ratio < 2.0, f"peak ratio {ratio:.2f}; small={p_small} large={p_large}"


def test_tight_guard_raises():
    """num_kernels=20000 -> state ~1.6 MB; guard 1 MB -> MemoryLimitExceeded."""
    df = _make_df(100)
    mg = MemoryGuard(max_memory_mb=1)
    with pytest.raises(MemoryLimitExceeded):
        fit_minirocket(df, "v", num_kernels=20000, memory_guard=mg)
