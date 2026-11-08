"""Component-wise verification for fit_kmeans (DEC-033, v0.9.138).

Model state = cluster_centers_(k*d*f64) + inertia_(8) + header(256).
Fit peak = 2*k*batch*row_size + k*d*f64 + k*f64 + f64 + 256
         (KMeansStepOp per-batch distance matrix + reduce state).
bounded_state: true.
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
from cadw.ml.kmeans import fit_kmeans


def _make_df(n, d, seed=0):
    rng = np.random.RandomState(seed)
    return CadwDataFrame.from_table(pa.table({
        f"x{j}": pa.array(rng.randn(n), type=pa.float64())
        for j in range(d)
    }))


def _cols(d):
    return [f"x{j}" for j in range(d)]


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


_warmup_df = _make_df(100, 5)
fit_kmeans(_warmup_df, _cols(5), k=3,
           memory_guard=MemoryGuard(max_memory_mb=500))


@pytest.mark.parametrize("k,d", [(3, 5), (5, 20), (10, 50)])
def test_state_nbytes_exact(k, d):
    """model.state_nbytes() == k*d*f64 + 8 (inertia) + 256 (header)."""
    df = _make_df(500, d)
    m = fit_kmeans(df, _cols(d), k=k,
                   memory_guard=MemoryGuard(max_memory_mb=500))
    expected = k * d * 8 + 8 + 256
    assert m.state_nbytes() == expected, (
        f"k={k} d={d}: state={m.state_nbytes()} expected={expected}")


def test_state_independent_of_n():
    """state(k,d fixed) unchanged across n=1e3 -> 1e5."""
    k, d = 5, 20
    s_small = fit_kmeans(_make_df(1_000, d), _cols(d), k=k,
                         memory_guard=MemoryGuard(max_memory_mb=500)
                         ).state_nbytes()
    s_large = fit_kmeans(_make_df(100_000, d), _cols(d), k=k,
                         memory_guard=MemoryGuard(max_memory_mb=500)
                         ).state_nbytes()
    assert s_small == s_large, (s_small, s_large)


def test_ooc_batch_fixed():
    """Peak ratio n=50k/n=10k < 2.0 at fixed batch_size=1000.

    STABILITY_PROTOCOL §2.1: fixed batch, vary n. Median-of-3 per
    cell (DEC-034 precedent: measure_peak_traced noise ~1-2 MB; at
    fit peak ~2 MB SNR < 2, median filters flake). Without explicit
    batch_size, auto-derive at budget=10 MB gives bs=9174, and at
    n=5000 n<bs (1 batch) — not an OOC test.
    """
    import statistics
    k, d = 5, 10
    bs = 1000
    peaks = []
    for n in (10_000, 50_000):
        df = _make_df(n, d)
        mg = MemoryGuard(max_memory_mb=10)
        _malloc_trim(); gc.collect()
        cell = []
        for _ in range(3):
            gc.collect()
            r = measure_peak_traced(
                fit_kmeans, df, _cols(d), k=k,
                max_iter=10, n_init=1, batch_size=bs,
                memory_guard=mg, mg=mg)
            cell.append(int(r["peak_bytes"]))
        peaks.append(int(statistics.median(cell)))
    ratio = peaks[-1] / max(peaks[0], 1)
    assert ratio < 2.0, f"ratio={ratio:.2f}; peaks={peaks}"


def test_guard_refuses_at_tight():
    """Tight guard + large k*d → MemoryLimitExceeded, not OOM."""
    d, k = 500, 20
    df = _make_df(2000, d)
    df._memory_guard = MemoryGuard(max_memory_mb=1)
    with pytest.raises(MemoryLimitExceeded):
        fit_kmeans(df, _cols(d), k=k)
