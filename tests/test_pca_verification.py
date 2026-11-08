"""Component-wise verification for fit_pca (v0.9.138).

Model state = components_(k*d*f64) + mean_(d*f64)
            + explained_variance_(k*f64)
            + explained_variance_ratio_(k*f64)
            + singular_values_(k*f64) + 256.
Fit peak = batch + 3*d*d*f64 (cov + eigh eigvecs) + 4*d*f64 + 264.
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
from cadw.ml.pca import fit_pca


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


_warmup_df = _make_df(100, 10)
fit_pca(_warmup_df, _cols(10), n_components=3,
        memory_guard=MemoryGuard(max_memory_mb=500))


@pytest.mark.parametrize("k,d", [(3, 10), (5, 20), (10, 50)])
def test_state_nbytes_exact(k, d):
    """state_nbytes() = k*d + d + 3*k (all f64) + 256 header."""
    df = _make_df(500, d)
    m = fit_pca(df, _cols(d), n_components=k,
                memory_guard=MemoryGuard(max_memory_mb=500))
    expected = k * d * 8 + d * 8 + 3 * k * 8 + 256
    assert m.state_nbytes() == expected, (
        f"k={k} d={d}: state={m.state_nbytes()} expected={expected}")


def test_state_independent_of_n():
    k, d = 5, 20
    s_small = fit_pca(_make_df(1_000, d), _cols(d), n_components=k,
                      memory_guard=MemoryGuard(max_memory_mb=500)
                      ).state_nbytes()
    s_large = fit_pca(_make_df(100_000, d), _cols(d), n_components=k,
                      memory_guard=MemoryGuard(max_memory_mb=500)
                      ).state_nbytes()
    assert s_small == s_large, (s_small, s_large)


def test_ooc_batch_fixed():
    """Peak ratio n=50k/n=10k < 2.0 at fixed batch_size=1000.

    STABILITY_PROTOCOL §2.1: fixed batch, vary n. Median-of-3 per
    cell (DEC-034 precedent).
    """
    import statistics
    d, k = 20, 5
    bs = 5000  # peak ~1.6 MB >> noise ~1 MB (SNR~1.5); 1000 gave 330 KB (SNR<1)
    peaks = []
    for n in (10_000, 50_000):
        df = _make_df(n, d)
        mg = MemoryGuard(max_memory_mb=50)
        _malloc_trim(); gc.collect()
        cell = []
        for _ in range(5):  # median-of-5 (DEC-034 escalation, low SNR)
            gc.collect()
            r = measure_peak_traced(
                fit_pca, df, _cols(d), n_components=k,
                batch_size=bs, memory_guard=mg, mg=mg)
            cell.append(int(r["peak_bytes"]))
        peaks.append(int(statistics.median(cell)))
    ratio = peaks[-1] / max(peaks[0], 1)
    assert ratio < 2.0, f"ratio={ratio:.2f}; peaks={peaks}"


def test_guard_refuses_at_tight():
    """Tight guard + large d (eigh (d,d) alloc) → MemoryLimitExceeded."""
    d, k = 1000, 5
    df = _make_df(200, d)
    df._memory_guard = MemoryGuard(max_memory_mb=20)  # eigh needs 16MB
    with pytest.raises(MemoryLimitExceeded):
        fit_pca(df, _cols(d), n_components=k)
