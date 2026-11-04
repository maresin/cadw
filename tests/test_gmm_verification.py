"""Component-wise verification for fit_gmm (DEC-033).

Formula (v0.9.110g):
  peak = state_nbytes(k, d) + 3*S*d*f64 + coef_batch*batch*row_size
  where:
    state_nbytes = k*(1+2d)*f64    (weights k + means k*d + covars k*d)
    S = min(n, max_samples)         (reservoir sample size)
    3*S*d*f64                       (kmeans++ temp + _responsibilities
                                     temp + M-step temp)
    coef_batch = 3.5                (X_b + column_stack + X[mask]
                                     copies; measured)
"""
from __future__ import annotations
import ctypes
import gc

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ml.gmm import fit_gmm


def _make_df(n, d, seed=0):
    rng = np.random.RandomState(seed)
    return CadwDataFrame.from_table(pa.table({
        f"x{j}": pa.array(rng.normal(size=n), type=pa.float64())
        for j in range(d)
    }))


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _warmup():
    df = _make_df(64, 2, seed=999)
    fit_gmm(df, ["x0", "x1"], n_components=2, max_samples=64,
            random_state=0, memory_guard=MemoryGuard(max_memory_mb=500),
            batch_size=32)


_warmup()


def _state_expected(K, d):
    return K * (1 + 2 * d) * 8


# ---- CW-1: state exact ----

@pytest.mark.parametrize("K,d", [(2, 10), (3, 50), (5, 100), (3, 200)])
def test_state_nbytes_exact(K, d):
    """model.state_nbytes() must equal k*(1+2d)*8 exactly."""
    df = _make_df(200, d)
    cols = [f"x{i}" for i in range(d)]
    m = fit_gmm(df, cols, n_components=K, max_samples=200,
                random_state=0,
                memory_guard=MemoryGuard(max_memory_mb=500),
                batch_size=200)
    actual = m.state_nbytes()
    expected = _state_expected(K, d)
    assert actual == expected, (
        f"state_nbytes={actual}, expected={expected} (K={K}, d={d})")


# ---- CW-2: state bounded / n-independent ----

def test_state_independent_of_n():
    """state_nbytes depends only on (K, d), not n or max_samples saturation."""
    d, K, max_S = 50, 3, 1000
    cols = [f"x{i}" for i in range(d)]
    m_small = fit_gmm(_make_df(500, d), cols, n_components=K,
                      max_samples=max_S, random_state=0,
                      memory_guard=MemoryGuard(max_memory_mb=500),
                      batch_size=200)
    m_large = fit_gmm(_make_df(5000, d), cols, n_components=K,
                      max_samples=max_S, random_state=0,
                      memory_guard=MemoryGuard(max_memory_mb=500),
                      batch_size=200)
    assert m_small.state_nbytes() == m_large.state_nbytes()


# ---- CW-3: batch coef isolated ----

def test_batch_coef_isolated():
    """d=200, max_samples=1000, batches 1000/2000/4000.

    Signal: batch term 1.6-6.4 MB vs RSS noise ~1 MB. Coef ~3.5
    (X_b stack + column_stack + X[mask] per row).

    v0.9.114 (DEC-034 precedent): median-of-3 per batch. In full-suite
    (4400+ prior tests) baseline RSS from Python allocator + Arrow
    pools raises `measure_peak_traced` noise to ~3-5 MB; single-run
    coef drifts outside [1.5, 3.0] (flake observed in v0.9.114 full
    fast; passes in isolation). Median-of-3 filters sporadic spikes.
    """
    import statistics
    d, K, max_S = 200, 5, 1000
    cols = [f"x{i}" for i in range(d)]
    row_size = 8 * d
    batches = [1000, 2000, 4000]
    peaks = []
    for b in batches:
        df = _make_df(max(b * 4, 4000), d)
        cell_peaks = []
        for _ in range(3):
            _malloc_trim()
            gc.collect()
            mg = MemoryGuard(max_memory_mb=500)
            r = measure_peak_traced(
                fit_gmm, df, cols,
                n_components=K, max_samples=max_S, random_state=0,
                memory_guard=mg, batch_size=b, mg=mg,
            )
            cell_peaks.append(int(r["peak_bytes"]))
            del r
            gc.collect()
        peaks.append(int(statistics.median(cell_peaks)))
        del df
        gc.collect()
    xs = np.array(batches, dtype=float)
    ys = np.array(peaks, dtype=float)
    slope, _ = np.polyfit(xs, ys, 1)
    coef = slope / row_size
    # Formula v2: coef 2.0 (arrays list + column_stack). Median-of-3
    # measurement: 1.75-2.0 typical; tolerance [1.5, 3.0] (unchanged
    # bound, only noise filtered via median).
    assert 1.5 <= coef <= 3.0, (
        f"coef_batch={coef:.2f} (expected ~2.0); peaks={peaks}")


# ---- CW-4: formula small scale (median-of-3) ----

def test_formula_small_scale():
    """predicted/measured in [0.7, 1.4] at d=200, batch=1000, S=1000.

    predicted = state_nbytes + 3*S*d*f64 + 3.5*batch*row_size
             = 16040 + 4.8M + 5.6M = 10.4 MB
    measured ~10 MB, RSS noise ±1 MB → ratio 0.9-1.1.
    """
    d, K, S, batch, n = 200, 5, 1000, 1000, 4000
    cols = [f"x{i}" for i in range(d)]
    row_size = 8 * d
    df = _make_df(n, d)
    peaks = []
    for _ in range(3):
        _malloc_trim()
        gc.collect()
        mg = MemoryGuard(max_memory_mb=500)
        r = measure_peak_traced(
            fit_gmm, df, cols,
            n_components=K, max_samples=S, random_state=0,
            memory_guard=mg, batch_size=batch, mg=mg,
        )
        peaks.append(int(r["peak_bytes"]))
    import statistics
    measured = int(statistics.median(peaks))
    # v2: sample (S*d*8) + M-step temp (4*S*d*8) + coef 2.0*batch*row_size
    predicted = _state_expected(K, d) + 5 * S * d * 8 + 2.0 * batch * row_size
    ratio = predicted / max(measured, 1)
    assert 0.7 <= ratio <= 1.4, (
        f"predicted={predicted:,} measured={measured:,} ratio={ratio:.2f}; "
        f"peaks={peaks}")


# ---- CW-5: OOC ----

def test_ooc_batch_fixed():
    """Peak does not grow with n at fixed batch (bounded_state: true)."""
    d, K, S, batch = 200, 5, 1000, 500
    cols = [f"x{i}" for i in range(d)]
    peaks = []
    for n in [1000, 5000]:
        df = _make_df(n, d)
        _malloc_trim()
        gc.collect()
        mg = MemoryGuard(max_memory_mb=500)
        r = measure_peak_traced(
            fit_gmm, df, cols,
            n_components=K, max_samples=S, random_state=0,
            memory_guard=mg, batch_size=batch, mg=mg,
        )
        peaks.append(int(r["peak_bytes"]))
        del df
        gc.collect()
    ratio = peaks[-1] / max(peaks[0], 1)
    assert ratio < 2.0, f"peak ratio {ratio:.2f}; peaks={peaks}"
