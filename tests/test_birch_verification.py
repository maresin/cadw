"""Component-wise verification for fit_birch (DEC-033).

BIRCH CF-tree state is O(#CF nodes), bounded by branching_factor
(1-level tree split policy). No measurement tooling needed — state
is exact by construction.

Verification decomposed into (DEC-033 pattern, adapted for bounded
Python-object state):
  1. State exact via model.state_nbytes() (cf_count known from tree)
  2. CF count structural invariant: cf_count <= branching_factor
  3. State upper-bound: state_nbytes() <= formula upper for n=1k..20k
  4. Batch coef isolated: slope/row_size in [1.5, 4.0]
  5. Formula small-scale: predicted/measured in [0.5, 2.0]
  6. OOC: peak ratio < 2.0 at fixed batch (median-of-3)

Runtime target: <10 seconds (small dims).
"""
from __future__ import annotations
import ctypes
import gc
import statistics

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ml.birch import fit_birch


def _make_df(n, d, seed=0):
    rng = np.random.RandomState(seed)
    return CadwDataFrame.from_table(pa.table({
        f"x{j}": pa.array(rng.normal(size=n), type=pa.float64())
        for j in range(d)
    }))


def _cols(d):
    return [f"x{j}" for j in range(d)]


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _warmup():
    df = _make_df(64, 2, seed=999)
    fit_birch(df, ["x0", "x1"], n_clusters=2,
              memory_guard=MemoryGuard(max_memory_mb=500),
              batch_size=32)


_warmup()


def _state_upper(bf, k_bound, d):
    """Upper bound on state_nbytes: cf_count <= bf, k <= k_bound."""
    return bf * ((2 * d + 1) * 8 + 32) + k_bound * d * 8 + 512


# ---- 1: state_nbytes exact (cf_count known from tree) ----

@pytest.mark.parametrize("d,n_clusters,bf,n", [
    (2, 4, 50, 500),
    (10, 8, 50, 2000),
    (50, 8, 20, 1000),
])
def test_state_nbytes_exact(d, n_clusters, bf, n):
    """state_nbytes() == cf_count*((2d+1)*8+32) + k*d*8 + 512, bit-exact."""
    df = _make_df(n, d)
    m = fit_birch(df, _cols(d), n_clusters=n_clusters,
                  branching_factor=bf, threshold=0.5,
                  memory_guard=MemoryGuard(max_memory_mb=500),
                  batch_size=500)
    cf_count = m._cf_count()
    k = m.n_clusters_
    expected = cf_count * ((2 * d + 1) * 8 + 32) + k * d * 8 + 512
    actual = m.state_nbytes()
    assert actual == expected, f"actual={actual} expected={expected} (cf={cf_count}, k={k})"


# ---- 2: cf_count structural invariant ----

@pytest.mark.parametrize("n", [500, 5000, 20000])
def test_cf_count_bounded(n):
    """cf_count <= branching_factor for all n (1-level tree split policy)."""
    d, bf = 10, 20
    df = _make_df(n, d)
    m = fit_birch(df, _cols(d), n_clusters=8, branching_factor=bf,
                  memory_guard=MemoryGuard(max_memory_mb=500),
                  batch_size=1000)
    cf = m._cf_count()
    assert cf <= bf, f"cf_count={cf} > branching_factor={bf} at n={n}"


# ---- 3: state upper bound across n ----

def test_state_bounded_for_large_n():
    """state_nbytes() <= formula upper for all n (structure: O(model), not O(n))."""
    d, k_bound, bf = 50, 8, 20
    upper = _state_upper(bf, k_bound, d)
    states = []
    for n in [1000, 5000, 20000]:
        df = _make_df(n, d)
        m = fit_birch(df, _cols(d), n_clusters=k_bound,
                      branching_factor=bf,
                      memory_guard=MemoryGuard(max_memory_mb=500),
                      batch_size=1000)
        s = m.state_nbytes()
        assert s <= upper, f"state={s} > upper={upper} at n={n}"
        states.append(s)
    # state is data-dependent but bounded; ratio across n should be
    # small (< 3x) because cf_count varies slowly with data density.
    ratio = max(states) / max(min(states), 1)
    assert ratio <= 3.0, f"state ratio {ratio:.2f}x across n (expected <=3x); states={states}"


# ---- 4: batch coef isolated ----

def test_batch_coef_isolated():
    """Extract coef_batch from isolated sweep. Expected ~1.16 (see comment)."""
    d, bf, n_clusters = 200, 50, 8
    cols = _cols(d)
    row_size = 8 * d
    batches = [1000, 2000, 4000]
    peaks = []
    for b in batches:
        df = _make_df(max(b * 4, 4000), d)
        _malloc_trim()
        gc.collect()
        mg = MemoryGuard(max_memory_mb=500)
        r = measure_peak_traced(
            fit_birch, df, cols,
            n_clusters=n_clusters, branching_factor=bf, threshold=0.5,
            memory_guard=mg, batch_size=b, mg=mg,
        )
        peaks.append(int(r["peak_bytes"]))
        del df
        gc.collect()
    xs = np.array(batches, dtype=float)
    ys = np.array(peaks, dtype=float)
    slope, _intercept = np.polyfit(xs, ys, 1)
    coef = slope / row_size
    # BIRCH: _stack_numeric on f64 columns is np.column_stack only
    # (per-column astype(copy=False) is zero-copy for f64). No additional
    # per-batch ops (unlike iforest, which does reservoir + per-tree
    # scoring -> 2.77 coef). Empirical coef ~1.16 (probe v0.9.111:
    # peaks=[8.39, 10.89, 14.10] MB for batch=[1000, 2000, 4000]).
    # [0.8, 2.5] covers: RSS noise (SNR 1.2), allocator variance from
    # per-row _insert churn, moderate over-estimate headroom.
    # Formula uses 3.0 conservative (fit_linear v0.9.107 precedent;
    # see DEC-035 for the intercept discrepancy).
    assert 0.8 <= coef <= 2.5, (
        f"coef_batch={coef:.2f} (expected ~1.16); peaks={peaks}"
    )


# ---- 5: formula small-scale ----

def test_formula_small_scale():
    """predicted/measured in [0.5, 2.0] at d=200, batch=1000, n=4000.

    DEC-034: at d=200, batch=1000, peak ~6-8 MB, full-suite RSS noise
    ~5 MB -> SNR ~1.2. [0.5, 2.0] is honest band (SNR 1.2 -> ~1.5-sigma).
    Primary evidence: test_state_nbytes_exact (bit-exact) + test_batch_coef_isolated.

    DEC-035: formula uses coef_batch=3.0 (conservative); measured slope
    is 1.16. Additionally there is ~5-6 MB unexplained intercept
    (glibc arena fragmentation from 4000 per-row _insert transients of
    ~4x80 KB each). Formula 3.0*batch*f64 at batch=1000 = 5.0 MB;
    measured 8.4 MB -> ratio 0.60 (within [0.5, 2.0]). At batch=4000
    formula = 19.4 MB > measured 14.1 MB -> ratio 1.38 (also in band).
    """
    d, bf, n_clusters, batch, n = 200, 50, 8, 1000, 4000
    cols = _cols(d)
    row_size = 8 * d
    df = _make_df(n, d)
    peaks = []
    for _ in range(3):
        _malloc_trim()
        gc.collect()
        mg = MemoryGuard(max_memory_mb=500)
        r = measure_peak_traced(
            fit_birch, df, cols,
            n_clusters=n_clusters, branching_factor=bf, threshold=0.5,
            memory_guard=mg, batch_size=batch, mg=mg,
        )
        peaks.append(int(r["peak_bytes"]))
    measured = int(statistics.median(peaks))
    predicted = (
        _state_upper(bf, n_clusters, d)
        + 3.0 * batch * row_size
    )
    ratio = predicted / max(measured, 1)
    assert 0.5 <= ratio <= 2.0, (
        f"predicted={predicted:,} measured={measured:,} ratio={ratio:.2f}; "
        f"peaks={peaks}"
    )


# ---- 6: OOC (median-of-3 per DEC-034) ----

def test_ooc_batch_fixed():
    """Peak must not grow with n at fixed batch (bounded_state: true).

    DEC-034: median-of-3 per n to filter RSS noise in full-suite.
    """
    d, bf, n_clusters, batch = 200, 50, 8, 500
    cols = _cols(d)
    peaks = []
    for n in [1000, 5000]:
        df = _make_df(n, d)
        cell_peaks = []
        for _ in range(3):
            _malloc_trim()
            gc.collect()
            mg = MemoryGuard(max_memory_mb=500)
            r = measure_peak_traced(
                fit_birch, df, cols,
                n_clusters=n_clusters, branching_factor=bf, threshold=0.5,
                memory_guard=mg, batch_size=batch, mg=mg,
            )
            cell_peaks.append(int(r["peak_bytes"]))
        peaks.append(int(statistics.median(cell_peaks)))
        del df
        gc.collect()
    ratio = peaks[-1] / max(peaks[0], 1)
    assert ratio < 2.0, f"peak ratio {ratio:.2f}; peaks={peaks}"
