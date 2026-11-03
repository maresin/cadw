"""Component-wise verification for fit_linucb (DEC-033).

Rationale: RSS-based combined matrix is unmeasurable when baseline
(python + arrow + numpy + parquet mmap) >> fit peak — RSS doesn't grow
because allocator reuses existing pool. See DEC-033.

Verification decomposed into:
  1. State: exact via model.state_nbytes() (no measurement needed)
  2. Batch coef: isolated sweep (small d, small baseline)
  3. Combined: formula = state + 2*K*d^2*f64 + coef*batch*row_size
  4. OOC: peak ratio at fixed batch, varying n

Total runtime target: <30 seconds (small dims).
"""
from __future__ import annotations
import ctypes
import gc

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ml.linucb import fit_linucb


def _make_df(n, d, K=5, seed=0):
    rng = np.random.RandomState(seed)
    theta = rng.normal(size=(K, d)) * 0.5
    X = rng.normal(size=(n, d))
    arms = rng.randint(0, K, size=n).astype(np.int64)
    r = np.einsum("ij,ij->i", X, theta[arms]) + 0.05 * rng.normal(size=n)
    tbl = pa.table({
        **{f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(d)},
        "arm": pa.array(arms, type=pa.int64()),
        "r": pa.array(r, type=pa.float64()),
    })
    return CadwDataFrame.from_table(tbl)


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _warmup(d=2, K=2):
    """Trigger lazy imports / pools."""
    rng = np.random.RandomState(999)
    tbl = pa.table({
        "x0": pa.array(rng.normal(size=64), type=pa.float64()),
        "x1": pa.array(rng.normal(size=64), type=pa.float64()),
        "arm": pa.array(rng.randint(0, 2, 64).astype(np.int64), type=pa.int64()),
        "r": pa.array(rng.normal(size=64), type=pa.float64()),
    })
    df = CadwDataFrame.from_table(tbl)
    fit_linucb(df, ["x0", "x1"], arm_col="arm", reward_col="r",
               memory_guard=MemoryGuard(max_memory_mb=500),
               batch_size=32, k_max_hint=K)


_warmup()


def _expected_state(K, d):
    # from linucb.py: A = K*d*d*8 + b = K*d*8 + 256 (header)
    return K * d * d * 8 + K * d * 8 + 256


# ---- test 1: state_nbytes exact ----

@pytest.mark.parametrize("K,d", [(2, 10), (5, 50), (5, 100), (10, 200)])
def test_state_nbytes_exact(K, d):
    """model.state_nbytes() must equal exact formula (K*d^2*8 + K*d*8 + 256)."""
    n = 100
    df = _make_df(n, d, K=K, seed=0)
    cols = [f"x{i}" for i in range(d)]
    m = fit_linucb(df, cols, arm_col="arm", reward_col="r",
                   memory_guard=MemoryGuard(max_memory_mb=500),
                   batch_size=100, k_max_hint=K)
    assert m.n_arms_ == K
    actual = m.state_nbytes()
    expected = _expected_state(K, d)
    # state_nbytes includes A.nbytes + b.nbytes + 256
    assert actual == expected, f"state_nbytes={actual} expected={expected}"


# ---- test 2: state is n-independent ----

def test_state_independent_of_n():
    """State must not grow with n (bounded state guarantee)."""
    d, K = 50, 5
    cols = [f"x{i}" for i in range(d)]
    df_small = _make_df(200, d, K=K, seed=0)
    df_large = _make_df(2000, d, K=K, seed=0)
    m_small = fit_linucb(df_small, cols, arm_col="arm", reward_col="r",
                         memory_guard=MemoryGuard(max_memory_mb=500),
                         batch_size=200, k_max_hint=K)
    m_large = fit_linucb(df_large, cols, arm_col="arm", reward_col="r",
                         memory_guard=MemoryGuard(max_memory_mb=500),
                         batch_size=200, k_max_hint=K)
    assert m_small.state_nbytes() == m_large.state_nbytes()


# ---- test 3: batch coef isolated ----

def test_batch_coef_isolated():
    """Extract coef_batch from isolated sweep (small baseline, small dims).

    Formula: peak(batch) = state + 2*K*d^2*8 + coef_batch*batch*row_size
    At d=200, K=5:
      state = 5*40000*8 + 5*200*8 = 1,608,000 B
      temp  = 2*5*40000*8 = 3,200,000 B  (constant per cell)
      slope(batch) = coef * 1616 B
    """
    d, K = 200, 5
    cols = [f"x{i}" for i in range(d)]
    row_size = 8 * d + 16
    # batch >= 1000: batch term (b*row_size) = 1.6-6.4 MB >> RSS noise (~1 MB)
    batches = [1000, 2000, 4000]
    peaks = []
    for b in batches:
        # n >= 4*b: enough rows for 4 full batches
        df = _make_df(max(b * 4, 4000), d, K=K, seed=0)
        _malloc_trim()
        gc.collect()
        mg = MemoryGuard(max_memory_mb=500)
        r = measure_peak_traced(
            fit_linucb, df, cols,
            arm_col="arm", reward_col="r",
            memory_guard=mg, batch_size=b, k_max_hint=K,
            mg=mg,
        )
        peaks.append(int(r["peak_bytes"]))
        del df
        gc.collect()

    # Fit slope only (intercept from slope-fit unreliable - see NOTE).
    xs = np.array(batches, dtype=float)
    ys = np.array(peaks, dtype=float)
    slope, _intercept = np.polyfit(xs, ys, 1)
    coef = slope / row_size

    # Formula v6: coef_batch = 3.5 (measured 3.08-3.64 across 5 trials).
    # Components from code:
    #   Arrow batch materialization (1x batch*row_size)
    #   + X=column_stack (1x, batch*d*8)
    #   + per-arm Xa (1x, sum_a batch_a = batch)
    # = 3 physical copies.
    assert 2.5 <= coef <= 4.5, (
        f"coef_batch={coef:.2f} (expected 3.0-3.5, range 2.5-4.5); "
        f"peaks={peaks}, batches={batches}, row_size={row_size}"
    )

    # NOTE on intercept: intercept = p_low - slope*batch_low (extrapolation
    # to batch=0). At p_low~7 MB, slope*batch~5 MB, intercept~2 MB. RSS
    # noise +-1 MB -> intercept +-50% (5 trials: 0.18-1.47 ratio). Not
    # reliable as assertion at this signal size. Combined formula sanity
    # is covered by test_formula_small_scale (direct predicted/measured,
    # not extrapolated). See DEC-033.


# ---- test 4: formula sanity at small scale ----

def test_formula_small_scale():
    """At small scale (measurable RSS), formula v6 predicts peak within ±40%.

    Uses d=200, batch=1000, n=4000: state=1.6MB, temp=0.32MB, batch=3.5*1000*1616
    =5.66MB, total ~7.6 MB. RSS noise ±1 MB -> ±13%. Better signal-to-noise
    than batch=200 (where signal was 3 MB and noise ±33%).

    Median-of-3 runs: sporadic GC/allocator timing shifts (~1 MB) filtered
    by median.
    """
    d, K, batch, n = 200, 5, 1000, 4000
    cols = [f"x{i}" for i in range(d)]
    row_size = 8 * d + 16
    df = _make_df(n, d, K=K, seed=0)
    peaks = []
    for _ in range(3):
        _malloc_trim()
        gc.collect()
        mg = MemoryGuard(max_memory_mb=500)
        r = measure_peak_traced(
            fit_linucb, df, cols,
            arm_col="arm", reward_col="r",
            memory_guard=mg, batch_size=batch, k_max_hint=K,
            mg=mg,
        )
        peaks.append(int(r["peak_bytes"]))
    import statistics
    measured = int(statistics.median(peaks))
    predicted = (
        _expected_state(K, d)    # A(K*d^2*8) + b(K*d*8) + 256
        + d * d * 8              # one temp in Xa.T @ Xa
        + 3.5 * batch * row_size # X + Xa + arrow batch, empirical
    )
    ratio = predicted / max(measured, 1)
    assert 0.7 <= ratio <= 1.4, (
        f"predicted={predicted:,} measured={measured:,} ratio={ratio:.2f}; "
        f"peaks={peaks}"
    )


# ---- test 5: OOC proven ----

def test_ooc_batch_fixed():
    """Peak must not grow with n at fixed batch (OOC guarantee)."""
    d, K, batch = 200, 5, 200
    cols = [f"x{i}" for i in range(d)]
    peaks = []
    for n in [1000, 5000]:
        df = _make_df(n, d, K=K, seed=0)
        _malloc_trim()
        gc.collect()
        mg = MemoryGuard(max_memory_mb=500)
        r = measure_peak_traced(
            fit_linucb, df, cols,
            arm_col="arm", reward_col="r",
            memory_guard=mg, batch_size=batch, k_max_hint=K,
            mg=mg,
        )
        peaks.append(int(r["peak_bytes"]))
        del df
        gc.collect()
    ratio = peaks[-1] / max(peaks[0], 1)
    assert ratio < 2.0, f"peak ratio {ratio:.2f} (n=5000/n=1000); peaks={peaks}"
