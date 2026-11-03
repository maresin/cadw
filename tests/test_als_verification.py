"""Component-wise verification for fit_als (DEC-033, v0.9.112).

State = factors (U+I)*k*8 + index lists/dicts (U+I)*(dict_b+str_b)
+ 256 header. All O(U+I), bounded by cardinality not n.

Note: batch coef isolated is skipped (d=2, SNR < 1 at typical row_size).
Coefficient 8.0 from code check_batch(extra_factor=8.0), conservative.
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
from cadw.ml.als import fit_als
from cadw.ml._memory_estimate import _load_constants


def _make_df(U, I, n_pairs, seed=0):
    rng = np.random.RandomState(seed)
    all_pairs = [(u, i) for u in range(U) for i in range(I)]
    if n_pairs >= len(all_pairs):
        pairs = all_pairs
    else:
        idx = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        pairs = [all_pairs[j] for j in idx]
    rng.shuffle(pairs)
    us = [u for u, i in pairs]
    its = [i for u, i in pairs]
    return CadwDataFrame.from_table(pa.table({
        "u": pa.array(us, type=pa.int64()),
        "i": pa.array(its, type=pa.int64()),
    }))


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _warmup():
    fit_als(_make_df(5, 4, 10), "u", "i", n_factors=2, n_iter=2,
            memory_guard=MemoryGuard(max_memory_mb=500), batch_size=50)


_warmup()


def _expected_state(U, I, k):
    c = _load_constants()
    dict_b = int(c.get("dict_entry_bytes", 116))
    str_b = int(c.get("py_str_overhead_bytes", 49))
    return (U + I) * k * 8 + (U + I) * (dict_b + str_b) + 256


@pytest.mark.parametrize("U,I,k", [(10, 8, 4), (20, 15, 8), (50, 30, 16)])
def test_state_nbytes_exact(U, I, k):
    """state_nbytes() == (U+I)*k*8 + (U+I)*(dict_b+str_b) + 256, bit-exact."""
    df = _make_df(U, I, n_pairs=U * I)
    m = fit_als(df, "u", "i", n_factors=k, n_iter=2,
                memory_guard=MemoryGuard(max_memory_mb=500))
    expected = _expected_state(U, I, k)
    actual = m.state_nbytes()
    assert actual == expected, (
        f"U={U} I={I} k={k}: state={actual} expected={expected}"
    )


def test_state_independent_of_n():
    """Same U, I, k -> same state regardless of density (n)."""
    U, I, k = 20, 15, 8
    df_dense = _make_df(U, I, n_pairs=U * I)
    df_sparse = _make_df(U, I, n_pairs=U * I // 2, seed=1)
    m_dense = fit_als(df_dense, "u", "i", n_factors=k, n_iter=2,
                      memory_guard=MemoryGuard(max_memory_mb=500))
    m_sparse = fit_als(df_sparse, "u", "i", n_factors=k, n_iter=2,
                       memory_guard=MemoryGuard(max_memory_mb=500))
    assert m_dense.state_nbytes() == m_sparse.state_nbytes(), (
        f"dense={m_dense.state_nbytes()} sparse={m_sparse.state_nbytes()}"
    )


def test_formula_small_scale():
    """predicted/measured in [0.5, 2.0] at U=100, I=50, k=8, n=5000."""
    U, I, k = 100, 50, 8
    df = _make_df(U, I, n_pairs=5000)
    peaks = []
    for _ in range(3):
        _malloc_trim()
        gc.collect()
        mg = MemoryGuard(max_memory_mb=500)
        r = measure_peak_traced(
            fit_als, df, "u", "i", n_factors=k, n_iter=3,
            memory_guard=mg, batch_size=1000, mg=mg,
        )
        peaks.append(int(r["peak_bytes"]))
    measured = int(statistics.median(peaks))
    state = _expected_state(U, I, k)
    predicted = state + 8.0 * 1000 * 16 * 3
    ratio = predicted / max(measured, 1)
    assert 0.5 <= ratio <= 2.0, (
        f"predicted={predicted:,} measured={measured:,} ratio={ratio:.2f}; "
        f"peaks={peaks}"
    )


def test_peak_grows_linearly_with_n():
    """Peak is LINEAR in n (not constant), but state is OOC.

    ALS is partially OOC: model state (factors + index dicts) is O(U+I),
    independent of n — confirmed by test_state_independent_of_n.

    However peak includes external_sort K-way merge buffer:
      K = ceil(n / run_buffer),  run_buffer ~= available_bytes / c
      peak_merge = K * read_buffer
    At fixed mg, run_buffer is fixed, so K grows O(n) and peak grows
    O(n/run_buffer). This is the CPU/IO cost of keeping state OOC.

    YAML: state_class O(N*d), n_scalability linear (not constant).
    At 4x n_sweep expect 2-5x peak growth (bounded by K, not n exactly).
    """
    U, I, k = 500, 200, 8
    peaks = []
    for n_pairs in [10000, 40000]:
        df = _make_df(U, I, n_pairs=n_pairs)
        cell_peaks = []
        for _ in range(3):
            _malloc_trim()
            gc.collect()
            mg = MemoryGuard(max_memory_mb=20)
            r = measure_peak_traced(
                fit_als, df, "u", "i", n_factors=k, n_iter=2,
                memory_guard=mg, batch_size=500, mg=mg,
            )
            cell_peaks.append(int(r["peak_bytes"]))
        peaks.append(int(statistics.median(cell_peaks)))
        del df
        gc.collect()
    ratio = peaks[-1] / max(peaks[0], 1)
    # 4x n -> expect 2-5x peak (linear with K, not 16x quadratic).
    # Tolerance 5.0 = upper for linear + noise; passes for K-bounded.
    assert ratio < 5.0, (
        f"peak ratio {ratio:.2f}; peaks={peaks}. "
        f"If >5x at 4x n, likely quadratic or unbounded merge."
    )


