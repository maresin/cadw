"""Component-wise verification for fit_coxph / predict_risk (DEC-033, v0.9.114).

State = coef(d) + cov(d,d if closed) + baseline_t/h(n_events*2) + 256.
fit_coxph: OOC on n via external_sort Newton. predict_risk: OOC on n
via SpillArray (output="auto"|"disk"). concordance_index: O(n) intrinsic
(Ts/Es/Rs concat + chunked pair matrix); v0.9.97 do-not-re-solve; bounded
by n_events (<= n) and n<=1e6 (statistical stability).
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
from cadw.ml.coxph import fit_coxph


def _make_df(n, d, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    # True linear predictor
    beta = rng.normal(size=d) * 0.3
    eta = X @ beta
    T = rng.exponential(scale=np.exp(-eta))
    # 30% censoring
    C = rng.exponential(scale=np.exp(-eta) * 2.0)
    T_obs = np.minimum(T, C)
    E = (T <= C).astype(np.float64)
    tbl = pa.table({
        **{f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(d)},
        "T": pa.array(T_obs, type=pa.float64()),
        "E": pa.array(E, type=pa.float64()),
    })
    return CadwDataFrame.from_table(tbl)


def _cols(d):
    return [f"x{j}" for j in range(d)]


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _warmup():
    df = _make_df(200, 3, seed=999)
    m = fit_coxph(df, _cols(3), "T", "E", max_iter=5,
                  memory_guard=MemoryGuard(max_memory_mb=500),
                  batch_size=100)
    m.predict_risk(df, output="memory")


_warmup()


# ---- 1: state_nbytes exact ----
@pytest.mark.parametrize("d,n", [(3, 500), (10, 1000), (50, 2000)])
def test_state_nbytes_exact(d, n):
    """state_nbytes() = coef(d) + cov(d,d if closed) + baseline_t/h(n_events) + 256."""
    df = _make_df(n, d, seed=1)
    m = fit_coxph(df, _cols(d), "T", "E", max_iter=10,
                  solver="closed",
                  memory_guard=MemoryGuard(max_memory_mb=500),
                  batch_size=200)
    expected = (
        d * 8                                  # coef
        + (d * d * 8 if m.covariance_ is not None else 0)  # cov
        + m.baseline_t_.nbytes + m.baseline_h_.nbytes     # baseline
        + 256
    )
    actual = m.state_nbytes()
    assert actual == expected, (
        f"d={d} n={n}: state={actual} expected={expected}")


# ---- 2: state bounded by n_events, not n_rows ----
def test_state_bounded_by_n_events():
    """state ~ n_events (event count), not n_rows (total). Censoring reduces."""
    d = 10
    # 90% censoring: n_events ~ 0.1 * n
    n_rows = 5000
    rng = np.random.RandomState(2)
    X = rng.normal(size=(n_rows, d))
    T_obs = rng.exponential(size=n_rows)
    E = (rng.uniform(size=n_rows) < 0.1).astype(np.float64)  # 90% censored
    df = CadwDataFrame.from_table(pa.table({
        **{f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(d)},
        "T": pa.array(T_obs, type=pa.float64()),
        "E": pa.array(E, type=pa.float64()),
    }))
    m = fit_coxph(df, _cols(d), "T", "E", max_iter=10, solver="closed",
                  memory_guard=MemoryGuard(max_memory_mb=500),
                  batch_size=200)
    # state ≤ coef + cov + 2*n_events*8 + 256
    n_events = len(m.baseline_t_)
    upper = d * 8 + d * d * 8 + 2 * n_events * 8 + 256
    assert m.state_nbytes() == upper
    # n_events << n_rows due to 90% censoring
    assert n_events < n_rows * 0.5, (
        f"n_events={n_events} not << n_rows={n_rows}")


# ---- 3: fit OOC (external_sort Newton) ----
