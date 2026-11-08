"""v0.9.143-batch5: verification tests for cadw.ml.reservoir_sample (DEC-033).

State = k'*d*f64 + 256. CW-1..5 + correctness + reservoir_sample primitive.
"""
import ctypes
import gc

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml.reservoir_sample import fit_reservoir_sample, ReservoirSampleModel
from cadw.ops.sampling import reservoir_sample


def _make_df(n, d, seed=0):
    rng = np.random.RandomState(seed)
    return CadwDataFrame.from_table(pa.table({
        f"x{j}": pa.array(rng.normal(size=n), type=pa.float64())
        for j in range(d)
    }))


def _cols(d):
    return [f"x{j}" for j in range(d)]


def _warmup():
    df = _make_df(200, 3, seed=999)
    fit_reservoir_sample(df, feature_cols=_cols(3), k=32,
                         memory_guard=MemoryGuard(max_memory_mb=500),
                         batch_size=100)


_warmup()


# ---- primitive tests ----

def test_primitive_reservoir_returns_k():
    """k < n -> sample size == k."""
    n, d, k = 1000, 3, 50
    sample, n_seen = reservoir_sample(
        _make_df(n, d)._iter_filtered_batches(), _cols(d), k)
    assert sample.shape == (k, d)
    assert n_seen == n


def test_primitive_reservoir_k_gt_n():
    """k > n -> sample size == n."""
    n, d, k = 50, 3, 100
    sample, n_seen = reservoir_sample(
        _make_df(n, d)._iter_filtered_batches(), _cols(d), k)
    assert sample.shape == (n, d)
    assert n_seen == n


def test_primitive_reservoir_uniform():
    """Sample from uniform [0,1] should have mean ~0.5."""
    n, d, k = 5000, 1, 500
    rng = np.random.RandomState(0)
    df = CadwDataFrame.from_table(pa.table({
        "x": pa.array(rng.random(n), type=pa.float64())
    }))
    sample, _ = reservoir_sample(
        df._iter_filtered_batches(), ["x"], k, seed=1)
    assert 0.4 < sample.mean() < 0.6


def test_primitive_empty_raises():
    df = CadwDataFrame.from_table(pa.table({
        "x": pa.array([], type=pa.float64())
    }))
    with pytest.raises(Exception):
        reservoir_sample(df._iter_filtered_batches(), ["x"], 10)


# ---- model tests ----

@pytest.mark.parametrize("k,d,n", [(10, 2, 500), (50, 5, 5000), (100, 10, 2000)])
def test_state_nbytes_exact(k, d, n):
    df = _make_df(n, d, seed=0)
    m = fit_reservoir_sample(df, feature_cols=_cols(d), k=k,
                             memory_guard=MemoryGuard(max_memory_mb=500),
                             batch_size=200)
    expected = min(k, n) * d * 8 + 256
    assert m.state_nbytes() == expected, (
        f"k={k} d={d} n={n}: got {m.state_nbytes()}, expected {expected}")


def test_state_independent_of_n():
    k, d = 50, 5
    m1 = fit_reservoir_sample(_make_df(500, d, seed=0),
                              feature_cols=_cols(d), k=k,
                              memory_guard=MemoryGuard(max_memory_mb=500),
                              batch_size=100)
    m2 = fit_reservoir_sample(_make_df(5000, d, seed=0),
                              feature_cols=_cols(d), k=k,
                              memory_guard=MemoryGuard(max_memory_mb=500),
                              batch_size=100)
    assert m1.state_nbytes() == m2.state_nbytes()


def test_guard_refuses_at_tight():
    df = _make_df(200, 20)
    with pytest.raises((MemoryLimitExceeded, Exception)):
        fit_reservoir_sample(df, feature_cols=_cols(20), k=100_000,
                             memory_guard=MemoryGuard(max_memory_mb=0.01),
                             batch_size=100)


def test_sample_batches_streaming():
    k, d = 32, 3
    df = _make_df(500, d, seed=1)
    m = fit_reservoir_sample(df, feature_cols=_cols(d), k=k,
                             memory_guard=MemoryGuard(max_memory_mb=500),
                             batch_size=100)
    total = 0
    for rb in m.sample_batches():
        assert rb.num_columns == d
        total += rb.num_rows
    assert total == k
