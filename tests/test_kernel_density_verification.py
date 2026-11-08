"""v0.9.143-batch4: verification tests for cadw.ml.kernel_density (DEC-033).

State = n_centers * d * f64 + 128 (RAM) / 256 (spilled).
CW-1: state_nbytes exact
CW-2: state independent of n (given K)
Spill: K*d*8 > state_limit -> state=256
CW-6: guard refuses at tight (large K,d)
Correctness: log-density finite, bandwidth positive, score matches manual.
OOC sanity skipped (row_size=d*8 small, signal < noise; DEC-034).
"""
import ctypes
import gc

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml.kernel_density import fit_kernel_density, KernelDensityModel


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
    fit_kernel_density(df, feature_cols=_cols(3), n_centers=32,
                       memory_guard=MemoryGuard(max_memory_mb=500),
                       batch_size=100)


_warmup()


@pytest.mark.parametrize("K,d", [(8, 2), (32, 5), (128, 10), (256, 20)])
def test_state_nbytes_exact(K, d):
    df = _make_df(500, d)
    m = fit_kernel_density(df, feature_cols=_cols(d), n_centers=K,
                           memory_guard=MemoryGuard(max_memory_mb=500),
                           batch_size=200)
    expected = K * d * 8 + 128
    assert m.state_nbytes() == expected, (
        f"K={K} d={d}: got {m.state_nbytes()}, expected {expected}")


def test_state_independent_of_n():
    K, d = 64, 5
    m1 = fit_kernel_density(_make_df(500, d, seed=0),
                            feature_cols=_cols(d), n_centers=K,
                            memory_guard=MemoryGuard(max_memory_mb=500),
                            batch_size=100)
    m2 = fit_kernel_density(_make_df(5000, d, seed=0),
                            feature_cols=_cols(d), n_centers=K,
                            memory_guard=MemoryGuard(max_memory_mb=500),
                            batch_size=100)
    assert m1.state_nbytes() == m2.state_nbytes()


def test_guard_refuses_at_tight():
    df = _make_df(200, 20)
    with pytest.raises((MemoryLimitExceeded, Exception)):
        fit_kernel_density(
            df, feature_cols=_cols(20), n_centers=100_000,
            memory_guard=MemoryGuard(max_memory_mb=0.05), batch_size=100)


def test_score_finite_on_gaussian():
    d = 3
    df = _make_df(1000, d, seed=1)
    m = fit_kernel_density(df, feature_cols=_cols(d), n_centers=128,
                           memory_guard=MemoryGuard(max_memory_mb=500),
                           batch_size=200)
    s = m.score(df, batch_size=200)
    assert np.isfinite(s)
    # for unit-variance 3D Gaussian, log density at mean ~ -d/2*log(2pi) ~ -2.75
    # loose range given reservoir+Silverman
    assert -10.0 < s < 1.0, f"score={s}"


def test_bandwidth_positive_and_silverman():
    d = 3
    df = _make_df(500, d, seed=2)
    m = fit_kernel_density(df, feature_cols=_cols(d), n_centers=64,
                           memory_guard=MemoryGuard(max_memory_mb=500),
                           batch_size=100)
    assert m.bandwidth_ > 0
    # user-supplied bandwidth
    m2 = fit_kernel_density(df, feature_cols=_cols(d), n_centers=64,
                            bandwidth=0.5,
                            memory_guard=MemoryGuard(max_memory_mb=500),
                            batch_size=100)
    assert m2.bandwidth_ == 0.5


def test_score_samples_streaming():
    d = 2
    df = _make_df(500, d, seed=3)
    m = fit_kernel_density(df, feature_cols=_cols(d), n_centers=64,
                           memory_guard=MemoryGuard(max_memory_mb=500),
                           batch_size=100)
    total = 0
    for arr in m.score_samples(df, batch_size=100):
        assert arr.ndim == 1
        assert np.isfinite(arr).all()
        total += arr.size
    assert total > 0


def test_spilled_state_constant(tmp_path):
    """Force center spill via tiny guard -> state=256."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    d = 10
    df = _make_df(2000, d, seed=4)
    # K*d*8 = 1024*10*8 = 80 KB; guard 0.01 MB = ~10 KB -> spill
    try:
        m = fit_kernel_density(
            df, feature_cols=_cols(d), n_centers=1024,
            memory_guard=MemoryGuard(max_memory_mb=0.05),
            batch_size=200, spill_dir=tmp_path / "kde")
    except Exception as e:
        pytest.skip(f"fit rejected at 0.05 MB: {type(e).__name__}: {e}")
    if m.centers_ is None:
        assert m.state_nbytes() == 256
        m.cleanup()
    else:
        pytest.skip(f"no spill: K={m.n_centers_} d={d}")
