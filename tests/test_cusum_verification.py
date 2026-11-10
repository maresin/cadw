"""Component-wise verification for fit_cusum (DEC-033, v0.9.113).

CUSUM is stateless sequential detector: state = O(1) (s_pos_cur,
s_neg_cur scalars + Welford n/mean/M2). Output O(n) scores+labels ->
SpillArray (df path) or in-memory (ndarray path).
"""
from __future__ import annotations
import ctypes
import gc

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ml.cusum import fit_cusum


def _make_df(x):
    return CadwDataFrame.from_table(pa.table({
        "v": pa.array(x, type=pa.float64()),
    }))


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _series(n, seed=0, shift_at=None, shift=2.0, noise=0.1):
    rng = np.random.RandomState(seed)
    x = rng.standard_normal(n) * noise
    if shift_at is None:
        shift_at = n // 2
    x[shift_at:] += shift
    return x


def _warmup():
    x = np.sin(np.arange(200) * 0.1)
    fit_cusum(x, k=0.5, h=5.0)
    fit_cusum(_make_df(x), "v", k=0.5, h=5.0, output="memory")


_warmup()


def test_bit_identical_df_vs_legacy():
    """df + output='memory' == legacy np.ndarray path (bit-exact)."""
    x = _series(2000, seed=1)
    r_legacy = fit_cusum(x, k=0.5, h=5.0)
    r_df = fit_cusum(_make_df(x), "v", k=0.5, h=5.0, output="memory")
    assert not r_df.is_spilled
    assert isinstance(r_df.decision_scores, np.ndarray)
    np.testing.assert_array_equal(r_legacy.decision_scores,
                                  r_df.decision_scores)
    np.testing.assert_array_equal(r_legacy.labels, r_df.labels)
    assert abs(r_legacy.target_mean - r_df.target_mean) < 1e-9
    assert abs(r_legacy.target_std - r_df.target_std) < 1e-9


def test_ooc_df_path():
    """df + output='disk': peak does not grow with n (median-of-3).

    output='disk' forces SpillArray(spill_ratio=0.0) -> spill on first
    write for any n. Peak = O(batch), independent of total n.
    (output='auto' with large guard keeps small outputs in memory; the
    small-guard case is covered by test_spill_auto_small_guard.)
    """
    def _peak(n):
        x = _series(n, seed=2, shift_at=n // 2, shift=2.0)
        df = _make_df(x)
        peaks = []
        for _ in range(3):
            _malloc_trim()
            gc.collect()
            mg = MemoryGuard(max_memory_mb=50)
            r = measure_peak_traced(
                fit_cusum, df, "v",
                k=0.5, h=5.0, output="disk",
                guard=mg, batch_size=1000, mg=mg,
            )
            peaks.append(int(r["peak_bytes"]))
            del r
            gc.collect()
        import statistics
        return int(statistics.median(peaks))

    p_small = _peak(5000)
    p_large = _peak(50000)
    ratio = p_large / max(p_small, 1)
    # state O(1), output spilled -> peak ~constant across n.
    assert ratio < 2.0, (
        f"peak ratio {ratio:.2f} (small={p_small}, large={p_large})")


def test_spill_auto_small_guard():
    """Small guard + df + output='auto' -> is_spilled=True."""
    x = _series(100_000, seed=3)
    df = _make_df(x)
    mg = MemoryGuard(max_memory_mb=2)
    r = fit_cusum(df, "v", k=0.5, h=5.0, output="auto", guard=mg)
    assert r.is_spilled
    total = sum(chunk.shape[0] for chunk in r.decision_scores)
    assert total == len(x)


def test_disk_forces_spill():
    """output='disk' -> is_spilled=True for any n."""
    x = _series(2000, seed=4)
    df = _make_df(x)
    mg = MemoryGuard(max_memory_mb=500)
    r = fit_cusum(df, "v", k=0.5, h=5.0, output="disk", guard=mg)
    assert r.is_spilled
    total = sum(chunk.shape[0] for chunk in r.decision_scores)
    assert total == len(x)


def test_contamination_refuses_auto():
    """contamination + output!='memory' -> NotImplementedError."""
    x = _series(5000, seed=5)
    df = _make_df(x)
    mg = MemoryGuard(max_memory_mb=500)
    with pytest.raises(NotImplementedError, match="contamination"):
        fit_cusum(df, "v", k=0.5, h=5.0, contamination=0.01,
                  output="auto", guard=mg)
    r = fit_cusum(df, "v", k=0.5, h=5.0, contamination=0.01,
                  output="memory", guard=mg)
    assert not r.is_spilled


def test_legacy_np_path_unchanged():
    """np.ndarray path ignores `output` kwarg (back-compat)."""
    x = _series(1000, seed=6)
    for out in ("auto", "memory", "disk"):
        r = fit_cusum(x, k=0.5, h=5.0, output=out)
        assert isinstance(r.decision_scores, np.ndarray)
        assert not r.is_spilled

# ---- S1.5 (v0.9.194): CW-1 state_nbytes exact (DEC-033 proper) ----

@pytest.mark.parametrize("n", [500, 2000, 5000])
def test_state_nbytes_exact(n):
    """CW-1: memory path formula bit-exact (256 + 2*n*8)."""
    from cadw.core.memory_guard import MemoryGuard
    x = _series(n, seed=0)
    r = fit_cusum(
        x, contamination=None,
        guard=MemoryGuard(max_memory_mb=500), output="memory")
    assert not r.is_spilled
    expected = (256
                + int(r.decision_scores.nbytes)
                + int(r.labels.nbytes))
    assert r.state_nbytes() == expected, (
        f"n={n}: state={r.state_nbytes()} expected={expected}")


def test_state_spilled_is_o1():
    """Spilled output -> O(1) RAM state (256 B); disk not counted.

    df path + output='disk' guarantees is_spilled=True regardless of n
    (see test_disk_forces_spill).
    """
    from cadw.core.memory_guard import MemoryGuard
    x = _series(2000, seed=4)
    df = _make_df(x)
    r = fit_cusum(df, "v", k=0.5, h=5.0, output="disk",
                  guard=MemoryGuard(max_memory_mb=500))
    assert r.is_spilled
    assert r.state_nbytes() == 256

