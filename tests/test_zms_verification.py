"""Component-wise verification for fit_zms (DEC-033, v0.9.113).

ZMS is stateless rolling detector: state = O(w) tail buffer, no model.
Output = O(n) scores+labels -> SpillArray (df path) or in-memory.
"""
from __future__ import annotations
import ctypes
import gc

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ml.zms import fit_zms


def _make_df(x):
    return CadwDataFrame.from_table(pa.table({
        "v": pa.array(x, type=pa.float64()),
    }))


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _series(n, seed=0, n_spikes=20):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.05 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += 20.0
    return x


def _warmup():
    x = np.sin(np.arange(200) * 0.1)
    fit_zms(x, window_size=20)
    fit_zms(_make_df(x), "v", window_size=20, output="memory")


_warmup()


def test_bit_identical_df_vs_legacy():
    """df + output='memory' == legacy np.ndarray path (bit-exact)."""
    x = _series(2000, seed=1)
    r_legacy = fit_zms(x, window_size=100, threshold=3.0)
    r_df = fit_zms(_make_df(x), "v", window_size=100, threshold=3.0,
                   output="memory")
    assert not r_df.is_spilled
    assert isinstance(r_df.decision_scores, np.ndarray)
    np.testing.assert_array_equal(r_legacy.decision_scores,
                                  r_df.decision_scores)
    np.testing.assert_array_equal(r_legacy.labels, r_df.labels)
    assert r_legacy.threshold == r_df.threshold


def test_ooc_df_path():
    """df + output='disk': peak ~constant at fixed batch (state O(w))."""
    def _peak(n):
        x = _series(n, seed=2)
        df = _make_df(x)
        peaks = []
        for _ in range(3):
            _malloc_trim()
            gc.collect()
            mg = MemoryGuard(max_memory_mb=50)
            r = measure_peak_traced(
                fit_zms, df, "v",
                window_size=100, threshold=3.0,
                output="disk", guard=mg, batch_size=1000, mg=mg,
            )
            peaks.append(int(r["peak_bytes"]))
            del r
            gc.collect()
        import statistics
        return int(statistics.median(peaks))

    p_small = _peak(5000)
    p_large = _peak(50000)
    ratio = p_large / max(p_small, 1)
    # state O(w), output spilled to disk; 10x n -> <2x peak at fixed batch.
    assert ratio < 2.0, (
        f"peak ratio {ratio:.2f} (small={p_small}, large={p_large})")


def test_spill_auto_small_guard():
    """Small guard + df + output='auto' -> is_spilled=True."""
    x = _series(100_000, seed=3)
    df = _make_df(x)
    mg = MemoryGuard(max_memory_mb=2)
    r = fit_zms(df, "v", window_size=100, threshold=3.0,
                output="auto", guard=mg)
    assert r.is_spilled
    total = sum(chunk.shape[0] for chunk in r.decision_scores)
    assert total == len(x)


def test_disk_forces_spill():
    """output='disk' -> is_spilled=True for any n."""
    x = _series(2000, seed=4)
    df = _make_df(x)
    mg = MemoryGuard(max_memory_mb=500)
    r = fit_zms(df, "v", window_size=50, output="disk", guard=mg)
    assert r.is_spilled
    total = sum(chunk.shape[0] for chunk in r.decision_scores)
    assert total == len(x)


def test_contamination_refuses_auto():
    """contamination + output!='memory' -> NotImplementedError."""
    x = _series(5000, seed=5)
    df = _make_df(x)
    mg = MemoryGuard(max_memory_mb=500)
    with pytest.raises(NotImplementedError, match="contamination"):
        fit_zms(df, "v", window_size=100, contamination=0.01,
                output="auto", guard=mg)
    r = fit_zms(df, "v", window_size=100, contamination=0.01,
                output="memory", guard=mg)
    assert not r.is_spilled
    assert isinstance(r.decision_scores, np.ndarray)


def test_legacy_np_path_unchanged():
    """np.ndarray path ignores `output` kwarg (back-compat)."""
    x = _series(1000, seed=6)
    for out in ("auto", "memory", "disk"):
        r = fit_zms(x, window_size=50, output=out)
        assert isinstance(r.decision_scores, np.ndarray)
        assert not r.is_spilled
