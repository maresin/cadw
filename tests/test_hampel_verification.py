"""Component-wise verification for fit_hampel (DEC-033, v0.9.115).

State = decision_scores(n) + labels(n) + header.
Formula: 2*n*f64 + 264.
CoxPH-precedent (bocpd/eif/kmeans_ad v0.9.114): bounded_by_domain n<=1e6.
"""
from __future__ import annotations

import ctypes
import gc
import statistics

import numpy as np
import pytest

from cadw.core.memory_guard import (
    MemoryGuard,
    MemoryLimitExceeded,
    measure_peak_traced,
)
from cadw.ml.hampel import fit_hampel




def _make_df(x, col="v"):
    """Wrap ndarray into CadwDataFrame with single column."""
    import pyarrow as pa
    from cadw import CadwDataFrame
    return CadwDataFrame.from_table(
        pa.table({col: pa.array(np.asarray(x, dtype=np.float64),
                                 type=pa.float64())}))


def _collect(x):
    """ndarray or AutoArrayResult -> flat ndarray."""
    if isinstance(x, np.ndarray):
        return x
    parts = []
    for chunk in x:
        parts.append(np.asarray(chunk).ravel())
    return np.concatenate(parts) if parts else np.array([])


def _make_series(n, seed=0, n_outliers=5):
    rng = np.random.RandomState(seed)
    x = rng.standard_normal(n)
    if n > 100:
        idx = rng.choice(n, size=n_outliers, replace=False)
        x[idx] = 10.0
    return x


def _expected_state(n):
    return 2 * n * 8 + 264


@pytest.mark.parametrize("n", [500, 2000, 5000])
def test_state_nbytes_exact(n):
    """state_nbytes() = 2*n*f64 + 264 (bit-exact)."""
    m = fit_hampel(_make_series(n), window_size=7, n_sigma=3.0)
    expected = _expected_state(n)
    assert m.state_nbytes() == expected, (
        f"n={n}: state={m.state_nbytes()} expected={expected}")


def test_state_ratio_linear():
    """state(n2) - state(n1) == (n2 - n1) * 16."""
    n1, n2 = 1000, 10000
    s1 = fit_hampel(_make_series(n1)).state_nbytes()
    s2 = fit_hampel(_make_series(n2)).state_nbytes()
    expected_delta = (n2 - n1) * 16
    actual_delta = s2 - s1
    assert 0.95 * expected_delta <= actual_delta <= 1.05 * expected_delta


def test_bounded_at_domain():
    """bounded_by_domain n<=1e6: formula at 1e6 must fit in 50 MB."""
    n = 1_000_000
    formula_at_1e6 = _expected_state(n)
    assert formula_at_1e6 < 50 * 1024 * 1024, (
        f"state at n=1e6 = {formula_at_1e6} B > 50 MB")
    m = fit_hampel(_make_series(10_000))
    extrapolated = m.state_nbytes() + (n - 10_000) * 16
    assert abs(extrapolated - formula_at_1e6) < 1000


def test_tight_guard_raises():
    class _FakeMB:
        max_memory_mb = 0.001
    with pytest.raises(MemoryLimitExceeded):
        fit_hampel(_make_series(5000), window_size=50, guard=_FakeMB())


def test_df_matches_series():
    """fit_hampel(df, col) == fit_hampel(series) bit-identical."""
    import pyarrow as pa
    from cadw import CadwDataFrame

    x = _make_series(1000, seed=7)
    df = CadwDataFrame.from_table(
        pa.table({"y": pa.array(x, type=pa.float64())}))
    m_series = fit_hampel(x, window_size=7)
    m_df = fit_hampel(df, "y", window_size=7)
    np.testing.assert_array_equal(m_series.decision_scores,
                                  m_df.decision_scores)
    np.testing.assert_array_equal(m_series.labels, m_df.labels)
    assert m_series.threshold == m_df.threshold


def test_output_memory_and_auto():
    """G9: output="memory" and "auto" produce equivalent results."""
    x = _make_series(1000, seed=5)
    m_mem = fit_hampel(x, window_size=7, output="memory")
    m_auto = fit_hampel(x, window_size=7, output="auto")
    np.testing.assert_array_equal(
        m_mem.decision_scores, m_auto.decision_scores)


def test_output_disk_raises():
    """output="disk" -> NotImplementedError (SpillArray is v0.9.116+)."""
    x = _make_series(1000, seed=5)
    with pytest.raises(NotImplementedError, match="requires df input"):
        fit_hampel(x, window_size=7, output="disk")


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _measure_peak_median3(fn, x, **kwargs):
    """median-of-3 peak measurement (DEC-034).

    Per-cell: malloc_trim + gc.collect, MemoryGuard(50), measure_peak_traced.
    Sporadic GC/allocator shifts filtered by median (A24/CW-AP2).
    """
    peaks = []
    for _ in range(3):
        _malloc_trim()
        gc.collect()
        mg = MemoryGuard(max_memory_mb=50)
        r = measure_peak_traced(fn, x, guard=mg, mg=mg, **kwargs)
        peaks.append(int(r["peak_bytes"]))
        del r
        gc.collect()
    return int(statistics.median(peaks))


def test_peak_grows_linearly_with_n():
    """CW-4 (DEC-033): fit peak grows linearly with n at fixed params.

    state_class O(n) honest; bounded_by_domain n<=1e6 accepts this.
    10x n -> 8-15x peak (linear, not quadratic). Not strict OOC —
    documented limitation. median-of-3 per DEC-034.
    """
    def _peak(n):
        x = _make_series(n, seed=4)
        return _measure_peak_median3(
            fit_hampel, x, window_size=50)

    p_small = _peak(3000)
    p_large = _peak(30000)
    ratio = p_large / max(p_small, 1)
    assert ratio < 15.0, (
        f"peak ratio {ratio:.2f} (small={p_small}, large={p_large}); "
        f"expected ~10x for linear; >15 suggests quadratic or formula gap")


def test_guard_refuses_at_domain_bound():
    """bounded_by_domain n<=1e6 honest: fit must REFUSE, not OOM.

    at n=1e6 (declared domain bound) + guard=10MB (insufficient for
    transient peak 16-42 MB): fit must raise MemoryLimitExceeded
    BEFORE allocating output arrays. If this test OOMs — the YAML
    bounded_by_domain claim is false (state_nbytes excludes transient).

    v0.9.116: all 7 series-path O(n)-output fns verified honest.
    """
    x = _make_series(1_000_000)
    try:
        with pytest.raises(MemoryLimitExceeded):
            fit_hampel(x, window_size=50,
                     guard=MemoryGuard(max_memory_mb=10))
    finally:
        del x


# ============================================================
# v0.9.217: streaming (df + output='auto'|'disk')
# ============================================================

@pytest.mark.parametrize("n,w", [
    (10, 3), (10, 7), (100, 7), (100, 15), (1000, 7), (1000, 51)])
def test_streaming_vs_memory_bitexact(n, w):
    """Streaming df path bit-identical to memory df path."""
    x = _make_series(n)
    df = _make_df(x)
    r_mem = fit_hampel(df, "v", window_size=w, output="memory",
                        guard=MemoryGuard(max_memory_mb=500))
    r_auto = fit_hampel(df, "v", window_size=w, output="auto",
                         guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(r_mem.decision_scores,
                                  _collect(r_auto.decision_scores))
    np.testing.assert_array_equal(r_mem.labels,
                                  _collect(r_auto.labels).astype(int))


@pytest.mark.parametrize("batch_size", [3, 7, 10, 100, 1000])
def test_streaming_batch_size_invariant(batch_size):
    """Streaming independent of batch_size."""
    x = _make_series(100)
    df = _make_df(x)
    r1 = fit_hampel(df, "v", window_size=7, output="auto", batch_size=1,
                     guard=MemoryGuard(max_memory_mb=500))
    rN = fit_hampel(df, "v", window_size=7, output="auto",
                     batch_size=batch_size,
                     guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(_collect(r1.decision_scores),
                                  _collect(rN.decision_scores))


def test_streaming_state_spilled_o1():
    x = _make_series(5000)
    df = _make_df(x)
    r = fit_hampel(df, "v", output="auto",
                    guard=MemoryGuard(max_memory_mb=500))
    assert r.is_spilled is True
    assert r.state_nbytes() == 256


def test_streaming_contamination_refuses():
    x = _make_series(500)
    df = _make_df(x)
    for out in ("auto", "disk"):
        with pytest.raises(NotImplementedError, match="contamination"):
            fit_hampel(df, "v", contamination=0.1, output=out,
                        guard=MemoryGuard(max_memory_mb=500))


def test_streaming_guard_refuses_tight():
    x = _make_series(500)
    df = _make_df(x)
    with pytest.raises(MemoryLimitExceeded):
        fit_hampel(df, "v", output="auto",
                    guard=MemoryGuard(max_memory_mb=0.01))


def test_streaming_disk_forced_spill():
    x = _make_series(200)
    df = _make_df(x)
    r = fit_hampel(df, "v", output="disk",
                    guard=MemoryGuard(max_memory_mb=500))
    assert r.is_spilled is True
    assert r.state_nbytes() == 256


def test_streaming_result_iterable():
    x = _make_series(200)
    df = _make_df(x)
    r = fit_hampel(df, "v", output="auto",
                    guard=MemoryGuard(max_memory_mb=500))
    a = _collect(r.decision_scores)
    b = _collect(r.decision_scores)
    np.testing.assert_array_equal(a, b)

