"""Component-wise verification for fit_dwt_mlead (DEC-033, v0.9.115).

State = anomaly_scores(n f64) + is_anomaly(n bool) + level_counts(max_level i64) + 264.
Formula: n*f64 + n*bool + max_level*i64 + 264.
max_level = log2(next_pow2(n)) (DWT depth). bounded_by_domain n<=1e6.
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
from cadw.ml.dwt_mlead import fit_dwt_mlead


def _make_series(n, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / 50.0) + 0.1 * rng.standard_normal(n)


def _max_level_for_n(n):
    """Mirror code: max_level = log2(next_pow2(n)) for n >= 2, else 0."""
    if n <= 1:
        return 0
    n_padded = 1 << (n - 1).bit_length()
    return n_padded.bit_length() - 1


def _expected_state(n):
    return n * 8 + n * 1 + _max_level_for_n(n) * 8 + 264


@pytest.mark.parametrize("n", [500, 2000, 5000])
def test_state_nbytes_exact(n):
    """state_nbytes() = n*f64 + n*bool + max_level*i64 + 264 (bit-exact)."""
    m = fit_dwt_mlead(_make_series(n), start_level=3)
    expected = _expected_state(n)
    assert m.state_nbytes() == expected, (
        f"n={n}: state={m.state_nbytes()} expected={expected} "
        f"max_level={_max_level_for_n(n)}")


def test_state_ratio_linear():
    """state(n2) - state(n1) ~= (n2-n1)*9 (max_level delta negligible)."""
    n1, n2 = 1000, 10000
    s1 = fit_dwt_mlead(_make_series(n1), start_level=3).state_nbytes()
    s2 = fit_dwt_mlead(_make_series(n2), start_level=3).state_nbytes()
    expected_delta = (n2 - n1) * 9
    actual_delta = s2 - s1
    assert 0.95 * expected_delta <= actual_delta <= 1.05 * expected_delta, (
        f"delta={actual_delta} expected~{expected_delta} "
        f"(s1={s1}, s2={s2})")


def test_bounded_at_domain():
    """bounded_by_domain n<=1e6: formula at 1e6 must fit in 50 MB."""
    n = 1_000_000
    formula_at_1e6 = _expected_state(n)
    assert formula_at_1e6 < 50 * 1024 * 1024, (
        f"state at n=1e6 = {formula_at_1e6} B > 50 MB")
    m = fit_dwt_mlead(_make_series(10_000), start_level=3)
    extrapolated = m.state_nbytes() + (n - 10_000) * 9
    assert abs(extrapolated - formula_at_1e6) < 1000, (
        f"extrapolated={extrapolated} formula={formula_at_1e6}")


def test_tight_guard_raises():
    """Duck-typed guard (max_memory_mb) -> MemoryLimitExceeded."""
    class _FakeMB:
        max_memory_mb = 0.001
    with pytest.raises(MemoryLimitExceeded):
        fit_dwt_mlead(_make_series(5000), start_level=3, guard=_FakeMB())


def test_df_matches_series():
    """fit_dwt_mlead(df, col) == fit_dwt_mlead(series) bit-identical."""
    import pyarrow as pa
    from cadw import CadwDataFrame

    x = _make_series(1000, seed=7)
    df = CadwDataFrame.from_table(
        pa.table({"y": pa.array(x, type=pa.float64())}))
    m_series = fit_dwt_mlead(x, start_level=3)
    m_df = fit_dwt_mlead(df, "y", start_level=3)
    np.testing.assert_array_equal(m_series.anomaly_scores,
                                  m_df.anomaly_scores)
    np.testing.assert_array_equal(m_series.is_anomaly, m_df.is_anomaly)
    np.testing.assert_array_equal(m_series.level_counts, m_df.level_counts)


def test_output_memory_and_auto():
    """G9: output="memory" and "auto" produce equivalent results."""
    x = _make_series(1000, seed=5)
    m_mem = fit_dwt_mlead(x, start_level=3, output="memory")
    m_auto = fit_dwt_mlead(x, start_level=3, output="auto")
    np.testing.assert_array_equal(m_mem.anomaly_scores, m_auto.anomaly_scores)


def test_output_disk_raises():
    """output="disk" -> NotImplementedError (SpillArray is v0.9.116+)."""
    x = _make_series(1000, seed=5)
    with pytest.raises(NotImplementedError, match="SpillArray.*v0.9.116"):
        fit_dwt_mlead(x, start_level=3, output="disk")


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
            fit_dwt_mlead, x, start_level=3)

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
            fit_dwt_mlead(x, start_level=3,
                     guard=MemoryGuard(max_memory_mb=10))
    finally:
        del x
