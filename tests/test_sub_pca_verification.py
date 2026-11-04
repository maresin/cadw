"""Component-wise verification for fit_sub_pca (DEC-033, v0.9.116).

State = decision_scores(n f64) + labels(n i64)
      + components(k*window_size f64) + mean(window_size f64)
      + header.
Formula (1D): 2*n*f64 + n_components*window_size*f64 + window_size*f64 + 264.
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
from cadw.ml.sub_pca import fit_sub_pca


def _make_series(n, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / 50.0) + 0.1 * rng.standard_normal(n)


@pytest.mark.parametrize("n", [300, 600, 1200])
def test_state_nbytes_exact(n):
    """state_nbytes() == sum of .nbytes + header (bit-exact)."""
    m = fit_sub_pca(_make_series(n), window_size=50, n_components=10)
    expected = (
        m.decision_scores.nbytes + m.labels.nbytes
        + m.components.nbytes + m.mean.nbytes
        + 8 + 256
    )
    assert m.state_nbytes() == expected, (
        f"n={n}: state={m.state_nbytes()} expected={expected}")


def test_state_ratio_linear():
    """state(n2) - state(n1) ~= (n2-n1)*16 (decision + labels only)."""
    n1, n2 = 500, 1000
    s1 = fit_sub_pca(_make_series(n1), window_size=50, n_components=10).state_nbytes()
    s2 = fit_sub_pca(_make_series(n2), window_size=50, n_components=10).state_nbytes()
    expected_delta = (n2 - n1) * 16
    actual_delta = s2 - s1
    assert 0.95 * expected_delta <= actual_delta <= 1.05 * expected_delta, (
        f"delta={actual_delta} expected~{expected_delta}")


def test_bounded_at_domain():
    """bounded_by_domain n<=1e6: formula at 1e6 must fit in 50 MB."""
    n = 1_000_000
    k, w = 25, 50
    formula_at_1e6 = (
        2 * n * 8 + k * w * 8 + w * 8 + 264
    )
    assert formula_at_1e6 < 50 * 1024 * 1024, (
        f"state at n=1e6 = {formula_at_1e6} B > 50 MB")


def test_tight_guard_raises():
    """Duck-typed guard (max_memory_mb) -> MemoryLimitExceeded."""
    class _FakeMB:
        max_memory_mb = 0.001
    with pytest.raises(MemoryLimitExceeded):
        fit_sub_pca(_make_series(2000), window_size=50, guard=_FakeMB())


def test_df_matches_series():
    """fit_sub_pca(df, col) == fit_sub_pca(series) bit-identical."""
    import pyarrow as pa
    from cadw import CadwDataFrame
    x = _make_series(500, seed=7)
    df = CadwDataFrame.from_table(
        pa.table({"y": pa.array(x, type=pa.float64())}))
    m_series = fit_sub_pca(x, window_size=50, n_components=10)
    m_df = fit_sub_pca(df, "y", window_size=50, n_components=10)
    np.testing.assert_array_equal(m_series.decision_scores,
                                  m_df.decision_scores)
    np.testing.assert_array_equal(m_series.labels, m_df.labels)
    np.testing.assert_array_equal(m_series.components, m_df.components)
    np.testing.assert_array_equal(m_series.mean, m_df.mean)


def test_df_multichannel_matches_series():
    """fit_sub_pca(df, [c1, c2, c3]) == fit_sub_pca(2D ndarray)."""
    import pyarrow as pa
    from cadw import CadwDataFrame
    n = 500
    X = np.stack([_make_series(n, seed=s) for s in (1, 2, 3)], axis=1)
    df = CadwDataFrame.from_table(pa.table({
        f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(3)
    }))
    m_series = fit_sub_pca(X, window_size=50, n_components=10)
    m_df = fit_sub_pca(df, ["x0", "x1", "x2"],
                       window_size=50, n_components=10)
    np.testing.assert_array_equal(m_series.decision_scores,
                                  m_df.decision_scores)
    np.testing.assert_array_equal(m_series.components, m_df.components)


def test_output_memory_and_auto():
    """G9: output="memory" and "auto" produce equivalent results."""
    x = _make_series(500, seed=5)
    m_mem = fit_sub_pca(x, window_size=50, n_components=10, output="memory")
    m_auto = fit_sub_pca(x, window_size=50, n_components=10, output="auto")
    np.testing.assert_array_equal(m_mem.decision_scores, m_auto.decision_scores)


def test_output_disk_raises():
    """output="disk" -> NotImplementedError (SpillArray is v0.9.116+)."""
    x = _make_series(500, seed=5)
    with pytest.raises(NotImplementedError, match="SpillArray.*v0.9.116"):
        fit_sub_pca(x, window_size=50, output="disk")


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
            fit_sub_pca, x, window_size=50, n_components=10)

    p_small = _peak(3000)
    p_large = _peak(30000)
    ratio = p_large / max(p_small, 1)
    assert ratio < 15.0, (
        f"peak ratio {ratio:.2f} (small={p_small}, large={p_large}); "
        f"expected ~10x for linear; >15 suggests quadratic or formula gap")
