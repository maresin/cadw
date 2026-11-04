"""Component-wise verification for fit_kshape (DEC-033, v0.9.115).

State = decision_scores(n) + labels(n) + centroids(k*w) + header.
Formula: 2*n*f64 + k*w*f64 + 264.
CoxPH-precedent (bocpd/eif/kmeans_ad v0.9.114): bounded_by_domain n<=1e6.
"""
from __future__ import annotations

import numpy as np
import pytest

from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ml.kshape import fit_kshape


def _make_series(n, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / 50.0) + 0.1 * rng.standard_normal(n)


def _expected_state(n, k, w):
    return 2 * n * 8 + k * w * 8 + 264


@pytest.mark.parametrize("n,k,w", [(500, 3, 30), (2000, 5, 50), (5000, 8, 80)])
def test_state_nbytes_exact(n, k, w):
    """model.state_nbytes() must equal 2*n*f64 + k*w*f64 + 264 (bit-exact)."""
    m = fit_kshape(_make_series(n), n_clusters=k, window_size=w,
                   max_iter=5, contamination=0.1)
    expected = _expected_state(n, k, w)
    assert m.state_nbytes() == expected, (
        f"n={n} k={k} w={w}: state={m.state_nbytes()} expected={expected}")


def test_state_ratio_linear():
    """state(n2) - state(n1) == (n2 - n1) * 16 (fixed kw*8 cancels)."""
    k, w = 3, 50
    n1, n2 = 1000, 10000
    s1 = fit_kshape(_make_series(n1), n_clusters=k, window_size=w,
                    max_iter=5).state_nbytes()
    s2 = fit_kshape(_make_series(n2), n_clusters=k, window_size=w,
                    max_iter=5).state_nbytes()
    expected_delta = (n2 - n1) * 16
    actual_delta = s2 - s1
    assert 0.95 * expected_delta <= actual_delta <= 1.05 * expected_delta, (
        f"delta={actual_delta} expected={expected_delta}")


def test_bounded_at_domain():
    """bounded_by_domain n<=1e6: formula at 1e6 must fit in 50 MB.

    At n=1e6, k=8, w=50: 16*1e6 + 8*50*8 + 264 ~ 15.3 MB.
    Sanity via extrapolation from measured state at n=10000.
    """
    k, w = 8, 50
    n = 1_000_000
    formula_at_1e6 = _expected_state(n, k, w)
    assert formula_at_1e6 < 50 * 1024 * 1024, (
        f"state at n=1e6 = {formula_at_1e6} B > 50 MB")
    m = fit_kshape(_make_series(10_000), n_clusters=k, window_size=w,
                   max_iter=5)
    extrapolated = m.state_nbytes() + (n - 10_000) * 16
    assert abs(extrapolated - formula_at_1e6) < 1000, (
        f"extrapolated {extrapolated} vs formula {formula_at_1e6}")


def test_tight_guard_raises():
    """Duck-typed guard (max_memory_mb only) -> est_bytes check -> raise."""
    class _FakeMB:
        max_memory_mb = 0.001
    with pytest.raises(MemoryLimitExceeded):
        fit_kshape(_make_series(5000), n_clusters=3, window_size=50,
                   max_iter=5, guard=_FakeMB())


def test_df_matches_series():
    """fit_kshape(df, col) equals fit_kshape(series) bit-identically."""
    import pyarrow as pa
    from cadw import CadwDataFrame

    x = _make_series(1000, seed=7)
    df = CadwDataFrame.from_table(
        pa.table({"y": pa.array(x, type=pa.float64())}))
    m_series = fit_kshape(x, n_clusters=3, window_size=30,
                          max_iter=10, random_state=0)
    m_df = fit_kshape(df, "y", n_clusters=3, window_size=30,
                      max_iter=10, random_state=0)
    np.testing.assert_array_equal(m_series.decision_scores,
                                  m_df.decision_scores)
    np.testing.assert_array_equal(m_series.labels, m_df.labels)
    assert m_series.threshold == m_df.threshold


def test_output_memory_and_auto():
    """G9: output="memory" and output="auto" produce equivalent results.

    v0.9.115: output="auto" aliased to "memory" (SpillArray is v0.9.116+).
    """
    x = _make_series(1000, seed=5)
    m_mem = fit_kshape(x, n_clusters=3, window_size=30, max_iter=5,
                       output="memory")
    m_auto = fit_kshape(x, n_clusters=3, window_size=30, max_iter=5,
                        output="auto")
    np.testing.assert_array_equal(
        m_mem.decision_scores, m_auto.decision_scores)
    np.testing.assert_array_equal(m_mem.labels, m_auto.labels)
    assert m_mem.threshold == m_auto.threshold


def test_output_disk_raises():
    """output="disk" -> NotImplementedError (SpillArray is v0.9.116+)."""
    x = _make_series(1000, seed=5)
    with pytest.raises(NotImplementedError, match="SpillArray.*v0.9.116"):
        fit_kshape(x, n_clusters=3, window_size=30, max_iter=5,
                   output="disk")

