"""Component-wise verification for fit_hampel (DEC-033, v0.9.115).

State = decision_scores(n) + labels(n) + header.
Formula: 2*n*f64 + 264.
CoxPH-precedent (bocpd/eif/kmeans_ad v0.9.114): bounded_by_domain n<=1e6.
"""
from __future__ import annotations

import numpy as np
import pytest

from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ml.hampel import fit_hampel


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
    with pytest.raises(NotImplementedError, match="SpillArray.*v0.9.116"):
        fit_hampel(x, window_size=7, output="disk")
