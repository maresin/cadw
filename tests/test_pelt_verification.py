"""Component-wise verification for fit_pelt (DEC-033, v0.9.140).

PELT: fit returns 256B container (validation only); real DP runs in
detect() = O(n). Honest single-series with bounded_by_domain n<=1e6.

Tests (CW adapted for container + detect):
- test_state_nbytes_exact: always 256
- test_state_independent_of_n: always 256 regardless n
- test_detect_guard_refuses: n=1e6 + tight guard -> MemoryLimitExceeded
"""
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml.pelt import fit_pelt


def _make_df(n, seed=0):
    rng = np.random.RandomState(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = np.sin(t * 0.05) + rng.normal(0, 0.1)
    return CadwDataFrame.from_table(
        pa.table({"y": pa.array(y, type=pa.float64())}))


@pytest.mark.parametrize("n", [100, 1000, 5000])
def test_state_nbytes_exact(n):
    """Container state = 256 (validation-only)."""
    df = _make_df(n)
    m = fit_pelt(df, "y", memory_guard=MemoryGuard(max_memory_mb=500))
    assert m.state_nbytes() == 256, m.state_nbytes()


def test_state_independent_of_n():
    s_small = fit_pelt(_make_df(100), "y",
                       memory_guard=MemoryGuard(max_memory_mb=500)
                       ).state_nbytes()
    s_large = fit_pelt(_make_df(5000), "y",
                       memory_guard=MemoryGuard(max_memory_mb=500)
                       ).state_nbytes()
    assert s_small == s_large == 256


def test_fit_refuses_at_domain_bound():
    """fit_pelt pre-passes _collect_series -> MLE at n=1e6 + tiny budget.

    Honest refuse-at-guard (bounded_by_domain n<=1e6). fit_pelt
    validates via _collect_series _iter_filtered_batches loop;
    check_batch/check_state fires before DP ever runs.
    """
    df = _make_df(1_000_000)
    df._memory_guard = MemoryGuard(max_memory_mb=1)  # < 8 MB for 1e6 f64
    with pytest.raises(MemoryLimitExceeded):
        fit_pelt(df, "y", memory_guard=df._memory_guard)


def test_detect_ooc_for_n_below_bound():
    """Small n fits within budget -> detect succeeds (honest)."""
    df = _make_df(500)
    m = fit_pelt(df, "y", memory_guard=MemoryGuard(max_memory_mb=500))
    d = m.detect(df)
    assert 0 in d
    assert isinstance(d[0], list)
