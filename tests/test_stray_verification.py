"""Component-wise verification for fit_stray (v0.9.140).

STRAY: O(n) output (decision_scores + labels). intrinsic O(n²) CPU
(kNN distances). tier1_core_exception documented n<=20k.

State = 2*n*f64 + 256.
"""
from __future__ import annotations
import numpy as np
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml.stray import fit_stray


def _make_series(n, seed=0):
    rng = np.random.RandomState(seed)
    return (np.sin(np.arange(n) * 0.1) + rng.normal(0, 0.1, n))


@pytest.mark.parametrize("n", [500, 2000, 5000])
def test_state_nbytes_exact(n):
    """state = 2n*8 (decision_scores + labels) + 256."""
    y = _make_series(n)
    m = fit_stray(y, k=10, max_mb=50)
    expected = 2 * n * 8 + 256
    assert m.state_nbytes() == expected, (
        f"n={n}: state={m.state_nbytes()} expected={expected}")


def test_state_ratio_linear():
    """n=1000 -> 5000 => ratio ~5 (O(n) output dominates)."""
    y_small = _make_series(1000)
    y_large = _make_series(5000)
    s_small = fit_stray(y_small, k=10, max_mb=50).state_nbytes()
    s_large = fit_stray(y_large, k=10, max_mb=50).state_nbytes()
    ratio = s_large / max(s_small, 1)
    assert 4.0 <= ratio <= 6.0, f"ratio={ratio:.2f}; {s_small}->{s_large}"


def test_guard_refuses_at_domain_bound():
    """tier1_core_exception: n=1e6 + tiny guard -> MLE from STRAY's internal check."""
    y = _make_series(1_000_000)
    with pytest.raises(MemoryLimitExceeded):
        fit_stray(y, k=10, max_mb=1, guard=MemoryGuard(max_memory_mb=1))


def test_slow_operation_warning_absent():
    """v0.9.140 verification: SlowOperationWarning (AP22) not yet implemented.

    STRAY at n>1e5 takes ~1h (CPU O(n²)). Tier1_core_exception documents
    the bound; AP22 warning is backlog. Test asserts current state —
    update when warning lands.
    """
    # Current: no warning, but documented in YAML tier1_core_exception.
    # No assert on warning; just verify small n works fast.
    y = _make_series(500)
    m = fit_stray(y, k=10, max_mb=50)
    assert m.decision_scores.shape == (500,)
