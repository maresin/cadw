"""Component-wise verification for fit_t2q (DEC-033, v0.9.140).

T2Q: O(n) output (decision_scores, labels, t2_scores, q_scores) — DEC-039
pattern (same as hampel/sub_pca v0.9.116). bounded_state: false,
bounded_by_domain n<=1e6.

State = 4*n*f64 + components(k*d) + eigvals(k) + mean(d) + 256
Fit peak = state + chunk_nw*w*8 (chunked windows).
"""
from __future__ import annotations
import ctypes, gc

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import (
    MemoryGuard, MemoryLimitExceeded, measure_peak_traced,
)
from cadw.ml.tspq import fit_t2q


def _make_series(n, seed=0):
    rng = np.random.RandomState(seed)
    return (np.sin(np.arange(n) * 0.1) + rng.normal(0, 0.1, n))


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


@pytest.mark.parametrize("n", [500, 2000, 5000])
def test_state_nbytes_exact(n):
    """state = 4n*8 (output arrays) + k*d*8 + k*8 + d*8 + 256."""
    y = _make_series(n)
    m = fit_t2q(y, window_size=20, stride=5)
    k = m.components.shape[0]
    d = m.components.shape[1]
    expected = (4 * n * 8 + k * d * 8 + k * 8 + d * 8 + 256)
    assert m.state_nbytes() == expected, (
        f"n={n}: state={m.state_nbytes()} expected={expected}")


def test_state_ratio_linear():
    """n=1000 -> 5000 => ratio ~4 (O(n) output dominates)."""
    y_small = _make_series(1000)
    y_large = _make_series(5000)
    s_small = fit_t2q(y_small, window_size=20, stride=5).state_nbytes()
    s_large = fit_t2q(y_large, window_size=20, stride=5).state_nbytes()
    ratio = s_large / max(s_small, 1)
    assert 3.5 <= ratio <= 5.5, f"ratio={ratio:.2f}; {s_small}->{s_large}"


def test_guard_refuses_at_domain_bound():
    """bounded_by_domain n<=1e6: n=1e6 + tight guard -> MLE."""
    y = _make_series(1_000_000)
    with pytest.raises(MemoryLimitExceeded):
        fit_t2q(y, window_size=20, stride=10,
                guard=MemoryGuard(max_memory_mb=5))


