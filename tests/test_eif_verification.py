"""Component-wise verification for fit_eif (DEC-033, v0.9.114).

EIF state = O(n): 3 n-sized output arrays (decision_scores, labels,
window_scores) + scalar fields + 256. Forest is hyperparameter-bounded
(n_estimators*psi Python objects) — O(model), not O(n). Honest per
bounded_by_domain n<=1e6 (CoxPH v0.9.114 precedent).
"""
from __future__ import annotations
import ctypes
import gc
import statistics

import numpy as np
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded, measure_peak_traced
from cadw.ml.eif import fit_eif


def _sine(n, seed=0, n_spikes=10, amplitude=20.0):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 100.0) + 0.05 * rng.standard_normal(n)
    pos = rng.choice(n, size=n_spikes, replace=False)
    x[pos] += amplitude
    return x


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _warmup():
    x = np.sin(np.arange(500) * 0.1)
    fit_eif(x, n_estimators=10, window_size=30,
            guard=MemoryGuard(max_memory_mb=200), random_state=0)


_warmup()


# ---- 1: state_nbytes exact ----
@pytest.mark.parametrize("n,w,stride", [
    (500, 30, 1),
    (2000, 50, 1),
    (2000, 50, 5),
])
def test_state_nbytes_exact(n, w, stride):
    """state_nbytes() = decision_scores + labels + window_scores + 256, bit-exact."""
    x = _sine(n, seed=1)
    m = fit_eif(x, n_estimators=20, window_size=w, stride=stride,
                guard=MemoryGuard(max_memory_mb=500), random_state=0)
    expected = (
        m.decision_scores.nbytes
        + m.labels.nbytes
        + m.window_scores.nbytes
        + 256
    )
    actual = m.state_nbytes()
    assert actual == expected, (
        f"n={n} w={w} stride={stride}: state={actual} expected={expected}")


# ---- 2: state_ratio linear ----
def test_state_ratio_linear():
    """state grows linearly (~10x) at 10x n."""
    def _state(n):
        x = _sine(n, seed=2)
        m = fit_eif(x, n_estimators=20, window_size=50,
                    guard=MemoryGuard(max_memory_mb=500), random_state=0)
        return m.state_nbytes()

    s_small = _state(1000)
    s_large = _state(10000)
    ratio = s_large / max(s_small, 1)
    # 10x n -> ~10x state (linear; quadratic would be 100x, constant 1x).
    # stride=1 → window_scores n_w ≈ n-49 → state(10x) ~ 10x.
    assert 8.0 <= ratio <= 12.0, (
        f"ratio {ratio:.2f} (small={s_small}, large={s_large}); "
        f"expected ~10x for linear n")


# ---- 3: peak not quadratic ----
def test_peak_not_quadratic():
    """Fit peak grows linearly or sublinearly with n; not quadratic.

    EIF chunked fit: Phase 1-3 forest gather is O(n_est*psi*d) fixed;
    chunked scoring output O(n_w~n) linear. At 10x n, ratio should be
    2-10x (fixed forest + linear output). Bound <15 catches quadratic.
    """
    def _peak(n):
        x = _sine(n, seed=3)
        peaks = []
        for _ in range(3):
            _malloc_trim()
            gc.collect()
            mg = MemoryGuard(max_memory_mb=100)
            r = measure_peak_traced(
                fit_eif, x,
                n_estimators=30, window_size=50,
                guard=mg, random_state=0, mg=mg,
            )
            peaks.append(int(r["peak_bytes"]))
            del r
            gc.collect()
        return int(statistics.median(peaks))

    p_small = _peak(3000)
    p_large = _peak(30000)
    ratio = p_large / max(p_small, 1)
    assert ratio < 15.0, (
        f"peak ratio {ratio:.2f} (small={p_small}, large={p_large}); "
        f"expected 2-10x; >15 suggests quadratic")


# ---- 4: tight_guard raises ----
def test_tight_guard_raises():
    """Duck-typed guard (max_memory_mb only) -> raise via est_bytes check."""
    class _FakeMB:
        max_memory_mb = 0.001

    x = _sine(2000, seed=4)
    with pytest.raises(MemoryLimitExceeded):
        fit_eif(x, n_estimators=100, window_size=30, guard=_FakeMB(),
                random_state=0)
