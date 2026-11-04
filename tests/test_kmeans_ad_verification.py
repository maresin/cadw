"""Component-wise verification for fit_kmeans_ad (DEC-033, v0.9.114).

KMeansAD state = O(n): 4 n-sized output arrays (decision_scores, labels,
window_scores, labels_per_window) + centroids (k,d). Honest per
bounded_by_domain n<=1e6 (CoxPH precedent). True OOC (streaming _map_max
+ reservoir + SpillArray output) deferred to v0.9.115.

Three tests:
  1. state_nbytes exact (bit-exact)
  2. state_ratio linear at 10x n
  3. tight_guard raises MemoryLimitExceeded
  4. fit peak grows linearly (not quadratic)
"""
from __future__ import annotations
import ctypes
import gc
import statistics

import numpy as np
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded, measure_peak_traced
from cadw.ml.kmeans_ad import fit_kmeans_ad


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
    fit_kmeans_ad(x, window_size=30, n_clusters=3,
                  guard=MemoryGuard(max_memory_mb=200))


_warmup()


# ---- 1: state_nbytes exact ----
@pytest.mark.parametrize("n,w,k", [(500, 30, 3), (2000, 50, 5), (5000, 50, 8)])
def test_state_nbytes_exact(n, w, k):
    """state_nbytes() = nbytes of 4 output arrays + centroids + 256, bit-exact."""
    x = _sine(n, seed=1)
    m = fit_kmeans_ad(x, window_size=w, n_clusters=k,
                      guard=MemoryGuard(max_memory_mb=500))
    expected = (
        m.decision_scores.nbytes + m.labels.nbytes
        + m.window_scores.nbytes + m.labels_per_window.nbytes
        + m.centroids.nbytes + 256
    )
    actual = m.state_nbytes()
    assert actual == expected, (
        f"n={n} w={w} k={k}: state={actual} expected={expected}")


# ---- 2: state_ratio linear ----
def test_state_ratio_linear():
    """state grows linearly (not quadratically) at 10x n."""
    def _state(n):
        x = _sine(n, seed=2)
        m = fit_kmeans_ad(x, window_size=50, n_clusters=5,
                          guard=MemoryGuard(max_memory_mb=500))
        return m.state_nbytes()

    s_small = _state(1000)
    s_large = _state(10000)
    ratio = s_large / max(s_small, 1)
    # 10x n -> ~10x state (linear). Bound [8, 12] = "definitely linear,
    # not quadratic (would be 100x) and not constant".
    assert 8.0 <= ratio <= 12.0, (
        f"ratio {ratio:.2f} (small={s_small}, large={s_large}); "
        f"expected ~10x for linear n")


# ---- 3: tight_guard raises ----
def test_tight_guard_raises():
    """Duck-typed guard (max_memory_mb only, no check_state) -> raise.

    Real MemoryGuard triggers OOC chunked path (has_check_state=True),
    so it does not raise — that's the desired OOC behavior. Duck-typed
    legacy guard (only max_memory_mb attribute) uses old full-window
    path with est_bytes vs limit_mb check -> raises. Matches
    test_kmeans_ad.py::test_ram_guard_raises_when_tight (existing test).
    """
    class _FakeMB:
        max_memory_mb = 0.001

    x = _sine(2000, seed=3)
    with pytest.raises(MemoryLimitExceeded):
        fit_kmeans_ad(x, window_size=30, n_clusters=5, guard=_FakeMB())


# ---- 4: peak grows linearly ----
def test_peak_grows_linearly_with_n():
    """Fit peak grows linearly with n (output n_w arrays intrinsic).

    state_class O(n) honest; bounded_by_domain n<=1e6 accepts this
    (CoxPH v0.9.114 precedent). 10x n -> 8-15x peak (linear, not
    quadratic). Not strict OOC — documented limitation.
    """
    def _peak(n):
        x = _sine(n, seed=4)
        peaks = []
        for _ in range(3):
            _malloc_trim()
            gc.collect()
            mg = MemoryGuard(max_memory_mb=50)
            r = measure_peak_traced(
                fit_kmeans_ad, x,
                window_size=50, n_clusters=5,
                guard=mg, mg=mg,
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
        f"expected ~10x for linear; >15 suggests quadratic")
