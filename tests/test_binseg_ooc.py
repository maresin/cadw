"""v0.9.181: binseg cumsum-once + explicit guard tests."""
import numpy as np
import pytest

from cadw.ml.binseg import (
    fit_binseg, _best_split, _best_split_from_cs, _segment_cost,
)


def _make_series(n=200, seed=0):
    rng = np.random.RandomState(seed)
    return np.concatenate([
        rng.normal(0, 1, n // 3),
        rng.normal(5, 1, n // 3),
        rng.normal(-2, 1, n - 2 * (n // 3)),
    ])


def test_best_split_from_cs_matches_legacy():
    """_best_split_from_cs on global cs/cs2 == _best_split on slice."""
    y = _make_series(200)
    cs = np.cumsum(y)
    cs2 = np.cumsum(y * y)

    # Full segment
    gi, gain = _best_split_from_cs(cs, cs2, 0, len(y), 20)
    li, lgain = _best_split(y, 20)
    assert gi == li
    assert abs(gain - lgain) < 1e-10

    # Sub-segment [50, 150)
    gi2, gain2 = _best_split_from_cs(cs, cs2, 50, 150, 20)
    li2, lgain2 = _best_split(y[50:150], 20)
    # li2 is local index; global = 50 + li2
    assert gi2 == 50 + li2
    assert abs(gain2 - lgain2) < 1e-10


def test_fit_binseg_detects_mean_shifts():
    y = _make_series(200)
    r = fit_binseg(y, n_bkps=2, min_size=20)
    assert len(r.changepoints) == 2
    # Approximate expected positions (66, 133)
    assert abs(int(r.changepoints[0]) - 66) <= 3
    assert abs(int(r.changepoints[1]) - 133) <= 3


def test_fit_binseg_bit_identical_to_pre_v09181():
    """Same output as old _best_split-based implementation."""
    y = _make_series(200)
    r_new = fit_binseg(y, n_bkps=2, min_size=20)

    # Reference: old-style (local cumsum per segment)
    segments = [(0, len(y))]
    while len(segments) < 2 + 1:
        best_seg_idx = -1
        best_split_local = -1
        best_gain = 0.0
        for k, (lo, hi) in enumerate(segments):
            i, g = _best_split(y[lo:hi], 20)
            if i < 0:
                continue
            if g > best_gain:
                best_gain = g
                best_seg_idx = k
                best_split_local = i
        if best_seg_idx < 0 or best_gain <= 1e-12:
            break
        lo, hi = segments[best_seg_idx]
        mid = lo + best_split_local
        segments[best_seg_idx] = (lo, mid)
        segments.insert(best_seg_idx + 1, (mid, hi))
    ref_cps = np.array([hi for (_, hi) in segments[:-1]], dtype=np.int64)

    np.testing.assert_array_equal(r_new.changepoints, ref_cps)


# === Explicit guard ===

def test_guard_refuses_above_budget():
    """Peak = 5*n*8 + 512; tight guard -> MemoryLimitExceeded."""
    from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
    y = _make_series(200_000)  # peak ~8 MB
    mg = MemoryGuard(max_memory_mb=0.1)  # 100 KB
    with pytest.raises(MemoryLimitExceeded):
        fit_binseg(y, n_bkps=2, min_size=20, guard=mg)


def test_guard_accepts_within_budget():
    from cadw.core.memory_guard import MemoryGuard
    y = _make_series(200)
    mg = MemoryGuard(max_memory_mb=10)  # 10 MB, fits 5*200*8+512 ~8 KB
    r = fit_binseg(y, n_bkps=2, min_size=20, guard=mg)
    assert len(r.changepoints) >= 1


def test_guard_duck_type_still_works():
    """Legacy fake-MB guard (only max_memory_mb) — two-path fallback."""
    class _FakeMB:
        max_memory_mb = 0.001
    from cadw.core.memory_guard import MemoryLimitExceeded
    y = _make_series(200_000)
    with pytest.raises(MemoryLimitExceeded):
        fit_binseg(y, n_bkps=2, min_size=20, guard=_FakeMB())


def test_no_guard_works():
    y = _make_series(200)
    r = fit_binseg(y, n_bkps=2, min_size=20, guard=None)
    assert len(r.changepoints) >= 1
