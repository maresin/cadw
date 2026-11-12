"""Component-wise verification for fit_bocpd (DEC-033, v0.9.114).

BOCPD state = O(n): 3 n-sized output arrays (decision_scores, labels,
argmax_run_length) + run_length_posterior (compact: posterior_keep*
(max_run+1)*f64 = ~103KB constant). Honest per bounded_by_domain n<=1e6
(CoxPH v0.9.114 precedent). full_posterior=True -> O(n*max_run)
quadratic, documented limitation.
"""
from __future__ import annotations
import ctypes
import gc
import statistics

import numpy as np
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded, measure_peak_traced
from cadw.ml.bocpd import fit_bocpd


def _sine(n, seed=0, n_cp=5):
    """Synthetic series with mean-shift change points."""
    rng = np.random.RandomState(seed)
    x = np.sin(2 * np.pi * np.arange(n) / 100.0) + 0.05 * rng.standard_normal(n)
    if n_cp > 0 and n > 100:
        positions = rng.choice(range(50, n - 50), size=n_cp, replace=False)
        for p in positions:
            x[p:] += 0.5
    return x


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _warmup():
    x = np.sin(np.arange(500) * 0.1)
    fit_bocpd(x, max_run=30, guard=MemoryGuard(max_memory_mb=200),
              )


_warmup()


# ---- 1: state_nbytes exact ----
@pytest.mark.parametrize("n,posterior_keep,max_run", [
    (500, 64, 100),
    (2000, 64, 200),
    (2000, 128, 100),
])
def test_state_nbytes_exact(n, posterior_keep, max_run):
    """state_nbytes() = 3*n*8 + run_length_posterior.nbytes + 256, bit-exact."""
    x = _sine(n, seed=1)
    m = fit_bocpd(x, max_run=max_run, posterior_keep=posterior_keep,
                  guard=MemoryGuard(max_memory_mb=500))
    expected = (
        m.decision_scores.nbytes
        + m.labels.nbytes
        + m.argmax_run_length.nbytes
        + m.run_length_posterior.nbytes
        + 256
    )
    actual = m.state_nbytes()
    assert actual == expected, (
        f"n={n} pk={posterior_keep} mr={max_run}: "
        f"state={actual} expected={expected}")


# ---- 2: state linear in n (delta test, cancels fixed posterior) ----
def test_state_ratio_linear():
    """state(n) = 3*n*8 + fixed_posterior + 256. Verify DELTA is linear.

    Direct ratio at small n is misleading: fixed posterior term
    (posterior_keep*max_run*8 ~26KB at defaults) dominates when
    n*24 < 26KB, i.e. n < ~1100. Delta test cancels it exactly:
        state(n2) - state(n1) = (n2 - n1) * 3 * 8 (bit-exact for linear)
    Quadratic would give (n2^2 - n1^2)*c, also caught by abs.
    """
    def _state(n):
        x = _sine(n, seed=2)
        m = fit_bocpd(x, max_run=50, posterior_keep=64,
                      guard=MemoryGuard(max_memory_mb=500))
        return m.state_nbytes()

    n1, n2 = 1000, 10000
    s_small = _state(n1)
    s_large = _state(n2)
    delta = s_large - s_small
    # Expected linear delta: 3 n-sized outputs * 8 bytes * delta_n
    expected_delta = (n2 - n1) * 3 * 8
    assert 0.95 * expected_delta <= delta <= 1.05 * expected_delta, (
        f"delta={delta}, expected={expected_delta} "
        f"(s_small={s_small}, s_large={s_large}); "
        f"linear n-scaled state expected")


# ---- 3: peak not quadratic ----
def test_peak_not_quadratic():
    """Peak ~linear; not O(n*max_run) because compact mode keeps R_hist fixed.

    Use max_run=30 to keep Python loop fast (n*max_run iterations).
    """
    def _peak(n):
        x = _sine(n, seed=3)
        peaks = []
        for _ in range(3):
            _malloc_trim()
            gc.collect()
            mg = MemoryGuard(max_memory_mb=100)
            r = measure_peak_traced(
                fit_bocpd, x,
                max_run=30, posterior_keep=64,
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
        f"expected ~10x linear; >15 suggests quadratic")


# ---- 4: tight_guard raises ----
def test_tight_guard_raises():
    """Duck-typed guard (max_memory_mb only) raises via est_bytes check."""
    class _FakeMB:
        max_memory_mb = 0.001

    x = _sine(2000, seed=4)
    with pytest.raises(MemoryLimitExceeded):
        fit_bocpd(x, max_run=50, guard=_FakeMB())


# ============================================================
# v0.9.219-b: streaming (output='auto'|'disk')
# ============================================================

def _collect(x):
    """ndarray or AutoArrayResult -> flat ndarray."""
    import numpy as np
    if isinstance(x, np.ndarray):
        return x
    parts = [np.asarray(c).ravel() for c in x]
    return np.concatenate(parts) if parts else np.array([])


def _make_df_bocpd(x):
    import numpy as np
    import pyarrow as pa
    from cadw import CadwDataFrame
    return CadwDataFrame.from_table(
        pa.table({"v": pa.array(np.asarray(x, dtype=np.float64),
                                 type=pa.float64())}))


def test_streaming_vs_memory_bitexact():
    """Streaming bit-identical to memory (scores/labels/argmax/cp)."""
    import numpy as np
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ml.bocpd import fit_bocpd
    rng = np.random.RandomState(0)
    x = np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 100)])
    df = _make_df_bocpd(x)
    mg = MemoryGuard(max_memory_mb=500)
    r_mem = fit_bocpd(df, "v", output="memory", guard=mg)
    r_auto = fit_bocpd(df, "v", output="auto", guard=mg)
    np.testing.assert_array_equal(
        r_mem.decision_scores, _collect(r_auto.decision_scores))
    np.testing.assert_array_equal(
        r_mem.labels, _collect(r_auto.labels).astype(int))
    np.testing.assert_array_equal(
        r_mem.argmax_run_length, _collect(r_auto.argmax_run_length).astype(int))
    np.testing.assert_array_equal(
        r_mem.changepoints, r_auto.changepoints)


def test_streaming_state_spilled_o1():
    """state_nbytes = 256 + posterior only (OOC in n) when spilled."""
    import numpy as np
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ml.bocpd import fit_bocpd
    rng = np.random.RandomState(1)
    x = rng.normal(size=200)
    df = _make_df_bocpd(x)
    r = fit_bocpd(df, "v", output="auto",
                   guard=MemoryGuard(max_memory_mb=500))
    assert r.is_spilled is True
    # state = 256 (header) + run_length_posterior bytes (O(keep*max_run))
    assert r.state_nbytes() < 1024 * 1024  # < 1 MB regardless of n


def test_streaming_contamination_refuses():
    """contamination + output='auto' -> NotImplementedError."""
    import numpy as np
    import pytest
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ml.bocpd import fit_bocpd
    rng = np.random.RandomState(2)
    x = rng.normal(size=100)
    df = _make_df_bocpd(x)
    for out in ("auto", "disk"):
        with pytest.raises(NotImplementedError, match="contamination"):
            fit_bocpd(df, "v", contamination=0.1, output=out,
                       guard=MemoryGuard(max_memory_mb=500))


def test_streaming_full_posterior_refuses():
    """full_posterior=True + output='auto' -> NotImplementedError."""
    import numpy as np
    import pytest
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ml.bocpd import fit_bocpd
    rng = np.random.RandomState(3)
    x = rng.normal(size=100)
    df = _make_df_bocpd(x)
    with pytest.raises(NotImplementedError, match="full_posterior"):
        fit_bocpd(df, "v", full_posterior=True, output="auto",
                   guard=MemoryGuard(max_memory_mb=500))


def test_streaming_disk_forced_spill():
    """output='disk' -> SpillArray spill_ratio=0.0 forced."""
    import numpy as np
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ml.bocpd import fit_bocpd
    rng = np.random.RandomState(4)
    x = rng.normal(size=100)
    df = _make_df_bocpd(x)
    r = fit_bocpd(df, "v", output="disk",
                   guard=MemoryGuard(max_memory_mb=500))
    assert r.is_spilled is True
    # verify correctness: same as memory on same data
    r_mem = fit_bocpd(df, "v", output="memory",
                       guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(
        r_mem.decision_scores, _collect(r.decision_scores))


def test_streaming_ndarray_disk_raises():
    """ndarray input + output='disk' -> NotImplementedError."""
    import numpy as np
    import pytest
    from cadw.core.memory_guard import MemoryGuard
    from cadw.ml.bocpd import fit_bocpd
    rng = np.random.RandomState(5)
    x = rng.normal(size=100)
    with pytest.raises(NotImplementedError, match="requires df input"):
        fit_bocpd(x, output="disk",
                   guard=MemoryGuard(max_memory_mb=500))

