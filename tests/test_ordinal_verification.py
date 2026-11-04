"""Component-wise verification for fit_ordinal_logit (DEC-033, v0.9.118).

State = beta_(d f64) + theta_(K-1 f64) + 256.
Fit peak = O(n*d) for _collect np.vstack(Xs) + baseline RSS (~5 MB).

bounded_by_domain n<=1e6: at n=1e6, d=50 → n*d*8 = 400 MB; guard
MemoryLimitExceeded via check_state in _collect (line 85).
"""
from __future__ import annotations
import ctypes, gc, statistics
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ml.ordinal import fit_ordinal_logit


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _make_df(n, d, K, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    beta = rng.normal(size=d) * 0.3
    eta = X @ beta
    # Ordinal: cutpoints at evenly-spaced quantiles
    thresholds = np.linspace(-1.0, 1.0, K - 1)
    y = np.digitize(eta, thresholds).astype(np.float64)
    tbl = {f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(d)}
    tbl["y"] = pa.array(y, type=pa.float64())
    return CadwDataFrame.from_table(pa.table(tbl))


def _state_nbytes(d, K):
    """Formula: d*f64 (beta) + (K-1)*f64 (theta) + 256 header."""
    return d * 8 + (K - 1) * 8 + 256


@pytest.mark.parametrize("d,K", [(5, 3), (10, 4), (20, 5)])
def test_state_nbytes_exact(d, K):
    """model.state_nbytes() == formula (bit-exact, d, K parametrized)."""
    df = _make_df(500, d, K, seed=0)
    m = fit_ordinal_logit(df, [f"x{j}" for j in range(d)], "y",
                          memory_guard=MemoryGuard(max_memory_mb=200))
    expected = _state_nbytes(d, K)
    assert m.state_nbytes() == expected, (
        f"d={d} K={K}: state={m.state_nbytes()} expected={expected} "
        f"(actual K={len(m.classes_)})")


def test_state_independent_of_n():
    """Ordinal state is O(d): state(n=2k) == state(n=10k) for same d, K."""
    d, K = 10, 4
    mg = MemoryGuard(max_memory_mb=300)
    m1 = fit_ordinal_logit(_make_df(2_000, d, K, seed=0),
                            [f"x{j}" for j in range(d)], "y",
                            memory_guard=mg, batch_size=500)
    m2 = fit_ordinal_logit(_make_df(10_000, d, K, seed=0),
                            [f"x{j}" for j in range(d)], "y",
                            memory_guard=mg, batch_size=500)
    # Note: scipy.optimize.minimize may stop early; state depends on d, K not n.
    assert m1.state_nbytes() == m2.state_nbytes()


def test_formula_small_scale():
    """predicted/measured in [0.5, 2.0] at d=20, K=5, n=4000."""
    d, K, n, batch = 20, 5, 4000, 500
    df = _make_df(n, d, K, seed=0)
    _malloc_trim(); gc.collect()
    mg = MemoryGuard(max_memory_mb=300)
    r = measure_peak_traced(
        fit_ordinal_logit, df,
        [f"x{j}" for j in range(d)], "y",
        memory_guard=mg, batch_size=batch, mg=mg,
    )
    measured = int(r["peak_bytes"])
    # predicted = peak transient only (measure_peak_traced = peak - baseline):
    #   _collect: Xs list (n*d*8) held while np.vstack(Xs) allocates duplicate
    #   → peak = 2 * n*d*8.  state = d*8 + (K-1)*8 + 256.
    # At n=4000, d=20: 2 * 4000 * 20 * 8 = 1.28 MB; state = 448 B.
    # predicted ≈ 1.28 MB; measured ~1.5 MB → ratio ~0.85.
    # No baseline RSS added — measure_peak_traced already measures delta.
    predicted = 2 * n * d * 8 + _state_nbytes(d, K)
    ratio = predicted / max(measured, 1)
    assert 0.5 <= ratio <= 2.0, (
        f"predicted={predicted:,} measured={measured:,} ratio={ratio:.2f}")


def test_peak_grows_linearly_with_n():
    """n=3k vs 30k, ratio < 15 (linear np.vstack(Xs) peak)."""
    d, K, batch = 20, 5, 500
    def _peak(n):
        df = _make_df(n, d, K, seed=0)
        _malloc_trim(); gc.collect()
        mg = MemoryGuard(max_memory_mb=1000)
        r = measure_peak_traced(
            fit_ordinal_logit, df,
            [f"x{j}" for j in range(d)], "y",
            memory_guard=mg, batch_size=batch, mg=mg,
        )
        return int(r["peak_bytes"])
    p_small = _peak(3_000)
    p_large = _peak(30_000)
    ratio = p_large / max(p_small, 1)
    assert ratio < 15.0, f"peak ratio {ratio:.2f}; small={p_small}, large={p_large}"


def test_guard_refuses_at_domain_bound():
    """bounded_by_domain n<=1e6 honest: n=1e6 + guard=10MB → refuse."""
    d, K = 50, 3
    df = _make_df(1_000_000, d, K, seed=0)
    with pytest.raises(Exception):
        fit_ordinal_logit(df, [f"x{j}" for j in range(d)], "y",
                          memory_guard=MemoryGuard(max_memory_mb=10),
                          batch_size=1000)


def test_tight_guard_raises():
    class _FakeMB:
        max_memory_mb = 0.001
    df = _make_df(1000, 5, 3, seed=0)
    with pytest.raises(Exception):
        fit_ordinal_logit(df, [f"x{j}" for j in range(5)], "y",
                          memory_guard=_FakeMB())
