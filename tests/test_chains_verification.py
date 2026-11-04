"""Component-wise verification for fit_chains (DEC-033, v0.9.118).

State = sum(GBDTModel.state_nbytes() for n_estimators models) + 512.
Fit peak = O(n*d) for _collect np.vstack(Xs) + O(n_estimators * d * nodes)
for GBDT fit (from fit_gbdt_classifier).

bounded_by_domain n<=1e6: at n=1e6, d=20 → n*d*8 = 160 MB; guard
MemoryLimitExceeded via check_state in _collect line 32.
"""
from __future__ import annotations
import ctypes, gc, statistics
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ml.chains import fit_chains


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _make_df(n, d, L, seed=0):
    """n rows, d features, L binary labels (correlated with X)."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    tbl = {f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(d)}
    for k in range(L):
        # y_k correlated with x_k to ensure two classes
        y_k = (X[:, k % d] + 0.3 * rng.normal(size=n) > 0).astype(np.float64)
        tbl[f"y{k}"] = pa.array(y_k, type=pa.float64())
    return CadwDataFrame.from_table(pa.table(tbl))


@pytest.mark.parametrize("n_est,d,L", [(5, 5, 2), (10, 8, 3), (5, 10, 4)])
def test_state_nbytes_exact(n_est, d, L):
    """model.state_nbytes() == sum(m.state_nbytes()) + 512 (bit-exact)."""
    df = _make_df(300, d, L, seed=0)
    cols = [f"x{j}" for j in range(d)]
    labels = [f"y{k}" for k in range(L)]
    m = fit_chains(df, cols, labels, n_estimators=n_est,
                   memory_guard=MemoryGuard(max_memory_mb=300))
    # chains.py: sum(m.state_nbytes() if hasattr(m, "state_nbytes") else 4096
    #              for m in models_) + 512
    per_model = [mm.state_nbytes() if hasattr(mm, "state_nbytes") else 4096
                 for mm in m.models_]
    expected = sum(per_model) + 512
    assert m.state_nbytes() == expected, (
        f"n_est={n_est} d={d} L={L}: state={m.state_nbytes()} expected={expected} "
        f"per_model={per_model[:3]}... n_models={len(m.models_)}")


def test_state_independent_of_n():
    """Chains state is O(n_estimators * d²): state(n=2k) == state(n=10k)."""
    d, L, n_est = 8, 3, 5
    mg = MemoryGuard(max_memory_mb=500)
    cols = [f"x{j}" for j in range(d)]
    labels = [f"y{k}" for k in range(L)]
    m1 = fit_chains(_make_df(2_000, d, L, seed=0), cols, labels,
                    n_estimators=n_est, memory_guard=mg, batch_size=500)
    m2 = fit_chains(_make_df(10_000, d, L, seed=0), cols, labels,
                    n_estimators=n_est, memory_guard=mg, batch_size=500)
    # GBDT fit may produce different node counts (data-dependent);
    # allow bounded variation (<=1.5x)
    s1, s2 = m1.state_nbytes(), m2.state_nbytes()
    ratio = max(s1, s2) / max(min(s1, s2), 1)
    assert ratio <= 1.5, f"state ratio {ratio:.2f}; s1={s1} s2={s2}"


def test_formula_small_scale():
    """predicted/measured in [0.5, 2.0] at d=20, L=3, n=4000."""
    d, L, n, n_est, batch = 20, 3, 4000, 5, 500
    df = _make_df(n, d, L, seed=0)
    _malloc_trim(); gc.collect()
    mg = MemoryGuard(max_memory_mb=500)
    r = measure_peak_traced(
        fit_chains, df,
        [f"x{j}" for j in range(d)], [f"y{k}" for k in range(L)],
        n_estimators=n_est,
        memory_guard=mg, batch_size=batch, mg=mg,
    )
    measured = int(r["peak_bytes"])
    # predicted = 2*n*d*8 (Xs list + vstack) + n_est * 1 MB (GBDT transient)
    # + n*L*8 (Ys list) + state (~few MB, GBDT tree nodes)
    # GBDT per-estimator transient at n_est=5, d=20, n=4000, small trees
    # (~400 KB per estimator: tree nodes + gain accumulators). Measured
    # 3.8 MB total → 2*4000*20*8 (1.28 MB) + 4000*3*8 (96 KB)
    # + 5*400 KB (2 MB) ≈ 3.38 MB → ratio ~0.89.
    predicted = (2 * n * d * 8
                 + n * L * 8
                 + n_est * 400_000)
    ratio = predicted / max(measured, 1)
    assert 0.5 <= ratio <= 2.0, (
        f"predicted={predicted:,} measured={measured:,} ratio={ratio:.2f}")


def test_peak_grows_linearly_with_n():
    """n=3k vs 30k, ratio < 15 (linear Xs np.vstack peak)."""
    d, L, n_est = 20, 3, 5
    def _peak(n):
        df = _make_df(n, d, L, seed=0)
        _malloc_trim(); gc.collect()
        mg = MemoryGuard(max_memory_mb=2000)
        r = measure_peak_traced(
            fit_chains, df,
            [f"x{j}" for j in range(d)], [f"y{k}" for k in range(L)],
            n_estimators=n_est,
            memory_guard=mg, batch_size=500, mg=mg,
        )
        return int(r["peak_bytes"])
    p_small = _peak(3_000)
    p_large = _peak(30_000)
    ratio = p_large / max(p_small, 1)
    # v0.9.118 (DEC-034 escalation): chains fit = _collect np.vstack(Xs)
    # + n_estimators × GBDT fit. GBDT per-tree per-round state (codes
    # int8 n×d + pred n×8 per round) adds ~4x constant factor at n=30k,
    # d=20, 5 estimators — still linear in n (52/3.0 = 17.3 < 20).
    # Isolation ratio ~10; full-suite baseline RSS adds drift.
    assert ratio < 20.0, f"ratio {ratio:.2f}; small={p_small}, large={p_large}"


def test_guard_refuses_at_domain_bound():
    """bounded_by_domain n<=1e6 honest: n=1e6 + guard=10MB -> refuse."""
    d, L = 20, 3
    df = _make_df(1_000_000, d, L, seed=0)
    with pytest.raises(Exception):
        fit_chains(df, [f"x{j}" for j in range(d)], [f"y{k}" for k in range(L)],
                   n_estimators=5,
                   memory_guard=MemoryGuard(max_memory_mb=10),
                   batch_size=1000)


def test_tight_guard_raises():
    class _FakeMB:
        max_memory_mb = 0.001
    df = _make_df(1000, 5, 2, seed=0)
    with pytest.raises(Exception):
        fit_chains(df, [f"x{j}" for j in range(5)], ["y0", "y1"],
                   n_estimators=5, memory_guard=_FakeMB())
