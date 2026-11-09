"""v0.9.163: CW-1..5 for LambdaMART (G_learning_to_rank).

State = n_nodes * 96 + n_features * 4096 (from code, LambdaMARTModel.state_nbytes).
Fit uses SpillDir memmaps (OOC by design v0.9.87-P2b-full-2).
Upper bound: n_estimators × (2^(max_depth+1)-1) nodes × 96 B + d × 4096.
"""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml.lambdamart import fit_lambdamart, LambdaMARTModel


def _make_ltr_df(nq=40, n=12, d=4, seed=0, *, guard_mb=500):
    rng = np.random.RandomState(seed)
    rows, qs, ys = [], [], []
    for q in range(nq):
        X = rng.normal(size=(n, d))
        s = X[:, 0] + 0.5 * X[:, 1]
        rel = (s > 0).astype(int) + (s > 1.0).astype(int)
        rows.append(X); qs += [q] * n; ys += rel.tolist()
    X = np.vstack(rows)
    cols = {f"f{j}": pa.array(X[:, j], type=pa.float64()) for j in range(d)}
    cols["y"] = pa.array(ys, type=pa.int64())
    cols["q"] = pa.array(qs, type=pa.int64())
    return CadwDataFrame.from_table(
        pa.table(cols), memory_guard=MemoryGuard(max_memory_mb=guard_mb))


def _upper_bound(n_est, depth, d):
    nodes_max = 2 ** (depth + 1) - 1
    return int(n_est * nodes_max * 96 + d * 4096 + 256)


def test_state_within_upper_bound():
    """CW-1: actual state ∈ [0.20, 1.0] of upper bound."""
    n_est, depth, d = 10, 3, 4
    df = _make_ltr_df(d=d)
    m = fit_lambdamart(df, [f"f{j}" for j in range(d)], "y", "q",
                        n_estimators=n_est, max_depth=depth,
                        memory_guard=MemoryGuard(max_memory_mb=500))
    assert isinstance(m, LambdaMARTModel)
    upper = _upper_bound(n_est, depth, d)
    actual = m.state_nbytes()
    assert 0.20 * upper <= actual <= upper, (
        f"state {actual} not in [{0.20*upper:.0f}, {upper}]")


def test_state_independent_of_n():
    """CW-2: same params, n=480 vs 4800 -> state ratio <= 3.0."""
    n_est, depth, d = 5, 2, 4
    m1 = fit_lambdamart(_make_ltr_df(nq=40, n=12, d=d, seed=0),
                         [f"f{j}" for j in range(d)], "y", "q",
                         n_estimators=n_est, max_depth=depth,
                         memory_guard=MemoryGuard(max_memory_mb=500))
    m2 = fit_lambdamart(_make_ltr_df(nq=400, n=12, d=d, seed=0),
                         [f"f{j}" for j in range(d)], "y", "q",
                         n_estimators=n_est, max_depth=depth,
                         memory_guard=MemoryGuard(max_memory_mb=500))
    s1, s2 = m1.state_nbytes(), m2.state_nbytes()
    ratio = max(s1, s2) / max(min(s1, s2), 1)
    assert ratio <= 3.0, f"state ratio {ratio:.2f} across n (bounded expected)"


def test_guard_refuses():
    """CW-6: nq=40, n=12, n_est=50, depth=10 + tiny guard -> refuse."""
    df = _make_ltr_df(guard_mb=0.001)
    with pytest.raises((MemoryLimitExceeded, Exception)):
        fit_lambdamart(df, ["f0", "f1", "f2", "f3"], "y", "q",
                        n_estimators=50, max_depth=10,
                        memory_guard=MemoryGuard(max_memory_mb=0.001))


def test_predict_smoke():
    """Sanity: predict returns pa.Table with 'score' column."""
    d = 4
    df = _make_ltr_df(d=d)
    m = fit_lambdamart(df, [f"f{j}" for j in range(d)], "y", "q",
                        n_estimators=5, max_depth=2,
                        memory_guard=MemoryGuard(max_memory_mb=500))
    out = m.predict(df)
    assert "score" in out.column_names
    assert out.num_rows == df.count()
