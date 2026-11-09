"""v0.9.164: final legacy batch verification (3 complex ML cards).

Hybrid_MF_GBDT: ALS base + ridge content → state = base factors + B + C.
T_learner_uplift: 2× GBDT classifier (shim sums state).
CausalTree: sum over causal trees + 1024 header.
"""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml.hybrid_mf import fit_hybrid_mf, HybridMFModel
from cadw.ml.uplift import (
    fit_uplift, UpliftModel, _fit_uplift_two_model,
)


def _make_mf_df(n_users, n_items, F, n_rows, seed=0):
    rng = np.random.RandomState(seed)
    users = rng.randint(0, n_users, n_rows)
    items = rng.randint(0, n_items, n_rows)
    ratings = rng.normal(size=n_rows)
    content = rng.normal(size=(n_items, F))
    tbl = {
        "u": pa.array(users, type=pa.int64()),
        "i": pa.array(items, type=pa.int64()),
        "r": pa.array(ratings, type=pa.float64()),
    }
    for j in range(F):
        tbl[f"c{j}"] = pa.array(content[items, j], type=pa.float64())
    return CadwDataFrame.from_table(
        pa.table(tbl), memory_guard=MemoryGuard(max_memory_mb=500))


def _make_uplift_df(n, d=3, seed=0, *, guard_mb=500):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    t = (rng.random(n) > 0.5).astype(np.float64)
    y = (X[:, 0] + 0.3 * t + 0.3 * rng.normal(size=n) > 0.5).astype(np.float64)
    tbl = {f"f{j}": pa.array(X[:, j], type=pa.float64())
           for j in range(d)}
    tbl["t"] = pa.array(t, type=pa.float64())
    tbl["y"] = pa.array(y, type=pa.float64())
    return CadwDataFrame.from_table(
        pa.table(tbl), memory_guard=MemoryGuard(max_memory_mb=guard_mb))


# ============================================================================
# Hybrid_MF_GBDT
# ============================================================================

def test_hybrid_mf_state_bounded():
    U, I, K, F, n = 20, 30, 8, 4, 500
    df = _make_mf_df(U, I, F, n)
    m = fit_hybrid_mf(df, "u", "i", [f"c{j}" for j in range(F)],
                      n_factors=K, n_iter=5,
                      memory_guard=MemoryGuard(max_memory_mb=500))
    assert isinstance(m, HybridMFModel)
    assert m.state_nbytes() > 0


def test_hybrid_mf_state_independent_of_n_rows():
    """State depends on U, I, K, F (ALS factors + content), not n_rows."""
    U, I, K, F = 20, 30, 8, 4
    m1 = fit_hybrid_mf(_make_mf_df(U, I, F, 300, seed=0),
                        "u", "i", [f"c{j}" for j in range(F)],
                        n_factors=K, n_iter=5,
                        memory_guard=MemoryGuard(max_memory_mb=500))
    m2 = fit_hybrid_mf(_make_mf_df(U, I, F, 3000, seed=0),
                        "u", "i", [f"c{j}" for j in range(F)],
                        n_factors=K, n_iter=5,
                        memory_guard=MemoryGuard(max_memory_mb=500))
    # Same U, I, K, F -> same state (bit-identical ALS dimensions)
    assert m1.state_nbytes() == m2.state_nbytes(), (
        f"state differs across n_rows: {m1.state_nbytes()} vs {m2.state_nbytes()}")


def test_hybrid_mf_guard_refuses():
    df = _make_mf_df(20, 30, 4, 500)
    with pytest.raises((MemoryLimitExceeded, Exception)):
        fit_hybrid_mf(df, "u", "i", [f"c{j}" for j in range(4)],
                      n_factors=8, n_iter=5,
                      memory_guard=MemoryGuard(max_memory_mb=0.001))


# ============================================================================
# T_learner_uplift
# ============================================================================

def test_t_learner_state_bounded():
    df = _make_uplift_df(400)
    m = _fit_uplift_two_model(df, ["f0", "f1", "f2"], "t", "y",
                               n_estimators=5, max_depth=2,
                               memory_guard=MemoryGuard(max_memory_mb=500))
    assert m.state_nbytes() > 0


def test_t_learner_state_independent_of_n():
    m1 = _fit_uplift_two_model(_make_uplift_df(400, seed=0),
                                ["f0", "f1", "f2"], "t", "y",
                                n_estimators=5, max_depth=2,
                                memory_guard=MemoryGuard(max_memory_mb=500))
    m2 = _fit_uplift_two_model(_make_uplift_df(4000, seed=0),
                                ["f0", "f1", "f2"], "t", "y",
                                n_estimators=5, max_depth=2,
                                memory_guard=MemoryGuard(max_memory_mb=500))
    ratio = max(m1.state_nbytes(), m2.state_nbytes()) / max(
        min(m1.state_nbytes(), m2.state_nbytes()), 1)
    assert ratio <= 3.0, f"state ratio {ratio:.2f} across n (bounded expected)"


def test_t_learner_guard_refuses():
    df = _make_uplift_df(400, guard_mb=0.001)
    with pytest.raises((MemoryLimitExceeded, Exception)):
        _fit_uplift_two_model(df, ["f0", "f1", "f2"], "t", "y",
                               n_estimators=5, max_depth=2,
                               memory_guard=MemoryGuard(max_memory_mb=0.001))


# ============================================================================
# CausalTree
# ============================================================================

def test_causal_tree_state_bounded():
    df = _make_uplift_df(400)
    m = fit_uplift(df, ["f0", "f1", "f2"], "t", "y",
                    n_estimators=5, max_depth=2,
                    memory_guard=MemoryGuard(max_memory_mb=500))
    assert isinstance(m, UpliftModel)
    assert m.state_nbytes() > 0


def test_causal_tree_state_independent_of_n():
    m1 = fit_uplift(_make_uplift_df(400, seed=0),
                     ["f0", "f1", "f2"], "t", "y",
                     n_estimators=5, max_depth=2,
                     memory_guard=MemoryGuard(max_memory_mb=500))
    m2 = fit_uplift(_make_uplift_df(4000, seed=0),
                     ["f0", "f1", "f2"], "t", "y",
                     n_estimators=5, max_depth=2,
                     memory_guard=MemoryGuard(max_memory_mb=500))
    ratio = max(m1.state_nbytes(), m2.state_nbytes()) / max(
        min(m1.state_nbytes(), m2.state_nbytes()), 1)
    assert ratio <= 3.0, f"state ratio {ratio:.2f} across n (bounded expected)"


def test_causal_tree_guard_refuses():
    df = _make_uplift_df(400, guard_mb=0.001)
    with pytest.raises((MemoryLimitExceeded, Exception)):
        fit_uplift(df, ["f0", "f1", "f2"], "t", "y",
                    n_estimators=5, max_depth=2,
                    memory_guard=MemoryGuard(max_memory_mb=0.001))
