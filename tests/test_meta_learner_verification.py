"""Component-wise verification for T_learner / S_learner (DEC-033, v0.9.141).

Both use fit_meta_learner (same code path, different kind='t'/'s').

State = n_estimators * K * 4096 + n * 8 + 1024
  K = 2 (T_learner: mu1 + mu0), K = 1 (S_learner: s_model)
  propensity_ = O(n) per-row array (honest O(n) state).

Fit transient O(n*d) via _collect + np.vstack (AP2, backlog v0.9.142).
bounded_by_domain n<=1e6.
"""
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml.meta_learners import fit_meta_learner


def _make_df(n, d=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    t = (rng.uniform(size=n) < 0.5).astype(float)
    y = 2.0 * X[:, 0] + t + rng.normal(scale=0.1, size=n)
    tbl = {f"x{j}": pa.array(X[:, j]) for j in range(d)}
    tbl["t"] = pa.array(t)
    tbl["y"] = pa.array(y)
    return CadwDataFrame.from_table(
        pa.table(tbl), memory_guard=MemoryGuard(max_memory_mb=500))


def _cols(d):
    return [f"x{j}" for j in range(d)]


@pytest.mark.parametrize("kind,K", [("t", 2), ("s", 1)])
@pytest.mark.parametrize("n,n_est", [(500, 5), (2000, 10), (5000, 5)])
def test_state_nbytes_exact(kind, K, n, n_est):
    """state = n_estimators*K*4096 + n*8 + 1024."""
    df = _make_df(n, d=3)
    m = fit_meta_learner(
        df, _cols(3), "t", "y",
        kind=kind, n_estimators=n_est, max_depth=3,
        memory_guard=MemoryGuard(max_memory_mb=500))
    expected = n_est * K * 4096 + n * 8 + 1024
    assert m.state_nbytes() == expected, (
        f"kind={kind} n={n} n_est={n_est}: "
        f"state={m.state_nbytes()} expected={expected}")


@pytest.mark.parametrize("kind", ["t", "s"])
def test_state_ratio_linear(kind):
    """5x n -> ratio 2-5 (propensity n*8 dominates for small trees)."""
    s_small = fit_meta_learner(
        _make_df(1000, d=3), _cols(3), "t", "y",
        kind=kind, n_estimators=5, max_depth=3,
        memory_guard=MemoryGuard(max_memory_mb=500)).state_nbytes()
    s_large = fit_meta_learner(
        _make_df(5000, d=3), _cols(3), "t", "y",
        kind=kind, n_estimators=5, max_depth=3,
        memory_guard=MemoryGuard(max_memory_mb=500)).state_nbytes()
    ratio = s_large / max(s_small, 1)
    # n_est=5 -> trees const: T=40960, S=20480. propensity grows
    # 8K->40K (5x n). ratio = (trees + 40K) / (trees + 8K).
    #   T: (40960+40000)/(40960+8000) = 1.65
    #   S: (20480+40000)/(20480+8000) = 2.12
    # Linear growth in n; ratio lower than 5 because trees add
    # constant. Healthy range [1.5, 5.0].
    assert 1.5 <= ratio <= 5.0, (
        f"kind={kind}: ratio={ratio:.2f}; {s_small}->{s_large}")


@pytest.mark.parametrize("kind", ["t", "s"])
def test_fit_refuses_at_tight(kind):
    """n=1e6 + tight guard -> MLE via _collect check_state."""
    df = _make_df(1_000_000, d=3)
    df._memory_guard = MemoryGuard(max_memory_mb=1)
    with pytest.raises(MemoryLimitExceeded):
        fit_meta_learner(
            df, _cols(3), "t", "y",
            kind=kind, n_estimators=5, max_depth=3,
            memory_guard=df._memory_guard)
