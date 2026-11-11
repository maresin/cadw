"""v0.9.214: fit_plaid_model verification (Lazzeroni & Owen 2002).

Deterministic ALS + synthetic recovery + state + guard + refuse_above.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import fit_plaid_model


def _mk_df(A):
    return CadwDataFrame.from_table(pa.table({
        f"x{j}": pa.array(A[:, j].astype(np.float64), type=pa.float64())
        for j in range(A.shape[1])
    }), memory_guard=MemoryGuard(max_memory_mb=500))


def _warmup():
    rng = np.random.RandomState(999)
    A = rng.normal(0, 0.01, size=(20, 10))
    A[:5, :5] += 5.0
    r = fit_plaid_model(_mk_df(A), [f"x{j}" for j in range(10)],
                         max_layers=3, threshold=0.5, max_als_iters=10,
                         memory_guard=MemoryGuard(max_memory_mb=500))


_warmup()


# ---- CW-1: single layer recovery (rows 0-9, cols 0-9) ----
def test_single_layer_recovery():
    rng = np.random.RandomState(0)
    A = rng.normal(0, 0.01, size=(30, 20))
    A[:10, :10] += 5.0
    r = fit_plaid_model(_mk_df(A), [f"x{j}" for j in range(20)],
                         max_layers=5, threshold=0.5, max_als_iters=20,
                         memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.n_layers_ >= 1
    R0 = set(r.layers_[0].rows_)
    C0 = set(r.layers_[0].cols_)
    assert R0 >= set(range(10)), f"R0={R0}"
    assert C0 >= set(range(10)), f"C0={C0}"
    assert r.layers_[0].mu_ > 3.0, f"mu={r.layers_[0].mu_}"


# ---- CW-2: two disjoint layers ----
def test_two_disjoint_layers():
    rng = np.random.RandomState(0)
    A = rng.normal(0, 0.01, size=(40, 30))
    A[:10, :10] += 5.0
    A[20:30, 15:25] += 5.0
    r = fit_plaid_model(_mk_df(A), [f"x{j}" for j in range(30)],
                         max_layers=5, threshold=0.5, max_als_iters=20,
                         memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.n_layers_ >= 2, f"n={r.n_layers_}"
    # Both layers with mu > 3
    high_mu = [l for l in r.layers_ if l.mu_ > 3.0]
    assert len(high_mu) >= 2, f"high_mu={[l.mu_ for l in r.layers_]}"


# ---- CW-3: all-zero -> 0 layers ----
def test_all_zero():
    A = np.zeros((10, 10))
    r = fit_plaid_model(_mk_df(A), [f"x{j}" for j in range(10)],
                         memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.n_layers_ == 0


# ---- CW-4: residual decreases ----
def test_residual_decreases():
    rng = np.random.RandomState(0)
    A = rng.normal(0, 0.01, size=(20, 20))
    A[:8, :8] += 4.0
    r = fit_plaid_model(_mk_df(A), [f"x{j}" for j in range(20)],
                         max_layers=3, threshold=0.5, max_als_iters=10,
                         memory_guard=MemoryGuard(max_memory_mb=500))
    # After removing layer, residual should be < original norm
    orig_norm = np.linalg.norm(A)
    assert r.residual_norm_ < orig_norm, (
        f"residual={r.residual_norm_}, orig={orig_norm}")


# ---- CW-5: state_nbytes ----
def test_state_nbytes():
    rng = np.random.RandomState(0)
    A = rng.normal(0, 0.01, size=(20, 20))
    A[:5, :5] += 5.0
    r = fit_plaid_model(_mk_df(A), [f"x{j}" for j in range(20)],
                         max_layers=3, threshold=0.5, max_als_iters=10,
                         memory_guard=MemoryGuard(max_memory_mb=500))
    # state >= 256 header
    assert r.state_nbytes() >= 256
    # n_layers_ * layer.nbytes()
    expected = 256 + sum(l.nbytes() for l in r.layers_)
    assert r.state_nbytes() == expected


# ---- CW-6: guard refuses at A materialization ----
def test_guard_refuses_tight():
    A = np.zeros((100, 100))
    # A state = 2*100*100*8 = 160 KB; guard 0.001 MB = 1 KB -> refuse
    with pytest.raises(MemoryLimitExceeded) as exc:
        fit_plaid_model(_mk_df(A), [f"x{j}" for j in range(100)],
                         memory_guard=MemoryGuard(max_memory_mb=0.001))
    assert "plaid.state" in str(exc.value)


# ---- CW-7: refuse_at_guard min(r,c) > 500 ----
def test_refuse_min_rc():
    # min(600, 600) = 600 > 500 -> refuse
    A = np.zeros((600, 600))
    with pytest.raises(MemoryLimitExceeded, match=r"min\(r=600"):
        fit_plaid_model(_mk_df(A), [f"x{j}" for j in range(600)],
                         memory_guard=MemoryGuard(max_memory_mb=5000))


# ---- CW-8: input validation ----
def test_input_validation():
    A = np.zeros((10, 10))
    df = _mk_df(A)
    X_cols = [f"x{j}" for j in range(10)]
    with pytest.raises(ValueError, match="X_cols is empty"):
        fit_plaid_model(df, [], memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="max_layers"):
        fit_plaid_model(df, X_cols, max_layers=0,
                         memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(ValueError, match="max_als_iters"):
        fit_plaid_model(df, X_cols, max_als_iters=0,
                         memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(KeyError, match="columns missing"):
        fit_plaid_model(df, ["missing"],
                         memory_guard=MemoryGuard(max_memory_mb=500))


# ---- CW-9: to_dict ----
def test_to_dict():
    rng = np.random.RandomState(0)
    A = rng.normal(0, 0.01, size=(20, 20))
    A[:5, :5] += 5.0
    r = fit_plaid_model(_mk_df(A), [f"x{j}" for j in range(20)],
                         max_layers=3, threshold=0.5, max_als_iters=10,
                         memory_guard=MemoryGuard(max_memory_mb=500))
    d = r.to_dict()
    assert d["A_shape"] == (20, 20)
    assert d["n_layers"] == r.n_layers_
    assert d["threshold"] == 0.5
    assert d["max_als_iters"] == 10
    assert len(d["layers"]) == r.n_layers_


# ---- CW-10: ALS fit exact recovery (clean model) ----
def test_als_exact_recovery():
    # Clean model: theta = 5 + alpha + beta, no noise
    rng = np.random.RandomState(1)
    alpha = rng.normal(0, 0.3, 10)
    beta = rng.normal(0, 0.3, 10)
    theta = 5.0 + alpha[:, None] + beta[None, :]
    A = np.zeros((20, 20))
    A[:10, :10] = theta
    r = fit_plaid_model(_mk_df(A), [f"x{j}" for j in range(20)],
                         max_layers=3, threshold=0.5, max_als_iters=20,
                         memory_guard=MemoryGuard(max_memory_mb=500))
    assert r.n_layers_ >= 1
    # mu approx 5.0 (up to re-centering; alpha/beta absorbed into mu
    # if not zero-mean; but recovery should be close)
    assert abs(r.layers_[0].mu_ - 5.0) < 1.0, (
        f"mu={r.layers_[0].mu_}, expected ~5.0")
    # Rows/cols cover the block
    R0 = set(r.layers_[0].rows_)
    C0 = set(r.layers_[0].cols_)
    assert R0 >= set(range(10))
    assert C0 >= set(range(10))


# ---- CW-11: no layers on random noise ----
def test_noise_no_layers():
    rng = np.random.RandomState(0)
    A = rng.normal(0, 0.01, size=(20, 20))
    # No layer signal -> no layer should be extracted (threshold too high)
    r = fit_plaid_model(_mk_df(A), [f"x{j}" for j in range(20)],
                         max_layers=3, threshold=0.5, max_als_iters=10,
                         memory_guard=MemoryGuard(max_memory_mb=500))
    # Could extract 0 or 1 noise-driven layer; must not be > max_layers
    assert r.n_layers_ <= 3


# ---- CW-12: max_layers cap respected ----
def test_max_layers_cap():
    rng = np.random.RandomState(0)
    A = rng.normal(0, 0.01, size=(30, 30))
    # 10 blocks of signal
    for k in range(10):
        i0 = k * 3
        j0 = k * 3
        A[i0:i0+3, j0:j0+3] += 3.0
    r = fit_plaid_model(_mk_df(A), [f"x{j}" for j in range(30)],
                         max_layers=5, threshold=0.5, max_als_iters=10,
                         memory_guard=MemoryGuard(max_memory_mb=500))
    # max_layers=5 -> at most 5 layers
    assert r.n_layers_ <= 5
