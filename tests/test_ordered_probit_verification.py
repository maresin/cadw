"""v0.9.229-c: ordered_probit multi-method verification (DEC-033).

Methods (4 in fit_ordered_probit):
  - sghmc (default): OOC any n; verified P63/P64
  - srvr-hmc: OOC any n; verified P76
  - gibbs: exact, n<=1e5; verified P46
  - nuts: exact, n<=1e5; primitive verified in test_nuts.py

This file:
  _METHODS = 3 scalable/exact methods (nuts excluded — see test_nuts.py)
  CW-1..4 for 3 methods; correctness/monotone/parity under @slow
  test_nuts_method_smoke: single small nuts fit (integration only)
"""
from __future__ import annotations
import gc

import numpy as np
import pyarrow as pa
import pytest
from scipy.stats import norm

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard
from cadw.ml.ordered_probit import (
    OrderedProbitModel, fit_ordered_probit)


_METHODS = ("gibbs", "sghmc", "srvr-hmc")


def _n_burnin(method):
    """Per-method n_burnin (heavy test values, P46/P63/P76)."""
    return {"gibbs": 1000, "sghmc": 3000, "srvr-hmc": 2000}[method]


def _make_df(n, d=3, K=3, beta_true=None, cut_true=(-1.0, 1.0), seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    if beta_true is None:
        beta_true = rng.normal(size=d) * 1.5
    else:
        beta_true = np.asarray(beta_true, dtype=np.float64)
        if beta_true.shape != (d,):
            raise ValueError(
                f"_make_df: beta_true shape {beta_true.shape}, expected ({d},)")
    eta = X @ beta_true
    z = eta + rng.normal(size=n)
    y = np.zeros(n, dtype=np.int64)
    cut_arr = np.array(cut_true)
    for k in range(K - 1):
        y[z > cut_arr[k]] = k + 1
    cols = {f"x{j}": pa.array(X[:, j], type=pa.float64())
            for j in range(d)}
    cols["y"] = pa.array(y, type=pa.int64())
    return CadwDataFrame.from_table(
        pa.table(cols), memory_guard=MemoryGuard(max_memory_mb=500))


def _extract_Xy(df, d):
    X_parts, y_parts = [], []
    for b in df._iter_filtered_batches():
        X_parts.append(np.column_stack([
            b.column(f"x{j}").to_numpy(zero_copy_only=False)
            for j in range(d)]))
        y_parts.append(b.column("y").to_numpy(zero_copy_only=False))
    X = np.concatenate(X_parts) if len(X_parts) > 1 else X_parts[0]
    y = np.concatenate(y_parts) if len(y_parts) > 1 else y_parts[0]
    return X, y


def _bayes_accuracy(X, y, beta_true, cut_true, K):
    eta = X @ np.array(beta_true)
    c_full = [-np.inf] + list(cut_true) + [np.inf]
    P = np.zeros((len(y), K))
    for k in range(K):
        P[:, k] = (norm.cdf(c_full[k+1] - eta)
                   - norm.cdf(c_full[k] - eta))
    return float((P.argmax(axis=1) == y).mean())


def _warmup():
    df = _make_df(200, d=2, K=3, seed=999)
    for method in _METHODS:
        fit_ordered_probit(
            df, ["x0", "x1"], "y",
            method=method, n_samples=10, n_burnin=20,
            memory_guard=MemoryGuard(max_memory_mb=500))


_warmup()


# ---- CW-1: state_nbytes exact ----

@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("d", [2, 5, 20])
def test_state_nbytes_exact(method, d):
    df = _make_df(500, d=d, K=3, seed=0)
    n_samples = 50
    m = fit_ordered_probit(
        df, [f"x{j}" for j in range(d)], "y",
        method=method, n_samples=n_samples, n_burnin=100,
        memory_guard=MemoryGuard(max_memory_mb=500))
    K = m.n_classes_
    expected = (d * 8 + (K - 1) * 8
                + n_samples * d * 8 + n_samples * (K - 1) * 8 + 256)
    assert m.state_nbytes() == expected, (
        f"{method} d={d}: state={m.state_nbytes()}, expected={expected}")


# ---- CW-2: state independent of n ----

@pytest.mark.parametrize("method", _METHODS)
def test_state_independent_of_n(method):
    states = []
    for n in [500, 1000, 2000]:
        df = _make_df(n, d=3, K=3, seed=0)
        m = fit_ordered_probit(
            df, [f"x{j}" for j in range(3)], "y",
            method=method, n_samples=30, n_burnin=50,
            memory_guard=MemoryGuard(max_memory_mb=500))
        states.append(m.state_nbytes())
        del df
        gc.collect()
    assert max(states) == min(states), (
        f"{method} states vary: {states}")


# ---- CW-3: state bounded (larger n) ----

@pytest.mark.parametrize("method", _METHODS)
def test_state_bounded_for_n_ge_s(method):
    states = []
    for n in [500, 2000, 5000]:
        df = _make_df(n, d=3, K=3, seed=0)
        m = fit_ordered_probit(
            df, [f"x{j}" for j in range(3)], "y",
            method=method, n_samples=30, n_burnin=50,
            memory_guard=MemoryGuard(max_memory_mb=500))
        states.append(m.state_nbytes())
        del df
        gc.collect()
    ratio = max(states) / max(min(states), 1)
    assert ratio <= 1.5, (
        f"{method} state ratio {ratio:.2f}; states={states}")


# ---- CW-4: guard refuses ----

@pytest.mark.parametrize("method", _METHODS)
def test_guard_refuses_at_tight(method):
    df = _make_df(500, d=3, K=3, seed=0)
    with pytest.raises(Exception):
        fit_ordered_probit(
            df, [f"x{j}" for j in range(3)], "y",
            method=method, n_samples=100, n_burnin=100,
            memory_guard=MemoryGuard(max_memory_mb=0.0005))


# ---- @slow: correctness (recover β, cutpoints; Bayes pre-computed) ----

@pytest.mark.slow
@pytest.mark.parametrize("method", _METHODS)
def test_recovers_ordered_classes(method):
    n, d, K = 1000, 3, 3
    beta_true = np.array([2.0, -1.5, 1.0])
    cut_true = np.array([-1.0, 1.0])
    df = _make_df(n, d=d, K=K, beta_true=beta_true,
                  cut_true=tuple(cut_true), seed=0)
    X, y = _extract_Xy(df, d)
    bayes = _bayes_accuracy(X, y, beta_true, cut_true, K)
    print(f"\n[{method}] Bayes = {bayes:.3f}")

    m = fit_ordered_probit(
        df, [f"x{j}" for j in range(d)], "y",
        method=method, n_samples=200, n_burnin=_n_burnin(method),
        seed=0, memory_guard=MemoryGuard(max_memory_mb=500))
    coef_err = float(np.abs(m.coef_mean_ - beta_true).max())
    cp_err = float(np.abs(m.cutpoints_mean_ - cut_true).max())
    assert coef_err < 0.5, f"{method}: coef_err={coef_err:.3f}"
    assert cp_err < 0.5, f"{method}: cp_err={cp_err:.3f}"
    assert (np.diff(m.cutpoints_mean_) > 0).all(), (
        f"{method}: cutpoints not monotone: {m.cutpoints_mean_}")


# ---- @slow: cutpoints monotone (K=4) ----

@pytest.mark.slow
@pytest.mark.parametrize("method", _METHODS)
def test_cutpoints_monotone(method):
    df = _make_df(500, d=3, K=4,
                  beta_true=(2.0, -1.5, 1.0),
                  cut_true=(-1.5, -0.5, 0.5),
                  seed=0)
    m = fit_ordered_probit(
        df, [f"x{j}" for j in range(3)], "y",
        method=method, n_samples=100, n_burnin=_n_burnin(method),
        seed=0, memory_guard=MemoryGuard(max_memory_mb=500))
    diffs = np.diff(m.cutpoints_mean_)
    assert (diffs > 0).all(), (
        f"{method}: cutpoints not strictly increasing: "
        f"{m.cutpoints_mean_}")


# ---- @slow: 3-method parity ----

@pytest.mark.slow
def test_methods_agree():
    df = _make_df(1000, d=3, K=3, seed=0)
    results = {}
    for method in _METHODS:
        m = fit_ordered_probit(
            df, [f"x{j}" for j in range(3)], "y",
            method=method, n_samples=200, n_burnin=_n_burnin(method),
            seed=0, memory_guard=MemoryGuard(max_memory_mb=500))
        results[method] = (m.coef_mean_.copy(), m.cutpoints_mean_.copy())

    ml = list(_METHODS)
    for i in range(len(ml)):
        for j in range(i + 1, len(ml)):
            a, b = ml[i], ml[j]
            coef_diff = float(np.abs(
                results[a][0] - results[b][0]).max())
            cp_diff = float(np.abs(
                results[a][1] - results[b][1]).max())
            assert coef_diff < 0.7, f"{a} vs {b}: coef_diff={coef_diff:.3f}"
            assert cp_diff < 0.7, f"{a} vs {b}: cp_diff={cp_diff:.3f}"


# ---- nuts integration smoke (small; primitive in test_nuts.py) ----

def test_nuts_method_smoke():
    """fit_ordered_probit(method='nuts') wires correctly (integration only).

    Full NUTS correctness/parity: test_nuts.py (10 tests, green,
    bit-identical to reference). This test: fit_ordered_probit branch
    produces sane shapes and monotone cutpoints.
    """
    df = _make_df(500, d=3, K=3,
                  beta_true=(2.0, -1.5, 1.0),
                  cut_true=(-1.0, 1.0), seed=0)
    m = fit_ordered_probit(
        df, [f"x{j}" for j in range(3)], "y",
        method="nuts", n_samples=30, n_burnin=20,
        max_tree_depth=3, lr=0.1, seed=0,
        memory_guard=MemoryGuard(max_memory_mb=500))
    assert m.coef_mean_.shape == (3,)
    assert m.cutpoints_mean_.shape == (2,)
    assert (np.diff(m.cutpoints_mean_) > 0).all(), (
        f"cutpoints not monotone: {m.cutpoints_mean_}")


# ---- determinism ----

@pytest.mark.parametrize("method", _METHODS)
def test_deterministic(method):
    df = _make_df(300, d=3, K=3, seed=0)
    m1 = fit_ordered_probit(
        df, [f"x{j}" for j in range(3)], "y",
        method=method, n_samples=30, n_burnin=50,
        seed=42, memory_guard=MemoryGuard(max_memory_mb=500))
    m2 = fit_ordered_probit(
        df, [f"x{j}" for j in range(3)], "y",
        method=method, n_samples=30, n_burnin=50,
        seed=42, memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(m1.coef_samples_, m2.coef_samples_)


# ---- invalid input ----

def test_invalid_method():
    df = _make_df(100, d=2, K=3, seed=0)
    with pytest.raises(ValueError, match="method"):
        fit_ordered_probit(
            df, ["x0", "x1"], "y",
            method="bad", n_samples=10, n_burnin=0,
            memory_guard=MemoryGuard(max_memory_mb=500))


def test_x_cols_overlap():
    df = _make_df(100, d=2, K=3, seed=0)
    with pytest.raises(ValueError, match="must not be in X_cols"):
        fit_ordered_probit(
            df, ["x0", "y"], "y", method="gibbs",
            n_samples=10, n_burnin=0,
            memory_guard=MemoryGuard(max_memory_mb=500))
