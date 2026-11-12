"""v0.9.225-a: LinearQuantReg component-wise verification (DEC-033).

State = coef_(d)*f64 + intercept_*f64 + 256B header. O(d).
Fit peak = O(batch * d); streams n via _iter_filtered_batches.

CW-1 state exact / CW-2 state bounded by n / CW-5 OOC + correctness.
"""
from __future__ import annotations
import ctypes
import gc
import statistics

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, measure_peak_traced
from cadw.ml.quantile_linear import (
    QuantileLinearModel, fit_linear_quantile)


def _make_df(n, d, seed=0, slope=2.0, noise=1.0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    y = slope * X[:, 0] + noise * rng.normal(size=n)
    cols = {f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(d)}
    cols["y"] = pa.array(y, type=pa.float64())
    return CadwDataFrame.from_table(
        pa.table(cols), memory_guard=MemoryGuard(max_memory_mb=500))


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _warmup():
    df = _make_df(64, 2, seed=999)
    fit_linear_quantile(
        df, ["x0", "x1"], "y", quantile=0.5, n_epochs=2, lr=0.01,
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=32)


_warmup()


# ---- CW-1: state exact ----

@pytest.mark.parametrize("d", [2, 10, 50, 200])
def test_state_nbytes_exact(d):
    df = _make_df(200, d, seed=1)
    m = fit_linear_quantile(
        df, [f"x{j}" for j in range(d)], "y",
        quantile=0.5, n_epochs=2, lr=0.01,
        memory_guard=MemoryGuard(max_memory_mb=500))
    assert m.state_nbytes() == d * 8 + 8 + 256, (
        f"d={d}: state={m.state_nbytes()}, expected={d * 8 + 8 + 256}")


# ---- CW-2: state bounded by n ----

def test_state_independent_of_n():
    d = 20
    cols = [f"x{j}" for j in range(d)]
    states = []
    for n in [1000, 5000, 20000]:
        m = fit_linear_quantile(
            _make_df(n, d, seed=0), cols, "y",
            quantile=0.5, n_epochs=2, lr=0.01,
            memory_guard=MemoryGuard(max_memory_mb=500), batch_size=500)
        states.append(m.state_nbytes())
    assert max(states) == min(states), (
        f"state varies with n: {states}")


# ---- CW-5: OOC peak ----

def test_ooc_batch_fixed():
    d = 20
    cols = [f"x{j}" for j in range(d)]
    peaks = []
    for n in [10_000, 100_000]:
        df = _make_df(n, d, seed=0)
        cell_peaks = []
        for _ in range(3):
            _malloc_trim(); gc.collect()
            mg = MemoryGuard(max_memory_mb=500)
            r = measure_peak_traced(
                fit_linear_quantile, df, cols, "y",
                quantile=0.5, n_epochs=2, lr=0.01,
                memory_guard=mg, batch_size=500, mg=mg)
            cell_peaks.append(int(r["peak_bytes"]))
        peaks.append(int(statistics.median(cell_peaks)))
        del df
        gc.collect()
    ratio = peaks[-1] / max(peaks[0], 1)
    assert ratio < 4.0, (
        f"peak ratio {ratio:.2f} (n=10k -> 100k); peaks={peaks}")


# ---- correctness: pinball coverage ----

@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_pinball_coverage(q):
    rng = np.random.RandomState(0)
    n, d = 5000, 3
    X = rng.normal(size=(n, d))
    y = 2.0 * X[:, 0] + (0.5 + np.exp(X[:, 1] * 0.3)) * rng.normal(size=n)
    tbl = pa.table({
        **{f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(d)},
        "y": pa.array(y, type=pa.float64())})
    df = CadwDataFrame.from_table(
        tbl, memory_guard=MemoryGuard(max_memory_mb=500))
    m = fit_linear_quantile(
        df, [f"x{j}" for j in range(d)], "y",
        quantile=q, n_epochs=30, lr=0.05,
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=500)
    pred_df = m.predict(df)
    _parts = []
    for _b in pred_df._iter_filtered_batches():
        _parts.append(
            _b.column(f"y_q{q:g}").to_numpy(zero_copy_only=False))
    pred = np.concatenate(_parts) if len(_parts) > 1 else _parts[0]
    frac_le = float((y <= pred).mean())
    tol = 0.10 if 0.3 <= q <= 0.7 else 0.15
    assert abs(frac_le - q) < tol, (
        f"q={q}: coverage={frac_le:.3f}, expected ~{q} (tol {tol})")


# ---- correctness: linear median ----

def test_correctness_median_linear():
    rng = np.random.RandomState(42)
    n, d = 3000, 2
    X = rng.normal(size=(n, d))
    coef_true = np.array([2.0, -1.0])
    y = X @ coef_true + 0.5 * rng.normal(size=n)
    tbl = pa.table({
        **{f"x{j}": pa.array(X[:, j], type=pa.float64()) for j in range(d)},
        "y": pa.array(y, type=pa.float64())})
    df = CadwDataFrame.from_table(
        tbl, memory_guard=MemoryGuard(max_memory_mb=500))
    m = fit_linear_quantile(
        df, [f"x{j}" for j in range(d)], "y",
        quantile=0.5, n_epochs=50, lr=0.05,
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=500)
    rel_err = np.abs(m.coef_ - coef_true) / (np.abs(coef_true) + 1e-9)
    assert rel_err.max() < 0.15, (
        f"coef error {rel_err.max():.3f} > 0.15; coef={m.coef_}")


# ---- determinism ----

def test_deterministic():
    df = _make_df(500, 3, seed=0)
    cols = [f"x{j}" for j in range(3)]
    m1 = fit_linear_quantile(
        df, cols, "y", quantile=0.5, n_epochs=5, lr=0.05, seed=42,
        memory_guard=MemoryGuard(max_memory_mb=500))
    m2 = fit_linear_quantile(
        df, cols, "y", quantile=0.5, n_epochs=5, lr=0.05, seed=42,
        memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_allclose(m1.coef_, m2.coef_, atol=1e-10)


# ---- invalid input ----

def test_invalid_quantile():
    df = _make_df(100, 3, seed=0)
    with pytest.raises(ValueError, match="quantile must be in"):
        fit_linear_quantile(df, ["x0", "x1", "x2"], "y", quantile=0.0)
    with pytest.raises(ValueError, match="quantile must be in"):
        fit_linear_quantile(df, ["x0", "x1", "x2"], "y", quantile=1.0)


def test_invalid_epochs():
    df = _make_df(100, 3, seed=0)
    with pytest.raises(ValueError, match="n_epochs must be"):
        fit_linear_quantile(df, ["x0", "x1", "x2"], "y", n_epochs=0)


def test_invalid_lr():
    df = _make_df(100, 3, seed=0)
    with pytest.raises(ValueError, match="lr must be"):
        fit_linear_quantile(df, ["x0", "x1", "x2"], "y", lr=0.0)


# ---- guard refuses ----

def test_guard_refuses_at_tight():
    df = _make_df(1000, 20, seed=0)
    with pytest.raises(Exception):
        fit_linear_quantile(
            df, [f"x{j}" for j in range(20)], "y",
            quantile=0.5, n_epochs=2, lr=0.05,
            memory_guard=MemoryGuard(max_memory_mb=0.0005))
