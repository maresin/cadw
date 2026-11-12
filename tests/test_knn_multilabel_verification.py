"""v0.9.225-b: ML-kNN component-wise verification (DEC-033).

State = n_train*(d+L)*f64 + L*(k+1)*f64*2 + L*f64 + 256. O(max_samples).
Fit peak = O(max_samples) constant + O(batch*row_size) transient.
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
from cadw.ml.knn_multilabel import (
    MLkNNModel, fit_knn_multilabel)


def _make_df(n, d, L, seed=0):
    """Synthetic multilabel: y0 = X[:,0]>0, y1 = (X[:,1]+0.5*X[:,0])>0,
    y2 = random 30% (if L>=3)."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    cols = {f"x{j}": pa.array(X[:, j], type=pa.float64())
            for j in range(d)}
    y0 = (X[:, 0] > 0).astype(np.int64)
    cols["y0"] = pa.array(y0, type=pa.int64())
    if L >= 2:
        y1 = (X[:, 1] + 0.5 * X[:, 0] > 0).astype(np.int64)
        cols["y1"] = pa.array(y1, type=pa.int64())
    if L >= 3:
        y2 = (rng.random(n) < 0.3).astype(np.int64)
        cols["y2"] = pa.array(y2, type=pa.int64())
    return CadwDataFrame.from_table(
        pa.table(cols), memory_guard=MemoryGuard(max_memory_mb=500))


def _extract_X(df, X_cols):
    parts = []
    for b in df._iter_filtered_batches():
        parts.append(np.column_stack([
            b.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
            for c in X_cols]))
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def _extract_pred(df, label_names, suffix="_pred"):
    parts = []
    for b in df._iter_filtered_batches():
        parts.append(np.column_stack([
            b.column(f"{c}{suffix}").to_numpy(zero_copy_only=False)
            for c in label_names]))
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _warmup():
    df = _make_df(100, 3, 2, seed=999)
    fit_knn_multilabel(
        df, ["x0", "x1", "x2"], ["y0", "y1"],
        k=3, max_samples=50, seed=0,
        memory_guard=MemoryGuard(max_memory_mb=500))


_warmup()


# ---- CW-1: state exact ----

@pytest.mark.parametrize("n_train,d,L,k", [
    (500, 3, 2, 3),
    (1000, 5, 2, 5),
    (800, 10, 3, 7),
])
def test_state_nbytes_exact(n_train, d, L, k):
    df = _make_df(n_train, d, L, seed=0)
    m = fit_knn_multilabel(
        df, [f"x{j}" for j in range(d)],
        [f"y{j}" for j in range(L)],
        k=k, max_samples=n_train, seed=42,
        memory_guard=MemoryGuard(max_memory_mb=500))
    expected = (
        m.n_train_ * (d + L) * 8
        + L * (k + 1) * 8 * 2
        + L * 8
        + 256)
    assert m.state_nbytes() == expected, (
        f"n_train={n_train} d={d} L={L} k={k}: "
        f"state={m.state_nbytes()} expected={expected}")


# ---- CW-2: state independent of n ----

def test_state_independent_of_n():
    d, L, k, max_samples = 5, 2, 3, 500
    states = []
    for n in [2000, 5000, 20000]:
        df = _make_df(n, d, L, seed=0)
        m = fit_knn_multilabel(
            df, [f"x{j}" for j in range(d)],
            [f"y{j}" for j in range(L)],
            k=k, max_samples=max_samples, seed=42,
            memory_guard=MemoryGuard(max_memory_mb=500))
        states.append(m.state_nbytes())
    assert max(states) == min(states), f"states differ: {states}"


# ---- CW-5: OOC peak ----

def test_ooc_batch_fixed():
    d, L, k, max_samples = 5, 2, 3, 500
    peaks = []
    for n in [10_000, 100_000]:
        df = _make_df(n, d, L, seed=0)
        cell_peaks = []
        for _ in range(3):
            _malloc_trim(); gc.collect()
            mg = MemoryGuard(max_memory_mb=500)
            r = measure_peak_traced(
                fit_knn_multilabel, df,
                [f"x{j}" for j in range(d)],
                [f"y{j}" for j in range(L)],
                k=k, max_samples=max_samples, seed=42,
                memory_guard=mg, batch_size=1000, mg=mg)
            cell_peaks.append(int(r["peak_bytes"]))
        peaks.append(int(statistics.median(cell_peaks)))
        del df
        gc.collect()
    ratio = peaks[-1] / max(peaks[0], 1)
    assert ratio < 4.0, (
        f"peak ratio {ratio:.2f} (n=10k -> 100k); peaks={peaks}")


# ---- Reference parity: vectorized vs slow per-sample ----

def _reference_predict(train_X, train_Y, test_X, k, prior, P_pos, P_neg):
    n_test, L = test_X.shape[0], train_Y.shape[1]
    pred = np.zeros((n_test, L), dtype=np.int64)
    for i in range(n_test):
        dists = ((train_X - test_X[i]) ** 2).sum(axis=1)
        nn = np.argsort(dists)[:k]
        for j in range(L):
            C_j = int(train_Y[nn, j].sum())
            p1 = prior[j] * P_pos[j, C_j]
            p0 = (1 - prior[j]) * P_neg[j, C_j]
            pred[i, j] = 1 if p1 > p0 else 0
    return pred


def test_reference_parity():
    d, L, k = 4, 2, 3
    df_train = _make_df(500, d, L, seed=0)
    df_test = _make_df(100, d, L, seed=99)
    m = fit_knn_multilabel(
        df_train, [f"x{j}" for j in range(d)],
        [f"y{j}" for j in range(L)],
        k=k, max_samples=500, seed=42,
        memory_guard=MemoryGuard(max_memory_mb=500))
    pred_df = m.predict(df_test)
    pred_fast = _extract_pred(pred_df, [f"y{j}" for j in range(L)])
    test_X = _extract_X(df_test, [f"x{j}" for j in range(d)])
    ref = _reference_predict(
        m.train_X_, m.train_Y_, test_X, m.k_,
        m.prior_, m.P_pos_, m.P_neg_)
    np.testing.assert_array_equal(pred_fast, ref)


# ---- Hamming loss on synthetic ----

def test_hamming_loss():
    d, L, k = 5, 3, 5
    df_train = _make_df(1000, d, L, seed=0)
    df_test = _make_df(300, d, L, seed=1)
    m = fit_knn_multilabel(
        df_train, [f"x{j}" for j in range(d)],
        [f"y{j}" for j in range(L)],
        k=k, max_samples=500, seed=42,
        memory_guard=MemoryGuard(max_memory_mb=500))
    pred_df = m.predict(df_test)
    pred = _extract_pred(pred_df, [f"y{j}" for j in range(L)])
    true = _extract_pred(df_test, [f"y{j}" for j in range(L)],
                         suffix="")
    hamming = float((pred != true).mean())
    assert hamming < 0.30, f"hamming_loss={hamming:.3f} > 0.30"


# ---- Determinism ----

def test_deterministic():
    d, L, k = 3, 2, 3
    df = _make_df(500, d, L, seed=0)
    m1 = fit_knn_multilabel(
        df, [f"x{j}" for j in range(d)],
        [f"y{j}" for j in range(L)],
        k=k, max_samples=300, seed=42,
        memory_guard=MemoryGuard(max_memory_mb=500))
    m2 = fit_knn_multilabel(
        df, [f"x{j}" for j in range(d)],
        [f"y{j}" for j in range(L)],
        k=k, max_samples=300, seed=42,
        memory_guard=MemoryGuard(max_memory_mb=500))
    np.testing.assert_array_equal(m1.train_X_, m2.train_X_)
    np.testing.assert_allclose(m1.prior_, m2.prior_, atol=1e-12)


# ---- Invalid input ----

def test_invalid_k():
    df = _make_df(100, 3, 2, seed=0)
    with pytest.raises(ValueError, match="k >= 1"):
        fit_knn_multilabel(
            df, ["x0", "x1", "x2"], ["y0", "y1"], k=0,
            memory_guard=MemoryGuard(max_memory_mb=500))


def test_invalid_max_samples():
    df = _make_df(100, 3, 2, seed=0)
    with pytest.raises(ValueError, match="max_samples >= k"):
        fit_knn_multilabel(
            df, ["x0", "x1", "x2"], ["y0", "y1"], k=5, max_samples=3,
            memory_guard=MemoryGuard(max_memory_mb=500))


def test_invalid_overlap():
    df = _make_df(100, 3, 2, seed=0)
    with pytest.raises(ValueError, match="overlap"):
        fit_knn_multilabel(
            df, ["x0", "x1", "y0"], ["y0", "y1"],
            memory_guard=MemoryGuard(max_memory_mb=500))


def test_guard_refuses_at_tight():
    df = _make_df(1000, 5, 2, seed=0)
    with pytest.raises(Exception):
        fit_knn_multilabel(
            df, [f"x{j}" for j in range(5)],
            ["y0", "y1"],
            k=3, max_samples=500,
            memory_guard=MemoryGuard(max_memory_mb=0.0005))
