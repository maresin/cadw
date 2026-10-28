"""Tests for D.4.x ROCKET (v0.9.89, SGD classifier)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import rocket as rk
from cadw.ml.rocket import fit_rocket, make_kernels, _rocket_features_series


def _binary_dataset(n_series=20, length=200, seed=0):
    rng = np.random.RandomState(seed)
    ids, vals, tgts = [], [], []
    for sid in range(n_series):
        label = sid % 2
        t = np.arange(length)
        base = np.sin(2 * np.pi * t / (10 + 5 * label))
        x = base + 0.1 * rng.randn(length)
        ids += [sid] * length
        vals += list(x)
        tgts += [label] * length
    return CadwDataFrame.from_table(pa.table({
        "id": pa.array(ids, type=pa.int64()),
        "v": pa.array(vals, type=pa.float64()),
        "y": pa.array(tgts, type=pa.int64()),
    }))


def _multi_dataset(n_series=15, length=200, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    ids, vals, tgts = [], [], []
    for sid in range(n_series):
        label = sid % n_classes
        t = np.arange(length)
        base = (np.sin(2 * np.pi * t / (8 + 6 * label))
                * (1.0 + 0.3 * label))
        x = base + 0.1 * rng.randn(length)
        ids += [sid] * length
        vals += list(x)
        tgts += [label] * length
    return CadwDataFrame.from_table(pa.table({
        "id": pa.array(ids, type=pa.int64()),
        "v": pa.array(vals, type=pa.float64()),
        "y": pa.array(tgts, type=pa.int64()),
    }))


def test_kernels_deterministic_and_valid():
    W1, b1, d1 = make_kernels(20, seed=3)
    W2, b2, d2 = make_kernels(20, seed=3)
    assert np.array_equal(W1, W2) and np.array_equal(b1, b2)
    assert np.array_equal(d1, d2)
    assert W1.shape == (20, rk.KERNEL_LEN)
    assert b1.shape == (20,)
    assert np.all(np.isin(d1, [1, 2, 4, 8, 16, 32]))
    W3, _, _ = make_kernels(20, seed=4)
    assert not np.array_equal(W1, W3)
    with pytest.raises(ValueError, match="num_kernels"):
        make_kernels(0)
    with pytest.raises(ValueError, match="max_dilation"):
        make_kernels(4, max_dilation=0)


def test_features_reference_manual_conv():
    rng = np.random.RandomState(7)
    x = rng.randn(50)
    W = rng.randn(1, rk.KERNEL_LEN)
    b = np.array([0.1])
    dils = np.array([2], dtype=np.int64)
    mg = MemoryGuard(max_memory_mb=100)
    f = _rocket_features_series(x, W, b, dils, rk.KERNEL_LEN, mg, "t")
    n_out = x.size - (rk.KERNEL_LEN - 1) * 2
    ref = np.array([
        sum(W[0, j] * x[i + j * 2] for j in range(rk.KERNEL_LEN)) + b[0]
        for i in range(n_out)
    ])
    assert abs(f[0, 0] - float(np.mean(ref > 0))) < 1e-10
    assert abs(f[0, 1] - float(ref.max())) < 1e-10


def test_features_shape_and_range():
    x = np.random.RandomState(0).randn(100)
    W, b, dils = make_kernels(16)
    mg = MemoryGuard(max_memory_mb=100)
    f = _rocket_features_series(x, W, b, dils, rk.KERNEL_LEN, mg, "t")
    assert f.shape == (16, 2)
    assert np.all((f[:, 0] >= 0.0) & (f[:, 0] <= 1.0))
    assert np.all(np.isfinite(f[:, 1]))


def test_binary_classification_accuracy():
    df = _binary_dataset(20, 200, seed=0)
    m = fit_rocket(df, "v", "y", id_col="id", num_kernels=50,
                   seed=0, sgd_epochs=60, sgd_lr=0.02)
    preds = m.predict(df)
    truth = np.array([sid % 2 for sid in range(20)])
    acc = float((preds == truth).mean())
    assert acc >= 0.90, f"binary acc={acc}"


def test_multiclass_classification_accuracy():
    df = _multi_dataset(15, 200, n_classes=3, seed=0)
    m = fit_rocket(df, "v", "y", id_col="id", num_kernels=50,
                   seed=0, sgd_epochs=100, sgd_lr=0.02)
    preds = m.predict(df)
    truth = np.array([sid % 3 for sid in range(15)])
    acc = float((preds == truth).mean())
    assert acc >= 0.80, f"multiclass acc={acc}"


def test_determinism_same_seed():
    df = _binary_dataset(10, 150, seed=1)
    m1 = fit_rocket(df, "v", "y", id_col="id", num_kernels=20,
                    seed=5, sgd_epochs=10)
    m2 = fit_rocket(df, "v", "y", id_col="id", num_kernels=20,
                    seed=5, sgd_epochs=10)
    assert np.allclose(m1.weights_, m2.weights_)
    assert np.array_equal(m1.predict(df), m2.predict(df))


def test_multi_batch_invariance():
    df = _binary_dataset(6, 120, seed=2)
    m = fit_rocket(df, "v", "y", id_col="id", num_kernels=10,
                   seed=1, sgd_epochs=10)
    tbl = pa.Table.from_batches(list(df._iter_filtered_batches()))
    df2 = CadwDataFrame.from_table(tbl)
    assert np.array_equal(m.predict(df), m.predict(df2))


def test_state_bounded_across_n():
    """state_nbytes depends on n_kernels only, not on n_series."""
    df_s = _binary_dataset(4, 100, seed=3)
    df_l = _binary_dataset(40, 100, seed=3)
    m_s = fit_rocket(df_s, "v", "y", id_col="id", num_kernels=20,
                     seed=1, sgd_epochs=5)
    m_l = fit_rocket(df_l, "v", "y", id_col="id", num_kernels=20,
                     seed=1, sgd_epochs=5)
    assert m_s.state_nbytes() == m_l.state_nbytes()


def test_state_bounded_for_large_n_kernels():
    """KEY constraint: n_kernels=10000 (paper default) must not blow up state.

    Closed-form ridge (XTX) would be 3.2 GB. SGD state is ~1 MB.
    """
    df = _binary_dataset(4, 100, seed=6)
    m = fit_rocket(df, "v", "y", id_col="id", num_kernels=10000,
                   seed=1, sgd_epochs=1)
    # Kernel bank ~880 KB, weights ~160 KB, overhead 4 KB → ~1.05 MB.
    assert m.state_nbytes() < 5 * 1024 * 1024, \
        f"state={m.state_nbytes()} should be < 5 MB at n_kernels=10000"


def test_guard_raises_on_absurdly_tight_budget():
    df = _binary_dataset(50, 300, seed=4)
    mg = MemoryGuard(max_memory_mb=1)
    with pytest.raises((MemoryLimitExceeded, MemoryError)):
        fit_rocket(df, "v", "y", id_col="id", num_kernels=1000,
                   memory_guard=mg, sgd_epochs=1)


def test_source_contract():
    src = inspect.getsource(rk)
    assert "iter_series" in src
    assert "SpillDir" in src
    assert "check_state" in src
    assert "sgd_streaming" in src
    # Sanity: no closed-form XTX in this module.
    assert "XTX" not in src
    assert "np.linalg.solve" not in src
    # SGD primitive lives in cadw.ops.optim (not cadw.ml).
    from cadw.ops import optim as optim_mod
    assert "sgd_streaming" in inspect.getsource(optim_mod)
