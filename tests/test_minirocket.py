"""Tests for D.4.1 MiniRocket streaming features (v0.9.37)."""
from __future__ import annotations
import inspect
import numpy as np
import pyarrow as pa
import pytest
from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml import minirocket as mr
from cadw.ml.minirocket import fit_minirocket, make_kernels, _ppv_series, KERNEL_LEN


def _single(n=300):
    x = np.sin(np.arange(n) * 0.1)
    return CadwDataFrame.from_table(pa.table({"v": pa.array(x)}))


def _multi(n_series=5, length=200):
    ids, vals = [], []
    for sid in range(n_series):
        ids += [sid] * length
        vals += list(np.sin(np.arange(length) * 0.05 + 0.1 * sid))
    return CadwDataFrame.from_table(pa.table({
        "id": pa.array(ids, type=pa.int64()), "v": pa.array(vals)}))


def test_ppv_reference_direct_convolution():
    w = np.array([-1, -1, 1, -1, 1, -1, 1, -1, -1], dtype=np.float64)
    x = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, 0.2, 0.4])
    d = 1
    n_out = x.size - (KERNEL_LEN - 1) * d
    ref = np.array([sum(w[j] * x[i + j * d] for j in range(KERNEL_LEN))
                    for i in range(n_out)])
    assert abs(_ppv_series(x, w, d) - float(np.mean(ref > 0))) < 1e-12


def test_kernels_deterministic_and_valid():
    k1 = make_kernels(20, seed=3)
    k2 = make_kernels(20, seed=3)
    assert len(k1) == 20
    for (w1, d1), (w2, d2) in zip(k1, k2):
        assert np.array_equal(w1, w2) and d1 == d2
        assert set(np.unique(w1)).issubset({-1.0, 1.0})
        assert (w1 == 1.0).sum() == 3
    k3 = make_kernels(20, seed=4)
    assert any(not np.array_equal(a[0], b[0]) for a, b in zip(k1, k3))


def test_single_series_feature_matrix_shape():
    df = _single()
    m = fit_minirocket(df, "v", num_kernels=16)
    F = m.transform(df)
    assert F.shape == (1, 16)
    assert np.all((F >= 0.0) & (F <= 1.0))


def test_multi_series_one_row_per_series():
    df = _multi(5, 200)
    m = fit_minirocket(df, "v", id_col="id", num_kernels=12)
    F = m.transform(df)
    assert F.shape == (5, 12)


def test_determinism_same_seed_same_features():
    df = _multi(4, 150)
    m1 = fit_minirocket(df, "v", id_col="id", num_kernels=10, seed=5)
    m2 = fit_minirocket(df, "v", id_col="id", num_kernels=10, seed=5)
    assert np.allclose(m1.transform(df), m2.transform(df))


def test_edge_short_series_ppv_zero():
    w = np.ones(KERNEL_LEN)
    assert _ppv_series(np.array([1.0, 2.0]), w, 1) == 0.0


def test_guards_bad_kernel_count_and_missing_col():
    with pytest.raises(ValueError, match="num_kernels"):
        make_kernels(0)
    with pytest.raises(ValueError, match="max_dilation"):
        make_kernels(4, max_dilation=0)
    df = _single()
    with pytest.raises(KeyError):
        fit_minirocket(df, "nope")
    with pytest.raises(KeyError):
        fit_minirocket(df, "v", id_col="nope")


def test_guard_unsorted_id_raises():
    ids = [0, 0, 1, 1, 0, 0]
    vals = list(np.sin(np.arange(6)))
    df = CadwDataFrame.from_table(pa.table({
        "id": pa.array(ids, type=pa.int64()), "v": pa.array(vals)}))
    m = fit_minirocket(df, "v", id_col="id", num_kernels=4)
    with pytest.raises(ValueError, match="grouped/sorted"):
        m.transform(df)


def test_ram_guard_raises_when_tight():
    df = _multi(3, 300)
    mg = MemoryGuard(max_memory_bytes=512)
    m = fit_minirocket(df, "v", id_col="id", num_kernels=8)
    with pytest.raises(MemoryLimitExceeded):
        m.transform(df, memory_guard=mg)


def test_multi_batch_consistent(tmp_path):
    import pyarrow.parquet as pq
    df = _multi(4, 200)
    tbl = pa.Table.from_batches(list(df._iter_filtered_batches()))
    path = str(tmp_path / "mr.parquet")
    pq.write_table(tbl, path, row_group_size=37)
    dmb = CadwDataFrame.from_parquet(path, base_dir="/tmp", chunk_size=50)
    m1 = fit_minirocket(df, "v", id_col="id", num_kernels=8)
    m2 = fit_minirocket(dmb, "v", id_col="id", num_kernels=8)
    assert np.allclose(m1.transform(df), m2.transform(dmb))


def test_source_contract_guard_and_log_fit():
    src = inspect.getsource(mr)
    assert "check_batch" in src and "extra_factor=3.0" in src
    assert "log_fit" in src
    assert "import pandas" not in src
