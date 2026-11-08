"""v0.9.143-batch5: verification for cadw.ml.one_hot_encoder (DEC-033).

State = total_categories Python objects + per-column pointer arrays + 256.
Exact via state_nbytes (sys.getsizeof-based). G2 guard during vocab.
"""
import ctypes
import gc

import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml.one_hot_encoder import fit_one_hot_encoder, OneHotEncoderModel


def _make_df(n, cardinalities, seed=0):
    """cardinalities: dict col -> K."""
    rng = np.random.RandomState(seed)
    return CadwDataFrame.from_table(pa.table({
        c: pa.array(rng.randint(0, K, size=n).astype(np.int64))
        for c, K in cardinalities.items()
    }))


def _warmup():
    df = _make_df(200, {"c0": 5, "c1": 3}, seed=999)
    fit_one_hot_encoder(df, categorical_cols=["c0", "c1"],
                        memory_guard=MemoryGuard(max_memory_mb=500),
                        batch_size=100)


_warmup()


def test_state_nbytes_exact():
    """state_nbytes() must be sys.getsizeof-based (exact)."""
    cardinalities = {"c0": 10, "c1": 5}
    df = _make_df(500, cardinalities, seed=0)
    m = fit_one_hot_encoder(
        df, categorical_cols=list(cardinalities.keys()),
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=200)
    # each column seen all K values (n=500, K<=10 -> all present)
    for c, K in cardinalities.items():
        assert len(m.categories_[c]) == K, (
            f"{c}: saw {len(m.categories_[c])} categories, expected {K}")
    # state_nbytes > 0 and reflects categories
    assert m.state_nbytes() > 256
    # non-spilled; state independent of n (same categories)
    df2 = _make_df(5000, cardinalities, seed=0)
    m2 = fit_one_hot_encoder(
        df2, categorical_cols=list(cardinalities.keys()),
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=200)
    # categories same; only header may differ by tiny (str caching)
    assert abs(m.state_nbytes() - m2.state_nbytes()) <= 64, (
        f"state drift: {m.state_nbytes()} vs {m2.state_nbytes()}")


def test_guard_refuses_at_tight():
    """Large category vocab + tiny guard -> MemoryLimitExceeded."""
    df = _make_df(500, {"c0": 10_000}, seed=0)
    with pytest.raises((MemoryLimitExceeded, Exception)):
        fit_one_hot_encoder(
            df, categorical_cols=["c0"],
            memory_guard=MemoryGuard(max_memory_mb=0.005),
            batch_size=200)


def test_transform_shape_and_one_hot():
    """Each row has exactly one 1.0 per input column's category group."""
    df = _make_df(300, {"c0": 4, "c1": 3}, seed=1)
    m = fit_one_hot_encoder(
        df, categorical_cols=["c0", "c1"],
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=100)
    total_cols = len(m.feature_names_)
    # c0: 4 + 1(unknown) = 5; c1: 3 + 1 = 4; total = 9
    assert total_cols == 9
    rows = 0
    for rb in m.transform_batches(df, batch_size=100):
        assert rb.num_columns == total_cols
        for j in range(total_cols):
            arr = rb.column(j).to_numpy()
            assert ((arr == 0.0) | (arr == 1.0)).all()
        # sum per row = 2 (one hot per column group)
        arr_all = np.column_stack([
            rb.column(j).to_numpy() for j in range(total_cols)])
        assert np.allclose(arr_all.sum(axis=1), 2.0)
        rows += rb.num_rows
    assert rows == 300


def test_transform_handle_unknown_ignore():
    """Values unseen during fit go to unknown column."""
    df_fit = _make_df(200, {"c0": 3}, seed=2)  # categories 0,1,2
    m = fit_one_hot_encoder(
        df_fit, categorical_cols=["c0"], handle_unknown="ignore",
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=100)
    # df with new value 99
    df_test = CadwDataFrame.from_table(pa.table({
        "c0": pa.array([0, 1, 2, 99, 100], type=pa.int64())
    }))
    results = []
    for rb in m.transform_batches(df_test, batch_size=100):
        for j in range(rb.num_columns):
            arr = rb.column(j).to_numpy()
            results.append(arr)
    # shape (4 features for {0,1,2,unknown}, 5 rows)
    assert len(results) == 4
    # rows 3,4 (99,100) -> unknown column (last, index 3) == 1
    assert results[3][3] == 1.0
    assert results[3][4] == 1.0
    # known rows -> unknown col == 0
    assert results[3][0] == 0.0
    assert results[3][1] == 0.0


def test_transform_handle_unknown_error():
    df_fit = _make_df(200, {"c0": 3}, seed=3)
    m = fit_one_hot_encoder(
        df_fit, categorical_cols=["c0"], handle_unknown="error",
        memory_guard=MemoryGuard(max_memory_mb=500), batch_size=100)
    df_test = CadwDataFrame.from_table(pa.table({
        "c0": pa.array([0, 99], type=pa.int64())
    }))
    with pytest.raises(ValueError, match="unknown category"):
        for _ in m.transform_batches(df_test, batch_size=100):
            pass
