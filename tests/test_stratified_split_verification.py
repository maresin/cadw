"""v0.9.143-batch5: verification for cadw.ml.stratified_split (DEC-033).

Thin wrapper over verified KFold. State = O(1) = 128 B.
CW-1: state_nbytes exact
Equivalence: model.split == KFold.split with same params
Correctness: strata preserved, union == n
"""
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame, KFold
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml.stratified_split import fit_stratified_split, StratifiedSplitModel


def _make_df(n, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    return CadwDataFrame.from_table(pa.table({
        "x": pa.array(rng.normal(size=n), type=pa.float64()),
        "y": pa.array(rng.randint(0, n_classes, size=n).astype(np.int64),
                      type=pa.int64()),
    }))


def _rows(df):
    out = []
    for b in df._iter_filtered_batches():
        out.extend(b.to_pylist())
    return out


def _warmup():
    df = _make_df(500, n_classes=3, seed=999)
    m = fit_stratified_split(df, stratify_col="y", n_splits=5, seed=42,
                              memory_guard=MemoryGuard(max_memory_mb=500))
    m.split(df, 0)


_warmup()


@pytest.mark.parametrize("n_splits", [2, 5, 10])
def test_state_nbytes_exact(n_splits):
    df = _make_df(500, n_classes=3, seed=0)
    m = fit_stratified_split(df, stratify_col="y", n_splits=n_splits,
                              seed=42,
                              memory_guard=MemoryGuard(max_memory_mb=500))
    assert m.state_nbytes() == 128


def test_state_independent_of_n():
    m1 = fit_stratified_split(_make_df(500, seed=0),
                               stratify_col="y", n_splits=5,
                               memory_guard=MemoryGuard(max_memory_mb=500))
    m2 = fit_stratified_split(_make_df(5000, seed=0),
                               stratify_col="y", n_splits=5,
                               memory_guard=MemoryGuard(max_memory_mb=500))
    assert m1.state_nbytes() == m2.state_nbytes() == 128


def test_guard_refuses_at_tight():
    """Very tight guard (0.0001 MB = ~100 B < 128 B header)."""
    df = _make_df(500, seed=0)
    with pytest.raises((MemoryLimitExceeded, Exception)):
        fit_stratified_split(
            df, stratify_col="y", n_splits=5,
            memory_guard=MemoryGuard(max_memory_mb=0.00005))


def test_split_delegates_to_kfold():
    """model.split(df, fold) == KFold(stratify_by=...).split(df)[fold]."""
    df = _make_df(500, n_classes=3, seed=1)
    model = fit_stratified_split(
        df, stratify_col="y", n_splits=5, seed=42,
        memory_guard=MemoryGuard(max_memory_mb=500))
    train_m, test_m = model.split(df, fold=2)
    # direct KFold
    cv = KFold(n_splits=5, stratify_by="y", random_state=42)
    direct_train = direct_test = None
    for fid, train, test in cv.split(df):
        if fid == 2:
            direct_train, direct_test = train, test
            break
    assert direct_train is not None and direct_test is not None
    assert _rows(train_m) == _rows(direct_train)
    assert _rows(test_m) == _rows(direct_test)


def test_strata_preserved():
    """Every class appears in each fold's test."""
    rng = np.random.RandomState(2)
    n = 1000
    y = rng.randint(0, 3, size=n).astype(np.int64)
    # ensure each class present ~equally
    y[:30] = 0; y[30:60] = 1; y[60:90] = 2
    df = CadwDataFrame.from_table(pa.table({
        "x": pa.array(rng.normal(size=n), type=pa.float64()),
        "y": pa.array(y, type=pa.int64()),
    }))
    model = fit_stratified_split(
        df, stratify_col="y", n_splits=5, seed=42,
        memory_guard=MemoryGuard(max_memory_mb=500))
    total_rows = 0
    for fid, train, test in model.all_splits(df):
        rows = _rows(test)
        total_rows += len(rows)
        classes_in_test = {r["y"] for r in rows}
        assert classes_in_test == {0, 1, 2}, (
            f"fold {fid}: missing classes {classes_in_test}")
    assert total_rows == n


def test_split_invalid_fold():
    df = _make_df(500, seed=0)
    m = fit_stratified_split(df, stratify_col="y", n_splits=5,
                              memory_guard=MemoryGuard(max_memory_mb=500))
    with pytest.raises(IndexError):
        m.split(df, fold=5)
    with pytest.raises(IndexError):
        m.split(df, fold=-1)


def test_missing_stratify_col():
    df = _make_df(500, seed=0)
    with pytest.raises(KeyError):
        fit_stratified_split(df, stratify_col="nonexistent", n_splits=5,
                              memory_guard=MemoryGuard(max_memory_mb=500))
