"""F9 fix (AUDIT_v0931): GBDT inner ReduceOps declare
batch_memory_factor > 1 because their per-batch peak exceeds the
yielded batch.nbytes."""
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pytest
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.ml.gbdt import _AllNodeHistOp, _AssignLeavesOp, _LeafValuesOp
from cadw.ops.reduce import reduce


def test_all_node_hist_batch_memory_factor():
    op = _AllNodeHistOp(feature_cols=["a", "b"], node_ids=[0],
                        mn=np.zeros(2), mx=np.ones(2), bins=8)
    assert op.batch_memory_factor() == 7.0


def test_assign_leaves_batch_memory_factor():
    op = _AssignLeavesOp(feature_cols=["a", "b"], mn=np.zeros(2),
                         mx=np.ones(2), bins=8, tree={0: None})
    assert op.batch_memory_factor() == 4.0


def test_leaf_values_batch_memory_factor():
    assert _LeafValuesOp(reg_lambda=1.0).batch_memory_factor() == 3.5


def _make_batch(n, f, seed):
    rng = np.random.default_rng(seed)
    cols = [pa.array(rng.random(n)) for _ in range(f)]
    cols.append(pa.array(rng.random(n)))
    cols.append(pa.array(rng.random(n)))
    cols.append(pa.array(np.zeros(n, dtype=np.int64)))
    return pa.RecordBatch.from_arrays(
        cols, names=["a", "b", "__g", "__h", "__leaf"])


def test_all_node_hist_rejects_at_peak_not_at_batch_size():
    batch = _make_batch(n=4000, f=2, seed=0)
    budget = batch.nbytes * 3
    op = _AllNodeHistOp(feature_cols=["a", "b"], node_ids=[0],
                        mn=np.zeros(2), mx=np.ones(2), bins=8)
    mg = MemoryGuard(budget)
    with pytest.raises(MemoryLimitExceeded):
        reduce(iter([batch]), op, mg, log=False)


def test_all_node_hist_accepts_when_budget_covers_peak():
    batch = _make_batch(n=4000, f=2, seed=1)
    op = _AllNodeHistOp(feature_cols=["a", "b"], node_ids=[0],
                        mn=np.zeros(2), mx=np.ones(2), bins=8)
    mg = MemoryGuard(batch.nbytes * 10)
    out = reduce(iter([batch]), op, mg, log=False)
    assert "g_hist" in out
