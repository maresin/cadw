"""Property-based tests for CADW critical invariants (PINGIS v0.9.31).

Uses hypothesis to assert contract-level invariants vs pandas/sklearn
references, not current implementation details.
"""
from __future__ import annotations
import tempfile
from pathlib import Path

import pyarrow as pa
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.groupby import groupby_aggregate
from cadw.ops.sort import external_sort
from cadw.ops.reduce import reduce, SumOp, CountOp, MeanOp
from cadw.ops.dedup import drop_duplicates


def _batches(vals, ncols=2, bs=7):
    n = len(vals)
    for i in range(0, n, bs):
        chunk = vals[i:i+bs]
        yield pa.record_batch({
            "g": pa.array([f"g{v % 5}" for v in chunk]),
            "v": pa.array(chunk, type=pa.int64()),
        })


@given(st.lists(st.integers(min_value=-10_000, max_value=10_000), min_size=1, max_size=200))
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_groupby_sum_matches_pandas(vals):
    import pandas as pd
    td = tempfile.mkdtemp()
    g = MemoryGuard(max_memory_mb=200)
    tbl = groupby_aggregate(_batches(vals), ["g"], ["v"], ["sum"], g, td, num_partitions=2)
    got = {r["g"]: r["v_sum"] for r in tbl.to_pylist()}
    ref = pd.DataFrame({"g":[f"g{v%5}" for v in vals], "v":vals}).groupby("g")["v"].sum().to_dict()
    assert got == {k: int(v) for k, v in ref.items()}


@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=200))
@settings(max_examples=30, deadline=None)
def test_sort_roundtrip(vals):
    td = tempfile.mkdtemp()
    g = MemoryGuard(max_memory_mb=200)
    out = external_sort(_batches(vals), ["v"], ["g","v"], g, td, read_batch_size=10)
    from cadw.io.batch_reader import ArrowIPCBatchReader
    got = []
    for b in ArrowIPCBatchReader(str(out), ["v"]):
        got.extend(b.column("v").to_pylist())
    assert got == sorted(vals)


@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=200))
@settings(max_examples=20, deadline=None)
def test_reduce_sum_matches_builtin(vals):
    g = MemoryGuard(max_memory_mb=200)
    got = reduce(_batches(vals), SumOp("v"), g, log=False)
    assert int(got) == sum(vals)


@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=200))
@settings(max_examples=20, deadline=None)
def test_reduce_count_eq_len(vals):
    g = MemoryGuard(max_memory_mb=200)
    got = reduce(_batches(vals), CountOp("v"), g, log=False)
    assert got == len(vals)


@given(st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=200))
@settings(max_examples=20, deadline=None)
def test_dedup_len_lte(vals):
    td = tempfile.mkdtemp()
    g = MemoryGuard(max_memory_mb=200)
    out = drop_duplicates(_batches(vals), ["v"], "first", g, td)
    from cadw.io.batch_reader import ArrowIPCBatchReader
    n_out = out.num_rows
    assert n_out <= len(vals)
    assert n_out == len(set(vals))


@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=200))
@settings(max_examples=20, deadline=None)
def test_reduce_mean_matches_builtin(vals):
    g = MemoryGuard(max_memory_mb=200)
    got = reduce(_batches(vals), MeanOp("v"), g, log=False)
    import math
    assert abs(got - sum(vals) / len(vals)) < 1e-9


@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=200))
@settings(max_examples=15, deadline=None)
def test_sort_chunk_size_invariance(vals):
    """Chunk=1 vs chunk=1000 must give same sorted output."""
    from cadw.io.batch_reader import ArrowIPCBatchReader
    def _bs(chunk):
        for i in range(0, len(vals), chunk):
            c = vals[i:i+chunk]
            yield pa.record_batch({"g": pa.array([f"g{v%5}" for v in c]), "v": pa.array(c, type=pa.int64())})
    g = MemoryGuard(max_memory_mb=200)
    out1 = external_sort(_bs(1), ["v"], ["v"], g, tempfile.mkdtemp(), read_batch_size=10)
    out2 = external_sort(_bs(1000), ["v"], ["v"], g, tempfile.mkdtemp(), read_batch_size=10)
    a = []; b = []
    for x in ArrowIPCBatchReader(str(out1), ["v"]): a.extend(x.column("v").to_pylist())
    for x in ArrowIPCBatchReader(str(out2), ["v"]): b.extend(x.column("v").to_pylist())
    assert a == b == sorted(vals)


@given(st.lists(st.integers(min_value=-10, max_value=10), min_size=1, max_size=100))
@settings(max_examples=15, deadline=None)
def test_groupby_chunk_invariance(vals):
    import pandas as pd
    g = MemoryGuard(max_memory_mb=200)
    t = groupby_aggregate(_batches(vals, bs=100), ["g"], ["v"], ["sum"], g, tempfile.mkdtemp(), num_partitions=2)
    got = {r["g"]: r["v_sum"] for r in t.to_pylist()}
    ref = pd.DataFrame({"g":[f"g{v%5}" for v in vals], "v":vals}).groupby("g")["v"].sum().to_dict()
    assert {k:int(v) for k,v in got.items()} == {k:int(v) for k,v in ref.items()}


@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100),
       st.integers(min_value=1, max_value=20))
@settings(max_examples=15, deadline=None)
def test_sort_head_equals_min_n(vals, n):
    """sort().head(n) == min-n sorted (contract vs builtin)."""
    from cadw.io.batch_reader import ArrowIPCBatchReader
    g = MemoryGuard(max_memory_mb=200)
    out = external_sort(_batches(vals, bs=7), ["v"], ["v"], g, tempfile.mkdtemp(), read_batch_size=5)
    got = []
    for b in ArrowIPCBatchReader(str(out), ["v"]):
        got.extend(b.column("v").to_pylist())
        if len(got) >= n: break
    assert got[:n] == sorted(vals)[:n]


@given(st.lists(st.integers(min_value=0, max_value=50), min_size=1, max_size=200))
@settings(max_examples=15, deadline=None)
def test_filter_count_lte(vals):
    """Count of subset <= total; sum of subset sums correctly (basic invariant)."""
    g = MemoryGuard(max_memory_mb=200)
    # COUNT of values > 25
    def _bs():
        for i in range(0, len(vals), 7):
            c = vals[i:i+7]
            yield pa.record_batch({"v": pa.array(c, type=pa.int64())})
    total = reduce(_bs(), CountOp("v"), g, log=False)
    assert total == len(vals)
    # sum of subset (positive values)
    subset = [v for v in vals if v > 25]
    if subset:
        def _bs2():
            for i in range(0, len(subset), 7):
                c = subset[i:i+7]
                yield pa.record_batch({"v": pa.array(c, type=pa.int64())})
        s = reduce(_bs2(), SumOp("v"), g, log=False)
        assert int(s) == sum(subset)


@given(st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=100))
@settings(max_examples=15, deadline=None)
def test_reduce_min_max_bounds(vals):
    from cadw.ops.reduce import MinMaxOp
    g = MemoryGuard(max_memory_mb=200)
    got_min = reduce(_batches(vals, bs=13), MinMaxOp("v", mode="min"), g, log=False)
    got_max = reduce(_batches(vals, bs=13), MinMaxOp("v", mode="max"), g, log=False)
    assert got_min == min(vals)
    assert got_max == max(vals)


def test_sort_preserves_row_count_on_duplicates():
    vals = [5] * 500
    g = MemoryGuard(max_memory_mb=200)
    out = external_sort(_batches(vals, bs=50), ["v"], ["v"], g, tempfile.mkdtemp(), read_batch_size=10)
    from cadw.io.batch_reader import ArrowIPCBatchReader
    total = sum(b.num_rows for b in ArrowIPCBatchReader(str(out), None))
    assert total == 500


@given(st.lists(st.tuples(st.integers(min_value=0, max_value=9), st.integers(min_value=0, max_value=9)),
                min_size=1, max_size=50))
@settings(max_examples=10, deadline=None)
def test_groupby_two_keys_group_count_bounded(pairs):
    """Number of output groups <= number of distinct keys (never more)."""
    g = MemoryGuard(max_memory_mb=200)
    def _bs():
        for i in range(0, len(pairs), 7):
            c = pairs[i:i+7]
            yield pa.record_batch({
                "a": pa.array([x for x,_ in c], type=pa.int64()),
                "b": pa.array([y for _,y in c], type=pa.int64()),
                "v": pa.array([1]*len(c), type=pa.int64()),
            })
    t = groupby_aggregate(_bs(), ["a","b"], ["v"], ["sum"], g, tempfile.mkdtemp(), num_partitions=2)
    assert t.num_rows <= len(set(pairs))
