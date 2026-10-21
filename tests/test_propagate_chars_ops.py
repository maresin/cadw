"""v0.9.8c: B1 — propagate_chars supports hash_shuffle (+4 ops)."""
from __future__ import annotations

import inspect

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.core.models import DataChars
from cadw.core.models_join import JoinParams
from cadw.core.models_shuffle import HashShuffleParams
from cadw.core.planner import (
    OpSpec,
    plan_pipeline,
    propagate_chars,
)


def _dc(n_rows=10_000, k_est=1000, row_size=16, size_of_key_entry=80):
    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])
    return DataChars(
        schema=schema,
        n_rows=n_rows,
        row_size_bytes=row_size,
        k_est=k_est,
        avg_key_size_bytes=8,
        size_of_key_entry_bytes=size_of_key_entry,
    )


# ============ propagate_chars per op ============


def test_propagate_chars_hash_shuffle():
    dc = _dc(n_rows=100_000)
    out = propagate_chars(OpSpec(name="hash_shuffle"), dc, None)
    assert out.n_rows == 100_000
    assert out.row_size_bytes == dc.row_size_bytes
    assert out.order == "random"


def test_propagate_chars_dedup():
    dc = _dc(n_rows=100_000, k_est=1000)
    out = propagate_chars(OpSpec(name="dedup"), dc, None)
    assert out.n_rows == 1000
    assert out.k_est == 1000
    assert out.order == "random"


def test_propagate_chars_dedup_no_k_est():
    dc = _dc(n_rows=100_000, k_est=None)
    out = propagate_chars(OpSpec(name="dedup"), dc, None)
    # k_est=None → conservative n_rows
    assert out.n_rows == 100_000


def test_propagate_chars_rank():
    dc = _dc(n_rows=10_000, row_size=16)
    out = propagate_chars(OpSpec(name="rank"), dc, None)
    assert out.n_rows == 10_000
    assert out.row_size_bytes == 16 + 8  # +rank int64
    assert out.order == "random"


def test_propagate_chars_quantile():
    dc = _dc(n_rows=10_000, row_size=16)
    out = propagate_chars(OpSpec(name="quantile"), dc, None)
    assert out.n_rows == 1
    assert out.k_est == 1
    assert out.row_size_bytes == 8


def test_propagate_chars_resample():
    dc = _dc(n_rows=100_000, k_est=1000)
    out = propagate_chars(OpSpec(name="resample"), dc, None)
    assert out.n_rows == 1000
    assert out.order == "sorted"


# ============ audit case: chain no crash ============


def test_plan_pipeline_filter_hash_shuffle_groupby():
    """Audit B1: [filter, hash_shuffle, groupby] must plan, not crash."""
    dc = _dc(n_rows=100_000, k_est=1000)
    ops = [
        OpSpec(name="filter"),
        OpSpec(name="hash_shuffle"),
        OpSpec(name="groupby"),
    ]
    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)
    assert plan.predicted_outcome == "fits"
    assert len(plan.steps) == 3
    # Step 1 (filter): propagator-only, params None.
    assert plan.steps[0].params is None
    # Step 2 (hash_shuffle): model exists, params set.
    assert isinstance(plan.steps[1].params, HashShuffleParams)
    # Step 3 (groupby): model exists, params set.
    assert plan.steps[2].params is not None


def test_plan_pipeline_hash_shuffle_alone():
    """Audit B1: hash_shuffle alone — must not crash."""
    dc = _dc(n_rows=100_000, k_est=1000)
    plan = plan_pipeline(
        [OpSpec(name="hash_shuffle")], dc, 10 * 1024 * 1024,
    )
    assert plan.predicted_outcome == "fits"
    assert isinstance(plan.steps[0].params, HashShuffleParams)


def test_plan_pipeline_filter_hash_shuffle_sort():
    """Variation: [filter, hash_shuffle, sort] no crash."""
    dc = _dc(n_rows=100_000, k_est=1000)
    ops = [
        OpSpec(name="filter"),
        OpSpec(name="hash_shuffle"),
        OpSpec(name="sort"),
    ]
    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)
    assert plan.predicted_outcome == "fits"
    assert plan.steps[2].output_chars.order == "sorted"


def test_plan_pipeline_dedup_no_crash():
    """dedup in chain: no model → propagator-only, no crash."""
    dc = _dc(n_rows=100_000, k_est=1000)
    ops = [OpSpec(name="dedup"), OpSpec(name="groupby")]
    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)
    assert plan.predicted_outcome == "fits"
    # After dedup, N = k_est = 1000; groupby sees 1000.
    assert plan.steps[1].input_chars.n_rows == 1000


def test_plan_pipeline_rank_no_crash():
    dc = _dc(n_rows=10_000)
    ops = [OpSpec(name="rank"), OpSpec(name="sort")]
    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)
    assert plan.predicted_outcome == "fits"


def test_plan_pipeline_quantile_no_crash():
    dc = _dc(n_rows=10_000)
    plan = plan_pipeline(
        [OpSpec(name="quantile")], dc, 1 * 1024 * 1024,
    )
    assert plan.predicted_outcome == "fits"
    assert plan.steps[0].output_chars.n_rows == 1


def test_plan_pipeline_resample_no_crash():
    dc = _dc(n_rows=100_000, k_est=1000)
    plan = plan_pipeline(
        [OpSpec(name="resample"), OpSpec(name="sort")],
        dc, 10 * 1024 * 1024,
    )
    assert plan.predicted_outcome == "fits"
    assert plan.steps[0].output_chars.order == "sorted"


# ============ error message includes new ops ============


def test_propagate_chars_supported_list_mentions_new_ops():
    """Error message for unsupported op lists hash_shuffle + others."""
    dc = _dc()
    with pytest.raises(ValueError, match="Supported:"):
        propagate_chars(OpSpec(name="nonexistent_op"), dc, None)
    with pytest.raises(ValueError) as excinfo:
        propagate_chars(OpSpec(name="nonexistent_op"), dc, None)
    msg = str(excinfo.value)
    for name in (
        "hash_shuffle", "dedup", "rank", "quantile", "resample",
        "filter", "with_column", "groupby", "sort", "reduce",
        "join",
    ):
        assert name in msg, f"expected {name!r} in Supported list"


# ============ AST guard: branches present (§13.4/190) ============


def test_propagate_chars_has_all_new_branches():
    from cadw.core.planner import propagate_chars
    src = inspect.getsource(propagate_chars)
    for op_name in (
        "hash_shuffle", "dedup", "rank", "quantile", "resample",
    ):
        assert f'op.name == "{op_name}"' in src, (
            f"propagate_chars missing branch for {op_name!r}"
        )
