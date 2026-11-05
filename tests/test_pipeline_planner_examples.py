"""v0.9.119: P1-P8 pipeline examples for segmented planner.

P1-P5: primitives only (filter/with_column/groupby/sort/hash_shuffle).
P6-P8: terminal model steps (model_spec, v0.9.119).

Each test answers: does the plan segment correctly, keep retained state
across spill boundaries, and estimate time via throughput rates?
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.core.models import DataChars
from cadw.core.planner import OpSpec, plan_pipeline


def _dc(row_size: int = 16, n_rows: int = 10_000, k_est: int = 100) -> DataChars:
    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])
    return DataChars(
        schema=schema, n_rows=n_rows, row_size_bytes=row_size,
        k_est=k_est, avg_key_size_bytes=8, size_of_key_entry_bytes=80,
    )


# ---- P1: stateless chain, one segment ----

def test_p1_filter_two_with_column_one_segment():
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc()
    ops = [
        OpSpec(name="filter"),
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
    ]
    plan = mg.plan_pipeline(ops, dc)
    assert plan.segments == [(0, 3)]
    assert plan.spill_boundaries == []
    assert plan.predicted_outcome == "fits"
    assert plan.est_time_seconds > 0.0
    assert plan.time_class in ("fast", "medium", "slow", "very_slow")


# ---- P2: with_column spill_after -> groupby ----

def test_p2_spill_before_groupby_two_segments():
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc()
    ops = [
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8},
               spill_after=True),
        OpSpec(name="groupby"),
    ]
    plan = mg.plan_pipeline(ops, dc)
    assert plan.segments == [(0, 1), (1, 2)]
    assert plan.spill_boundaries == [0]
    assert plan.predicted_outcome in ("fits_with_spill", "does_not_fit")
    assert plan.peak_bytes > 0


# ---- P3: sort is a breaker ----

def test_p3_sort_spill_after():
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc()
    ops = [
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
        OpSpec(name="sort", spill_after=True),
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
    ]
    plan = mg.plan_pipeline(ops, dc)
    assert plan.segments == [(0, 2), (2, 3)]
    assert plan.spill_boundaries == [1]
    assert plan.peak_bytes > 0


# ---- P4: two breakers in a row ----

def test_p4_hash_shuffle_and_sort():
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc(k_est=50)
    ops = [
        OpSpec(name="hash_shuffle", spill_after=True),
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
        OpSpec(name="sort", spill_after=True),
    ]
    plan = mg.plan_pipeline(ops, dc)
    assert plan.segments == [(0, 1), (1, 3)]
    assert plan.spill_boundaries == [0, 2]
    assert plan.predicted_outcome in ("fits_with_spill", "does_not_fit")


# ---- P5: explicit spill boundary ----

def test_p5_explicit_spill_boundary():
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc()
    ops = [
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8},
               spill_after=True),
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
    ]
    plan = mg.plan_pipeline(ops, dc)
    assert plan.segments == [(0, 1), (1, 2)]
    assert plan.spill_boundaries == [0]


# ---- P6: classification (filter -> 2 features -> fit_logistic) ----

def test_p6_classification_with_model_spec():
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc()
    ops = [
        OpSpec(name="filter"),
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
        OpSpec(name="fit_logistic", model_spec={"state_bytes": 800}),
    ]
    plan = mg.plan_pipeline(ops, dc)
    assert plan.segments == [(0, 4)]
    assert plan.spill_boundaries == []
    # Model step output is a state, not rows
    assert plan.steps[-1].output_chars.n_rows == 1
    assert plan.steps[-1].output_chars.row_size_bytes == 800


# ---- P7: regression with sort + lag + fit_linear ----

def test_p7_regression_sort_then_model():
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc()
    ops = [
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
        OpSpec(name="sort", spill_after=True),
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
        OpSpec(name="fit_linear", model_spec={"state_bytes": 6400}),  # d=20: 2*d^2*8
    ]
    plan = mg.plan_pipeline(ops, dc)
    assert plan.segments == [(0, 2), (2, 4)]
    assert plan.spill_boundaries == [1]
    assert plan.steps[-1].output_chars.n_rows == 1


# ---- P8: groupby + nonlinear model ----

def test_p8_groupby_then_gbdt_model():
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc(k_est=20)
    ops = [
        OpSpec(name="groupby", spill_after=True),
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
        OpSpec(name="fit_gbdt_classifier", model_spec={"state_bytes": 50000}),
    ]
    plan = mg.plan_pipeline(ops, dc)
    assert plan.segments == [(0, 1), (1, 3)]
    assert plan.spill_boundaries == [0]


# ---- back-compat: direct plan_pipeline (no guard) uses v1 ----

def test_backcompat_direct_call_no_guard():
    """plan_pipeline without memory_guard -> v1, spill_boundaries=[]."""
    dc = _dc()
    plan = plan_pipeline(
        [OpSpec(name="filter"), OpSpec(name="with_column",
                output_overrides={"new_column_row_size": 8},
                spill_after=True)],  # spill_after ignored in v1
        dc, 10 * 1024 * 1024, safety=0.7,
    )
    assert plan.spill_boundaries == []
    assert plan.segments == []   # v1 default
    assert plan.predicted_outcome == "fits"


# ---- back-compat: guard path without spill_after -> v2 but identical ----

def test_backcompat_guard_no_spill_after():
    """Guard path with no spill_after -> one segment, outcome fits."""
    mg = MemoryGuard(max_memory_mb=100)
    dc = _dc()
    plan = mg.plan_pipeline([OpSpec(name="filter")], dc)
    assert plan.segments == [(0, 1)]
    assert plan.spill_boundaries == []
    assert plan.predicted_outcome == "fits"
