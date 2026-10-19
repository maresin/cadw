"""
Тесты PipelinePlanner (11-A-3, §9.3).

plan_pipeline — predict-only: проверяем outcomes, spill_boundaries,
propagate_chars. Фактическое выполнение (spill + runtime) — v0.2.
"""
import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.core.models import DataChars
from cadw.core.models_reduce import ReduceParams
from cadw.core.planner import (
    OpSpec,
    PipelineMemoryError,
    PipelinePlan,
    plan_pipeline,
    propagate_chars,
)


def _dc(
    row_size: int = 16,
    n_rows: int = 10_000,
    k_est: int = 100,
    size_of_key_entry: int = 80,
) -> DataChars:
    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])
    return DataChars(
        schema=schema,
        n_rows=n_rows,
        row_size_bytes=row_size,
        k_est=k_est,
        avg_key_size_bytes=8,
        size_of_key_entry_bytes=size_of_key_entry,
    )


# ---------- propagate_chars ----------

def test_propagate_filter_keeps_n():
    dc = _dc(n_rows=10_000)
    out = propagate_chars(OpSpec(name="filter"), dc, None)
    assert out.n_rows == 10_000
    assert out.row_size_bytes == dc.row_size_bytes


def test_propagate_with_column_adds_row_size():
    dc = _dc(row_size=16)
    op = OpSpec(name="with_column", output_overrides={"new_column_row_size": 8})
    out = propagate_chars(op, dc, None)
    assert out.row_size_bytes == 24


def test_propagate_groupby_sets_n_eq_k():
    dc = _dc(n_rows=10_000, k_est=100)
    op = OpSpec(name="groupby")
    out = propagate_chars(op, dc, None)
    assert out.n_rows == 100
    assert out.k_est == 100


def test_propagate_sort_sets_order():
    dc = _dc()
    op = OpSpec(name="sort")
    out = propagate_chars(op, dc, None)
    assert out.order == "sorted"


def test_propagate_reduce_sets_n_eq_1():
    dc = _dc()
    op = OpSpec(name="reduce", output_overrides={"state_bytes": 16})
    out = propagate_chars(op, dc, None)
    assert out.n_rows == 1
    assert out.row_size_bytes == 16


def test_propagate_unknown_op_raises():
    dc = _dc()
    with pytest.raises(ValueError, match="unsupported op"):
        propagate_chars(OpSpec(name="nonexistent"), dc, None)


# ---------- plan_pipeline: single op ----------

def test_pipeline_single_reduce_fits():
    dc = _dc()
    ops = [OpSpec(name="reduce", op_context={"state_bytes": 16})]
    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)
    assert isinstance(plan, PipelinePlan)
    assert plan.predicted_outcome == "fits"
    assert len(plan.steps) == 1
    assert isinstance(plan.steps[0].params, ReduceParams)
    assert plan.peak_bytes <= int(10 * 1024 * 1024 * 0.7)


def test_pipeline_single_groupby_fits():
    dc = _dc(n_rows=1_000_000, k_est=1000)
    ops = [OpSpec(name="groupby")]
    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)
    assert plan.predicted_outcome == "fits"
    assert plan.steps[0].output_chars.n_rows == 1000


def test_pipeline_single_op_none_raises():
    """reduce(state=200MB) при available=10MB -> PipelineMemoryError."""
    dc = _dc()
    ops = [OpSpec(name="reduce", op_context={"state_bytes": 200 * 1024 * 1024})]
    with pytest.raises(PipelineMemoryError) as excinfo:
        plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)
    err = excinfo.value
    assert err.step_idx == 0
    assert err.step_name == "reduce"
    assert "no params exist" in err.describe


# ---------- plan_pipeline: chains ----------

def test_pipeline_filter_groupby_fits():
    dc = _dc(n_rows=100_000, k_est=1000)
    ops = [OpSpec(name="filter"), OpSpec(name="groupby")]
    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)
    assert plan.predicted_outcome == "fits"
    assert len(plan.steps) == 2
    # filter: params=None (propagator-only), estimate=0
    assert plan.steps[0].params is None
    # groupby: params set
    assert plan.steps[1].params is not None
    # output: N=K после groupby
    assert plan.steps[1].output_chars.n_rows == 1000


def test_pipeline_filter_groupby_sort_fits():
    dc = _dc(n_rows=100_000, k_est=1000)
    ops = [OpSpec(name="filter"), OpSpec(name="groupby"), OpSpec(name="sort")]
    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)
    assert plan.predicted_outcome == "fits"
    assert len(plan.steps) == 3
    assert plan.steps[2].output_chars.order == "sorted"


def test_pipeline_with_column_groupby():
    dc = _dc(row_size=16, n_rows=10_000, k_est=100)
    ops = [
        OpSpec(name="with_column", output_overrides={"new_column_row_size": 8}),
        OpSpec(name="groupby"),
    ]
    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)
    assert plan.predicted_outcome == "fits"
    # row_size на входе groupby = 24 (после with_column)
    assert plan.steps[1].input_chars.row_size_bytes == 24


# ---------- spill boundaries ----------

def test_pipeline_spill_boundaries_empty_in_v0_1():
    """spill_boundaries reserved for v0.2: в v0.1.x всегда пуст."""
    dc = _dc(n_rows=1_000_000, k_est=1_000_000, size_of_key_entry=80)
    ops = [OpSpec(name="filter"), OpSpec(name="groupby")]
    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)
    assert plan.spill_boundaries == []
    assert plan.predicted_outcome == "fits"
    assert all(not s.spill_before for s in plan.steps)


def test_pipeline_solve_none_raises_regardless_of_allow_spill():
    """solve()=None -> PipelineMemoryError, allow_spill не помогает (v0.1.x)."""
    dc = _dc()
    ops = [OpSpec(name="reduce", op_context={"state_bytes": 200 * 1024 * 1024})]
    for allow in (True, False):
        with pytest.raises(PipelineMemoryError):
            plan_pipeline(
                ops, dc, 10 * 1024 * 1024, safety=0.7, allow_spill=allow
            )


def test_pipeline_invariant_violation_raises_runtime_error(monkeypatch):
    """Если solve вернул params с estimate > budget -> RuntimeError (баг модели)."""
    from cadw.core import planner as planner_mod
    from cadw.core.models_reduce import ReduceMemoryModel, ReduceParams

    class FakeModel(ReduceMemoryModel):
        op_name = "fake_reduce"

        def solve(self, available_bytes, data_chars, *, safety=0.7, op_context=None):
            # Возвращаем params, у которых estimate заведомо > budget
            return ReduceParams(batch_size=10**9, state_bytes=0)

        def estimate(self, params, data_chars):
            return 10**12  # 1 TB

        def describe(self):
            return "FakeModel (always huge)"

    monkeypatch.setattr(
        planner_mod, "get_model", lambda name: FakeModel()
    )
    dc = _dc()
    with pytest.raises(RuntimeError, match="invariant violated"):
        planner_mod.plan_pipeline(
            [OpSpec(name="fake_reduce")], dc, 10 * 1024 * 1024, safety=0.7
        )


# ---------- invariant ----------

@pytest.mark.parametrize(
    "available_mb, k_est, n_rows",
    [
        (10, 100, 10_000),
        (10, 1000, 100_000),
        (50, 10_000, 1_000_000),
        (100, 100_000, 1_000_000),
    ],
)
def test_pipeline_step_estimates_within_budget(available_mb, k_est, n_rows):
    """Каждый step: estimate(params) <= available * safety (или spill)."""
    dc = _dc(n_rows=n_rows, k_est=k_est)
    available = available_mb * 1024 * 1024
    ops = [OpSpec(name="filter"), OpSpec(name="groupby"), OpSpec(name="sort")]
    plan = plan_pipeline(ops, dc, available, safety=0.7, allow_spill=True)
    budget = int(available * 0.7)
    for step in plan.steps:
        if step.params is not None and not step.spill_before:
            assert step.estimate_bytes <= budget, (
                f"step {step.op_name}: estimate={step.estimate_bytes} > "
                f"budget={budget}, spill_before={step.spill_before}"
            )


def test_invalid_safety_raises():
    dc = _dc()
    with pytest.raises(ValueError, match="safety"):
        plan_pipeline([], dc, 10 * 1024 * 1024, safety=1.5)


def test_empty_pipeline_ok():
    dc = _dc()
    plan = plan_pipeline([], dc, 10 * 1024 * 1024, safety=0.7)
    assert plan.predicted_outcome == "fits"
    assert plan.peak_bytes == 0
    assert plan.steps == []


# ---------- MemoryGuard.plan / plan_pipeline ----------

def test_memory_guard_plan_reduce():
    mg = MemoryGuard(max_memory_mb=10)
    dc = _dc()
    params = mg.plan("reduce", dc, op_context={"state_bytes": 16})
    assert isinstance(params, ReduceParams)


def test_memory_guard_plan_none_raises():
    mg = MemoryGuard(max_memory_mb=10)
    dc = _dc()
    with pytest.raises(MemoryLimitExceeded):
        mg.plan("reduce", dc, op_context={"state_bytes": 200 * 1024 * 1024})


def test_memory_guard_plan_pipeline_wraps():
    mg = MemoryGuard(max_memory_mb=10)
    dc = _dc(n_rows=10_000, k_est=100)
    plan = mg.plan_pipeline([OpSpec(name="filter"), OpSpec(name="groupby")], dc)
    assert isinstance(plan, PipelinePlan)
    assert plan.predicted_outcome in ("fits", "fits_with_spill")
