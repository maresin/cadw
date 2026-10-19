"""
Тесты propagate_chars("join") и plan_pipeline с join (11-B-2c-2).

Проверяем:
  - propagate_chars(join) корректно считает N', k_est', row_size';
  - k_est' = N' (upper bound, безопасно для downstream groupby);
  - требует JoinParams (не принимает None или чужой Params);
  - plan_pipeline: filter → join → groupby и join → join.
"""
import pyarrow as pa
import pytest

from cadw.core.models import DataChars, get_model
from cadw.core.models_join import JoinParams
from cadw.core.planner import OpSpec, plan_pipeline, propagate_chars


def _dc(
    n_rows: int = 10_000,
    k_est: int = 1000,
    row_size: int = 16,
) -> DataChars:
    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])
    return DataChars(
        schema=schema,
        n_rows=n_rows,
        row_size_bytes=row_size,
        k_est=k_est,
        avg_key_size_bytes=8,
        size_of_key_entry_bytes=80,
    )


def _solve_join(left, right, how="inner", available=10 * 1024 * 1024):
    model = get_model("join")
    return model.solve(
        available, left,
        op_context={"right_chars": right, "how": how},
    )


# ---------- propagate_chars(join) ----------

def test_propagate_join_basic():
    """N' = N_left × avg_matches; k_est = N' (upper bound)."""
    left = _dc(n_rows=10_000, k_est=1000, row_size=16)
    right = _dc(n_rows=10_000, k_est=1000, row_size=16)
    params = _solve_join(left, right)
    assert params is not None
    # avg_matches = 10_000 / 1000 = 10 → N_out = 10_000 × 10 = 100_000
    assert params.avg_matches == pytest.approx(10.0, abs=0.1)

    out = propagate_chars(
        OpSpec(name="join"), left, params,
    )
    assert out.n_rows == 100_000
    assert out.k_est == 100_000  # upper bound
    assert out.row_size_bytes == params.output_row_size
    assert out.order == "random"


def test_propagate_join_pk_fk():
    """PK-FK: left = FK (unique), right = PK (1000 unique) → avg_matches = 1."""
    left = _dc(n_rows=10_000, k_est=10_000, row_size=16)  # all unique
    right = _dc(n_rows=1000, k_est=1000, row_size=16)     # 1000 unique
    params = _solve_join(left, right)
    assert params is not None
    # output = 10_000 × 1000 / max(10_000, 1000) = 10_000
    # avg = 10_000 / 10_000 = 1
    assert params.avg_matches == pytest.approx(1.0, abs=0.1)

    out = propagate_chars(OpSpec(name="join"), left, params)
    assert out.n_rows == 10_000
    assert out.k_est == 10_000


def test_propagate_join_requires_join_params():
    left = _dc()
    with pytest.raises(ValueError, match="JoinParams"):
        propagate_chars(OpSpec(name="join"), left, None)


def test_propagate_join_rejects_wrong_params_type():
    """Не JoinParams (например, ReduceParams) -> ValueError."""
    left = _dc()
    right = _dc()
    # Создаём Params не того типа через фейковый класс
    class FakeParams:
        pass
    with pytest.raises(ValueError, match="JoinParams"):
        propagate_chars(OpSpec(name="join"), left, FakeParams())


def test_propagate_join_size_of_key_entry_includes_state():
    """size_of_key_entry' >= row_size' (запас на agg state)."""
    left = _dc(n_rows=10_000, k_est=1000)
    right = _dc(n_rows=10_000, k_est=1000)
    params = _solve_join(left, right)
    out = propagate_chars(OpSpec(name="join"), left, params)
    assert out.size_of_key_entry_bytes >= out.row_size_bytes
    assert out.avg_key_size_bytes == out.row_size_bytes


# ---------- plan_pipeline с join ----------

def test_pipeline_join_fits():
    """Одна join-операция вписывается в бюджет."""
    left = _dc(n_rows=10_000, k_est=10_000)  # unique FK
    right = _dc(n_rows=1000, k_est=1000)     # unique PK
    ops = [
        OpSpec(name="join", op_context={"right_chars": right, "how": "inner"}),
    ]
    plan = plan_pipeline(ops, left, 10 * 1024 * 1024, safety=0.7)
    assert plan.predicted_outcome == "fits"
    assert len(plan.steps) == 1
    assert plan.steps[0].params is not None
    assert isinstance(plan.steps[0].params, JoinParams)
    # N_out = N_left × avg_matches = 10_000 × 1 = 10_000
    assert plan.steps[0].output_chars.n_rows == 10_000


def test_pipeline_filter_join_groupby():
    """Цепочка filter → join → groupby планируется."""
    left = _dc(n_rows=10_000, k_est=10_000)   # unique FK
    right = _dc(n_rows=1000, k_est=1000)      # unique PK
    ops = [
        OpSpec(name="filter"),
        OpSpec(name="join", op_context={"right_chars": right, "how": "inner"}),
        OpSpec(name="groupby"),
    ]
    plan = plan_pipeline(ops, left, 10 * 1024 * 1024, safety=0.7)
    assert plan.predicted_outcome == "fits"
    assert len(plan.steps) == 3
    # filter: N не меняется
    assert plan.steps[0].output_chars.n_rows == 10_000
    # join: N' = 10_000 (avg_matches = 1), k_est = 10_000 (upper)
    assert plan.steps[1].output_chars.n_rows == 10_000
    assert plan.steps[1].output_chars.k_est == 10_000
    # groupby: N' = k_est = 10_000 (upper bound)
    assert plan.steps[2].output_chars.n_rows == 10_000


def test_pipeline_join_join():
    """Две последовательные join-операции планируются."""
    left = _dc(n_rows=10_000, k_est=10_000)
    right1 = _dc(n_rows=1000, k_est=1000)
    right2 = _dc(n_rows=1000, k_est=1000)
    ops = [
        OpSpec(name="join", op_context={"right_chars": right1, "how": "inner"}),
        OpSpec(name="join", op_context={"right_chars": right2, "how": "inner"}),
    ]
    plan = plan_pipeline(ops, left, 10 * 1024 * 1024, safety=0.7)
    assert plan.predicted_outcome == "fits"
    assert len(plan.steps) == 2
    # Оба join-шага имеют params
    for step in plan.steps:
        assert isinstance(step.params, JoinParams)


def test_pipeline_join_none_raises():
    """build_partition не влезает при любом P → PipelineMemoryError."""
    left = _dc(n_rows=10_000_000, k_est=10_000_000, row_size=200)
    right = _dc(n_rows=10_000_000, k_est=10_000_000, row_size=200)
    ops = [
        OpSpec(name="join", op_context={"right_chars": right, "how": "inner"}),
    ]
    from cadw.core.planner import PipelineMemoryError
    with pytest.raises(PipelineMemoryError):
        plan_pipeline(ops, left, 1024, safety=0.7)  # 1 KB — слишком мало
