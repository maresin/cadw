#!/usr/bin/env bash
# patch_11a3_planner: PipelinePlan, plan_pipeline, propagate_chars,
# MemoryGuard.plan/plan_pipeline, PipelineMemoryError.
# Spill boundaries — predict-only (фактический spill — v0.2).
# AST guard (dict check) — отложен в 11-A-4.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-11a3-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_11a3"

BR="patch/11a3-planner"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- cadw/core/planner.py ----------
cat > cadw/core/planner.py <<'PY'
"""
Pipeline planner (§9.3).

Композиция OpMemoryModel в цепочку. Не материализует данные.
predict-only: spill_boundaries планируются, но фактический spill — v0.2.

    plan_pipeline(ops, data_chars, available, safety) -> PipelinePlan

`predicted_outcome`:
- "fits"            — все steps влезают, без spill;
- "fits_with_spill" — есть spill-boundary, после которого влезает;
- "does_not_fit"    — невозможен; PipelineMemoryError с обоснованием.

План не выполняется. Runtime (v0.2) читает PipelinePlan и исполняет.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.core.models import DataChars, Params, get_model


@dataclass
class OpSpec:
    """Одна операция цепочки для planner'а.

    name — ключ в реестре моделей ("reduce", "hash_shuffle", "sort",
    "groupby", или пропагатор-only имена "filter", "with_column" для
    планирования; для них модель не нужна, только `output_chars`).
    op_context — опциональный контекст (например, {"state_bytes": N}
    для reduce).
    output_overrides — опциональные переопределения для propagate:
    например, {"new_column_row_size": 8} для with_column.
    """
    name: str
    op_context: Optional[dict] = None
    output_overrides: Optional[dict] = None


@dataclass
class StepPlan:
    op_name: str
    params: Optional[Params]
    estimate_bytes: int
    input_chars: DataChars
    output_chars: DataChars
    spill_before: bool = False   # spill-boundary перед этим step


@dataclass
class PipelinePlan:
    steps: list[StepPlan]
    peak_bytes: int
    spill_boundaries: list[int]   # индексы steps (0-based), где spill_before=True
    estimated_io: int
    predicted_outcome: str        # "fits" | "fits_with_spill" | "does_not_fit"
    reason: Optional[str] = None
    available_bytes: int = 0
    safety: float = 0.7


class PipelineMemoryError(MemoryLimitExceeded):
    """План невозможен. Наследуется от MemoryLimitExceeded (совместимость)."""

    def __init__(
        self,
        step_idx: int,
        step_name: str,
        estimate_bytes: int,
        available_bytes: int,
        describe: str,
    ) -> None:
        self.step_idx = step_idx
        self.step_name = step_name
        self.estimate_bytes = estimate_bytes
        self.available_bytes = available_bytes
        self.describe = describe
        msg = (
            f"pipeline step {step_idx} ({step_name!r}) does not fit: "
            f"estimate={estimate_bytes} B > available*safety="
            f"{int(available_bytes * 0.7)} B. Model: {describe}"
        )
        super().__init__(msg)


# ---------- propagate_chars ----------

def _copy_chars(dc: DataChars, **overrides) -> DataChars:
    """Копия DataChars с переопределением полей."""
    return DataChars(
        schema=dc.schema,
        n_rows=overrides.get("n_rows", dc.n_rows),
        row_size_bytes=overrides.get("row_size_bytes", dc.row_size_bytes),
        k_est=overrides.get("k_est", dc.k_est),
        avg_key_size_bytes=overrides.get("avg_key_size_bytes", dc.avg_key_size_bytes),
        size_of_key_entry_bytes=overrides.get(
            "size_of_key_entry_bytes", dc.size_of_key_entry_bytes
        ),
        order=overrides.get("order", dc.order),
        extras=overrides.get("extras", dc.extras),
    )


def propagate_chars(
    op: OpSpec,
    input_chars: DataChars,
    params: Optional[Params],
) -> DataChars:
    """DataChars после op.

    Консервативно:
    - filter: N' = N (верхняя оценка; selectivity неизвестна без sample).
    - with_column: row_size' = row_size + type_size(new_column).
    - groupby: N' = K, row_size' = из output_overrides или input (агрегаты
      обычно узкие, но planner не знает тип agg).
    - sort: order = "sorted".
    - reduce: N' = 1, row_size' = state_bytes.
    """
    if op.name == "filter":
        return _copy_chars(input_chars)   # N' = N (консервативно)

    if op.name == "with_column":
        extra = (op.output_overrides or {}).get("new_column_row_size", 0)
        return _copy_chars(
            input_chars,
            row_size_bytes=input_chars.row_size_bytes + int(extra),
        )

    if op.name == "groupby":
        k = input_chars.k_est or 0
        ov = op.output_overrides or {}
        out_row = int(ov.get("output_row_size", input_chars.row_size_bytes))
        return _copy_chars(
            input_chars,
            n_rows=k,
            k_est=k,
            row_size_bytes=out_row,
            order="random",
        )

    if op.name == "sort":
        return _copy_chars(input_chars, order="sorted")

    if op.name == "reduce":
        ov = op.output_overrides or {}
        state_bytes = int(ov.get("state_bytes", input_chars.row_size_bytes))
        return _copy_chars(
            input_chars,
            n_rows=1,
            k_est=1,
            row_size_bytes=state_bytes,
        )

    # Для hash_shuffle и прочих пропагатор не определён
    raise ValueError(
        f"propagate_chars: unsupported op {op.name!r}. "
        f"Supported: filter, with_column, groupby, sort, reduce."
    )


# ---------- plan_pipeline ----------

def _io_estimate(n_rows: int, row_size: int) -> int:
    """Грубая оценка IO (байт): N * row_size."""
    return max(0, n_rows) * max(1, row_size)


def plan_pipeline(
    ops: list[OpSpec],
    data_chars: DataChars,
    available_bytes: int,
    *,
    safety: float = 0.7,
    allow_spill: bool = True,
) -> PipelinePlan:
    """Планирует цепочку.

    Для каждой op:
      1. model = get_model(op.name) — если op.name есть в реестре.
         Если нет ("filter", "with_column") — только propagate.
      2. params = model.solve(...); если None — попытка spill.
      3. estimate = model.estimate(params); если > budget — попытка spill.
      4. output_chars = propagate_chars(op, input, params).
    """
    if not (0 < safety < 1):
        raise ValueError(f"safety must be in (0,1), got {safety}")

    budget = int(available_bytes * safety)
    steps: list[StepPlan] = []
    spill_boundaries: list[int] = []
    current_chars = data_chars
    estimated_io_total = 0

    for idx, op in enumerate(ops):
        # Определить, есть ли модель для op
        model = None
        params: Optional[Params] = None
        estimate = 0
        try:
            model = get_model(op.name)
        except KeyError:
            # propagator-only op (filter, with_column) — модель не нужна
            model = None

        if model is not None:
            params = model.solve(
                available_bytes,
                current_chars,
                safety=safety,
                op_context=op.op_context,
            )
            if params is None:
                # spill не поможет: params не существует при любом бюджете
                raise PipelineMemoryError(
                    step_idx=idx,
                    step_name=op.name,
                    estimate_bytes=0,
                    available_bytes=available_bytes,
                    describe=model.describe() + " (no params exist)",
                )
            estimate = model.estimate(params, current_chars)

            if estimate > budget:
                if not allow_spill:
                    raise PipelineMemoryError(
                        step_idx=idx,
                        step_name=op.name,
                        estimate_bytes=estimate,
                        available_bytes=available_bytes,
                        describe=model.describe(),
                    )
                spill_boundaries.append(idx)

        output_chars = propagate_chars(op, current_chars, params)
        estimated_io_total += _io_estimate(current_chars.n_rows, current_chars.row_size_bytes)

        steps.append(
            StepPlan(
                op_name=op.name,
                params=params,
                estimate_bytes=int(estimate),
                input_chars=current_chars,
                output_chars=output_chars,
                spill_before=(idx in spill_boundaries),
            )
        )
        current_chars = output_chars

    peak = max((s.estimate_bytes for s in steps), default=0)
    if spill_boundaries:
        outcome = "fits_with_spill"
    else:
        outcome = "fits"

    return PipelinePlan(
        steps=steps,
        peak_bytes=int(peak),
        spill_boundaries=list(spill_boundaries),
        estimated_io=int(estimated_io_total),
        predicted_outcome=outcome,
        reason=None,
        available_bytes=int(available_bytes),
        safety=float(safety),
    )
PY

# ---------- MemoryGuard.plan / plan_pipeline ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/core/memory_guard.py")
s = p.read_text()

# Добавить импорты в шапку (после существующих)
if "from cadw.core.models import DataChars, Params, get_model" not in s:
    # Найти конец импортного блока (первый пустой перевод после импортов)
    import re
    # Вставить перед "logger = get_logger"
    anchor = "logger = get_logger("
    if anchor not in s:
        raise SystemExit("memory_guard: logger anchor not found")
    additions = (
        "from cadw.core.models import DataChars, Params, get_model\n\n\n"
    )
    s = s.replace(anchor, additions + anchor, 1)

# Добавить методы в класс MemoryGuard (перед "def log_step")
anchor = "    def log_step("
if anchor not in s:
    raise SystemExit("memory_guard: log_step anchor not found")

methods = '''    def plan(
        self,
        op_name: str,
        data_chars: DataChars,
        *,
        op_context: Optional[dict] = None,
    ) -> Params:
        """Запланировать одну операцию (§9.2). Raises MemoryLimitExceeded при None."""
        model = get_model(op_name)
        params = model.solve(
            self.available_bytes(),
            data_chars,
            safety=self._safety,
            op_context=op_context,
        )
        if params is None:
            raise MemoryLimitExceeded(
                f"plan({op_name!r}): no params exist. {model.describe()}"
            )
        return params

    def plan_pipeline(self, ops, data_chars: DataChars, *, allow_spill: bool = True):
        """Запланировать цепочку (§9.3). Raises PipelineMemoryError."""
        from cadw.core.planner import plan_pipeline
        return plan_pipeline(
            ops,
            data_chars,
            self.available_bytes(),
            safety=self._safety,
            allow_spill=allow_spill,
        )

'''
s = s.replace(anchor, methods + anchor, 1)
p.write_text(s)
print("memory_guard.py: plan() + plan_pipeline() добавлены")
PY

# ---------- cadw/core/__init__.py: экспорты ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/core/__init__.py")
s = p.read_text()
if "from .planner import" not in s:
    s = s.replace(
        "from .models_groupby import GroupByMemoryModel, GroupByParams",
        "from .models_groupby import GroupByMemoryModel, GroupByParams\n"
        "from .planner import (\n"
        "    OpSpec,\n"
        "    PipelineMemoryError,\n"
        "    PipelinePlan,\n"
        "    StepPlan,\n"
        "    plan_pipeline,\n"
        "    propagate_chars,\n"
        ")",
        1,
    )
    s = s.replace(
        '    "GroupByMemoryModel",\n    "GroupByParams",\n',
        '    "GroupByMemoryModel",\n    "GroupByParams",\n'
        '    "OpSpec",\n    "PipelineMemoryError",\n    "PipelinePlan",\n'
        '    "StepPlan",\n    "plan_pipeline",\n    "propagate_chars",\n',
        1,
    )
    p.write_text(s)
    print("cadw/core/__init__.py: planner exported")
PY

# ---------- tests/test_planner.py ----------
cat > tests/test_planner.py <<'EOF'
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

def test_pipeline_spill_boundary_when_estimate_exceeds_budget():
    """
    filter пропускает (estimate=0), groupby возвращает estimate > budget.
    Ожидается: fits_with_spill, spill_boundaries=[1] (перед groupby).
    """
    dc = _dc(n_rows=1_000_000, k_est=1_000_000, size_of_key_entry=80)
    ops = [OpSpec(name="filter"), OpSpec(name="groupby")]
    # available = 10 MB; groupby с K=1M и size_of_key=80 -> estimate ~25 MB
    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7, allow_spill=True)
    assert plan.predicted_outcome == "fits_with_spill"
    assert plan.spill_boundaries == [1]
    assert plan.steps[1].spill_before is True


def test_pipeline_no_spill_when_not_allowed():
    dc = _dc(n_rows=1_000_000, k_est=1_000_000, size_of_key_entry=80)
    ops = [OpSpec(name="filter"), OpSpec(name="groupby")]
    with pytest.raises(PipelineMemoryError):
        plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7, allow_spill=False)


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
EOF

# ---------- run ----------
echo "=== pytest tests/test_planner.py ==="
set +e
uv run pytest tests/test_planner.py -v
RC_MODELS=$?
set -e
if [[ $RC_MODELS -ne 0 ]]; then
  echo "test_planner.py failed (rc=$RC_MODELS)"
  git add -A
  git commit -m "WIP: 11-A-3 (test_planner red; fix follows)" || true
  exit 1
fi

echo "=== pytest (fast) ==="
set +e
uv run pytest -q -m "not slow"
RC_FAST=$?
set -e
if [[ $RC_FAST -ne 0 ]]; then
  echo "fast: rc=$RC_FAST"
  git add -A
  git commit -m "WIP: 11-A-3 (fast red; fix follows)" || true
  exit 1
fi

git add -A
git commit -m "11-A-3: PipelinePlan, plan_pipeline, propagate_chars, MemoryGuard.plan (§9.3); 20 tests"

echo "==> sha256"
sha256sum cadw/core/planner.py cadw/core/memory_guard.py tests/test_planner.py
echo "==> 11-A-3 applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"