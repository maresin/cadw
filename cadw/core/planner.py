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
    - join: N' = N_left × avg_matches, row_size' = output_row_size,
      k_est' = N' (upper bound); требует JoinParams в params.
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

    if op.name == "join":
        # params — JoinParams из get_model("join").solve(). Содержит
        # right_chars (right-side DataChars) и avg_matches (upper bound).
        from cadw.core.models_join import JoinParams
        if not isinstance(params, JoinParams):
            raise ValueError(
                f"propagate_chars(join): params must be JoinParams, "
                f"got {type(params).__name__ if params is not None else None}"
            )
        n_out = max(1, int(input_chars.n_rows * params.avg_matches))
        row_out = int(params.output_row_size)
        # k_est — upper bound: все выходные строки уникальны. Безопасно
        # для downstream groupby/shuffle (over-estimate → over-recursion,
        # не OOM). Точный k_est для конкретной колонки недоступен из
        # модели — groupby_aggregate сам измерит sample.
        return _copy_chars(
            input_chars,
            n_rows=n_out,
            row_size_bytes=row_out,
            k_est=n_out,
            avg_key_size_bytes=row_out,
            size_of_key_entry_bytes=row_out + 80,
            order="random",
        )

    # v0.9.8c (B1): hash_shuffle — audit case.
    # N unchanged (partition does not change row count), row_size
    # unchanged (key + value + row_id; row_id uint64 not modeled —
    # conservative on N, not row_size). Order becomes "random".
    if op.name == "hash_shuffle":
        return _copy_chars(input_chars, order="random")

    # v0.9.8c: dedup — conservative N = k_est or n_rows (no model yet).
    # Best estimate k_est; if None → n_rows (safe). row_size unchanged.
    if op.name == "dedup":
        k = input_chars.k_est or input_chars.n_rows
        return _copy_chars(
            input_chars,
            n_rows=k,
            k_est=k,
            order="random",
        )

    # v0.9.8c: rank — N unchanged, row_size + 8 (rank int64).
    # order = "random" (sort by key then re-sort by row_id).
    if op.name == "rank":
        return _copy_chars(
            input_chars,
            row_size_bytes=input_chars.row_size_bytes + 8,
            order="random",
        )

    # v0.9.8c: quantile — N = 1, row_size = 8 (scalar float64).
    if op.name == "quantile":
        return _copy_chars(
            input_chars,
            n_rows=1,
            k_est=1,
            row_size_bytes=8,
            order="random",
        )

    # v0.9.8c: resample — conservative N = k_est or n_rows (buckets),
    # row_size unchanged (on + agg cols; conservative upper bound).
    if op.name == "resample":
        k = input_chars.k_est or input_chars.n_rows
        return _copy_chars(
            input_chars,
            n_rows=k,
            k_est=k,
            order="sorted",
        )

    # Unsupported op.
    raise ValueError(
        f"propagate_chars: unsupported op {op.name!r}. "
        f"Supported: filter, with_column, groupby, sort, reduce, join, "
        f"hash_shuffle, dedup, rank, quantile, resample."
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
    """Планирует цепочку (predict-only, §9.3).

    Для каждой op:
      1. model = get_model(op.name) — если op.name есть в реестре
         (reduce, hash_shuffle, sort, groupby, join).
         Если нет ("filter", "with_column") — только propagate.
      2. params = model.solve(...). Если None — PipelineMemoryError.
      3. estimate = model.estimate(params). Инвариант solve гарантирует
         estimate <= budget; нарушение — RuntimeError (баг модели).
      4. output_chars = propagate_chars(op, input, params).

    `spill_boundaries` **reserved for v0.2** (chunked execution):
    в v0.1.x всегда пуст, `predicted_outcome` ∈ {"fits", "does_not_fit"}.
    `allow_spill` — no-op в v0.1.x, параметр сохранён для API v0.2.
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
                # Инвариант solve нарушен: params пришли из solve,
                # значит estimate должен быть <= budget. Это баг модели,
                # а не повод для spill (spill не уменьшает estimate для
                # той же op на тех же данных).
                raise RuntimeError(
                    f"planner invariant violated at step {idx} "
                    f"({op.name!r}): solve() returned params with "
                    f"estimate={estimate} > budget={budget}. "
                    f"Model: {model.describe()}"
                )

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
    # v0.1.x: spill_boundaries всегда пуст (см. docstring);
    # outcome ∈ {"fits", "does_not_fit"}. "fits_with_spill" — v0.2.
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
