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

    v0.9.119 additions (all default no-op for back-compat):
    - spill_after: marks a pipeline breaker boundary after this step.
      Only honored when plan_pipeline is called with a memory_guard;
      in old path (memory_guard=None) it's ignored.
    - retained_state_bytes: RAM that stays resident after the spill
      (typically 0 for primitives; >0 only for ops that cache state).
    - model_spec: terminal ML model step, e.g.
      {"ram_formula": "...", "ram_params": {...}, "state_bytes": int}.
      When set, `name` is ignored for propagation (model uses
      n_rows=1, row_size=state_bytes).
    """
    name: str
    op_context: Optional[dict] = None
    output_overrides: Optional[dict] = None
    spill_after: bool = False
    retained_state_bytes: int = 0
    model_spec: Optional[dict] = None


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

    # v0.9.119 additions (defaults preserve old behaviour).
    est_time_seconds: float = 0.0
    time_class: str = "unknown"
    segments: list = field(default_factory=list)   # [(start_idx, end_idx_excl), ...]


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
    # v0.9.119: terminal ML model step -> output is a model state, not rows.
    if op.model_spec is not None:
        sb = int(op.model_spec.get("state_bytes", 0))
        if sb < 0:
            raise ValueError(
                f"propagate_chars: model_spec.state_bytes must be >= 0, "
                f"got {sb}")
        return _copy_chars(
            input_chars, n_rows=1, k_est=1, row_size_bytes=sb)

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
    memory_guard=None,
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

    # v0.9.119: new code path when memory_guard is provided.
    if memory_guard is not None:
        return _plan_pipeline_v2(
            ops, data_chars, available_bytes, safety=safety,
            allow_spill=allow_spill, memory_guard=memory_guard,
        )

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


# ---------- v0.9.119: segmented pipeline planning ----------

def _split_segments(ops: list[OpSpec]) -> list[tuple[int, int]]:
    """Group op indices into segments delimited by spill_after=True.

    A segment is [start, end_exclusive). Last segment always ends at
    len(ops). Empty ops -> [].
    """
    if not ops:
        return []
    segments: list[tuple[int, int]] = []
    start = 0
    for i, op in enumerate(ops):
        if op.spill_after:
            segments.append((start, i + 1))
            start = i + 1
    if start < len(ops):
        segments.append((start, len(ops)))
    return segments


def _derive_batch_from_formula(fn: str, params: dict, budget: int,
                               row_size: int) -> "int | None":
    """Inverse of ram_formula w.r.t. batch. Returns None if formula is
    batch-independent or unresolvable.

    AP23-compliant: uses the same formula that verification measured.
    e1  = formula at batch=1  (state terms + 1 row)
    e1k = formula at batch=1000
    coef_per_row = (e1k - e1) / 999
    max_batch = (budget - e1) / coef_per_row
    """
    # v0.9.119-solve-batch: delegate to shared helpers (DEC-044).
    from cadw.ml._memory_estimate import (
        _is_linear_in_batch, _numeric_inverse_batch,
    )
    if not _is_linear_in_batch(fn, params, row_size):
        return None
    return _numeric_inverse_batch(fn, params, budget, row_size)


def _model_step_estimate(
    op: OpSpec,
    data_chars: DataChars,
    *,
    budget: int = 0,
    memory_guard=None,
) -> "int | None":
    """Estimate peak for a terminal model step (v0.9.119-rule2).

    Contract (AI_CONSTITUTION AP23, bounded_state):
      - bounded_by_domain: n > bound -> return None (caller raises
        PipelineMemoryError). This is the "rule 2 in action".
      - bounded_state: True -> batch = inverse from ram_formula;
        bounded_state: False -> batch = n_rows (model does not honour
        planner batch; the formula must account for that itself).
      - No formula -> fallback to spec.state_bytes.

    Returns None only for bounded_by_domain violation.
    """
    spec = op.model_spec or {}
    fn = spec.get("cadw_function", op.name)
    state_bytes = int(spec.get("state_bytes", 0))

    # Load YAML card for bounded_state / bounded_by_domain.
    card = None
    try:
        from cadw.ml._memory_estimate import _lookup_card
        card = _lookup_card(fn)
    except Exception:
        card = None

    if card is not None:
        bdd = card.get("bounded_by_domain") or {}
        if (bdd.get("axis") == "n"
                and data_chars.n_rows > int(bdd.get("bound", 10**9))):
            return None  # caller raises PipelineMemoryError
        bounded_state = bool(card.get("bounded_state", True))
    else:
        bounded_state = True  # default: honour planner batch

    # Decide batch size.
    row_size = max(1, int(data_chars.row_size_bytes))
    params = dict(spec.get("ram_params") or {})
    if not bounded_state:
        planner_batch = int(data_chars.n_rows)
    elif budget > 0:
        if memory_guard is not None:
            inv = memory_guard.solve_batch(
                fn, data_chars, budget=budget, **params)
        else:
            inv = _derive_batch_from_formula(fn, params, budget, row_size)
        planner_batch = int(data_chars.n_rows) if inv is None else int(inv)
        if planner_batch < 1:
            # state alone exceeds budget -> treat as does-not-fit.
            return None
    else:
        planner_batch = int(data_chars.n_rows)

    # Evaluate formula with concrete batch.
    try:
        from cadw.ml._memory_estimate import evaluate_formula
        out = evaluate_formula(fn, batch=planner_batch, row_size=row_size,
                               **params)
        return int(out.get("bytes", 0))
    except Exception:
        # Formula missing or params unresolvable -> fallback.
        return state_bytes


def _plan_pipeline_v2(
    ops: list[OpSpec],
    data_chars: DataChars,
    available_bytes: int,
    *,
    safety: float,
    allow_spill: bool,
    memory_guard,
) -> PipelinePlan:
    """Segmented planning with spill boundaries + carried state + time.

    Semantics (v0.9.119):
      - Ops split into segments at spill_after boundaries.
      - Within a segment, ops are pipelined (batched); segment peak =
        max(step_estimate) + carried_state.
      - Between segments, spill discards batch RAM; carried_state =
        sum(retained_state_bytes of steps in prior segments that set
        spill_after=True).
      - peak_bytes = max over segments of (carried_at_entry + seg_peak).
      - Time: sum(model.estimate_time or dc.n_rows/rate) per step.
    """
    budget = int(available_bytes * safety)
    segments = _split_segments(ops)
    if not segments:
        return PipelinePlan(
            steps=[], peak_bytes=0, spill_boundaries=[],
            estimated_io=0, predicted_outcome="fits",
            reason=None, available_bytes=int(available_bytes),
            safety=float(safety),
        )

    steps: list[StepPlan] = []
    spill_boundaries: list[int] = []
    current_chars = data_chars
    estimated_io_total = 0
    carried_state = 0
    overall_peak = 0
    est_time_total = 0.0

    for (seg_start, seg_end) in segments:
        seg_step_peaks: list[int] = []
        seg_retained_total = 0

        for idx in range(seg_start, seg_end):
            op = ops[idx]
            params = None
            estimate = 0

            model = None
            if op.model_spec is not None:
                estimate = _model_step_estimate(
                    op, current_chars, budget=budget,
                    memory_guard=memory_guard)
                if estimate is None:
                    raise PipelineMemoryError(
                        step_idx=idx, step_name=op.name,
                        estimate_bytes=0,
                        available_bytes=available_bytes,
                        describe=(
                            f"model {op.name!r}: bounded_by_domain or "
                            f"state exceeds budget (n={current_chars.n_rows})"
                        ),
                    )
                if estimate > budget:
                    raise PipelineMemoryError(
                        step_idx=idx, step_name=op.name,
                        estimate_bytes=estimate,
                        available_bytes=available_bytes,
                        describe="model_spec estimate > budget",
                    )
            else:
                try:
                    model = get_model(op.name)
                except KeyError:
                    model = None
                if model is not None:
                    params = model.solve(
                        available_bytes, current_chars,
                        safety=safety, op_context=op.op_context,
                    )
                    if params is None:
                        raise PipelineMemoryError(
                            step_idx=idx, step_name=op.name,
                            estimate_bytes=0,
                            available_bytes=available_bytes,
                            describe=model.describe() + " (no params exist)",
                        )
                    estimate = model.estimate(params, current_chars)
                    if estimate > budget:
                        raise RuntimeError(
                            f"planner invariant violated at step {idx} "
                            f"({op.name!r}): solve() returned params with "
                            f"estimate={estimate} > budget={budget}. "
                            f"Model: {model.describe()}"
                        )

            # Time estimate: model.estimate_time (if any), else n_rows/rate.
            # v0.9.119-fw4: applies to ALL steps, including propagator-only
            # (filter/with_column with model=None) — they take O(n) time too.
            t_model = None
            if model is not None:
                try:
                    t_model = model.estimate_time(params, current_chars)
                except Exception:
                    t_model = None
            if t_model is not None:
                est_time_total += float(t_model)
            else:
                rates = memory_guard._load_throughput_rates()
                entry = rates.get(op.name) or {}
                rate = entry.get("rate_per_sec")
                if rate and float(rate) > 0:
                    est_time_total += float(current_chars.n_rows) / float(rate)

            # v0.9.120-fix1: for model_spec, use computed `estimate` as
            # the output state size. propagate_chars would read
            # op.model_spec["state_bytes"] (0 if not passed), disagreeing
            # with estimate from solve_batch + evaluate_formula.
            if op.model_spec is not None and estimate > 0:
                output_chars = _copy_chars(
                    current_chars, n_rows=1, k_est=1,
                    row_size_bytes=int(estimate))
            else:
                output_chars = propagate_chars(op, current_chars, params)
            estimated_io_total += _io_estimate(
                current_chars.n_rows, current_chars.row_size_bytes)

            steps.append(StepPlan(
                op_name=op.name,
                params=params,
                estimate_bytes=int(estimate),
                input_chars=current_chars,
                output_chars=output_chars,
                spill_before=(idx in spill_boundaries),
            ))
            seg_step_peaks.append(int(estimate))
            if op.spill_after:
                spill_boundaries.append(idx)
                seg_retained_total += int(op.retained_state_bytes)
            current_chars = output_chars

        seg_max = max(seg_step_peaks, default=0)
        seg_peak_total = carried_state + seg_max
        if seg_peak_total > overall_peak:
            overall_peak = seg_peak_total
        carried_state = seg_retained_total

    # Determine outcome
    if not spill_boundaries:
        outcome = "fits"
    elif overall_peak <= budget:
        outcome = "fits_with_spill"
    else:
        outcome = "does_not_fit"

    # Time class
    from cadw.core.memory_guard import MemoryGuard as _MG
    time_class = _MG._classify_time(est_time_total)

    return PipelinePlan(
        steps=steps,
        peak_bytes=int(overall_peak),
        spill_boundaries=list(spill_boundaries),
        estimated_io=int(estimated_io_total),
        predicted_outcome=outcome,
        reason=None,
        available_bytes=int(available_bytes),
        safety=float(safety),
        est_time_seconds=float(est_time_total),
        time_class=time_class,
        segments=list(segments),
    )
