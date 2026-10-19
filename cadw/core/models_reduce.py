"""
ReduceMemoryModel: пик RAM для свёртки.

Формула (DECISIONS.md §9.2.1):

    f_reduce = batch_size * row_size + 2 * state_bytes

`2 * state_bytes` — старый и новый state на время `update` (merge).
Верхняя оценка, консервативно.

`solve`:
    available_for_batch = available * safety - 2 * state_bytes
    если available_for_batch < row_size -> None (state не влезает)
    batch_size = available_for_batch // row_size

`op_context`: {"state_bytes": int} — обязателен.
    state_bytes считает вызывающий (MemoryGuard.plan_reduce) через
    `estimate_state_size` в MemoryGuard или через `op.state_nbytes(init_state)`
    на конкретном ReduceOp. Модель не знает про reduce-операции, только
    про размер состояния.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cadw.core.models import (
    DataChars,
    OpMemoryModel,
    Params,
    register_model,
)


@dataclass
class ReduceParams(Params):
    batch_size: int
    state_bytes: int


class ReduceMemoryModel(OpMemoryModel):
    op_name = "reduce"

    def estimate(self, params: ReduceParams, data_chars: DataChars) -> int:
        if not isinstance(params, ReduceParams):
            raise TypeError(f"expected ReduceParams, got {type(params).__name__}")
        return params.batch_size * data_chars.row_size_bytes + 2 * params.state_bytes

    def solve(
        self,
        available_bytes: int,
        data_chars: DataChars,
        *,
        safety: float = 0.7,
        op_context: Optional[dict] = None,
    ) -> Optional[ReduceParams]:
        if op_context is None or "state_bytes" not in op_context:
            raise ValueError(
                "ReduceMemoryModel.solve: op_context['state_bytes'] required"
            )
        state_bytes = int(op_context["state_bytes"])
        if state_bytes < 0:
            raise ValueError(f"state_bytes must be >= 0, got {state_bytes}")
        if data_chars.row_size_bytes <= 0:
            raise ValueError(
                f"row_size_bytes must be > 0, got {data_chars.row_size_bytes}"
            )

        usable = int(available_bytes * safety)
        available_for_batch = usable - 2 * state_bytes
        if available_for_batch < data_chars.row_size_bytes:
            return None

        batch_size = available_for_batch // data_chars.row_size_bytes
        if batch_size < 1:
            return None
        return ReduceParams(batch_size=batch_size, state_bytes=state_bytes)

    def describe(self) -> str:
        return (
            "ReduceMemoryModel: "
            "estimate = batch_size * row_size + 2 * state_bytes"
        )


register_model(ReduceMemoryModel())
