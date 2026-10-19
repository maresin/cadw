"""
SortMemoryModel (§9.2.3).

Две фазы:
    f_phase1 = run_buffer_bytes
    f_phase2 = k * read_buffer_bytes
             + k * heap_entry_size
             + read_batch_size * row_size

estimate = max(phase1, phase2). Числовые константы — параметры
конструктора (CALIBRATION DEFAULT).
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
class SortParams(Params):
    run_buffer_bytes: int
    read_buffer_bytes: int
    k: int
    read_batch_size: int


class SortMemoryModel(OpMemoryModel):
    op_name = "sort"

    def __init__(
        self,
        *,
        run_buffer_ratio: float = 0.5,
        read_buffer_ratio: float = 0.3,
        read_batch_target: int = 1000,  # CALIBRATION DEFAULT
        heap_entry_size: int = 32,      # (key_size + row_id) approximation
        max_k: int = 256,
    ) -> None:
        if not (0 < run_buffer_ratio < 1):
            raise ValueError("run_buffer_ratio must be in (0,1)")
        if not (0 < read_buffer_ratio < 1):
            raise ValueError("read_buffer_ratio must be in (0,1)")
        self.run_buffer_ratio = float(run_buffer_ratio)
        self.read_buffer_ratio = float(read_buffer_ratio)
        self.read_batch_target = int(read_batch_target)
        self.heap_entry_size = int(heap_entry_size)
        self.max_k = int(max_k)

    def estimate(self, params: SortParams, data_chars: DataChars) -> int:
        if not isinstance(params, SortParams):
            raise TypeError(f"expected SortParams, got {type(params).__name__}")
        if params.k < 1:
            raise ValueError("SortParams.k must be >= 1")
        f_phase1 = params.run_buffer_bytes
        f_phase2 = (
            params.k * params.read_buffer_bytes
            + params.k * self.heap_entry_size
            + params.read_batch_size * data_chars.row_size_bytes
        )
        return max(f_phase1, f_phase2)

    def solve(
        self,
        available_bytes: int,
        data_chars: DataChars,
        *,
        safety: float = 0.7,
        op_context: Optional[dict] = None,
    ) -> Optional[SortParams]:
        if data_chars.row_size_bytes <= 0:
            raise ValueError("row_size_bytes must be > 0")
        budget = int(available_bytes * safety)
        if budget <= 0:
            return None

        run_buffer = int(budget * self.run_buffer_ratio)
        read_avail = int(budget * self.read_buffer_ratio)

        if read_avail < 2 * data_chars.row_size_bytes:
            return None

        k_target = read_avail // (self.read_batch_target * data_chars.row_size_bytes)
        k = max(2, min(self.max_k, int(k_target)))
        read_buffer = read_avail // k
        read_batch_size = read_buffer // data_chars.row_size_bytes
        if read_batch_size < 1:
            return None
        params = SortParams(
            run_buffer_bytes=run_buffer,
            read_buffer_bytes=int(read_buffer),
            k=int(k),
            read_batch_size=int(read_batch_size),
        )
        if self.estimate(params, data_chars) > budget:
            return None
        return params

    def describe(self) -> str:
        return (
            "SortMemoryModel: "
            "f = max(run_buffer, k*read_buffer + k*heap + read_batch*row_size) "
            "(run_ratio=%.2f, read_ratio=%.2f, max_k=%d)"
            % (self.run_buffer_ratio, self.read_buffer_ratio, self.max_k)
        )


register_model(SortMemoryModel())
