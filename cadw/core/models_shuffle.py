"""
HashShuffleMemoryModel (§9.2.2).

    f_shuffle = batch_size * row_size
              + P * writer_buffer_bytes
              + c_hash * batch_size

c_hash — overhead `compute_hashes` (`to_pylist`): `repr(v).encode()` +
Python int. Все числовые константы — параметры конструктора
(CALIBRATION DEFAULT), переопределяются в 11-A-4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from cadw.core.models import (
    DataChars,
    OpMemoryModel,
    Params,
    register_model,
)


@dataclass
class HashShuffleParams(Params):
    batch_size: int
    num_partitions: int
    writer_buffer_bytes: int


class HashShuffleMemoryModel(OpMemoryModel):
    op_name = "hash_shuffle"

    def __init__(
        self,
        *,
        python_int_overhead: int = 28,        # CALIBRATION DEFAULT (sys.getsizeof(int))
        writer_buffer_min: int = 4 * 1024,    # 4 KB
        writer_buffer_max: int = 1024 * 1024, # 1 MB
        writer_budget_ratio: float = 0.1,     # доля budget на writer buffers
        max_keys_ratio: float = 0.5,          # доля budget на dict partition
        max_keys_floor: int = 1,
    ) -> None:
        self.python_int_overhead = int(python_int_overhead)
        self.writer_buffer_min = int(writer_buffer_min)
        self.writer_buffer_max = int(writer_buffer_max)
        if not (0 < writer_budget_ratio < 1):
            raise ValueError("writer_budget_ratio must be in (0,1)")
        if not (0 < max_keys_ratio < 1):
            raise ValueError("max_keys_ratio must be in (0,1)")
        self.writer_budget_ratio = float(writer_budget_ratio)
        self.max_keys_ratio = float(max_keys_ratio)
        self.max_keys_floor = int(max_keys_floor)

    def _c_hash(self, data_chars: DataChars) -> int:
        """repr(v).encode() + Python int, если sample дал avg_key_size."""
        base = data_chars.avg_key_size_bytes or 0
        return base + self.python_int_overhead

    def estimate(self, params: HashShuffleParams, data_chars: DataChars) -> int:
        if not isinstance(params, HashShuffleParams):
            raise TypeError(
                f"expected HashShuffleParams, got {type(params).__name__}"
            )
        if params.batch_size < 0 or params.num_partitions < 1:
            raise ValueError("invalid HashShuffleParams")
        return (
            params.batch_size * data_chars.row_size_bytes
            + params.num_partitions * params.writer_buffer_bytes
            + self._c_hash(data_chars) * params.batch_size
        )

    def solve(
        self,
        available_bytes: int,
        data_chars: DataChars,
        *,
        safety: float = 0.7,
        op_context: Optional[dict] = None,
    ) -> Optional[HashShuffleParams]:
        if data_chars.k_est is None:
            raise ValueError(
                "HashShuffleMemoryModel.solve: data_chars.k_est required"
            )
        if data_chars.size_of_key_entry_bytes is None:
            raise ValueError(
                "HashShuffleMemoryModel.solve: "
                "data_chars.size_of_key_entry_bytes required"
            )
        if data_chars.row_size_bytes <= 0:
            raise ValueError("row_size_bytes must be > 0")

        budget = int(available_bytes * safety)

        # P: из K_est и max_keys_per_partition
        max_keys_per_partition = max(
            self.max_keys_floor,
            int(budget * self.max_keys_ratio / data_chars.size_of_key_entry_bytes),
        )
        if data_chars.k_est > 0:
            p_raw = math.ceil(
                data_chars.k_est * safety / max_keys_per_partition
            )
        else:
            p_raw = 2
        p = max(2, p_raw)
        # round up to power of 2 (fast path _mod_p)
        p = 1 << (p - 1).bit_length()

        # sanity: P * writer_buffer_min должно укладываться в 0.4 * budget
        if p * self.writer_buffer_min > budget * 0.4:
            return None

        writer_buffer = int(budget * self.writer_budget_ratio // p)
        writer_buffer = max(self.writer_buffer_min, min(self.writer_buffer_max, writer_buffer))

        c_hash = self._c_hash(data_chars)
        remaining = budget - p * writer_buffer
        if remaining <= data_chars.row_size_bytes:
            return None
        batch_size = remaining // (data_chars.row_size_bytes + c_hash)
        if batch_size < 1:
            return None
        return HashShuffleParams(
            batch_size=int(batch_size),
            num_partitions=int(p),
            writer_buffer_bytes=int(writer_buffer),
        )

    def describe(self) -> str:
        return (
            "HashShuffleMemoryModel: "
            "f = batch*row_size + P*writer_buffer + c_hash*batch "
            "(writer_budget_ratio=%.2f, max_keys_ratio=%.2f)"
            % (self.writer_budget_ratio, self.max_keys_ratio)
        )


register_model(HashShuffleMemoryModel())
