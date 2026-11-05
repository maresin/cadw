"""
OpMemoryModel: базовый слой планировщика памяти (DECISIONS.md §9).

Задача: по `available_bytes` и `data_chars` выбрать `params`, такие что
`estimate(params, data_chars) <= available_bytes * safety`.

Никаких литералов в размерах: всё — из pyarrow types или из sample.
Калибровочные коэффициенты (C_*) — из Calibration (v0.2), не хардкод.

Планировщик (PipelinePlanner) использует реестр `_MODELS` и композицию
моделей через `plan_pipeline`. Каждая операция — `OpMemoryModel`.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pyarrow as pa


# ---------- DataChars ----------

@dataclass
class DataChars:
    """Характеристики данных, вход планировщика.

    schema — pa.Schema; row_size_bytes — вычислено `type_size` по полям.
    k_est — оценка числа уникальных ключей (sample), для groupby/shuffle.
    avg_key_size_bytes — средний размер ключа после `repr().encode()`
                         (sample), для `compute_hashes` overhead.
    size_of_key_entry_bytes — tuple key + agg state, для dict groupby.
    order — подсказка о порядке: "random" | "sorted" | "clustered".
            Влияет на выбор sample-strategy в planner (v0.2), в моделях
            не используется.
    """
    schema: pa.Schema
    n_rows: int
    row_size_bytes: int
    k_est: Optional[int] = None
    avg_key_size_bytes: Optional[int] = None
    size_of_key_entry_bytes: Optional[int] = None
    order: str = "random"
    extras: dict = field(default_factory=dict)


# ---------- Params ----------

@dataclass
class Params:
    """Базовый класс параметров. Подклассы — в models_<op>.py."""
    pass

@dataclass
class CostEstimate:
    """Estimated RAM + time for one op (v0.9.119, AP22).

    Returned by MemoryGuard.estimate_cost(). `est_ram_bytes` = model
    estimate(params) under the guard's budget. `est_time_seconds` =
    est_ops / rate (rate from throughput.yaml), or 0.0 with
    LegacyFormulaWarning when no rate exists.

    time_class: "fast" < 1s | "medium" < 60s | "slow" < 600s |
                "very_slow" >= 600s | "unknown" (no rate).
    """
    op_name: str
    est_ram_bytes: int
    fits_ram: bool
    est_ops: int
    est_time_seconds: float
    time_class: str
    reason: Optional[str] = None



# ---------- OpMemoryModel ----------

class OpMemoryModel(ABC):
    """Модель памяти одной операции.

    Контракт:
      - estimate(params, data_chars) -> int: пик RAM (байты) при params.
        Верхняя оценка, не средняя.
      - solve(available_bytes, data_chars, *, safety, op_context) -> Params|None:
        находит params, такие что estimate(params) <= available * safety.
        None — если решения нет (state > available, k < 2 и т.п.).
      - describe() -> str: для error-сообщений.
    """

    op_name: str = "abstract"

    @abstractmethod
    def estimate(self, params: Params, data_chars: DataChars) -> int: ...

    @abstractmethod
    def solve(
        self,
        available_bytes: int,
        data_chars: DataChars,
        *,
        safety: float = 0.7,
        op_context: Optional[dict] = None,
    ) -> Optional[Params]: ...

    @abstractmethod
    def describe(self) -> str: ...

    def estimate_time(
        self,
        params: Params,
        data_chars: DataChars,
    ) -> Optional[float]:
        """Estimated wall-clock seconds for this op (AP22, v0.9.119).

        Default returns None -> MemoryGuard.estimate_cost falls back to
        throughput.yaml conservative rate + LegacyFormulaWarning.
        Override for model-specific estimates (e.g. n_epochs * n_rows).
        """
        return None


_MODELS: dict[str, OpMemoryModel] = {}


def register_model(model: OpMemoryModel) -> None:
    """Зарегистрировать модель. Идемпотентно (перезапись с warning)."""
    if model.op_name in _MODELS:
        warnings.warn(
            f"OpMemoryModel {model.op_name!r} overwritten",
            RuntimeWarning,
            stacklevel=2,
        )
    _MODELS[model.op_name] = model


def get_model(op_name: str) -> OpMemoryModel:
    if op_name not in _MODELS:
        raise KeyError(
            f"OpMemoryModel {op_name!r} not registered. "
            f"Available: {sorted(_MODELS)}"
        )
    return _MODELS[op_name]


def list_models() -> list[str]:
    return sorted(_MODELS)


# ---------- type sizes ----------

def _arrow_offset_bytes() -> int:
    """Размер offset в Arrow для variable-length (string/binary/list).

    Не литерал: pyarrow использует int32 (signed) для offset в `Array`
    и int64 для `LargeArray`. Здесь — консервативно int32 (4 B), т.к.
    наши данные — обычные `Array`. Для `large_*` — 8 B (см. отдельно).
    """
    return 4


def type_size(dtype: pa.DataType, sample: Optional[pa.Array] = None) -> int:
    """Размер одного значения в RAM (байт).

    - numeric/bool/timestamp/date/time: из bit_width.
    - string/binary: avg len из sample + offset (4 B).
      Без sample для string — ValueError (нельзя оценить без измерения).
    - dictionary/list/struct: пока NotImplementedError (v0.2).
    """
    if pa.types.is_boolean(dtype):
        # Arrow bool — bit-packed на диске, но в numpy/Arrow Array — 1 байт
        # на значение при маскировании. Консервативно 1.
        return 1
    if pa.types.is_integer(dtype) or pa.types.is_floating(dtype):
        return max(1, dtype.bit_width // 8)
    if (pa.types.is_timestamp(dtype) or pa.types.is_date(dtype)
            or pa.types.is_time(dtype)):
        return max(1, dtype.bit_width // 8)
    if pa.types.is_string(dtype) or pa.types.is_binary(dtype):
        if sample is None:
            raise ValueError(
                f"type_size({dtype}): sample required for variable-length; "
                f"cannot estimate avg_len without data"
            )
        n = len(sample)
        if n == 0:
            raise ValueError(f"type_size({dtype}): empty sample")
        lens = [len(s) for s in sample.to_pylist()]
        avg = sum(lens) / n
        return int(np.ceil(avg)) + _arrow_offset_bytes()
    if pa.types.is_dictionary(dtype):
        raise NotImplementedError("type_size: dictionary (v0.2)")
    if pa.types.is_list(dtype) or pa.types.is_struct(dtype):
        raise NotImplementedError(f"type_size: {dtype} (v0.2)")
    raise NotImplementedError(f"type_size: unsupported dtype {dtype}")


def row_size(
    schema: pa.Schema,
    sample_batch: Optional[pa.RecordBatch] = None,
) -> int:
    """Сумма type_size по полям schema.

    sample_batch — обязателен, если есть string/binary поля.
    """
    total = 0
    for field in schema:
        sub_sample = None
        if sample_batch is not None and field.name in sample_batch.schema.names:
            sub_sample = sample_batch.column(field.name)
        total += type_size(field.type, sub_sample)
    return total


def data_chars_from_sample(
    sample_batch: pa.RecordBatch,
    n_rows_total: int,
    *,
    k_est: Optional[int] = None,
    avg_key_size_bytes: Optional[int] = None,
    size_of_key_entry_bytes: Optional[int] = None,
    order: str = "random",
) -> DataChars:
    """Удобный конструктор DataChars из sample-батча."""
    rs = row_size(sample_batch.schema, sample_batch)
    return DataChars(
        schema=sample_batch.schema,
        n_rows=n_rows_total,
        row_size_bytes=rs,
        k_est=k_est,
        avg_key_size_bytes=avg_key_size_bytes,
        size_of_key_entry_bytes=size_of_key_entry_bytes,
        order=order,
    )
