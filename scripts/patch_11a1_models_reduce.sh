#!/usr/bin/env bash
# patch_11a1: OpMemoryModel + ReduceMemoryModel (базовый слой §9).
# Не трогает memory_guard / reduce / shuffle / sort / groupby —
# это 11-A-2..4.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-11a1-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_11a1"

BR="patch/11a1-models-reduce"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- cadw/core/models.py ----------
cat > cadw/core/models.py <<'PY'
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
PY

# ---------- cadw/core/models_reduce.py ----------
cat > cadw/core/models_reduce.py <<'PY'
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
PY

# ---------- cadw/core/__init__.py ----------
cat > cadw/core/__init__.py <<'PY'
from .memory_guard import MemoryGuard, MemoryLimitExceeded
from .models import (
    DataChars,
    OpMemoryModel,
    Params,
    data_chars_from_sample,
    get_model,
    list_models,
    register_model,
    row_size,
    type_size,
)
from .models_reduce import ReduceMemoryModel, ReduceParams

__all__ = [
    "MemoryGuard",
    "MemoryLimitExceeded",
    "DataChars",
    "OpMemoryModel",
    "Params",
    "data_chars_from_sample",
    "get_model",
    "list_models",
    "register_model",
    "row_size",
    "type_size",
    "ReduceMemoryModel",
    "ReduceParams",
]
PY

# ---------- tests/test_models.py ----------
cat > tests/test_models.py <<'EOF'
"""
Тесты OpMemoryModel и ReduceMemoryModel (11-A-1).

Инвариант модели: estimate(solve(x)) <= x * safety.
Плюс граничные случаи: state не влезает -> None; string без sample -> ValueError.
"""
import pyarrow as pa
import pytest

from cadw.core.models import (
    DataChars,
    data_chars_from_sample,
    get_model,
    list_models,
    row_size,
    type_size,
)
from cadw.core.models_reduce import ReduceMemoryModel, ReduceParams


# ---------- type_size / row_size ----------

def test_type_size_numeric():
    assert type_size(pa.int64()) == 8
    assert type_size(pa.int32()) == 4
    assert type_size(pa.float64()) == 8
    assert type_size(pa.float32()) == 4
    assert type_size(pa.bool_()) == 1


def test_type_size_timestamp():
    assert type_size(pa.timestamp("ns")) == 8
    assert type_size(pa.date32()) == 4


def test_type_size_string_requires_sample():
    with pytest.raises(ValueError, match="sample required"):
        type_size(pa.string(), sample=None)


def test_type_size_string_with_sample():
    sample = pa.array(["abc", "de", "f"])
    # avg_len = (3+2+1)/3 = 2, + offset 4 = 6
    assert type_size(pa.string(), sample=sample) == 6


def test_type_size_empty_sample_raises():
    with pytest.raises(ValueError, match="empty sample"):
        type_size(pa.string(), sample=pa.array([], type=pa.string()))


def test_row_size_mixed():
    schema = pa.schema([("a", pa.int64()), ("b", pa.float32())])
    assert row_size(schema) == 12  # 8 + 4


def test_row_size_with_string_requires_sample():
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    with pytest.raises(ValueError):
        row_size(schema, sample_batch=None)


def test_data_chars_from_sample():
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3]), pa.array(["a", "bb", "ccc"])],
        names=["n", "s"],
    )
    dc = data_chars_from_sample(batch, n_rows_total=1000, k_est=10)
    # row = 8 (int64) + (avg string (1+2+3)/3=2 + 4 offset = 6) = 14
    assert dc.row_size_bytes == 14
    assert dc.n_rows == 1000
    assert dc.k_est == 10


# ---------- registry ----------

def test_reduce_model_registered():
    m = get_model("reduce")
    assert isinstance(m, ReduceMemoryModel)
    assert "reduce" in list_models()


def test_get_unknown_model_raises():
    with pytest.raises(KeyError, match="not registered"):
        get_model("nonexistent_op")


# ---------- ReduceMemoryModel ----------

def _data_chars(row_size=16, n_rows=10_000) -> DataChars:
    schema = pa.schema([("x", pa.int64()), ("y", pa.int64())])
    return DataChars(schema=schema, n_rows=n_rows, row_size_bytes=row_size)


@pytest.mark.parametrize(
    "state_bytes, available_mb",
    [
        (16, 10),       # CountOp/SumOp/MeanOp
        (24, 10),       # StdOp (Welford)
        (8 + 800 + 80_000, 10),      # CovarianceOp(d=100): 8+8*100+8*10000
        (255 * 8 + 16, 10),          # HistogramOp(bins=255)
    ],
)
def test_reduce_estimate_within_safety(state_bytes, available_mb):
    """estimate(solve(x)) <= x * safety для всех реалистичных state."""
    model = ReduceMemoryModel()
    dc = _data_chars()
    available = available_mb * 1024 * 1024
    safety = 0.7

    params = model.solve(available, dc, safety=safety, op_context={"state_bytes": state_bytes})
    assert params is not None, (
        f"solve returned None for state={state_bytes}, available={available_mb}MB"
    )
    assert isinstance(params, ReduceParams)
    assert params.batch_size > 0
    assert params.state_bytes == state_bytes

    est = model.estimate(params, dc)
    assert est <= int(available * safety), (
        f"estimate {est} > available*safety {int(available * safety)} "
        f"for state={state_bytes}"
    )


def test_reduce_solve_returns_none_when_state_too_big():
    """CovarianceOp(d=5000) ~ 200 MB; available=10 MB -> None."""
    d = 5000
    state_bytes = 8 + 8 * d + 8 * d * d  # ~200 MB
    model = ReduceMemoryModel()
    dc = _data_chars()
    available = 10 * 1024 * 1024
    params = model.solve(available, dc, op_context={"state_bytes": state_bytes})
    assert params is None


def test_reduce_solve_requires_state_bytes():
    model = ReduceMemoryModel()
    dc = _data_chars()
    with pytest.raises(ValueError, match="state_bytes"):
        model.solve(10 * 1024 * 1024, dc, op_context=None)
    with pytest.raises(ValueError, match="state_bytes"):
        model.solve(10 * 1024 * 1024, dc, op_context={})


def test_reduce_solve_negative_state_raises():
    model = ReduceMemoryModel()
    dc = _data_chars()
    with pytest.raises(ValueError, match=">= 0"):
        model.solve(10 * 1024 * 1024, dc, op_context={"state_bytes": -1})


def test_reduce_estimate_type_check():
    model = ReduceMemoryModel()
    dc = _data_chars()
    # Параметр не того типа
    class FakeParams:
        pass
    with pytest.raises(TypeError, match="ReduceParams"):
        model.estimate(FakeParams(), dc)


def test_reduce_describe():
    model = ReduceMemoryModel()
    d = model.describe()
    assert "ReduceMemoryModel" in d
    assert "state_bytes" in d
EOF

# ---------- run ----------
echo "=== pytest tests/test_models.py ==="
set +e
uv run pytest tests/test_models.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "ERROR: test_models.py failed (rc=$RC)"
  exit 1
fi

# Полный прогон — убедиться, что ничего не сломали
echo "=== pytest (fast) ==="
uv run pytest -q -m "not slow"

# ---------- commit ----------
git add -A
git commit -m "11-A-1: OpMemoryModel + ReduceMemoryModel (§9.0-9.2.1); type_size без литералов; 16 tests"

echo "==> sha256"
sha256sum cadw/core/models.py cadw/core/models_reduce.py tests/test_models.py
echo "==> 11-A-1 applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"