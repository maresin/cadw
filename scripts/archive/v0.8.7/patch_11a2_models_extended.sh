#!/usr/bin/env bash
# patch_11a2_models_shuffle_sort_groupby: HashShuffle, Sort, GroupBy MemoryModel.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-11a2-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_11a2"

BR="patch/11a2-models-extended"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- models_shuffle.py ----------
cat > cadw/core/models_shuffle.py <<'PY'
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
PY

# ---------- models_sort.py ----------
cat > cadw/core/models_sort.py <<'PY'
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
PY

# ---------- models_groupby.py ----------
cat > cadw/core/models_groupby.py <<'PY'
"""
GroupByMemoryModel (§9.2.4).

Композиция: HashShuffle + _local_groupby + concat.

    f_groupby = max(f_shuffle, f_local_groupby)
    f_local_groupby = partition_bytes + dict_bytes + c_temp_ratio * dict_bytes
    partition_bytes = (K_est / P) * row_size
    dict_bytes = (K_est / P) * size_of_key_entry

recurse_depth: upper bound на число уровней split внутри партиции.
  Если K_est/P > max_dict_keys — split по P_recurse до тех пор, пока
  K_per_partition' <= max_dict_keys и partition_bytes' <= file_budget.
  depth = max(d_dict, d_file); None, если > MAX_DEPTH.

Параметры partition_ratio, dict_ratio, c_temp_ratio, p_recurse, max_depth —
CALIBRATION DEFAULT (11-A-4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from cadw.core.models import (
    DataChars,
    OpMemoryModel,
    Params,
    get_model,
    register_model,
)
from cadw.core.models_shuffle import HashShuffleMemoryModel, HashShuffleParams


@dataclass
class GroupByParams(Params):
    shuffle: HashShuffleParams
    max_dict_keys: int
    recurse_depth: int
    p_recurse: int


def _depth_for(needed: float, capacity: float, p_recurse: int) -> int:
    if capacity <= 0:
        return 10**6  # effectively infinite
    if needed <= capacity:
        return 0
    return int(math.ceil(math.log(needed / capacity) / math.log(p_recurse)))


class GroupByMemoryModel(OpMemoryModel):
    op_name = "groupby"

    def __init__(
        self,
        shuffle_model: Optional[HashShuffleMemoryModel] = None,
        *,
        p_recurse: int = 4,             # CALIBRATION DEFAULT
        max_depth: int = 5,             # CALIBRATION DEFAULT
        partition_ratio: float = 0.35,  # доля budget под partition_bytes
        dict_ratio: float = 0.25,       # доля budget под dict_bytes
        c_temp_ratio: float = 1.0,      # temp ≈ dict_bytes
    ) -> None:
        self.shuffle_model = shuffle_model or get_model("hash_shuffle")
        if not isinstance(self.shuffle_model, HashShuffleMemoryModel):
            raise TypeError("shuffle_model must be HashShuffleMemoryModel")
        if p_recurse < 2:
            raise ValueError("p_recurse must be >= 2")
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        for name, r in (
            ("partition_ratio", partition_ratio),
            ("dict_ratio", dict_ratio),
            ("c_temp_ratio", c_temp_ratio),
        ):
            if r < 0:
                raise ValueError(f"{name} must be >= 0")
        if partition_ratio + dict_ratio * (1 + c_temp_ratio) > 1.0:
            raise ValueError(
                "partition_ratio + dict_ratio*(1+c_temp_ratio) must be <= 1.0; "
                f"got {partition_ratio} + {dict_ratio}*(1+{c_temp_ratio})"
            )
        self.p_recurse = int(p_recurse)
        self.max_depth = int(max_depth)
        self.partition_ratio = float(partition_ratio)
        self.dict_ratio = float(dict_ratio)
        self.c_temp_ratio = float(c_temp_ratio)

    def estimate(self, params: GroupByParams, data_chars: DataChars) -> int:
        if not isinstance(params, GroupByParams):
            raise TypeError(f"expected GroupByParams, got {type(params).__name__}")
        if data_chars.k_est is None or data_chars.size_of_key_entry_bytes is None:
            raise ValueError(
                "GroupByMemoryModel.estimate: "
                "k_est and size_of_key_entry_bytes required"
            )

        f_shuffle = self.shuffle_model.estimate(params.shuffle, data_chars)

        k_per = data_chars.k_est / params.shuffle.num_partitions
        partition_bytes = k_per * data_chars.row_size_bytes
        dict_bytes = k_per * data_chars.size_of_key_entry_bytes
        f_local = partition_bytes + dict_bytes + self.c_temp_ratio * dict_bytes

        return int(max(f_shuffle, f_local))

    def solve(
        self,
        available_bytes: int,
        data_chars: DataChars,
        *,
        safety: float = 0.7,
        op_context: Optional[dict] = None,
    ) -> Optional[GroupByParams]:
        if data_chars.k_est is None:
            raise ValueError("GroupByMemoryModel.solve: k_est required")
        if data_chars.size_of_key_entry_bytes is None:
            raise ValueError(
                "GroupByMemoryModel.solve: size_of_key_entry_bytes required"
            )

        shuffle_params = self.shuffle_model.solve(
            available_bytes, data_chars, safety=safety
        )
        if shuffle_params is None:
            return None

        budget = int(available_bytes * safety)
        max_dict_keys = max(
            1, int(budget * self.dict_ratio / data_chars.size_of_key_entry_bytes)
        )
        file_budget = budget * self.partition_ratio
        if file_budget <= 0:
            return None

        k_per = data_chars.k_est / shuffle_params.num_partitions
        if k_per <= 0:
            # Пустой dataset
            return GroupByParams(
                shuffle=shuffle_params,
                max_dict_keys=max_dict_keys,
                recurse_depth=0,
                p_recurse=self.p_recurse,
            )

        d_dict = _depth_for(k_per, max_dict_keys, self.p_recurse)
        file_bytes = k_per * data_chars.row_size_bytes
        d_file = _depth_for(file_bytes, file_budget, self.p_recurse)
        recurse_depth = max(d_dict, d_file)
        if recurse_depth > self.max_depth:
            return None

        params = GroupByParams(
            shuffle=shuffle_params,
            max_dict_keys=int(max_dict_keys),
            recurse_depth=int(recurse_depth),
            p_recurse=self.p_recurse,
        )
        if self.estimate(params, data_chars) > budget:
            return None
        return params

    def describe(self) -> str:
        return (
            "GroupByMemoryModel: f = max(f_shuffle, f_local_groupby); "
            "recurse_depth = ceil(log((K/P)/max_dict_keys)/log(p_recurse)) "
            "(partition_ratio=%.2f, dict_ratio=%.2f, p_recurse=%d, max_depth=%d)"
            % (
                self.partition_ratio,
                self.dict_ratio,
                self.p_recurse,
                self.max_depth,
            )
        )


register_model(GroupByMemoryModel())
PY

# ---------- cadw/core/__init__.py (дополнить экспорты) ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/core/__init__.py")
s = p.read_text()
if "from .models_shuffle" not in s:
    s = s.replace(
        "from .models_reduce import ReduceMemoryModel, ReduceParams",
        "from .models_reduce import ReduceMemoryModel, ReduceParams\n"
        "from .models_shuffle import HashShuffleMemoryModel, HashShuffleParams\n"
        "from .models_sort import SortMemoryModel, SortParams\n"
        "from .models_groupby import GroupByMemoryModel, GroupByParams",
        1,
    )
    s = s.replace(
        '    "ReduceMemoryModel",\n    "ReduceParams",\n',
        '    "ReduceMemoryModel",\n    "ReduceParams",\n'
        '    "HashShuffleMemoryModel",\n    "HashShuffleParams",\n'
        '    "SortMemoryModel",\n    "SortParams",\n'
        '    "GroupByMemoryModel",\n    "GroupByParams",\n',
        1,
    )
    p.write_text(s)
    print("cadw/core/__init__.py: added models_shuffle, models_sort, models_groupby")
PY

# ---------- tests/test_models_extended.py ----------
cat > tests/test_models_extended.py <<'EOF'
"""
Тесты HashShuffleMemoryModel, SortMemoryModel, GroupByMemoryModel (11-A-2).

Инвариант для каждой модели: estimate(solve(x)) <= x * safety.
Граничные случаи: None когда params не существует; композиция GroupBy.
"""
import pyarrow as pa
import pytest

from cadw.core.models import DataChars, get_model, list_models
from cadw.core.models_shuffle import HashShuffleMemoryModel, HashShuffleParams
from cadw.core.models_sort import SortMemoryModel, SortParams
from cadw.core.models_groupby import GroupByMemoryModel, GroupByParams


def _dc(
    row_size: int = 16,
    n_rows: int = 1_000_000,
    k_est: int = 1000,
    avg_key_size: int = 8,
    size_of_key_entry: int = 80,
) -> DataChars:
    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])
    return DataChars(
        schema=schema,
        n_rows=n_rows,
        row_size_bytes=row_size,
        k_est=k_est,
        avg_key_size_bytes=avg_key_size,
        size_of_key_entry_bytes=size_of_key_entry,
    )


# ---------- registry ----------

def test_all_models_registered():
    names = list_models()
    for n in ("reduce", "hash_shuffle", "sort", "groupby"):
        assert n in names, f"{n} not registered; got {names}"


# ---------- HashShuffleMemoryModel ----------

@pytest.mark.parametrize(
    "available_mb, k_est, row_size",
    [
        (10, 1000, 16),
        (10, 100_000, 16),
        (100, 1_000_000, 16),
        (10, 1_000_000, 16),
        (50, 10, 16),
        (100, 1000, 1000),
    ],
)
def test_shuffle_invariant(available_mb, k_est, row_size):
    m = HashShuffleMemoryModel()
    dc = _dc(row_size=row_size, k_est=k_est)
    available = available_mb * 1024 * 1024
    safety = 0.7
    params = m.solve(available, dc, safety=safety)
    assert params is not None, (
        f"unexpected None: available={available_mb}MB k_est={k_est}"
    )
    assert isinstance(params, HashShuffleParams)
    assert params.num_partitions >= 2
    # power of 2
    assert (params.num_partitions & (params.num_partitions - 1)) == 0
    assert params.batch_size >= 1
    est = m.estimate(params, dc)
    assert est <= int(available * safety), (
        f"estimate {est} > safety {int(available*safety)}"
    )


def test_shuffle_none_when_too_many_partitions():
    """P * writer_buffer_min > budget * 0.4 -> None."""
    m = HashShuffleMemoryModel(writer_buffer_min=64 * 1024)  # 64 KB
    dc = _dc(k_est=10**9, size_of_key_entry=80)
    # available очень мал, K_est огромен -> P огромно -> P*64KB > budget*0.4
    params = m.solve(64 * 1024, dc, safety=0.7)
    assert params is None


def test_shuffle_requires_k_est():
    m = HashShuffleMemoryModel()
    dc = _dc(k_est=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k_est"):
        m.solve(10 * 1024 * 1024, dc)


def test_shuffle_requires_size_of_key_entry():
    m = HashShuffleMemoryModel()
    dc = _dc(size_of_key_entry=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="size_of_key_entry"):
        m.solve(10 * 1024 * 1024, dc)


def test_shuffle_type_check():
    m = HashShuffleMemoryModel()
    dc = _dc()
    with pytest.raises(TypeError, match="HashShuffleParams"):
        m.estimate("not a params", dc)  # type: ignore[arg-type]


# ---------- SortMemoryModel ----------

@pytest.mark.parametrize(
    "available_mb, row_size",
    [(10, 16), (10, 1000), (100, 16), (1, 16)],
)
def test_sort_invariant(available_mb, row_size):
    m = SortMemoryModel()
    dc = _dc(row_size=row_size)
    available = available_mb * 1024 * 1024
    safety = 0.7
    params = m.solve(available, dc, safety=safety)
    assert params is not None
    assert isinstance(params, SortParams)
    assert params.k >= 2
    assert params.read_batch_size >= 1
    est = m.estimate(params, dc)
    assert est <= int(available * safety)


def test_sort_none_when_too_tight():
    m = SortMemoryModel()
    dc = _dc(row_size=1000)
    # 100 KB budget, 1 KB rows: read_avail = 21 KB; read_batch_target=1000 -> нужно 1 MB
    params = m.solve(100 * 1024, dc, safety=0.7)
    # read_avail // (1000 * 1000) = 0 -> k = max(2, 0) = 2; read_buffer = 10 KB; read_batch = 10
    # f_phase2 = 2*10KB + 2*32 + 10*1000 = 20KB + 64 + 10KB ≈ 30 KB < 70 KB -> passes
    # Для точного None нужен ещё меньше budget:
    params2 = m.solve(10 * 1024, dc, safety=0.7)
    # 10 KB * 0.7 = 7 KB; read_avail = 2.1 KB; read_avail < 2 * 1000 -> None
    assert params2 is None


def test_sort_max_k_cap():
    m = SortMemoryModel(max_k=8)
    dc = _dc(row_size=16)
    params = m.solve(100 * 1024 * 1024, dc, safety=0.7)
    assert params is not None
    assert params.k <= 8


# ---------- GroupByMemoryModel ----------

@pytest.mark.parametrize(
    "available_mb, k_est, size_of_key_entry",
    [
        (10, 10, 80),
        (10, 1000, 80),
        (10, 10_000, 80),
        (10, 100_000, 80),
        (100, 1_000_000, 80),
    ],
)
def test_groupby_invariant(available_mb, k_est, size_of_key_entry):
    m = GroupByMemoryModel()
    dc = _dc(k_est=k_est, size_of_key_entry=size_of_key_entry)
    available = available_mb * 1024 * 1024
    safety = 0.7
    params = m.solve(available, dc, safety=safety)
    assert params is not None, (
        f"unexpected None: available={available_mb}MB k_est={k_est}"
    )
    assert isinstance(params, GroupByParams)
    assert params.recurse_depth >= 0
    assert params.recurse_depth <= 5  # max_depth default
    est = m.estimate(params, dc)
    assert est <= int(available * safety), (
        f"estimate {est} > safety {int(available*safety)} "
        f"for k_est={k_est}"
    )


def test_groupby_recurse_depth_upper_bound():
    """K_est / P / p_recurse^d <= max_dict_keys — при d=recurse_depth."""
    m = GroupByMemoryModel()
    dc = _dc(k_est=1_000_000, size_of_key_entry=80, row_size=16)
    params = m.solve(10 * 1024 * 1024, dc, safety=0.7)
    assert params is not None
    k_per = dc.k_est / params.shuffle.num_partitions
    k_after = k_per / (params.p_recurse ** params.recurse_depth)
    assert k_after <= params.max_dict_keys, (
        f"after {params.recurse_depth} splits: "
        f"{k_after:.0f} > max_dict_keys={params.max_dict_keys}"
    )


def test_groupby_none_when_exceeds_max_depth():
    """K_est/P/p_recurse^MAX_DEPTH > max_dict_keys -> None."""
    m = GroupByMemoryModel(max_depth=2)
    dc = _dc(k_est=10**12, size_of_key_entry=80, row_size=16)
    params = m.solve(10 * 1024 * 1024, dc, safety=0.7)
    assert params is None


def test_groupby_invalid_ratios():
    with pytest.raises(ValueError, match="<= 1.0"):
        GroupByMemoryModel(partition_ratio=0.5, dict_ratio=0.4, c_temp_ratio=1.0)


def test_groupby_requires_k_est():
    m = GroupByMemoryModel()
    dc = _dc(k_est=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k_est"):
        m.solve(10 * 1024 * 1024, dc)


# ---------- composition check ----------

def test_groupby_estimate_is_max_of_parts():
    """f_groupby = max(f_shuffle, f_local); проверяем, что >= f_shuffle."""
    m = GroupByMemoryModel()
    dc = _dc(k_est=10_000, size_of_key_entry=80, row_size=16)
    params = m.solve(10 * 1024 * 1024, dc, safety=0.7)
    assert params is not None
    f_shuffle = m.shuffle_model.estimate(params.shuffle, dc)
    f_total = m.estimate(params, dc)
    assert f_total >= f_shuffle
EOF

# ---------- run ----------
echo "=== pytest tests/test_models_extended.py ==="
set +e
uv run pytest tests/test_models_extended.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "ERROR: test_models_extended.py failed (rc=$RC)"
  exit 1
fi

echo "=== pytest (fast) ==="
uv run pytest -q -m "not slow"

# ---------- commit ----------
git add -A
git commit -m "11-A-2: HashShuffle/Sort/GroupBy MemoryModel (§9.2.2-9.2.4); closed form; recurse_depth upper bound; 29 tests"

echo "==> sha256"
sha256sum cadw/core/models_shuffle.py cadw/core/models_sort.py cadw/core/models_groupby.py tests/test_models_extended.py
echo "==> 11-A-2 applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"