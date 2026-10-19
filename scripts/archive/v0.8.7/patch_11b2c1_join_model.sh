#!/usr/bin/env bash
# patch_11b2c1_join_model: JoinMemoryModel (§9.2.5) + JoinParams + tests.
# Не трогает merge/planner — это 11-B-2c-2/3.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-11b2c1-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_11b2c1"

BR="patch/11b2c1-join-model"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- cadw/core/models_join.py ----------
cat > cadw/core/models_join.py <<'PY'
"""
JoinMemoryModel (§9.2.5).

Grace hash join: build = меньшая (inner) или right (left), probe = другая.

    f_join = build_partition_bytes              # materialized build per partition
           + probe_batch_bytes                  # batch_size × row_size_probe
           + output_batch_bytes                 # batch_size × avg_matches × row_size_output
           + c_merge × output_batch_bytes       # pandas temp in pd.merge

`build_partition_bytes` — state, сохраняется на всё время обработки пары
партиций. `probe_batch_bytes` + `output_batch_bytes` — пик на один шаг
`pd.merge`; суммируются с build.

`solve(available, left_chars, op_context={"right_chars": ..., "how": ...})`:
  - `P` = max(shuffle.solve(left), shuffle.solve(right))
  - `build_side` = "right" если how=="left", иначе "left"/"right" (меньший state)
  - budget split: build_ratio=0.4, probe_ratio=0.3, output_ratio=0.3
  - `max_output_rows` = output_budget / (row_size_out × (1 + c_merge))
  - `probe_batch_size` = min(probe_budget/row_size_probe,
                            output_budget / (avg_matches × row_size_out × (1+c)))
  - None если build_partition > build_budget при любом P
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cadw.core.models import (
    DataChars,
    OpMemoryModel,
    Params,
    get_model,
    register_model,
)
from cadw.core.models_shuffle import HashShuffleMemoryModel


@dataclass
class JoinParams(Params):
    build_side: str         # "left" | "right"
    how: str                # "inner" | "left"
    num_partitions: int
    probe_batch_size: int
    build_partition_bytes: int
    max_output_rows: int
    right_chars: DataChars
    probe_row_size: int
    output_row_size: int
    avg_matches: float


class JoinMemoryModel(OpMemoryModel):
    op_name = "join"

    def __init__(
        self,
        shuffle_model: Optional[HashShuffleMemoryModel] = None,
        *,
        build_ratio: float = 0.4,       # CALIBRATION DEFAULT (§9.5)
        probe_ratio: float = 0.3,       # CALIBRATION DEFAULT
        output_ratio: float = 0.3,      # CALIBRATION DEFAULT
        c_merge_ratio: float = 2.0,     # pandas temp ≈ 2 × output
    ) -> None:
        self.shuffle_model = shuffle_model or get_model("hash_shuffle")
        if not isinstance(self.shuffle_model, HashShuffleMemoryModel):
            raise TypeError("shuffle_model must be HashShuffleMemoryModel")
        for name, r in (
            ("build_ratio", build_ratio),
            ("probe_ratio", probe_ratio),
            ("output_ratio", output_ratio),
        ):
            if not (0 < r < 1):
                raise ValueError(f"{name} must be in (0,1), got {r}")
        if build_ratio + probe_ratio + output_ratio > 1.0:
            raise ValueError(
                "build_ratio + probe_ratio + output_ratio must be <= 1.0; "
                f"got {build_ratio} + {probe_ratio} + {output_ratio}"
            )
        if c_merge_ratio < 0:
            raise ValueError("c_merge_ratio must be >= 0")
        self.build_ratio = float(build_ratio)
        self.probe_ratio = float(probe_ratio)
        self.output_ratio = float(output_ratio)
        self.c_merge_ratio = float(c_merge_ratio)

    def estimate(self, params: JoinParams, data_chars: DataChars) -> int:
        """Пик RAM: build_partition + probe_batch + output_batch×(1+c_merge)."""
        if not isinstance(params, JoinParams):
            raise TypeError(f"expected JoinParams, got {type(params).__name__}")
        return int(
            params.build_partition_bytes
            + params.probe_batch_size * params.probe_row_size
            + params.probe_batch_size
            * params.avg_matches
            * params.output_row_size
            * (1.0 + self.c_merge_ratio)
        )

    def solve(
        self,
        available_bytes: int,
        data_chars: DataChars,
        *,
        safety: float = 0.7,
        op_context: Optional[dict] = None,
    ) -> Optional[JoinParams]:
        if op_context is None or "right_chars" not in op_context:
            raise ValueError(
                "JoinMemoryModel.solve: op_context['right_chars'] required"
            )
        right_chars: DataChars = op_context["right_chars"]
        how: str = op_context.get("how", "inner")
        if how not in ("inner", "left"):
            raise ValueError(
                f"JoinMemoryModel.solve: how must be 'inner'|'left', got {how!r}"
            )
        left_chars = data_chars

        # P: max от shuffle для обеих сторон
        left_shuffle = self.shuffle_model.solve(
            available_bytes, left_chars, safety=safety
        )
        right_shuffle = self.shuffle_model.solve(
            available_bytes, right_chars, safety=safety
        )
        if left_shuffle is None or right_shuffle is None:
            return None
        P = max(left_shuffle.num_partitions, right_shuffle.num_partitions)
        if P < 2:
            return None

        # build_side
        if how == "left":
            build_side = "right"  # left preserved → right = build
        else:  # inner
            left_size = (left_chars.k_est or left_chars.n_rows) * left_chars.row_size_bytes
            right_size = (right_chars.k_est or right_chars.n_rows) * right_chars.row_size_bytes
            build_side = "left" if left_size <= right_size else "right"

        build_chars = left_chars if build_side == "left" else right_chars
        probe_chars = right_chars if build_side == "left" else left_chars

        budget = int(available_bytes * safety)
        build_budget = int(budget * self.build_ratio)
        probe_budget = int(budget * self.probe_ratio)
        output_budget = int(budget * self.output_ratio)

        # build_partition per partition
        build_k = build_chars.k_est or build_chars.n_rows
        build_partition_bytes = (build_k / P) * build_chars.row_size_bytes
        if build_partition_bytes > build_budget:
            return None

        # avg_matches (upper bound)
        k_left = left_chars.k_est or left_chars.n_rows
        k_right = right_chars.k_est or right_chars.n_rows
        if k_left > 0 and k_right > 0 and left_chars.n_rows > 0:
            output_upper = (
                left_chars.n_rows * right_chars.n_rows / max(k_left, k_right)
            )
            avg_matches = max(1.0, output_upper / left_chars.n_rows)
        else:
            avg_matches = 1.0

        # row_size_output: консервативно = probe_row + build_row
        # (реально = probe_rest + build_rest + on_keys, но probe_row и
        # build_row включают on-keys, что даёт верхнюю оценку).
        row_size_out = probe_chars.row_size_bytes + build_chars.row_size_bytes

        # probe_batch: из двух ограничений
        probe_batch_by_probe = max(
            1, probe_budget // max(1, probe_chars.row_size_bytes)
        )
        denom = avg_matches * row_size_out * (1.0 + self.c_merge_ratio)
        probe_batch_by_output = max(1, int(output_budget / max(1.0, denom)))
        probe_batch_size = min(probe_batch_by_probe, probe_batch_by_output)

        # max_output_rows: картезиан-стопор (upper bound actual output).
        # worst-case pd.merge output = |build| × |probe| строк.
        # Допускаем не более max_output_rows при пике output_budget.
        max_output_rows = max(
            1, int(output_budget / max(1.0, row_size_out * (1.0 + self.c_merge_ratio)))
        )

        params = JoinParams(
            build_side=build_side,
            how=how,
            num_partitions=int(P),
            probe_batch_size=int(probe_batch_size),
            build_partition_bytes=int(build_partition_bytes),
            max_output_rows=int(max_output_rows),
            right_chars=right_chars,
            probe_row_size=int(probe_chars.row_size_bytes),
            output_row_size=int(row_size_out),
            avg_matches=float(avg_matches),
        )

        # Инвариант: estimate <= budget. Консервативные округления могут
        # дать крошечное превышение; уменьшаем probe_batch_size.
        if self.estimate(params, left_chars) > budget:
            params.probe_batch_size -= 1
            if params.probe_batch_size < 1:
                return None
            if self.estimate(params, left_chars) > budget:
                return None
        return params

    def describe(self) -> str:
        return (
            "JoinMemoryModel: "
            "f = build_partition + probe_batch + output_batch*(1+c_merge); "
            f"build_ratio={self.build_ratio:.2f}, "
            f"probe_ratio={self.probe_ratio:.2f}, "
            f"output_ratio={self.output_ratio:.2f}, "
            f"c_merge={self.c_merge_ratio:.2f}"
        )


register_model(JoinMemoryModel())
PY

# ---------- cadw/core/__init__.py: регистрация ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/core/__init__.py")
s = p.read_text()
if "from .models_join import JoinMemoryModel, JoinParams" not in s:
    s = s.replace(
        "from .models_groupby import GroupByMemoryModel, GroupByParams",
        "from .models_groupby import GroupByMemoryModel, GroupByParams\n"
        "from .models_join import JoinMemoryModel, JoinParams",
        1,
    )
    s = s.replace(
        '    "GroupByMemoryModel",\n    "GroupByParams",\n',
        '    "GroupByMemoryModel",\n    "GroupByParams",\n'
        '    "JoinMemoryModel",\n    "JoinParams",\n',
        1,
    )
    p.write_text(s)
    print("cadw/core/__init__.py: JoinMemoryModel, JoinParams exported")
else:
    print("cadw/core/__init__.py: already exported")
PY

# ---------- tests/test_models_join.py ----------
cat > tests/test_models_join.py <<'EOF'
"""Тесты JoinMemoryModel (11-B-2c-1, §9.2.5)."""
import pyarrow as pa
import pytest

from cadw.core.models import DataChars, get_model, list_models
from cadw.core.models_join import JoinMemoryModel, JoinParams


def _chars(
    n_rows: int = 100_000,
    k_est: int = 1000,
    row_size: int = 32,
    cols: tuple = ("k", "v"),
) -> DataChars:
    schema = pa.schema([(c, pa.int64()) for c in cols])
    return DataChars(
        schema=schema,
        n_rows=n_rows,
        row_size_bytes=row_size,
        k_est=k_est,
        avg_key_size_bytes=8,
        size_of_key_entry_bytes=80,
    )


# ---------- registry ----------

def test_join_model_registered():
    m = get_model("join")
    assert isinstance(m, JoinMemoryModel)
    assert "join" in list_models()


# ---------- invariant: estimate(solve) <= budget ----------

@pytest.mark.parametrize(
    "avail_mb, how, left_k, right_k",
    [
        (10, "inner", 100, 100),
        (10, "inner", 10_000, 1_000),
        (10, "inner", 100_000, 10_000),
        (10, "left", 100, 100),
        (100, "inner", 1_000_000, 100_000),
        (100, "left", 1_000_000, 100_000),
    ],
)
def test_invariant(avail_mb, how, left_k, right_k):
    m = JoinMemoryModel()
    left = _chars(n_rows=left_k * 10, k_est=left_k)
    right = _chars(n_rows=right_k * 10, k_est=right_k)
    available = avail_mb * 1024 * 1024
    safety = 0.7
    params = m.solve(
        available, left, safety=safety,
        op_context={"right_chars": right, "how": how},
    )
    assert params is not None, (
        f"unexpected None: avail={avail_mb}MB how={how} "
        f"left_k={left_k} right_k={right_k}"
    )
    assert isinstance(params, JoinParams)
    assert params.how == how
    assert params.num_partitions >= 2
    est = m.estimate(params, left)
    assert est <= int(available * safety), (
        f"estimate {est} > safety {int(available*safety)} "
        f"for avail={avail_mb}MB how={how}"
    )


# ---------- build_side ----------

def test_build_side_right_for_left_join():
    m = JoinMemoryModel()
    left = _chars(n_rows=10_000, k_est=100)
    right = _chars(n_rows=10_000, k_est=100)
    params = m.solve(
        10 * 1024 * 1024, left,
        op_context={"right_chars": right, "how": "left"},
    )
    assert params is not None
    assert params.build_side == "right"


def test_build_side_smaller_for_inner():
    m = JoinMemoryModel()
    left_big = _chars(n_rows=1_000_000, k_est=1_000_000, row_size=32)
    right_small = _chars(n_rows=100, k_est=10, row_size=16)
    params = m.solve(
        10 * 1024 * 1024, left_big,
        op_context={"right_chars": right_small, "how": "inner"},
    )
    assert params is not None
    assert params.build_side == "right"  # right smaller


def test_build_side_left_when_left_smaller():
    m = JoinMemoryModel()
    left_small = _chars(n_rows=100, k_est=10, row_size=16)
    right_big = _chars(n_rows=1_000_000, k_est=1_000_000, row_size=32)
    params = m.solve(
        10 * 1024 * 1024, left_small,
        op_context={"right_chars": right_big, "how": "inner"},
    )
    assert params is not None
    assert params.build_side == "left"


# ---------- max_output_rows из бюджета ----------

def test_max_output_rows_from_budget():
    """max_output_rows × row_size_out × (1+c_merge) <= output_budget."""
    m = JoinMemoryModel()
    left = _chars(n_rows=100_000, k_est=1000)
    right = _chars(n_rows=100_000, k_est=1000)
    available = 10 * 1024 * 1024
    safety = 0.7
    params = m.solve(
        available, left, safety=safety,
        op_context={"right_chars": right, "how": "inner"},
    )
    assert params is not None
    output_budget = int(available * safety * m.output_ratio)
    worst = (
        params.max_output_rows
        * params.output_row_size
        * (1 + m.c_merge_ratio)
    )
    assert worst <= output_budget + 1000  # допуск на округление вниз


# ---------- None когда build не влезает ----------

def test_none_when_build_partition_too_big():
    """Огромный build (K × row_size / P) > build_budget → None."""
    m = JoinMemoryModel()
    # left огромен (P будет большой, но K/P всё равно большой)
    left = _chars(n_rows=10_000_000, k_est=10_000_000, row_size=200)
    right = _chars(n_rows=10_000_000, k_est=10_000_000, row_size=200)
    params = m.solve(
        1024, left,  # 1 KB — заведомо мало
        op_context={"right_chars": right, "how": "inner"},
    )
    assert params is None


# ---------- requires right_chars ----------

def test_solve_requires_right_chars():
    m = JoinMemoryModel()
    left = _chars()
    with pytest.raises(ValueError, match="right_chars"):
        m.solve(10 * 1024 * 1024, left, op_context=None)
    with pytest.raises(ValueError, match="right_chars"):
        m.solve(10 * 1024 * 1024, left, op_context={})


def test_solve_invalid_how():
    m = JoinMemoryModel()
    left = _chars()
    right = _chars()
    with pytest.raises(ValueError, match="how must be"):
        m.solve(
            10 * 1024 * 1024, left,
            op_context={"right_chars": right, "how": "outer"},
        )


# ---------- type check ----------

def test_estimate_type_check():
    m = JoinMemoryModel()
    left = _chars()
    with pytest.raises(TypeError, match="JoinParams"):
        m.estimate("not a params", left)


# ---------- describe ----------

def test_describe():
    m = JoinMemoryModel()
    d = m.describe()
    assert "JoinMemoryModel" in d
    assert "build_ratio" in d


# ---------- invalid ratios ----------

def test_invalid_ratios():
    with pytest.raises(ValueError, match="must be in"):
        JoinMemoryModel(build_ratio=0)
    with pytest.raises(ValueError, match="must be in"):
        JoinMemoryModel(probe_ratio=1.5)
    with pytest.raises(ValueError, match="<= 1.0"):
        JoinMemoryModel(
            build_ratio=0.5, probe_ratio=0.4, output_ratio=0.4,
        )
EOF

# ---------- run ----------
echo "=== pytest tests/test_models_join.py -v ==="
set +e
uv run pytest tests/test_models_join.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "test_models_join.py: rc=$RC"
  git add -A
  git commit -m "WIP: 11-B-2c-1 (test red; fix follows)" || true
  exit 1
fi

echo "=== full fast + slow ==="
uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-B-2c-1: JoinMemoryModel (§9.2.5) + JoinParams + 14 tests"

echo "==> 11-B-2c-1 applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"