#!/usr/bin/env bash
# patch_11b2c2_planner_join: propagate_chars("join") в planner.py.
# Работает на той же ветке patch/11b2c1-join-model; squash с 2c-1 → v0.2.3.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

BR="patch/11b2c1-join-model"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11b2c2-${TS}"
git diff > ".backups/backup-pre-11b2c2-${TS}/wip.diff" || true

# ---------- planner.py: propagate_chars("join") ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/core/planner.py")
s = p.read_text()

# 1. Вставить case для "join" перед финальным raise.
old = '''    # Для hash_shuffle и прочих пропагатор не определён
    raise ValueError(
        f"propagate_chars: unsupported op {op.name!r}. "
        f"Supported: filter, with_column, groupby, sort, reduce."
    )'''

new = '''    if op.name == "join":
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

    # Для hash_shuffle и прочих пропагатор не определён
    raise ValueError(
        f"propagate_chars: unsupported op {op.name!r}. "
        f"Supported: filter, with_column, groupby, sort, reduce, join."
    )'''

if old not in s:
    raise SystemExit("propagate_chars final raise not found")
s = s.replace(old, new, 1)

# 2. Обновить docstring propagate_chars.
s = s.replace(
    '''    - sort: order = "sorted".
    - reduce: N' = 1, row_size' = state_bytes.
    """''',
    '''    - sort: order = "sorted".
    - reduce: N' = 1, row_size' = state_bytes.
    - join: N' = N_left × avg_matches, row_size' = output_row_size,
      k_est' = N' (upper bound); требует JoinParams в params.
    """''',
    1,
)

# 3. Обновить docstring plan_pipeline "Для каждой op" — упомянуть join.
s = s.replace(
    '''      1. model = get_model(op.name) — если op.name есть в реестре.
         Если нет ("filter", "with_column") — только propagate.''',
    '''      1. model = get_model(op.name) — если op.name есть в реестре
         (reduce, hash_shuffle, sort, groupby, join).
         Если нет ("filter", "with_column") — только propagate.''',
    1,
)

p.write_text(s)
print("planner.py: propagate_chars(join) добавлен")
PY

# ---------- tests/test_planner_join.py ----------
cat > tests/test_planner_join.py <<'EOF'
"""
Тесты propagate_chars("join") и plan_pipeline с join (11-B-2c-2).

Проверяем:
  - propagate_chars(join) корректно считает N', k_est', row_size';
  - k_est' = N' (upper bound, безопасно для downstream groupby);
  - требует JoinParams (не принимает None или чужой Params);
  - plan_pipeline: filter → join → groupby и join → join.
"""
import pyarrow as pa
import pytest

from cadw.core.models import DataChars, get_model
from cadw.core.models_join import JoinParams
from cadw.core.planner import OpSpec, plan_pipeline, propagate_chars


def _dc(
    n_rows: int = 10_000,
    k_est: int = 1000,
    row_size: int = 16,
) -> DataChars:
    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])
    return DataChars(
        schema=schema,
        n_rows=n_rows,
        row_size_bytes=row_size,
        k_est=k_est,
        avg_key_size_bytes=8,
        size_of_key_entry_bytes=80,
    )


def _solve_join(left, right, how="inner", available=10 * 1024 * 1024):
    model = get_model("join")
    return model.solve(
        available, left,
        op_context={"right_chars": right, "how": how},
    )


# ---------- propagate_chars(join) ----------

def test_propagate_join_basic():
    """N' = N_left × avg_matches; k_est = N' (upper bound)."""
    left = _dc(n_rows=10_000, k_est=1000, row_size=16)
    right = _dc(n_rows=10_000, k_est=1000, row_size=16)
    params = _solve_join(left, right)
    assert params is not None
    # avg_matches = 10_000 / 1000 = 10 → N_out = 10_000 × 10 = 100_000
    assert params.avg_matches == pytest.approx(10.0, abs=0.1)

    out = propagate_chars(
        OpSpec(name="join"), left, params,
    )
    assert out.n_rows == 100_000
    assert out.k_est == 100_000  # upper bound
    assert out.row_size_bytes == params.output_row_size
    assert out.order == "random"


def test_propagate_join_pk_fk():
    """PK-FK: left = FK (unique), right = PK (1000 unique) → avg_matches = 1."""
    left = _dc(n_rows=10_000, k_est=10_000, row_size=16)  # all unique
    right = _dc(n_rows=1000, k_est=1000, row_size=16)     # 1000 unique
    params = _solve_join(left, right)
    assert params is not None
    # output = 10_000 × 1000 / max(10_000, 1000) = 10_000
    # avg = 10_000 / 10_000 = 1
    assert params.avg_matches == pytest.approx(1.0, abs=0.1)

    out = propagate_chars(OpSpec(name="join"), left, params)
    assert out.n_rows == 10_000
    assert out.k_est == 10_000


def test_propagate_join_requires_join_params():
    left = _dc()
    with pytest.raises(ValueError, match="JoinParams"):
        propagate_chars(OpSpec(name="join"), left, None)


def test_propagate_join_rejects_wrong_params_type():
    """Не JoinParams (например, ReduceParams) -> ValueError."""
    left = _dc()
    right = _dc()
    # Создаём Params не того типа через фейковый класс
    class FakeParams:
        pass
    with pytest.raises(ValueError, match="JoinParams"):
        propagate_chars(OpSpec(name="join"), left, FakeParams())


def test_propagate_join_size_of_key_entry_includes_state():
    """size_of_key_entry' >= row_size' (запас на agg state)."""
    left = _dc(n_rows=10_000, k_est=1000)
    right = _dc(n_rows=10_000, k_est=1000)
    params = _solve_join(left, right)
    out = propagate_chars(OpSpec(name="join"), left, params)
    assert out.size_of_key_entry_bytes >= out.row_size_bytes
    assert out.avg_key_size_bytes == out.row_size_bytes


# ---------- plan_pipeline с join ----------

def test_pipeline_join_fits():
    """Одна join-операция вписывается в бюджет."""
    left = _dc(n_rows=10_000, k_est=10_000)  # unique FK
    right = _dc(n_rows=1000, k_est=1000)     # unique PK
    ops = [
        OpSpec(name="join", op_context={"right_chars": right, "how": "inner"}),
    ]
    plan = plan_pipeline(ops, left, 10 * 1024 * 1024, safety=0.7)
    assert plan.predicted_outcome == "fits"
    assert len(plan.steps) == 1
    assert plan.steps[0].params is not None
    assert isinstance(plan.steps[0].params, JoinParams)
    # N_out = N_left × avg_matches = 10_000 × 1 = 10_000
    assert plan.steps[0].output_chars.n_rows == 10_000


def test_pipeline_filter_join_groupby():
    """Цепочка filter → join → groupby планируется."""
    left = _dc(n_rows=10_000, k_est=10_000)   # unique FK
    right = _dc(n_rows=1000, k_est=1000)      # unique PK
    ops = [
        OpSpec(name="filter"),
        OpSpec(name="join", op_context={"right_chars": right, "how": "inner"}),
        OpSpec(name="groupby"),
    ]
    plan = plan_pipeline(ops, left, 10 * 1024 * 1024, safety=0.7)
    assert plan.predicted_outcome == "fits"
    assert len(plan.steps) == 3
    # filter: N не меняется
    assert plan.steps[0].output_chars.n_rows == 10_000
    # join: N' = 10_000 (avg_matches = 1), k_est = 10_000 (upper)
    assert plan.steps[1].output_chars.n_rows == 10_000
    assert plan.steps[1].output_chars.k_est == 10_000
    # groupby: N' = k_est = 10_000 (upper bound)
    assert plan.steps[2].output_chars.n_rows == 10_000


def test_pipeline_join_join():
    """Две последовательные join-операции планируются."""
    left = _dc(n_rows=10_000, k_est=10_000)
    right1 = _dc(n_rows=1000, k_est=1000)
    right2 = _dc(n_rows=1000, k_est=1000)
    ops = [
        OpSpec(name="join", op_context={"right_chars": right1, "how": "inner"}),
        OpSpec(name="join", op_context={"right_chars": right2, "how": "inner"}),
    ]
    plan = plan_pipeline(ops, left, 10 * 1024 * 1024, safety=0.7)
    assert plan.predicted_outcome == "fits"
    assert len(plan.steps) == 2
    # Оба join-шага имеют params
    for step in plan.steps:
        assert isinstance(step.params, JoinParams)


def test_pipeline_join_none_raises():
    """build_partition не влезает при любом P → PipelineMemoryError."""
    left = _dc(n_rows=10_000_000, k_est=10_000_000, row_size=200)
    right = _dc(n_rows=10_000_000, k_est=10_000_000, row_size=200)
    ops = [
        OpSpec(name="join", op_context={"right_chars": right, "how": "inner"}),
    ]
    from cadw.core.planner import PipelineMemoryError
    with pytest.raises(PipelineMemoryError):
        plan_pipeline(ops, left, 1024, safety=0.7)  # 1 KB — слишком мало
EOF

# ---------- DECISIONS: §11.1 + §13.4/42 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

# §11.1: добавить строки (2c-1 + 2c-2)
if "| `JoinMemoryModel` (v0.2.3: §9.2.5) |" not in s:
    s = s.replace(
        "| `hash_join` inner/left (v0.2.2: HashShuffle обеих сторон + pd.merge) | ✅ | 15 |",
        "| `hash_join` inner/left (v0.2.2: HashShuffle обеих сторон + pd.merge) | ✅ | 15 |\n"
        "| `JoinMemoryModel` (v0.2.3: §9.2.5) | ✅ | 17 |\n"
        "| `propagate_chars(\"join\")` (v0.2.3: planner chain с join) | ✅ | 9 |",
        1,
    )

# §12.6: v0.2 progress
s = s.replace(
    "§12.6. v0.2 (в работе: inner+left ✅; JoinMemoryModel, E2E — 11-B-2b/2c)",
    "§12.6. v0.2 (в работе: inner+left ✅, JoinMemoryModel ✅, propagate ✅; merge integration, E2E — 11-B-2c-3/2d)",
    1,
)

# §13.4 эпизод 42
marker_135 = "### 13.5. Открытые вопросы"
episode_42 = """**Эпизод 42 (planner vs runtime: expected vs cartesian).** Модель
`JoinMemoryModel.estimate` считает пик по **ожидаемому** выходу:
`probe_batch × avg_matches × row_size_out × (1+c_merge)`. Runtime `hash_join`
проверяет `len(build_df) × len(probe_df) > max_output_rows` — это
**cartesian worst case** (все ключи совпадают). Расхождение: при
`build=250`, `probe=2187`, `avg_matches=10` модель предсказывает output
~21 870 (влезает), runtime проверяет 250 × 2187 = 546 750 (не влезает).
**Симптом:** `plan_pipeline` говорит `"fits"`, а `merge` поднимает
`MemoryLimitExceeded`. **Причины:** модель оптимистичнее (правильно для
типичных данных), runtime консервативнее (правильно для защиты от
skew). **Статус:** 11-B-2c-3 — согласовать: либо runtime использует
`max_output_rows` как expected (а cartesian вынести в отдельный лимит
`max_cartesian_rows`), либо модель использует cartesian как upper bound
(over-conservative, отвергает валидные join'ы). Мой выбор — первое:
runtime проверяет `len(build) × len(probe) > max_cartesian_rows` (с
большим запасом, `c_cartesian ≈ 10×` от expected) и `expected_output >
max_output_rows`. Это разделяет «защиту от OOM» (expected) и «защиту от
skew» (cartesian). **Урок:** модель предсказывает **ожидаемый** пик,
runtime защищает от **worst case**. Это два разных числа, их нельзя
смешивать в одном параметре.

"""
if "Эпизод 42" not in s:
    s = s.replace(marker_135, episode_42 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §11.1, §12.6, §13.4/42")
PY

# ---------- CHANGELOG: v0.2.3 (2c-1 + 2c-2) ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.2.3] — 2026-10-19

### Добавлено
- `JoinMemoryModel` (§9.2.5): `f_join = build_partition + probe_batch +
  output_batch × (1 + c_merge)`.
  - `solve()`: `P` = max от shuffle обеих сторон; `build_side` по `how`
    (right для left join; меньший state для inner);
    `max_output_rows` из `output_budget / (row_size_out × (1+c_merge))`
    вместо ad-hoc `0.3 × available / 256`;
    `avg_matches` = upper bound `N_L × N_R / max(K_L, K_R)` (floor 1).
  - `JoinParams`: `build_side`, `how`, `num_partitions`,
    `probe_batch_size`, `build_partition_bytes`, `max_output_rows`,
    `right_chars`, `probe_row_size`, `output_row_size`, `avg_matches`.
- `propagate_chars("join")`: `N' = N_left × avg_matches`,
  `row_size' = output_row_size`, `k_est' = N'` (upper bound),
  `size_of_key_entry' = row_size' + 80`, `order = "random"`.
  Требует `JoinParams` в `params` (из `get_model("join").solve()`).
- `plan_pipeline` теперь планирует цепочки с join
  (`filter → join → groupby`, `join → join`).

### Открыто (11-B-2c-3)
- Модель считает **expected** output, runtime — **cartesian worst case**.
  `plan_pipeline` может предсказать `"fits"`, а `merge` — упасть.
  Fix: разделить `max_output_rows` (expected) и `max_cartesian_rows`
  (worst case, ~10× larger). §13.4 эпизод 42.

"""
if "## [0.2.3]" not in s:
    if "## [0.2.2]" in s:
        s = s.replace("## [0.2.2]", entry + "## [0.2.2]", 1)
    else:
        s = s.replace("## [0.2.1]", entry + "## [0.2.1]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.2.3 (2c-1 + 2c-2)")
PY

# ---------- run ----------
echo "=== pytest tests/test_planner_join.py -v ==="
set +e
uv run pytest tests/test_planner_join.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "test_planner_join.py: rc=$RC"
  git add -A
  git commit -m "WIP: 11-B-2c-2 (test red; fix follows)" || true
  exit 1
fi

echo "=== full fast + slow ==="
uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-B-2c-2: propagate_chars(join) + 9 tests; §13.4/42 (planner expected vs runtime cartesian)"

echo "==> 11-B-2c-2 applied on ${BR}"
echo "    Next: squash 2c-1 + 2c-2 → tag v0.2.3"