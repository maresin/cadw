#!/usr/bin/env bash
# patch_10_materialize_audit: must-fix «materialize before check_state».
#   1. _local_groupby: stat().st_size ДО чтения; helper _read_partition_safe.
#   2. reduce: estimate_state_size + check_state ДО op.init().
#   3. AST guards: list(reader) forbidden вне helper.
#   4. OOM E2E (slow): 10^6 групп, skew, cov(d=5000) под малым бюджетом.
#   5. DECISIONS §3.10, §11.4 (класс багов), §13.4/27-28.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-10-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_10 (materialize audit)"

BR="patch/10-materialize-audit"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- 1. groupby.py: helper + fix _local_groupby ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/groupby.py")
s = p.read_text()

# 1a. helper _read_partition_safe перед _local_groupby
if "def _read_partition_safe(" not in s:
    helper = '''def _read_partition_safe(
    partition_path: Path,
    memory_guard: "MemoryGuard | None",
) -> "pa.Table | None":
    """Прочитать всю партицию в память, только если она влезает в бюджет.

    Returns None если partition size > available*0.5 — caller должен
    рекурсивно разбить партицию (см. `_local_groupby`).

    ВАЖНО: pattern `list(ArrowIPCBatchReader(...))` разрешён ТОЛЬКО здесь.
    Любое materialize целой партиции в другом месте — must-fix
    (materialize-before-check, §11.4). Enforced AST guard'ом
    `test_materialize_guard.py::test_no_list_of_whole_reader_outside_helper`.

    Arrow IPC без сжатия: file size ≈ in-memory nbytes. Для сжатия
    (если появится) — увеличить safety или читать по-другому.
    """
    if memory_guard is not None:
        size = partition_path.stat().st_size
        limit = int(memory_guard.available_bytes() * 0.5)
        if size > limit:
            return None

    batches = list(ArrowIPCBatchReader(str(partition_path), None))
    if not batches:
        return pa.Table.from_batches([], schema=pa.schema([]))
    table = pa.Table.from_batches(batches)
    if memory_guard is not None:
        memory_guard.check_state(
            table.nbytes,
            name=f"read_partition_safe({partition_path.name})",
        )
    return table


'''
    anchor = "def _local_groupby("
    if anchor not in s:
        raise SystemExit("_local_groupby def not found")
    s = s.replace(anchor, helper + anchor, 1)

# 1b. заменить блок чтения в _local_groupby
old = '''    # 0. Пустая партиция
    batches = list(ArrowIPCBatchReader(str(partition_path), None))
    if not batches:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )
    table = pa.Table.from_batches(batches)
    if table.num_rows == 0:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )

    # 1. Размер партиции = сумма nbytes уже прочитанных батчей
    total_bytes = sum(b.nbytes for b in batches)
    safety_limit = int(memory_guard.available_bytes() * 0.5) if memory_guard else total_bytes

    # 2. Влезает — материализуем и агрегируем
    if total_bytes <= safety_limit:
        agg_specs = [(col, f) for col, f in zip(agg_columns, agg_funcs)]
        return table.group_by(list(by_columns)).aggregate(agg_specs)'''

new = '''    # 1. Оценка размера партиции ДО чтения (must-fix, §11.4).
    #    Arrow IPC без сжатия: file size ≈ in-memory nbytes. Читать весь
    #    файл в list() до проверки — OOM для партиций > бюджета.
    partition_bytes = (
        partition_path.stat().st_size if memory_guard is not None else 0
    )
    safety_limit = (
        int(memory_guard.available_bytes() * 0.5)
        if memory_guard is not None
        else partition_bytes
    )

    # 2. Влезает — материализуем через безопасный helper
    if partition_bytes <= safety_limit:
        table = _read_partition_safe(partition_path, memory_guard)
        if table is None or table.num_rows == 0:
            return pa.Table.from_pylist(
                [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
            )
        agg_specs = [(col, f) for col, f in zip(agg_columns, agg_funcs)]
        return table.group_by(list(by_columns)).aggregate(agg_specs)'''

if old not in s:
    raise SystemExit("_local_groupby read block: pattern not found")
s = s.replace(old, new, 1)

p.write_text(s)
print("groupby.py: _read_partition_safe + _local_groupby (stat before read)")
PY

# ---------- 2. reduce.py: estimate_state_size before init ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/reduce.py")
s = p.read_text()

old = '''    state = op.init()
    batch_id = 0
    for batch in batch_iter:
        memory_guard.check_batch(batch, name=f"{type(op).__name__}.batch")'''

new = '''    # Pre-flight: оценка размера state ДО op.init() (must-fix, §11.4).
    # CovarianceOp(d) = O(d²) — np.zeros((d,d)) до check_state → OOM для
    # d=10^5 (80 GB). HistogramOp(bins) — np.zeros(bins) — то же.
    # estimate_state_size знает про CovarianceOp (8+8d+8d²) и
    # HistogramOp (8*bins+16); для остальных — консервативный fallback.
    _estimated = memory_guard.estimate_state_size(
        type(op).__name__,
        schema=None,
        d=getattr(op, "d", None),
        bins=getattr(op, "bins", 255),
    )
    memory_guard.check_state(_estimated, name=f"{type(op).__name__}.estimated")

    state = op.init()
    batch_id = 0
    for batch in batch_iter:
        memory_guard.check_batch(batch, name=f"{type(op).__name__}.batch")'''

if old not in s:
    raise SystemExit("reduce: state=op.init() pattern not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("reduce.py: estimate_state_size before op.init()")
PY

# ---------- 3. AST guards ----------
cat > tests/test_materialize_guard.py <<'EOF'
"""
AST guards: «materialize-before-check» — must-fix class of bugs (§11.4).

Правило (§3.10): любая операция, читающая весь источник в память
(`list(reader)`, `Table.from_batches(all_batches)`), обязана
предварительно проверить размер. Исключение — materialize, ограниченный
`batch_size` (один батч, уже проверен `check_batch`).
"""
import ast
import inspect
from pathlib import Path

import pytest

OPS_DIR = Path(__file__).parent.parent / "cadw" / "ops"
FORBIDDEN_READERS = {"ArrowIPCBatchReader", "ParquetBatchReader"}
ALLOWED_FUNCTIONS = {"_read_partition_safe"}


def _iter_py_files(root: Path):
    yield from root.rglob("*.py")


def test_no_list_of_whole_reader_outside_helper():
    """
    `list(<Reader>(...))` разрешён ТОЛЬКО в `_read_partition_safe`.

    Причина: helper делает `partition_path.stat().st_size` ДО `list()`,
    поэтому materialize происходит только если партиция влезает в бюджет.
    В любом другом месте — это materialize-before-check, OOM-риск.
    """
    violations: list[str] = []
    for py in _iter_py_files(OPS_DIR):
        tree = ast.parse(py.read_text())
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            if fn.name in ALLOWED_FUNCTIONS:
                continue
            for node in ast.walk(fn):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "list"
                ):
                    continue
                if not node.args:
                    continue
                inner = node.args[0]
                if not isinstance(inner, ast.Call):
                    continue
                if not (isinstance(inner.func, ast.Name) and inner.func.id in FORBIDDEN_READERS):
                    continue
                violations.append(
                    f"{py.relative_to(OPS_DIR.parent.parent)}:{node.lineno}: "
                    f"list({inner.func.id}(...)) in function {fn.name!r}. "
                    f"Use _read_partition_safe() or partition_path.stat().st_size first."
                )
    assert not violations, (
        "materialize-before-check violations (§11.4 / §3.10):\n" + "\n".join(violations)
    )


def test_reduce_estimate_state_before_init():
    """reduce() должен вызвать estimate_state_size/check_state ДО op.init()."""
    from cadw.ops.reduce import reduce

    src = inspect.getsource(reduce)
    est_pos = src.find("estimate_state_size")
    init_pos = src.find("op.init()")
    check_pos = src.find("check_state")

    assert est_pos >= 0, "estimate_state_size not called in reduce()"
    assert init_pos >= 0, "op.init() not called in reduce()"
    assert check_pos >= 0, "check_state not called in reduce()"
    assert est_pos < init_pos, (
        f"reduce: estimate_state_size (pos {est_pos}) must come BEFORE "
        f"op.init() (pos {init_pos}); иначе cov(d²)/histogram(bins) OOM до check"
    )
    assert check_pos < init_pos, (
        f"reduce: check_state (pos {check_pos}) must come BEFORE "
        f"op.init() (pos {init_pos})"
    )
EOF

# ---------- 4. OOM E2E (slow) ----------
cat > tests/test_oom_resistance.py <<'EOF'
"""
OOM resistance: 10^6 строк под малым бюджетом.

Проверяем, что operations либо успешно завершаются (за счёт shuffle/скью),
либо поднимают MemoryLimitExceeded ДО OOM. OOM = падение процесса; тут
проверяем, что процесс выживает.
"""
import numpy as np
import pyarrow as pa
import pytest

from cadw import MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ops.groupby import groupby_aggregate
from cadw.ops.reduce import CovarianceOp, reduce


def make_iter(table: pa.Table, batch_size: int):
    for i in range(0, table.num_rows, batch_size):
        yield table.slice(i, batch_size).to_batches()[0]


# ---------- groupby: 10^6 групп под 10 MB ----------

@pytest.mark.slow
def test_groupby_many_groups_10_6_under_10mb(tmp_path):
    """10^6 уникальных групп, 10 MB budget. Успех или MemoryLimitExceeded, не OOM."""
    n = 1_000_000
    table = pa.table({
        "k": [f"g{i:07d}" for i in range(n)],
        "v": [1.0] * n,
    })
    guard = MemoryGuard(max_memory_mb=10)
    try:
        result = groupby_aggregate(
            make_iter(table, 10_000), ["k"], ["v"], ["count"],
            guard, tmp_path, num_partitions=64, log=False,
        )
        # Если успех — 10^6 строк, count=1 каждая
        assert result.num_rows == n
        for r in result.to_pylist()[:10]:
            assert r["v_count"] == 1
    except MemoryLimitExceeded:
        # Допустимо: budget слишком мал даже для P=64. Главное — не OOM.
        pass


# ---------- groupby: skew 90/10 под 10 MB ----------

@pytest.mark.slow
def test_groupby_skew_10_6_under_10mb(tmp_path):
    """90/10 skew, 10^6 строк, 10 MB. Skew isolation должна справиться."""
    n = 1_000_000
    heavy = n * 9 // 10
    table = pa.table({
        "k": ["A"] * heavy + ["B"] * (n - heavy),
        "v": [1.0] * n,
    })
    guard = MemoryGuard(max_memory_mb=10)
    result = groupby_aggregate(
        make_iter(table, 10_000), ["k"], ["v"], ["count"],
        guard, tmp_path, log=False,
    )
    rows = {r["k"]: r["v_count"] for r in result.to_pylist()}
    assert rows == {"A": heavy, "B": n - heavy}


# ---------- covariance: d=5000 под 10 MB ----------

@pytest.mark.slow
def test_covariance_oom_resistance(tmp_path):
    """
    CovarianceOp(d=5000) → d²·8 = 200 MB state. Budget 10 MB →
    MemoryLimitExceeded ДО np.zeros((d,d)). Не OOM.

    До патча 10: np.zeros((5000, 5000)) = 200 MB аллоцируется в op.init()
    до первого check_state. На budget 10 MB — проверка срабатывала бы
    после аллокации (но аллокация 200 MB не OOM сама по себе на машине
    с 8-16 GB; для демонстрации — d=50000 = 20 GB, но 50000 столбцов
    в RecordBatch медленно собирать; используем d=5000 как proxy —
    raise должен произойти ДО init()).
    """
    d = 5000
    cols = [f"c{i}" for i in range(d)]
    # tiny batch, d columns
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1.0]) for _ in cols], names=cols,
    )
    guard = MemoryGuard(max_memory_mb=10)
    op = CovarianceOp(cols)
    with pytest.raises(MemoryLimitExceeded):
        reduce(iter([batch]), op, guard, log=False)
EOF

# ---------- 5. DECISIONS.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

# 5.1 §3: новый принцип 3.10
if "10. **Check before materialize.**" not in s:
    s = s.replace(
        "9. **Никаких ML-адаптеров.**",
        "10. **Check before materialize.** Любая операция, читающая весь\n"
        "    источник в память (`list(reader)`, `Table.from_batches(all)`,\n"
        "    `to_pylist()` на полном датасете), обязана предварительно\n"
        "    вызвать `check_state(estimated)` или\n"
        "    `partition_path.stat().st_size`-проверку. Исключение —\n"
        "    materialize, ограниченный `batch_size` (один батч, уже\n"
        "    проверен `check_batch`). Нарушение — must-fix. Enforced\n"
        "    AST guard'ом `test_no_list_of_whole_reader_outside_helper`.\n"
        "    Также: `op.init()` может аллоцировать большой state\n"
        "    (`CovarianceOp(d²)` = 80 GB при d=10^5); `reduce` вызывает\n"
        "    `estimate_state_size` + `check_state` **до** `op.init()`.\n"
        "9. **Никаких ML-адаптеров.**",
        1,
    )

# 5.2 §11.4: добавить класс багов (после существующей регрессии)
old_regression = '''**Регрессия (обнаружено в патче 08, must-fix, не закрыто):** в
`_local_groupby` вызов `list(ArrowIPCBatchReader(...))` материализует
всю партицию **до** `check_state`. Для партиции ~500 MB — OOM до
срабатывания проверки. Это **не** закрывает пункт 1 из 02b1
(«materialize до check»), а воспроизводит его на другом уровне.
Настоящий streaming: `partition_path.stat().st_size` (Arrow IPC без
сжатия ≈ in-memory) → если > `available*0.5` — рекурсия без чтения.
**Статус: must-fix, патч 10. После 10 — аудит всех примитивов на
паттерн «materialize до check_state».**'''

new_regression = '''**Класс багов «materialize-before-check» (обнаружено 08, must-fix,
патч 10):** операция читает весь источник в память **до**
`check_state`/`check_batch`. Обнаружено в двух местах:

- **`_local_groupby` (groupby.py)**: `list(ArrowIPCBatchReader(...))` до
  size-check. 02b1 добавил `sum(b.nbytes)`, но **после** materialize —
  это не закрыло баг, а лишь добавило проверку после OOM. Fix (патч 10):
  `partition_path.stat().st_size` до чтения + helper
  `_read_partition_safe` (единственное место, где разрешён `list(reader)`).
- **`reduce` (reduce.py)**: `state = op.init()` до `estimate_state_size`.
  `CovarianceOp(d=10^5)` → `np.zeros((10^5,10^5))` = 80 GB → OOM
  **внутри** `init()`, до первого `check_state`. `HistogramOp(bins=10^9)`
  — то же. Fix (патч 10): `estimate_state_size` + `check_state` перед
  `op.init()`.

**Патч 10:** оба fixa + AST guards (`test_materialize_guard.py`) +
OOM E2E (`test_oom_resistance.py`: 10^6 групп, skew, cov(d=5000) под
10 MB) + ревизия `test_mustfix_local_groupby.py`. **Статус: закрыто
после патча 10.** Класс багов остаётся в §3.10 как принцип для будущих
операций.'''
s = s.replace(old_regression, new_regression, 1)

# 5.3 §13.4: эпизоды 27 и 28
episodes = """**Эпизод 27 (must-fix, класс).** 02b1 «закрыл» must-fix п.1
(`_local_groupby` без check) через `sum(b.nbytes for b in batches)`. Но
проверка была **после** `list(ArrowIPCBatchReader(...))` — то есть после
OOM. Тест `test_local_groupby_recursion_called` проверял, что
`hash_shuffle` вызывается, но **не** проверял, что `list(reader)` не
вызывается до него. **Урок:** «закрытие must-fix» требует теста, который
падает при **неправильном порядке** операций, а не только при
отсутствии проверки. AST guard `test_no_list_of_whole_reader_outside_helper`
+ `test_reduce_estimate_state_before_init` фиксируют **порядок**
(`estimate_state_size` до `init`, `stat` до `list`), а не только наличие
вызовов.

**Эпизод 28 (обнаружено аудитом).** `reduce` вызывает `op.init()`
**до** `estimate_state_size`/`check_state`. Для `CovarianceOp(d=10^5)`
`op.init()` = `np.zeros((10^5, 10^5))` = 80 GB — OOM **внутри** init,
до первого `check_state`. Аудит нашёл этот случай через grep по
`materialize`-паттернам (`np.zeros`, `Table.from_batches`, `list(reader)`)
+ анализ `op.init()`. **Урок:** аудит перед новыми функциями (join, GBDT)
находит унаследованные баги. Фикс: `estimate_state_size` перед init
(+ AST guard `test_reduce_estimate_state_before_init`).

"""
marker_135 = "### 13.5. Открытые вопросы"
if "Эпизод 27" not in s:
    s = s.replace(marker_135, episodes + marker_135, 1)

# 5.4 §11.1: добавить строки
s = s.replace(
    "| API availability guard (07, 18 assertions) | ✅ | 18 |",
    "| API availability guard (07, 18 assertions) | ✅ | 18 |\n"
    "| Materialize-before-check fix (10) | ✅ | (см. test_materialize_guard) |",
    1,
)

# 5.5 §14 резюме
s = s.replace(
    "**v0.1.3 (патч 09):** `partial_fit` удалён полностью (§3.9). ML-слой\n"
    "CADW = `to_batches(format=...)` только. Собственные ML-алгоритмы —\n"
    "v0.3 (GBDT, Linear, k-means, PCA, NB) на примитивах.",
    "**v0.1.3 (патч 09):** `partial_fit` удалён полностью (§3.9). ML-слой\n"
    "CADW = `to_batches(format=...)` только. Собственные ML-алгоритмы —\n"
    "v0.3 (GBDT, Linear, k-means, PCA, NB) на примитивах.\n\n"
    "**v0.1.4 (патч 10):** must-fix «materialize-before-check»: `_local_groupby`\n"
    "(stat до чтения) и `reduce` (estimate_state_size до init()). Добавлены\n"
    "AST guards + OOM E2E. §3.10 — новый принцип.",
    1,
)

p.write_text(s)
print("DECISIONS.md: §3.10, §11.4 (класс), §13.4/27-28, §11.1, §14")
PY

# ---------- 6. CHANGELOG v0.1.4 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()

entry = """## [0.1.4] — 2026-10-18

### Исправлено (must-fix §11.4, класс «materialize-before-check»)
- `_local_groupby` материализовал всю партицию (`list(ArrowIPCBatchReader)`)
  до size-check. 02b1 добавил проверку **после** materialize — это не
  закрыло баг. Fix: `partition_path.stat().st_size` до чтения + helper
  `_read_partition_safe` (единственное место, где разрешён `list(reader)`).
- `reduce` вызывал `op.init()` до `estimate_state_size`/`check_state`.
  `CovarianceOp(d=10^5)` → `np.zeros((10^5,10^5))` = 80 GB → OOM **внутри**
  init. Fix: `estimate_state_size` + `check_state` **до** `op.init()`.

### Добавлено
- §3.10 «Check before materialize» — новый обязательный принцип.
- `tests/test_materialize_guard.py` — AST guards:
  - `test_no_list_of_whole_reader_outside_helper`
  - `test_reduce_estimate_state_before_init`
- `tests/test_oom_resistance.py` (slow):
  - 10^6 групп, 10 MB budget;
  - 90/10 skew, 10^6 строк, 10 MB;
  - `CovarianceOp(d=5000)` под 10 MB → `MemoryLimitExceeded` до `init`.

### Уроки (DECISIONS §13.4)
- Эпизод 27: «закрытие must-fix» требует теста на **порядок** операций,
  а не только на наличие вызова. 02b1 проверял, что `hash_shuffle`
  вызывается, но не проверял, что `list(reader)` **не** вызывается до.
- Эпизод 28: аудит перед новыми функциями находит унаследованные баги
  (`reduce.init()` — не был замечен при 02a).

"""

marker = "## [0.1.3]"
if marker in s:
    s = s.replace(marker, entry + marker, 1)
else:
    s = s.replace("## [0.1.0]", entry + "## [0.1.0]", 1)

p.write_text(s)
print("CHANGELOG.md: v0.1.4 entry added")
PY

# ---------- 7. Run ----------
echo "=== pytest (fast) ==="
set +e
uv run pytest -q -m "not slow"
RC_FAST=$?
uv run pytest -q -m slow
RC_SLOW=$?
set -e
if [[ $RC_FAST -ne 0 ]]; then echo "fast: rc=$RC_FAST"; exit 1; fi
if [[ $RC_SLOW -ne 0 ]]; then echo "slow: rc=$RC_SLOW"; exit 1; fi

# ---------- 8. Commit ----------
git add -A
git commit -m "10: must-fix §11.4 (materialize-before-check) — _local_groupby stat до read; reduce estimate_state_size до init; AST guards + OOM E2E"

echo "==> sha256"
sha256sum cadw/ops/groupby.py cadw/ops/reduce.py tests/test_materialize_guard.py tests/test_oom_resistance.py
echo "==> 10 applied on ${BR}"