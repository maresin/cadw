#!/usr/bin/env bash
# patch_11b2c3_fix: вставить расчёт unique_build_keys/avg_matches_per_probe
# в _process_partition_pair (шаг 2d прошлого патча молча no-op).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/11b2c3-merge-planner"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11b2c3-fix-${TS}"
git diff > ".backups/backup-pre-11b2c3-fix-${TS}/wip.diff" || true

# ---------- Fix: вставить расчёт avg_matches_per_probe ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/join.py")
s = p.read_text()

# Проверяем: имя уже используется в теле, но не определено.
if "avg_matches_per_probe = (" in s:
    print("join.py: avg_matches_per_probe уже определён — no-op")
else:
    # Вставляем блок непосредственно перед "out_batches: list[pa.RecordBatch] = []"
    # (единственное вхождение в _process_partition_pair).
    old = '''    out_batches: list[pa.RecordBatch] = []
    empty_schema = _empty_result_schema(
        left_schema, right_schema, on, suffixes
    )'''

    new = '''    # Per-partition skew check: unique_build_keys + avg_matches_per_probe.
    # Ловит partition-level skew, которого глобальная модель не видит:
    # после hash_shuffle все строки с одним ключом идут в одну партицию,
    # и там build может дублироваться сильнее, чем в среднем.
    unique_build_keys = len(build_df)
    if len(build_df) > 0 and len(on) > 0:
        try:
            if len(on) == 1:
                unique_build_keys = int(build_df[on[0]].nunique(dropna=True))
            else:
                unique_build_keys = int(
                    build_df.drop_duplicates(subset=list(on)).shape[0]
                )
            unique_build_keys = max(1, unique_build_keys)
        except Exception:
            unique_build_keys = len(build_df)
    avg_matches_per_probe = (
        len(build_df) / max(1, unique_build_keys)
        if unique_build_keys > 0 else 1.0
    )

    out_batches: list[pa.RecordBatch] = []
    empty_schema = _empty_result_schema(
        left_schema, right_schema, on, suffixes
    )'''

    if old not in s:
        raise SystemExit(
            "out_batches block not found — file structure changed unexpectedly"
        )
    s = s.replace(old, new, 1)
    print("join.py: avg_matches_per_probe вставлен перед out_batches")
p.write_text(s)
PY

# ---------- AST guard: имя определено ДО первого использования ----------
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_join_oom_guard.py")
s = p.read_text()

extra = '''

def test_process_partition_pair_defines_avg_matches_before_use():
    """avg_matches_per_probe определён ДО использования (AST, lineno).

    Регрессия 11-B-2c-3: шаг 2d патча молча не сработал (str.replace с
    несуществующим шаблоном — no-op), и avg_matches_per_probe использовался
    без определения. NameError в runtime. Guard фиксирует порядок.
    """
    from cadw.ops.join import _process_partition_pair
    import ast

    src = inspect.getsource(_process_partition_pair)
    tree = ast.parse(inspect.cleandoc(src))

    # Найти все Store и Load для имени "avg_matches_per_probe"
    stores: list[int] = []
    loads: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "avg_matches_per_probe":
            if isinstance(node.ctx, ast.Store):
                stores.append(node.lineno)
            elif isinstance(node.ctx, ast.Load):
                loads.append(node.lineno)

    assert stores, (
        "avg_matches_per_probe never assigned in _process_partition_pair "
        "(NameError at runtime)"
    )
    assert loads, (
        "avg_matches_per_probe never read in _process_partition_pair"
    )
    first_store = min(stores)
    first_load = min(loads)
    assert first_store < first_load, (
        f"avg_matches_per_probe: first Store at line {first_store}, "
        f"first Load at line {first_load} — use before assignment"
    )

'''
s = s.rstrip() + extra
p.write_text(s)
print("test_join_oom_guard.py: +AST guard на порядок define->use")
PY

# ---------- DECISIONS: эпизод 44 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

marker_135 = "### 13.5. Открытые вопросы"
episode_44 = """**Эпизод 44 (str.replace silent no-op).** Патч 11-B-2c-3 содержал
две замены: (2d) вставить `unique_build_keys`/`avg_matches_per_probe`
перед `out_batches`, и (2e) заменить cartesian-check на два новых.
Шаг 2e сработал, шаг 2d — **молча нет**: шаблон `old` включал комментарий
`# how='left': пустой build — продолжаем (left preserved).`, которого в
файле после 11-B-2b-fix2 не было. `str.replace(old, new)` при отсутствии
`old` возвращает исходную строку без ошибки. Итог: код ссылается на
`avg_matches_per_probe`, которая никогда не вычисляется. **Fix:** guard
`assert old in s` перед каждым `s.replace(old, new, 1)`, или (лучше)
AST guard на порядок define→use (`test_process_partition_pair_defines_
avg_matches_before_use`). **Урок:** `str.replace` в патч-скриптах —
это DSL без проверки синтаксиса. Любой silent no-op допустим только
если он явно логируется; в нашем случае молчание скрыло баг от
`test_join.py` (там `avg_matches_per_probe` не использовался в
существующих тестах, только в новых из `test_merge_planner.py`).
**Более сильный фикс:** AST guard `define before use` — ловит **класс**
багов, а не конкретный случай.

"""
if "Эпизод 44" not in s:
    s = s.replace(marker_135, episode_44 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §13.4/44")
PY

# ---------- run ----------
echo "=== pytest tests/test_merge_planner.py tests/test_join_oom_guard.py -v ==="
set +e
uv run pytest tests/test_merge_planner.py tests/test_join_oom_guard.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "fix: rc=$RC"
  exit 1
fi

echo "=== full fast + slow ==="
uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-B-2c-3-fix: avg_matches_per_probe определён (str.replace silent no-op); AST guard define-before-use; §13.4/44"

echo "==> 11-B-2c-3-fix applied"
echo "    Next: squash → tag v0.2.4"