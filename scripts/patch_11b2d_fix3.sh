#!/usr/bin/env bash
# patch_11b2d_fix3: докончить — убрать docstring-упоминание, обновить
# AST guards, DECISIONS, CHANGELOG, запустить тесты.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/11b2d-e2e-join"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11b2d-fix3-${TS}"
git diff > ".backups/backup-pre-11b2d-fix3-${TS}/wip.diff" || true

# ---------- 1. dataframe.py: docstring без max_cartesian_rows ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

old_doc = """        - num_partitions: число партиций HashShuffle; None — из
          JoinMemoryModel (если use_planner), иначе дефолт 8.
        - use_planner: использовать JoinMemoryModel.solve() для выбора
          num_partitions / max_output_rows / max_cartesian_rows. Если
          False — runtime использует старые дефолты (0.3 * available / 256)."""
new_doc = """        - num_partitions: число партиций HashShuffle; None — из
          JoinMemoryModel (если use_planner), иначе дефолт 8.
        - use_planner: использовать JoinMemoryModel.solve() для выбора
          num_partitions / max_output_rows. Если False — runtime
          использует старые дефолты (0.3 * available / 256)."""
if old_doc in s:
    s = s.replace(old_doc, new_doc, 1)
    print("dataframe.py: docstring обновлён (без max_cartesian_rows)")
elif "max_cartesian_rows" in s:
    # fallback: убрать только упоминание в строке
    lines = s.splitlines()
    for i, line in enumerate(lines):
        if "max_cartesian_rows" in line:
            lines[i] = line.replace(" / max_cartesian_rows", "")
    s = "\n".join(lines) + "\n"
    print("dataframe.py: docstring fallback — убрано упоминание")
else:
    print("dataframe.py: max_cartesian_rows уже нет")

# Проверка
if "max_cartesian_rows" in s:
    offending = [(i+1, line) for i, line in enumerate(s.splitlines())
                 if "max_cartesian_rows" in line]
    raise SystemExit(
        "dataframe.py: всё ещё упоминает:\n" +
        "\n".join(f"  {i}: {line}" for i, line in offending)
    )
p.write_text(s)
print("dataframe.py: готово")
PY

# ---------- 2. tests/test_join_oom_guard.py: обновить AST guards ----------
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_join_oom_guard.py")
s = p.read_text()

# Заменить test_process_partition_pair_defines_avg_matches_before_use
if "def test_process_partition_pair_defines_avg_matches_before_use():" in s:
    old_start = s.find("def test_process_partition_pair_defines_avg_matches_before_use():")
    # Конец — до следующего "def test_" на верхнем уровне
    old_end = s.find("\ndef test_", old_start + 10)
    if old_end < 0:
        old_end = len(s)

    new_test = '''def test_actual_matches_checked_before_pd_merge():
    """Exact-match guard вызывается ДО pd.merge в _process_partition_pair.

    Регрессия 11-B-2d: cartesian check (`len(build) × len(probe) >
    max_cartesian_rows`) давал ложные reject'ы для PK-FK (build уникален).
    Fix: точная оценка `actual_matches = _exact_actual_matches(probe)`,
    сравнение с `max_output_rows` — то же, что `pd.merge` материализует.
    """
    from cadw.ops.join import _process_partition_pair
    import ast

    src = inspect.getsource(_process_partition_pair)
    tree = ast.parse(inspect.cleandoc(src))

    exact_call_lines: list[int] = []
    pd_merge_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "_exact_actual_matches":
                exact_call_lines.append(node.lineno)
            if isinstance(f, ast.Attribute) and f.attr == "merge":
                if isinstance(f.value, ast.Name) and f.value.id == "pd":
                    pd_merge_lines.append(node.lineno)

    assert exact_call_lines, (
        "_exact_actual_matches not called in _process_partition_pair"
    )
    assert pd_merge_lines, "pd.merge not called in _process_partition_pair"
    assert min(exact_call_lines) < min(pd_merge_lines), (
        "_exact_actual_matches must be called before pd.merge"
    )


def test_no_cartesian_check_in_join():
    """max_cartesian_rows не упоминается в join.py (cartesian не OOM-граница)."""
    from cadw.ops import join as join_mod
    src = inspect.getsource(join_mod)
    assert "max_cartesian_rows" not in src, (
        "max_cartesian_rows must be removed from join.py; use exact-match"
    )

'''
    s = s[:old_start] + new_test + s[old_end:]
    p.write_text(s)
    print("test_join_oom_guard.py: старый guard заменён 2 новыми")
else:
    print("test_join_oom_guard.py: старый guard уже заменён (no-op)")

if "test_actual_matches_checked_before_pd_merge" not in s:
    # fallback — дописать в конец
    extra = '''

def test_actual_matches_checked_before_pd_merge():
    """Exact-match guard вызывается ДО pd.merge в _process_partition_pair."""
    from cadw.ops.join import _process_partition_pair
    import ast

    src = inspect.getsource(_process_partition_pair)
    tree = ast.parse(inspect.cleandoc(src))

    exact_call_lines: list[int] = []
    pd_merge_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "_exact_actual_matches":
                exact_call_lines.append(node.lineno)
            if isinstance(f, ast.Attribute) and f.attr == "merge":
                if isinstance(f.value, ast.Name) and f.value.id == "pd":
                    pd_merge_lines.append(node.lineno)

    assert exact_call_lines, "_exact_actual_matches not called"
    assert pd_merge_lines, "pd.merge not called"
    assert min(exact_call_lines) < min(pd_merge_lines)


def test_no_cartesian_check_in_join():
    """max_cartesian_rows не упоминается в join.py."""
    from cadw.ops import join as join_mod
    src = inspect.getsource(join_mod)
    assert "max_cartesian_rows" not in src
'''
    s = s.rstrip() + extra
    p.write_text(s)
    print("test_join_oom_guard.py: guards дописаны (fallback)")
PY

# ---------- 3. DECISIONS: эпизоды 47, 47b ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

marker_135 = "### 13.5. Открытые вопросы"
episodes = """**Эпизод 47 (cartesian — не OOM-граница).** После 11-B-2c-3 runtime
проверял `len(build) × len(probe) > max_cartesian_rows` с
`max_cartesian_rows = 10 × max_output_rows`. Для PK-FK join 100k × 10k
(8 партиций): build ~5000 уникальных, probe ~2500 → cartesian = 12.6M
при **реальном output 2500 строк**. `max_cartesian_rows = 407k` →
ложный `MemoryLimitExceeded`. **Причина:** `pd.merge` материализует
**только совпадения**, не cartesian. `len(build) × len(probe)` —
верхняя граница совпадений, но для PK-FK (build уникален) она в тысячи
раз больше реального output. **Fix:** точная оценка
`actual_matches = sum over common keys(build_count × probe_count)`
через `build_df.groupby(on).size()` × `probe_df.groupby(on).size()` ×
`merge`. O(build + probe) на уникальных ключах — дёшево. Это **точный**
размер `pd.merge` output. `max_cartesian_rows` удалён из всех слоёв.
**Урок:** для защиты от OOM нужно оценивать **что материализуется**
(совпадения), а не **что могло бы материализоваться** (cartesian).
Верхняя граница должна быть **звуковой** (никогда не ниже реального
output), а cartesian over-reject'ит; exact match ни over-, ни
under-reject'ит — это ровно то, что `pd.merge` сделает.

**Эпизод 47b (regex в патч-скриптах).** Шаг 1d 11-B-2d-fix использовал
`s.replace(old, new)` с многострочным шаблоном. Шаблон не совпал
(символы `×`/`≈` / отступы), `str.replace` молча вернул исходную строку.
Файл остался в промежуточном состоянии: `avg_matches_per_probe` удалён
шагом 1c, но check'и продолжали на него ссылаться → NameError. **Fix:**
regex по маркерам (`# 1. Expected output` … `Likely extreme skew.`) +
явная проверка «что заменяем». **Урок (усиливает эпизод 44):** в патч-
скриптах для больших блоков — regex по маркерам, не точный многострочный
шаблон; любая замена — с assert на «что заменяем».

**Эпизод 47c (strict check on docstring).** Шаг 3 11-B-2d-fix2
проверял `if "max_cartesian_rows" in s` и падал на **docstring**
`merge` (`use_planner: num_partitions / max_output_rows /
max_cartesian_rows`). Код уже был очищен; проверка была слишком
строгой — включила комментарии. **Fix:** либо проверять AST (только
`Name`-узлы, не строки), либо пропускать строки с `#`/docstring.
**Урок:** «нет упоминаний» — это два разных контракта: «нет в коде»
(важно) и «нет в тексте» (часто не нужно). Default — AST.

"""
if "Эпизод 47 (" not in s:
    s = s.replace(marker_135, episodes + marker_135, 1)
    p.write_text(s)
    print("DECISIONS: §13.4/47, 47b, 47c")
else:
    print("DECISIONS: §13.4/47 уже есть")
PY

# ---------- 4. CHANGELOG: v0.2.5 fix note ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
old_marker = "## [0.2.5] — 2026-10-19"
if old_marker in s and "### Исправлено (§13.4 эпизод 47)" not in s:
    fix_note = """
### Исправлено (§13.4 эпизод 47)
- `hash_join`: заменён cartesian check (`len(build) × len(probe) >
  max_cartesian_rows`) на **точную** оценку совпадений:
  `actual_matches = sum over common keys(build_count × probe_count)`
  через `groupby.size()`. O(build + probe), дёшево. Устраняет ложные
  reject'ы для PK-FK.
- Удалены: `max_cartesian_rows` (все слои), `c_cartesian`.
- 2 AST guard'а: `test_actual_matches_checked_before_pd_merge`,
  `test_no_cartesian_check_in_join`.

"""
    s = s.replace(old_marker, old_marker + fix_note, 1)
    p.write_text(s)
    print("CHANGELOG: v0.2.5 fix note")
else:
    print("CHANGELOG: fix note уже есть или маркера нет")
PY

# ---------- 5. run ----------
echo "=== pytest tests/test_e2e_join.py -v (slow) ==="
set +e
uv run pytest tests/test_e2e_join.py -v
RC_E2E=$?
set -e
if [[ $RC_E2E -ne 0 ]]; then
  echo "test_e2e_join.py: rc=$RC_E2E"
  exit 1
fi

echo "=== pytest fast ==="
set +e
uv run pytest -q -m "not slow"
RC_FAST=$?
set -e
if [[ $RC_FAST -ne 0 ]]; then
  echo "fast: rc=$RC_FAST"
  exit 1
fi

echo "=== pytest slow (all) ==="
uv run pytest -q -m slow

git add -A
git commit -m "11-B-2d-fix3: exact-match OOM guard завершён; max_cartesian_rows удалён; §13.4/47+47b+47c"

echo "==> 11-B-2d-fix3 applied"