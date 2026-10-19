#!/usr/bin/env bash
# patch_11b2d_fix2: докончить 11-B-2d-fix — regex-замена check'ов + удаление
# max_cartesian_rows из всех слоёв.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/11b2d-e2e-join"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11b2d-fix2-${TS}"
git diff > ".backups/backup-pre-11b2d-fix2-${TS}/wip.diff" || true

# ---------- 1. join.py: regex-замена check'ов + удаление параметра ----------
python3 - <<'PY'
import re
from pathlib import Path

p = Path("cadw/ops/join.py")
s = p.read_text()

# 1a. Заменить блок check'ов (от маркера "# 1. Expected output" до
#     "Likely extreme skew." + закрывающая скобка) на exact-match.
if "_exact_actual_matches(probe_df)" in s and "avg_matches_per_probe" not in s:
    print("join.py: check'и уже заменены (no-op)")
else:
    start_marker = "# 1. Expected output"
    end_marker = "Likely extreme skew."
    start_idx = s.find(start_marker)
    if start_idx < 0:
        raise SystemExit(
            "start marker '# 1. Expected output' not found in join.py"
        )
    # найти закрывающую скобку после end_marker: ")"
    end_marker_idx = s.find(end_marker, start_idx)
    if end_marker_idx < 0:
        raise SystemExit("end marker 'Likely extreme skew.' not found")
    # от end_marker найти следующую ")": это конец raise MemoryLimitExceeded
    close_idx = s.find(")", end_marker_idx)
    if close_idx < 0:
        raise SystemExit("closing ) not found after 'Likely extreme skew.'")
    # закрывающая ) raise-вызова имеет отступ 16 (внутри if/for/func), ищем до \n
    # плюс перевод строки после
    end_idx = close_idx + 1
    # проверим, что заменяем именно то, что ожидали
    removed = s[start_idx:end_idx]
    if "cartesian = len(build_df) * len(probe_df)" not in removed:
        raise SystemExit(
            f"expected 'cartesian = len(build_df) * len(probe_df)' inside "
            f"removed block; got:\n{removed[:500]}"
        )

    # новое содержимое (без начального отступа — он уже есть в s[:start_idx])
    # indent для внутренних строк: 12 пробелов (for + func body)
    new_block = (
        "# Точное число строк, которое pd.merge материализует:\n"
        "            # sum over common keys(build_count × probe_count).\n"
        "            actual_matches = _exact_actual_matches(probe_df)\n"
        "            if actual_matches > max_output_rows:\n"
        "                raise MemoryLimitExceeded(\n"
        "                    f\"hash_join(how={how!r}): exact matches between this \"\n"
        "                    f\"probe batch ({len(probe_df)} rows) and build \"\n"
        "                    f\"({len(build_df)} rows) = {actual_matches} > \"\n"
        "                    f\"max_output_rows={max_output_rows}. Likely skew. \"\n"
        "                    f\"Увеличьте budget или уменьшите num_partitions.\"\n"
        "                )"
    )
    s = s[:start_idx] + new_block + s[end_idx:]
    print("join.py: check'и заменены на exact-match")

# 1b. hash_join signature: убрать max_cartesian_rows
s2, n = re.subn(
    r"(\n    max_output_rows: Optional\[int\] = None,)\n    max_cartesian_rows: Optional\[int\] = None,",
    r"\1",
    s,
)
if n:
    s = s2
    print(f"join.py: hash_join signature — убрано {n} max_cartesian_rows")

# 1c. Удалить default max_cartesian_rows
s2, n = re.subn(
    r"    if max_cartesian_rows is None:\n"
    r"        # Default: 10× expected \(c_cartesian из JoinMemoryModel\)\n"
    r"        max_cartesian_rows = max_output_rows \* 10\n\n",
    "",
    s,
)
if n:
    s = s2
    print(f"join.py: удалён default max_cartesian_rows ({n})")

# 1d. Убрать передачу max_cartesian_rows в _process_partition_pair и hash_join
s2, n = re.subn(
    r",\n(\s+)max_cartesian_rows=max_cartesian_rows,",
    ",",
    s,
)
if n:
    s = s2
    print(f"join.py: убрано {n} передач max_cartesian_rows")

# 1e. _process_partition_pair: убрать параметр (уже удалён шагом 1a)
#     Проверим:
if "max_cartesian_rows: int = 0," in s:
    s2, n = re.subn(r"\n    max_cartesian_rows: int = 0,", "", s)
    if n:
        s = s2
        print(f"join.py: _process_partition_pair — убрано {n} параметров max_cartesian_rows")

# 1f. Убедиться, что max_cartesian_rows больше не упоминается
if "max_cartesian_rows" in s:
    # оставшиеся упоминания — в docstring или комментариях, не критично?
    # Проверим: если в коде — фейл.
    offending = [
        (i + 1, line)
        for i, line in enumerate(s.splitlines())
        if "max_cartesian_rows" in line and not line.strip().startswith("#")
    ]
    if offending:
        raise SystemExit(
            f"max_cartesian_rows осталось в коде join.py:\n" +
            "\n".join(f"  {i}: {line}" for i, line in offending)
        )
    print("join.py: max_cartesian_rows осталось только в комментариях")

p.write_text(s)
print("join.py: готово")
PY

# ---------- 2. models_join.py: удалить max_cartesian_rows, c_cartesian ----------
python3 - <<'PY'
import re
from pathlib import Path
p = Path("cadw/core/models_join.py")
s = p.read_text()

# 2a. JoinParams: убрать поле
s2, n = re.subn(
    r"    max_output_rows: int\n"
    r"    max_cartesian_rows: int[^\n]*\n"
    r"    right_chars: DataChars",
    "    max_output_rows: int\n    right_chars: DataChars",
    s,
)
if n:
    s = s2
    print(f"models_join.py: JoinParams — убрано {n} max_cartesian_rows")

# 2b. __init__: убрать c_cartesian
s2, n = re.subn(
    r"        c_merge_ratio: float = 2\.0,[^\n]*\n"
    r"        c_cartesian: float = 10\.0,[^\n]*\n"
    r"    \) -> None:",
    "        c_merge_ratio: float = 2.0,     # pandas temp ≈ 2 × output\n    ) -> None:",
    s,
)
if n:
    s = s2
    print(f"models_join.py: __init__ — убрано c_cartesian ({n})")

# 2c. Убрать проверку/присваивание c_cartesian
s2, n = re.subn(
    r"        if c_cartesian < 1\.0:\n"
    r"            raise ValueError\(\"c_cartesian must be >= 1\.0\"\)\n",
    "",
    s,
)
if n:
    s = s2
    print(f"models_join.py: убрана проверка c_cartesian ({n})")

s2, n = re.subn(
    r"        self\.c_cartesian = float\(c_cartesian\)\n",
    "",
    s,
)
if n:
    s = s2
    print(f"models_join.py: убрано self.c_cartesian ({n})")

# 2d. solve: убрать max_cartesian_rows (вычисление + параметр)
s2, n = re.subn(
    r"        # worst-case: cartesian product\.[^\n]*\n"
    r"(?:        #[^\n]*\n)*"
    r"        max_cartesian_rows = max\(\n"
    r"(?:[^\n]*\n)*?"
    r"        \)\n\n",
    "",
    s,
)
if n:
    s = s2
    print(f"models_join.py: solve — убрано max_cartesian_rows ({n})")

s2, n = re.subn(
    r"            max_cartesian_rows=int\(max_cartesian_rows\),\n",
    "",
    s,
)
if n:
    s = s2
    print(f"models_join.py: JoinParams() — убран max_cartesian_rows ({n})")

# 2e. describe: убрать c_cartesian
s2, n = re.subn(
    r"            f\"c_merge=\{self\.c_merge_ratio:\.2f\}, \"\n"
    r"            f\"c_cartesian=\{self\.c_cartesian:\.2f}\"\n"
    r"        \)",
    "            f\"c_merge={self.c_merge_ratio:.2f}\"\n        )",
    s,
)
if n:
    s = s2
    print(f"models_join.py: describe — убрано c_cartesian ({n})")

# Проверка
if "max_cartesian_rows" in s or "c_cartesian" in s:
    offending = [
        (i + 1, line)
        for i, line in enumerate(s.splitlines())
        if "max_cartesian_rows" in line or "c_cartesian" in line
    ]
    raise SystemExit(
        "models_join.py: остались упоминания:\n" +
        "\n".join(f"  {i}: {line}" for i, line in offending)
    )

p.write_text(s)
print("models_join.py: готово")
PY

# ---------- 3. dataframe.py: убрать max_cartesian_rows ----------
python3 - <<'PY'
import re
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

s2, n = re.subn(
    r"        max_output_rows: int \| None = None\n"
    r"        max_cartesian_rows: int \| None = None\n\n",
    "        max_output_rows: int | None = None\n\n",
    s,
)
if n:
    s = s2
    print(f"dataframe.py: убрана переменная max_cartesian_rows ({n})")

s2, n = re.subn(
    r"                max_cartesian_rows = params\.max_cartesian_rows\n",
    "",
    s,
)
if n:
    s = s2
    print(f"dataframe.py: убрано присваивание max_cartesian_rows ({n})")

s2, n = re.subn(
    r"            max_cartesian_rows=max_cartesian_rows,\n",
    "",
    s,
)
if n:
    s = s2
    print(f"dataframe.py: убрана передача в hash_join ({n})")

if "max_cartesian_rows" in s:
    offending = [
        (i + 1, line) for i, line in enumerate(s.splitlines())
        if "max_cartesian_rows" in line
    ]
    raise SystemExit(
        "dataframe.py: остались упоминания:\n" +
        "\n".join(f"  {i}: {line}" for i, line in offending)
    )

p.write_text(s)
print("dataframe.py: готово")
PY

# ---------- 4. tests/test_join_oom_guard.py: обновить AST guard ----------
python3 - <<'PY'
import re
from pathlib import Path
p = Path("tests/test_join_oom_guard.py")
s = p.read_text()

# Заменить test_process_partition_pair_defines_avg_matches_before_use
old_test_start = "def test_process_partition_pair_defines_avg_matches_before_use():"
old_test_end_marker = "\n\ndef test_"

start_idx = s.find(old_test_start)
if start_idx < 0:
    print("test_join_oom_guard.py: старый guard уже удалён (no-op)")
else:
    end_idx = s.find(old_test_end_marker, start_idx + len(old_test_start))
    if end_idx < 0:
        end_idx = len(s)

    new_test = '''def test_actual_matches_checked_before_pd_merge():
    """Exact-match guard вызывается ДО pd.merge в _process_partition_pair.

    Регрессия 11-B-2d: cartesian check (`len(build) × len(probe) >
    max_cartesian_rows`) давал ложные reject'ы для PK-FK (build уникален).
    Fix: точная оценка `actual_matches = _exact_actual_matches(probe)`,
    сравнение с `max_output_rows` — то же, что `pd.merge` материализует.
    AST guard фиксирует порядок: check ДО merge.
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
    first_check = min(exact_call_lines)
    first_merge = min(pd_merge_lines)
    assert first_check < first_merge, (
        f"_exact_actual_matches at line {first_check} must precede "
        f"pd.merge at line {first_merge}"
    )


def test_no_cartesian_check_in_join():
    """cartesian check удалён: `max_cartesian_rows` не упоминается в join.py.

    Регрессия 11-B-2d: cartesian check ложно режет PK-FK. Правильный
    guard — exact matches.
    """
    from cadw.ops import join as join_mod
    src = inspect.getsource(join_mod)
    assert "max_cartesian_rows" not in src, (
        "max_cartesian_rows must be removed from join.py "
        "(cartesian is not an OOM bound); use exact-match estimate"
    )
'''

    s = s[:start_idx] + new_test + s[end_idx:]
    p.write_text(s)
    print("test_join_oom_guard.py: старый guard заменён 2 новыми")

# Убедиться, что новые функции определены
if "test_actual_matches_checked_before_pd_merge" not in s:
    raise SystemExit("новый test_actual_matches_checked_before_pd_merge не добавлен")
PY

# ---------- 5. DECISIONS: эпизод 47 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

marker_135 = "### 13.5. Открытые вопросы"
episode_47 = """**Эпизод 47 (cartesian — не OOM-граница).** После 11-B-2c-3 runtime
проверял `len(build) × len(probe) > max_cartesian_rows` с
`max_cartesian_rows = 10 × max_output_rows`. Для PK-FK join 100k × 10k
(8 партиций): build ~5000 уникальных, probe ~2500 → cartesian = 12.6M
при **реальном output 2500 строк** (каждый probe-ключ матчит ровно один
build-ключ). `max_cartesian_rows = 407k` → ложный `MemoryLimitExceeded`.
**Причина:** `pd.merge` материализует **только совпадения**, не
cartesian. `len(build) × len(probe)` — верхняя граница совпадений, но
для PK-FK (build уникален) она в тысячи раз больше реального output.
**Fix:** точная оценка `actual_matches = sum over common keys(build_count
× probe_count)` через `build_df.groupby(on).size()` (один раз до цикла)
× `probe_df.groupby(on).size()` (per batch) × `merge`. O(build + probe)
на уникальных ключах (не на всех строках) — дёшево. Это **точный**
размер `pd.merge` output, а не эвристика. `max_cartesian_rows` удалён
из всех слоёв. **Урок:** для защиты от OOM нужно оценивать **что
материализуется** (совпадения), а не **что могло бы материализоваться
в худшем случае** (cartesian). Верхняя граница должна быть **звуковой**
(никогда не ниже реального output), а cartesian over-reject'ит; exact
match ни over-, ни under-reject'ит — это ровно то, что `pd.merge`
сделает. Используем структуру данных **до** аллокации результата:
единственная точка, где мы знаем ключи, — это наш shuffle output.

**Эпизод 47b (regex в патч-скриптах).** Шаг 1d 11-B-2d-fix использовал
`s.replace(old, new)` с многострочным шаблоном, содержащим `×`, `≈`,
`avg_matches_per_probe`. Шаблон не совпал (символы или отступы), а
`str.replace` молча вернул исходную строку. Файл остался в промежуточном
состоянии: `avg_matches_per_probe` удалён шагом 1c, но check'и продолжали
на него ссылаться → NameError. **Fix:** regex по маркерам
(`# 1. Expected output` … `Likely extreme skew.`) + явная проверка
`if "cartesian = len(build_df) * len(probe_df)" not in removed: raise`.
**Урок (усиливает эпизод 44):** в патч-скриптах для больших блоков
использовать regex по маркерам, а не точный многострочный шаблон.
Любая замена — с assert на «что заменяем» (не только «что получилось»).

"""
if "Эпизод 47 (" not in s:
    s = s.replace(marker_135, episode_47 + marker_135, 1)
    p.write_text(s)
    print("DECISIONS: §13.4/47, 47b")
else:
    print("DECISIONS: §13.4/47 уже есть")
PY

# ---------- 6. CHANGELOG: v0.2.5 fix note ----------
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
PY

# ---------- run ----------
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
git commit -m "11-B-2d-fix2: exact-match OOM guard (regex-замена); max_cartesian_rows удалён из всех слоёв; §13.4/47+47b"

echo "==> 11-B-2d-fix2 applied"