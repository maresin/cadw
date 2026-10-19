#!/usr/bin/env bash
# patch_11b2c3_fix2: вынести _peek_first/_data_chars_for_join из
# cadw/dataframe.py в cadw/core/sampling.py — facade AST guard
# (test_facade_has_no_as_py) снова проходит.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/11b2c3-merge-planner"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11b2c3-fix2-${TS}"
git diff > ".backups/backup-pre-11b2c3-fix2-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: 11-B-2c-3-fix (facade AST guard red; move sampling helpers out)" || true
require_clean_tree || exit 1

# ---------- 1. cadw/core/sampling.py: helpers ----------
cat > cadw/core/sampling.py <<'PY'
"""
Sampling helpers для planner integration.

Первые батчи читаются в `CadwDataFrame.merge` для построения `DataChars`
и вызова `MemoryGuard.plan(...)`. `as_py()` здесь допустим: O(1) на
сэмпл (одна строка на колонку для оценки `avg_key_size`), не O(N).
Наружу отдаётся `DataChars` — фасад (`dataframe.py`) не вызывает `as_py`
(см. AST guard `test_facade_has_no_as_py`).
"""
from __future__ import annotations

import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.core.models import DataChars, data_chars_from_sample


def peek_first_batch(gen) -> "pa.RecordBatch | None":
    """Первый батч генератора или None (для sampling в merge)."""
    for b in gen:
        return b
    return None


def data_chars_for_join(
    sample: pa.RecordBatch,
    n_total: int,
    on_cols: list[str],
    memory_guard: MemoryGuard,
) -> DataChars:
    """DataChars для join: k_est — из unique на on-колонках sample,
    экстраполяция на n_total.

    `as_py()` на одной строке (repr().encode() для avg_key_size) — O(1),
    не нарушает гарантию памяти. `size_of_key_entry` — консервативно
    `row_size + 80` (agg state).
    """
    if sample.num_rows == 0:
        raise ValueError("data_chars_for_join: empty sample")
    on_sample = sample.select([c for c in on_cols if c in sample.schema.names])
    k_est = memory_guard.estimate_unique_keys(on_sample, n_total)

    # O(1): одна строка на колонку, не цикл по batch.num_rows.
    first = {
        c: sample.column(c)[0].as_py()
        for c in on_cols
        if c in sample.schema.names
    }
    avg_key_size = len(repr(tuple(first.values())).encode())

    return data_chars_from_sample(
        sample, n_total,
        k_est=k_est,
        avg_key_size_bytes=avg_key_size,
        size_of_key_entry_bytes=avg_key_size + 80,
    )
PY

# ---------- 2. dataframe.py: remove helpers, add import, rename calls ----------
python3 - <<'PY'
from pathlib import Path
import re

p = Path("cadw/dataframe.py")
s = p.read_text()

# 2a. Заменить импорт DataChars/data_chars_from_sample на sampling helpers.
old_import = "from cadw.core.models import DataChars, data_chars_from_sample"
if old_import in s:
    s = s.replace(
        old_import,
        "from cadw.core.sampling import (\n"
        "    data_chars_for_join,\n"
        "    peek_first_batch,\n"
        ")",
        1,
    )
    print("dataframe.py: import -> cadw.core.sampling")
elif "from cadw.core.sampling import" in s:
    print("dataframe.py: sampling import already present")
else:
    raise SystemExit(
        "expected 'from cadw.core.models import DataChars, data_chars_from_sample' "
        "not found; file structure changed"
    )

# 2b. Удалить helpers из dataframe.py.
# Ищем от "\n\ndef _peek_first(" до "class _GroupBy:" и заменяем на
# "\n\n\nclass _GroupBy:" (PEP8 два blank line перед top-level class).
pattern = re.compile(
    r"\n{2,}def _peek_first\(.*?(?=class _GroupBy:)",
    re.DOTALL,
)
m = pattern.search(s)
if not m:
    if "def _peek_first(" in s:
        raise SystemExit(
            "_peek_first found but regex did not match — check surrounding structure"
        )
    print("dataframe.py: helpers already removed — no-op")
else:
    s = s[:m.start()] + "\n\n\n" + s[m.end():]
    print("dataframe.py: helpers _peek_first/_data_chars_for_join removed")

# 2c. Переименовать вызовы в merge.
before = s
s = s.replace("_peek_first(", "peek_first_batch(")
s = s.replace("_data_chars_for_join(", "data_chars_for_join(")
if s == before:
    print("dataframe.py: no call-site renames (already renamed)")
else:
    print("dataframe.py: call sites renamed (peek_first_batch, data_chars_for_join)")

# 2d. Убедимся, что в dataframes.py нет оставшихся ссылок на старые имена
# (вне комментариев/строк — но в AST guard это ловится тестом).
for old in ("_peek_first", "_data_chars_for_join", "DataChars"):
    if old in s:
        raise SystemExit(
            f"dataframe.py still references {old!r} after move "
            f"(grep: {[l for l in s.splitlines() if old in l][:3]})"
        )

p.write_text(s)
print("dataframe.py: cleanup verified")
PY

# ---------- 3. AST guard: добавить проверку на отсутствие sampling helpers в facade ----------
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_dataframe.py")
s = p.read_text()

extra = '''

def test_facade_does_not_define_sampling_helpers():
    """`_peek_first`/`_data_chars_for_join` не живут в dataframe.py.

    Регрессия 11-B-2c-3-fix2: helpers были добавлены в dataframe.py и
    использовали as_py, нарушая test_facade_has_no_as_py. Правило:
    helpers для сэмплинга — в cadw/core/sampling.py; facade остаётся
    без as_py.
    """
    src = inspect.getsource(df_mod)
    for name in ("_peek_first", "_data_chars_for_join"):
        assert f"def {name}(" not in src, (
            f"{name!r} must live in cadw/core/sampling.py, "
            f"not in facade"
        )

'''
s = s.rstrip() + extra
p.write_text(s)
print("test_dataframe.py: +test_facade_does_not_define_sampling_helpers")
PY

# ---------- 4. DECISIONS: §13.4/45 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

marker_135 = "### 13.5. Открытые вопросы"
episode_45 = """**Эпизод 45 (facade AST guard scope).** `merge(use_planner=True)`
требует sample first batch → `DataChars` → `MemoryGuard.plan("join", ...)`.
Помощник `_data_chars_for_join` использует `sample.column(c)[0].as_py()` —
O(1) (одна строка на колонку), не O(N). Но был положен в
`cadw/dataframe.py` — AST guard `test_facade_has_no_as_py` проверяет
**весь модуль**, не только класс `CadwDataFrame`, и упал (2 теста:
`test_facade_has_no_as_py` + `test_facade_has_no_as_py_ast`).
**Fix:** вынести `peek_first_batch`/`data_chars_for_join` в новый
`cadw/core/sampling.py` с документированным «as_py здесь O(1)». Facade
(`dataframe.py`) остаётся без `as_py`. Guard-скоуп «весь модуль» —
намеренно строгий: помогает держать `dataframe.py` без любых row-level
примитивов, даже O(1). **Альтернатива** (сузить guard до класса) —
отвергнута: ослабила бы инвариант, который уже ловил регрессии (§13.4
эпизод 4b: `test_facade_has_no_as_py` был текстовым vs AST). **Урок:**
если хочется добавить `as_py` в facade «по-мелочи» — сигнал, что место
кода не в facade. AST guard фиксирует **слой**, а не только намерение.

"""
if "Эпизод 45" not in s:
    s = s.replace(marker_135, episode_45 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §13.4/45")
PY

# ---------- run ----------
echo "=== pytest tests/test_dataframe.py tests/test_dataframe_filters.py tests/test_merge_planner.py -v ==="
set +e
uv run pytest tests/test_dataframe.py tests/test_dataframe_filters.py tests/test_merge_planner.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "fix2: rc=$RC"
  exit 1
fi

echo "=== full fast + slow ==="
uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-B-2c-3-fix2: helpers → cadw/core/sampling.py; facade AST guard снова зелёный; §13.4/45"

echo "==> 11-B-2c-3-fix2 applied"
echo "    Next: squash → tag v0.2.4"