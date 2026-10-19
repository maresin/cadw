#!/usr/bin/env bash
# patch_v034_combiner_fix2: _combine_batches — собираем pa.Table, затем
# to_batches()[0], вместо pa.RecordBatch.from_arrays(ChunkedArray).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v034-combiner"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-v034-fix2-${TS}"
git diff > ".backups/backup-pre-v034-fix2-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: v0.3.4-fix2 (RecordBatch.from_arrays gets ChunkedArray; fix follows)" || true
require_clean_tree || exit 1

python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/groupby.py")
s = p.read_text()

old = '''        new_names = list(by_columns) + [
            f"{_COMB_PREFIX}{col}_{bf}" for col, bf in base_specs
        ]
        arrays = [grouped.column(c) for c in list(by_columns)] + [
            grouped.column(f"{col}_{bf}") for col, bf in base_specs
        ]
        out = pa.RecordBatch.from_arrays(
            arrays, names=new_names
        )
        yield out'''

new = '''        # group_by().aggregate() returns pa.Table; .column() -> ChunkedArray.
        # pa.RecordBatch.from_arrays requires Array, not ChunkedArray.
        # Collect via pa.Table, then yield its (single, combined) batch.
        orig_names = list(by_columns) + [
            f"{col}_{bf}" for col, bf in base_specs
        ]
        new_names = list(by_columns) + [
            f"{_COMB_PREFIX}{col}_{bf}" for col, bf in base_specs
        ]
        out_table = grouped.select(orig_names).rename_columns(new_names)
        for b in out_table.to_batches():
            yield b'''

if old not in s:
    raise SystemExit("_combine_batches yield block not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("groupby.py: _combine_batches — pa.Table -> to_batches()")
PY

# ---------- DECISIONS: episode 53 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

marker_135 = "### 13.5. Открытые вопросы"
episode_53 = """**Эпизод 53 (ChunkedArray vs Array в RecordBatch).** Первая версия
`_combine_batches` использовала `pa.RecordBatch.from_arrays(arrays, ...)`,
где `arrays` собирались из `grouped.column(c)`. `grouped` — `pa.Table`,
`.column()` возвращает **`ChunkedArray`** (может быть несколько chunks
после `group_by().aggregate()`). `pa.RecordBatch.from_arrays` требует
**`Array`** → `TypeError: Cannot convert ChunkedArray to Array`. 11 из 14
тестов упали.

**Fix:** `grouped.select(orig_names).rename_columns(new_names)` даёт
`pa.Table`, `.to_batches()` даёт `RecordBatch` (со всеми chunk'ами,
склеенными автоматически). Один batch на входной batch.

**Правило:** в out-of-core коде мы работаем с `RecordBatch` (батчи
фиксированного размера), но промежуточные pandas-подобные операции
(`group_by().aggregate()`) естественно возвращают `Table`. Склеивать
через `combine_chunks()` или `to_batches()` — не через ручной
`pa.array(...)`. Это тот же класс, что эпизод 44/47b: `.column()`
возвращает не то, что ожидалось; `from_arrays` строже, чем `from_pandas`.
"""

if "Эпизод 53" not in s:
    s = s.replace(marker_135, episode_53 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §13.4/53")
PY

# ---------- run ----------
echo "=== pytest tests/test_combiner.py -v ==="
set +e
uv run pytest tests/test_combiner.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "combiner fix2: rc=$RC"
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.3.4-fix2: _combine_batches — pa.Table.to_batches() вместо RecordBatch.from_arrays(ChunkedArray); §13.4/53"

echo "==> v0.3.4-fix2 applied"