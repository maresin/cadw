#!/usr/bin/env bash
# patch_v034_fix3: hash_shuffle + _shuffle_write — dedupe key_columns+value_columns
# (fixes unique() / groupby(same_col_in_by_and_agg); §13.4/54).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v034-combiner"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-v034-fix3-${TS}"
git diff > ".backups/backup-pre-v034-fix3-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: v0.3.4-fix3 (unique() failed due to hash_shuffle duplicate column)" || true
require_clean_tree || exit 1

# ---------- Fix 1: shuffle.py — dedupe key+value ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/shuffle.py")
s = p.read_text()

# Add helper near top
helper = '''def _dedup_columns(cols: Sequence[str]) -> list[str]:
    """Dedupe columns preserving order. FIX §13.4/54: hash_shuffle used to
    write duplicate columns when key ∩ value ≠ ∅ (e.g. groupby(same_col)).
    Caller contract unchanged: we still write key+value semantics; duplicates
    are just collapsed to one Arrow field."""
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


'''

if "def _dedup_columns" not in s:
    # Insert before _stable_bytes
    anchor = "def _stable_bytes("
    if anchor not in s:
        raise SystemExit("_stable_bytes not found")
    s = s.replace(anchor, helper + anchor, 1)
    print("shuffle.py: _dedup_columns helper added")

# Fix schema construction
old_schema = '''    # Схема: key + value + row_id
    fields: list[pa.Field] = []
    for col in list(key_columns) + list(value_columns):
        fields.append(first_batch.schema.field(col))
    fields.append(pa.field("row_id", pa.uint64()))
    schema = pa.schema(fields)'''
new_schema = '''    # Схема: key + value + row_id (dedup key ∩ value — §13.4/54)
    all_columns = _dedup_columns(list(key_columns) + list(value_columns))
    fields: list[pa.Field] = []
    for col in all_columns:
        fields.append(first_batch.schema.field(col))
    fields.append(pa.field("row_id", pa.uint64()))
    schema = pa.schema(fields)'''
if old_schema not in s:
    raise SystemExit("schema construction block not found")
s = s.replace(old_schema, new_schema, 1)
print("shuffle.py: schema construction — dedup")

# Fix select in _process
old_select = '''        sorted_batch = batch.select(list(key_columns) + list(value_columns)).take(order)'''
new_select = '''        sorted_batch = batch.select(all_columns).take(order)'''
if old_select not in s:
    raise SystemExit("select block not found")
s = s.replace(old_select, new_select, 1)
print("shuffle.py: select — use deduped all_columns")

p.write_text(s)
PY

# ---------- Fix 2: groupby.py — _shuffle_write dedupe ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/groupby.py")
s = p.read_text()

old = '''    hashes = compute_hashes(batch, key_columns, hash_seed)
    parts = _mod_p(hashes, num_partitions)
    order = pc.sort_indices(parts)
    sorted_batch = batch.select(list(key_columns) + list(value_columns)).take(order)'''

new = '''    hashes = compute_hashes(batch, key_columns, hash_seed)
    parts = _mod_p(hashes, num_partitions)
    order = pc.sort_indices(parts)
    # Dedup key ∩ value (§13.4/54): batch.select() на дубликатах имён
    # создаёт "cat, cat" → group_by("cat") падает с "Multiple matches".
    _seen: set[str] = set()
    _cols: list[str] = []
    for _c in list(key_columns) + list(value_columns):
        if _c not in _seen:
            _seen.add(_c)
            _cols.append(_c)
    sorted_batch = batch.select(_cols).take(order)'''

if old not in s:
    raise SystemExit("_shuffle_write select block not found")
s = s.replace(old, new, 1)
print("groupby.py: _shuffle_write — dedup key+value")

p.write_text(s)
PY

# ---------- Fix 3: tests/test_dataframe.py — add regression test ----------
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_dataframe.py")
s = p.read_text()

extra = '''

def test_groupby_same_col_as_by(tmp_path):
    """df.groupby('cat').agg('cat', 'count') — agg == by (§13.4/54)."""
    df = make_df(tmp_path)
    result = df.groupby("cat").agg("cat", "count")
    out = tmp_path / "same_col.parquet"
    result.write_parquet(out)
    rows = {r["cat"]: r["cat_count"] for r in pq.read_table(out).to_pylist()}
    # 20 rows, 5 groups → 4 each
    assert all(v == 4 for v in rows.values())
    assert len(rows) == 5

'''

s = s.rstrip() + extra
p.write_text(s)
print("test_dataframe.py: +test_groupby_same_col_as_by")
PY

# ---------- DECISIONS: episode 54 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

marker_135 = "### 13.5. Открытые вопросы"
episode_54 = """**Эпизод 54 (дубликат в hash_shuffle: key ∩ value ≠ ∅).** `df.unique(col)`
вызывает `groupby_aggregate([col], [col], ["count"])`. В `groupby_aggregate`
`value_columns = [c for c in agg_columns if c not in seen]` → пусто (все
agg в by); fallback `if not value_columns: value_columns = list(by_columns)`
→ `[col]`. Затем `hash_shuffle(key=[col], value=[col], ...)`: `fields =
[key, key, row_id]` — Arrow допускает duplicate field names в schema
(no validation), но downstream `Table.group_by(["col"])` падает с
`Multiple matches for FieldRef.Name(col)`.

**Почему баг существовал, но не всплывал:** `test_unique` был написан
ещё в v0.1.0 и проходил — вероятно, из-за особенностей Arrow 24–25 (или
порядка тестов, где уникальные partition-файлы не триггерили Acero).
Только на PyArrow 26 Acero стал строже валидировать duplicate input
columns. Это класс «latent bug, проявляется при обновлении зависимости».

**Fix:** dedupe `key_columns + value_columns` в `hash_shuffle` (и
`_shuffle_write`) через `list(dict.fromkeys(...))`. Контракт не меняется
— duplicate value column семантически избыточен (value уже присутствует
как key). Функция `_dedup_columns` вынесена в `shuffle.py`.

**Regression test:** `test_groupby_same_col_as_by` — прямой тест
`df.groupby('cat').agg('cat', 'count')`, agg == by. Плюс `test_unique`
уже есть (и теперь зелёный).

**Урок:** API Arrow `Table.group_by()` может быть строже, чем ожидается:
он валидирует duplicate columns в input. `hash_shuffle` — primitive,
который должен быть защищён от неаккуратных caller'ов (caller может
передать `value_columns=["cat"]` когда `"cat"` уже в `key_columns`).
Defensive dedup — обязательный шаг для reusable primitive. Не полагаться
на то, что caller dedup'нет.

**Следующее (v0.3.5):** тест на `hash_shuffle` с explicit duplicate
key/value — должен не падать, dedupe молча. Плюс regression `test_unique`
уже есть (был всегда).

"""
if "Эпизод 54" not in s:
    s = s.replace(marker_135, episode_54 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §13.4/54")
PY

# ---------- run ----------
echo "=== pytest tests/test_dataframe.py tests/test_combiner.py -v ==="
set +e
uv run pytest tests/test_dataframe.py tests/test_combiner.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "fix3: rc=$RC"
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.3.4-fix3: hash_shuffle/_shuffle_write dedupe key+value (fixes unique()/groupby(same-col)); §13.4/54"

echo "==> v0.3.4-fix3 applied"