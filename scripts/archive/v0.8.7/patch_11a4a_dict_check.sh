#!/usr/bin/env bash
# patch_11a4a_dict_check: must-fix §11.4 — dict check до group_by;
# _estimate_distinct_keys; in-memory resplit; AST guard; 4 tests.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-11a4a-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_11a4a"

BR="patch/11a4a-dict-check"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- groupby.py: _estimate_distinct_keys + fix _local_groupby ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/groupby.py")
s = p.read_text()

# 1. Константа + helper перед _read_partition_safe
if "_DEFAULT_KEY_ENTRY_BYTES" not in s:
    helper = '''_DEFAULT_KEY_ENTRY_BYTES = 80  # CALIBRATION DEFAULT (см. 11-A-4b)


def _estimate_distinct_keys(
    table: pa.Table, by_columns: Sequence[str]
) -> int:
    """Upper bound на число уникальных ключей.

    Для multi-key: min(prod(nunique), num_rows). Без построения полного
    hash-таблицы (pc.count_distinct — single int per column).
    """
    import pyarrow.compute as pc

    product = 1
    for col in by_columns:
        try:
            n = pc.count_distinct(table.column(col)).as_py()
        except Exception:
            return table.num_rows
        if not isinstance(n, int) or n <= 0:
            return table.num_rows
        product *= n
        if product >= table.num_rows:
            return table.num_rows
    return min(product, table.num_rows)


'''
    anchor = "def _read_partition_safe("
    if anchor not in s:
        raise SystemExit("_read_partition_safe def not found")
    s = s.replace(anchor, helper + anchor, 1)
    print("groupby.py: _estimate_distinct_keys + _DEFAULT_KEY_ENTRY_BYTES")

# 2. Сигнатура _local_groupby: size_of_key_entry_bytes
old_sig = '''def _local_groupby(
    partition_path: Path,
    by_columns: Sequence[str],
    agg_columns: Sequence[str],
    agg_funcs: Sequence[str] | str,
    memory_guard: MemoryGuard | None = None,
    spill_dir: Path | None = None,
    depth: int = 0,
    max_depth: int = _MAX_DEPTH,
) -> pa.Table:'''
new_sig = '''def _local_groupby(
    partition_path: Path,
    by_columns: Sequence[str],
    agg_columns: Sequence[str],
    agg_funcs: Sequence[str] | str,
    memory_guard: MemoryGuard | None = None,
    spill_dir: Path | None = None,
    depth: int = 0,
    max_depth: int = _MAX_DEPTH,
    *,
    size_of_key_entry_bytes: int | None = None,
) -> pa.Table:'''
if old_sig not in s:
    raise SystemExit("_local_groupby signature not found")
s = s.replace(old_sig, new_sig, 1)

# 3. Заменить блок "1. Оценка ... / 2. Влезает ... / 3. Не влезает ..."
# от "# 1. Оценка размера" до "def _read_for_resplit():".
import re
start_marker = "    # 1. Оценка размера партиции ДО чтения (must-fix, §11.4)."
end_marker = "    def _read_for_resplit():"
m = re.search(
    re.escape(start_marker) + r".*?(?=" + re.escape(end_marker) + ")",
    s,
    re.DOTALL,
)
if not m:
    raise SystemExit("_local_groupby main block not found")

new_block = '''    if size_of_key_entry_bytes is None:
        size_of_key_entry_bytes = _DEFAULT_KEY_ENTRY_BYTES

    # 1. Оценка размера файла ДО чтения.
    partition_bytes = (
        partition_path.stat().st_size if memory_guard is not None else 0
    )
    safety_limit = (
        int(memory_guard.available_bytes() * 0.5)
        if memory_guard is not None
        else partition_bytes
    )

    table: pa.Table | None = None
    estimated_dict_bytes: int = 0
    dict_fits: bool = True

    # 2. Файл влезает -> materialize -> dict check -> group_by
    if partition_bytes <= safety_limit:
        table = _read_partition_safe(partition_path, memory_guard)
        if table is None or table.num_rows == 0:
            return pa.Table.from_pylist(
                [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
            )

        # must-fix §11.4: dict check до group_by.
        # _read_partition_safe проверяет file_size, но group_by строит
        # hash-таблицу K_i * size_of_key_entry, которая для узких строк
        # больше файла -> OOM внутри pyarrow.
        if memory_guard is not None:
            estimated_k = _estimate_distinct_keys(table, by_columns)
            estimated_dict_bytes = estimated_k * size_of_key_entry_bytes
            dict_limit = int(memory_guard.available_bytes() * 0.5)
            dict_fits = estimated_dict_bytes <= dict_limit

        if dict_fits:
            agg_specs = [(col, f) for col, f in zip(agg_columns, agg_funcs)]
            return table.group_by(list(by_columns)).aggregate(agg_specs)
        # fall through: file fits, dict overflow -> split

    # 3. Файл или dict не влезает -> рекурсивное партиционирование
    if memory_guard is None or spill_dir is None:
        raise MemoryLimitExceeded(
            f"_local_groupby: file={partition_bytes} B, "
            f"dict~{estimated_dict_bytes} B, safety={safety_limit} B; "
            f"memory_guard/spill_dir not provided."
        )
    if depth >= max_depth:
        raise MemoryLimitExceeded(
            f"_local_groupby: depth={depth} >= max_depth={max_depth}; "
            f"file={partition_bytes} B, dict~{estimated_dict_bytes} B, "
            f"safety={safety_limit} B. Likely skew (DECISIONS §11.4)."
        )

    def _read_for_resplit():
        # Если таблица уже в памяти (file fits, dict overflow) — не
        # перечитываем с диска: сплитим in-memory batches.
        if table is not None:
            keep = [
                c for c in list(by_columns) + list(agg_columns)
                if c in table.schema.names
            ]
            for b in table.select(keep).to_batches():
                yield b
            return
        for b in ArrowIPCBatchReader(str(partition_path), None):
            cols = [
                c for c in list(by_columns) + list(agg_columns)
                if c in b.schema.names
            ]
            yield b.select(cols)

'''
s = s[:m.start()] + new_block + s[m.end():]
p.write_text(s)
print("groupby.py: _local_groupby — dict check, in-memory resplit")
PY

# ---------- tests/test_mustfix_dict_check.py ----------
cat > tests/test_mustfix_dict_check.py <<'EOF'
"""
Must-fix §11.4: dict check в `_local_groupby` до `group_by`.

Проверяем:
  - AST: dict-check marker (`estimated_dict_bytes`/`dict_fits`/
    `_estimate_distinct_keys`) предшествует `group_by(`.
  - behavior: file fits, dict overflow -> split (hash_shuffle called);
  - behavior: dict fits -> no recursion;
  - error: max_depth исчерпан при dict overflow -> MemoryLimitExceeded.
"""
import inspect
from pathlib import Path

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops import groupby as gb_mod
from cadw.ops.groupby import _local_groupby


def write_partition(path: Path, table: pa.Table, chunk: int = 200) -> Path:
    writer = ArrowIPCBatchWriter(str(path), table.schema)
    try:
        for b in table.to_batches(max_chunksize=chunk):
            writer.write_batch(b)
    finally:
        writer.close()
    return path


# ---------- AST guard ----------

def test_local_groupby_dict_check_before_group_by():
    """AST: dict check предшествует group_by в _local_groupby."""
    src = inspect.getsource(_local_groupby)
    gb_pos = src.find("group_by(")
    assert gb_pos >= 0, "group_by( not found in _local_groupby"
    markers = [
        src.find("_estimate_distinct_keys"),
        src.find("estimated_dict_bytes"),
        src.find("dict_fits"),
    ]
    valid = [p for p in markers if p >= 0]
    assert valid, (
        "no dict-check marker found in _local_groupby "
        "(must-fix §11.4: estimated_dict_bytes/dict_fits/_estimate_distinct_keys)"
    )
    earliest = min(valid)
    assert earliest < gb_pos, (
        f"dict check at pos {earliest} must precede group_by( at pos {gb_pos}. "
        f"Source: {src[:200]}..."
    )


# ---------- behavior: dict overflow triggers split ----------

def test_dict_overflow_triggers_split(tmp_path, monkeypatch):
    """File fits, dict overflow -> hash_shuffle called (recursion)."""
    called = []
    orig_hash_shuffle = gb_mod.hash_shuffle

    def spy(*args, **kwargs):
        called.append(kwargs.get("hash_seed"))
        return orig_hash_shuffle(*args, **kwargs)

    monkeypatch.setattr(gb_mod, "hash_shuffle", spy)

    # N=1000, K=1000 (unique int32 keys). row_size=8 B.
    n = 1000
    table = pa.table({
        "cat": pa.array(list(range(n)), type=pa.int32()),
        "val": pa.array(list(range(n)), type=pa.int32()),
    })
    p = write_partition(tmp_path / "p.arrow", table)

    # budget=100 KB -> safety=50 KB; file ~10 KB fits, dict=80 KB overflow
    guard = MemoryGuard(max_memory_bytes=100_000)
    result = _local_groupby(
        p, ["cat"], ["val"], ["count"],
        memory_guard=guard,
        spill_dir=tmp_path,
        size_of_key_entry_bytes=80,
    )

    assert called, "dict overflow must trigger hash_shuffle (recursion)"
    assert result.num_rows == n, (
        f"expected {n} groups, got {result.num_rows}"
    )


# ---------- behavior: dict fits, no recursion ----------

def test_dict_fits_no_recursion(tmp_path, monkeypatch):
    """K мал, dict fits -> hash_shuffle не вызывается."""
    called = []
    orig_hash_shuffle = gb_mod.hash_shuffle

    def spy(*args, **kwargs):
        called.append(True)
        return orig_hash_shuffle(*args, **kwargs)

    monkeypatch.setattr(gb_mod, "hash_shuffle", spy)

    # N=100, K=10 (10 unique keys). dict = 10*80 = 800 B
    n = 100
    keys = [i % 10 for i in range(n)]
    table = pa.table({
        "cat": pa.array(keys, type=pa.int32()),
        "val": pa.array(list(range(n)), type=pa.int32()),
    })
    p = write_partition(tmp_path / "p.arrow", table)

    guard = MemoryGuard(max_memory_bytes=100_000)  # 100 KB; dict_limit=50 KB
    result = _local_groupby(
        p, ["cat"], ["val"], ["count"],
        memory_guard=guard,
        spill_dir=tmp_path,
        size_of_key_entry_bytes=80,
    )

    assert not called, "dict fits -> no hash_shuffle expected"
    assert result.num_rows == 10


# ---------- error: max_depth exceeded ----------

def test_max_depth_raises_on_dict_overflow(tmp_path):
    """dict overflow + depth >= max_depth -> MemoryLimitExceeded."""
    n = 1000
    table = pa.table({
        "cat": pa.array(list(range(n)), type=pa.int32()),
        "val": pa.array(list(range(n)), type=pa.int32()),
    })
    p = write_partition(tmp_path / "p.arrow", table)

    guard = MemoryGuard(max_memory_bytes=100_000)
    with pytest.raises(MemoryLimitExceeded, match="max_depth"):
        _local_groupby(
            p, ["cat"], ["val"], ["count"],
            memory_guard=guard,
            spill_dir=tmp_path,
            depth=0,
            max_depth=0,
            size_of_key_entry_bytes=80,
        )


# ---------- invariants ----------

def test_dict_check_estimate_upper_bound(tmp_path):
    """_estimate_distinct_keys — верхняя оценка: <= num_rows, >= K_true."""
    from cadw.ops.groupby import _estimate_distinct_keys

    n = 200
    keys = [i % 50 for i in range(n)]  # K_true = 50
    table = pa.table({"cat": pa.array(keys, type=pa.int32())})
    est = _estimate_distinct_keys(table, ["cat"])
    assert est == 50, f"expected 50, got {est}"

    # multi-key
    table2 = pa.table({
        "a": pa.array([i % 5 for i in range(100)], type=pa.int32()),
        "b": pa.array([i % 7 for i in range(100)], type=pa.int32()),
    })
    est2 = _estimate_distinct_keys(table2, ["a", "b"])
    # product(5, 7) = 35; K_true <= 35
    assert est2 == 35, f"expected 35, got {est2}"
    assert est2 <= 100
EOF

# ---------- DECISIONS.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

# §11.1: строка
if "Materialize-before-check fix (10)" in s and "dict check в `_local_groupby` (11-A-4a)" not in s:
    s = s.replace(
        "| Materialize-before-check fix (10) | ✅ | (см. test_materialize_guard) |",
        "| Materialize-before-check fix (10) | ✅ | (см. test_materialize_guard) |\n"
        "| dict check в `_local_groupby` (11-A-4a, must-fix §11.4) | ✅ | 5 |",
        1,
    )

# §11.4: закрытие dict must-fix (вставить после блока про 02b2)
marker_114 = "**Must-fix, связанные с §9:**"
if marker_114 in s and "dict check" not in s.split(marker_114)[1][:200]:
    old = marker_114
    new = (
        "**Must-fix: dict check в `_local_groupby` (закрыто 11-A-4a).**\n"
        "`_read_partition_safe` проверяет `file_size`, но `group_by` строит\n"
        "hash-таблицу `K_i × size_of_key_entry`. Для узких строк (16 B)\n"
        "dict больше файла: 10^6 уникальных ключей дают dict 80 MB при файле\n"
        "16 MB. Если файл проходит по `file_size`, но dict нет — OOM внутри\n"
        "pyarrow после `check_state`, который проверял только file. **Fix:**\n"
        "`_estimate_distinct_keys` (product of nunique, upper bound) ×\n"
        "`size_of_key_entry` проверяется до `group_by`; при overflow —\n"
        "in-memory recursion через `hash_shuffle`; при `depth >= max_depth`\n"
        "— `MemoryLimitExceeded` с указанием file/dict/safety. **AST guard:**\n"
        "`test_local_groupby_dict_check_before_group_by`.\n"
        "\n"
        + old
    )
    s = s.replace(old, new, 1)

# §13.4: эпизод 35
marker_135 = "### 13.5. Открытые вопросы"
episode_35 = """**Эпизод 35 (must-fix, dict check).** До 11-A-4a `_local_groupby`
проверял только `file_size <= safety`. Но `group_by` строит hash-таблицу
`K_i × size_of_key_entry`; для узких строк dict больше файла: 10^6
уникальных ключей при `row_size=16` дают file 16 MB, dict 80 MB. При
budget 50 MB: file проходит (16 ≤ 25), dict нет (80 > 25) — OOM **внутри**
pyarrow после `check_state`, который проверял только file. Это третья
форма класса «materialize-before-check»: file check есть, но не тот
объект, который materialize-ится. **Fix (11-A-4a):** `_estimate_distinct_keys`
(product of nunique, upper bound, без построения hash-таблицы) ×
`size_of_key_entry` проверяется до `group_by`; при overflow — in-memory
recursion (не перечитываем с диска: `table.to_batches()`); при
`depth >= max_depth` — `MemoryLimitExceeded`. **Урок:** `check_state`
должен проверять **тот** объект, который реально строится следующей
операцией. File size и dict size — разные объекты; проверять нужно оба.

"""
if "Эпизод 35" not in s:
    s = s.replace(marker_135, episode_35 + marker_135, 1)

# §14 резюме
s = s.replace(
    "**v0.1.6 (патч 11-A-3):** планировщик памяти `MemoryGuard.plan`.",
    "**v0.1.6 (патч 11-A-3):** планировщик памяти `MemoryGuard.plan`.\n\n"
    "**v0.1.7 (патч 11-A-4a):** must-fix dict check в `_local_groupby`\n"
    "(AST guard + behavior tests + max_depth error).",
    1,
) if "v0.1.6 (патч 11-A-3)" in s else s

p.write_text(s)
print("DECISIONS: §11.1, §11.4, §13.4/35, §14")
PY

# ---------- CHANGELOG ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.1.7] — 2026-10-19

### Исправлено (must-fix §11.4)
- `_local_groupby`: dict check до `group_by`. `_read_partition_safe`
  проверял только `file_size`, но `group_by` строит hash-таблицу
  `K_i × size_of_key_entry` — для узких строк dict больше файла, OOM
  внутри pyarrow. `_estimate_distinct_keys` (product of nunique) ×
  `size_of_key_entry` проверяется до `group_by`; overflow -> in-memory
  recursion через `hash_shuffle`; `depth >= max_depth` ->
  `MemoryLimitExceeded` с file/dict/safety.

### Добавлено
- `_estimate_distinct_keys(table, by_columns) -> int` — upper bound без
  построения hash-таблицы (`pc.count_distinct` per column).
- AST guard `test_local_groupby_dict_check_before_group_by`.
- Behavior tests: `test_dict_overflow_triggers_split`,
  `test_dict_fits_no_recursion`, `test_max_depth_raises_on_dict_overflow`.

"""
if "## [0.1.7]" not in s:
    s = s.replace("## [0.1.6]", entry + "## [0.1.6]", 1) if "## [0.1.6]" in s else s.replace("## [0.1.0]", entry + "## [0.1.0]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.1.7")
PY

# ---------- run ----------
echo "=== pytest tests/test_mustfix_dict_check.py ==="
set +e
uv run pytest tests/test_mustfix_dict_check.py -v
RC_MODELS=$?
set -e
if [[ $RC_MODELS -ne 0 ]]; then
  echo "test_mustfix_dict_check.py: rc=$RC_MODELS"
  git add -A
  git commit -m "WIP: 11-A-4a (test_mustfix_dict_check red; fix follows)" || true
  exit 1
fi

uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-A-4a: must-fix dict check в _local_groupby (§11.4) + AST guard + behavior tests"

echo "==> sha256"
sha256sum cadw/ops/groupby.py tests/test_mustfix_dict_check.py
echo "==> 11-A-4a applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"