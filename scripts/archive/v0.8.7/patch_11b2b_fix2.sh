#!/usr/bin/env bash
# patch_11b2b_fix2: _safe_materialize_partition — schema-correct для пустых
# файлов (schema-only); fallback в _process_partition_pair для 0-columns.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/11b2b-join-oom-safety"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11b2b-fix2-${TS}"
git diff > ".backups/backup-pre-11b2b-fix2-${TS}/wip.diff" || true

# Закоммитить WIP от прошлого патча (patch 11b2b_fix упал до commit).
git add -A
git commit -m "WIP: 11-B-2b-fix (join left empty partitions; correction follows)" || true

# Теперь дерево чистое.
require_clean_tree || exit 1

# ---------- Fix 1: _safe_materialize_partition — schema-correct ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/join.py")
s = p.read_text()

old = '''def _safe_materialize_partition(
    path: Path,
    memory_guard: MemoryGuard,
    *,
    safety_ratio: float = 0.5,
) -> "pa.Table | None":
    """Прочитать партицию в память, только если она влезает в бюджет.

    Returns None если file size > available * safety_ratio — caller должен
    рекурсивно разбить партицию (см. _process_partition_pair).

    Arrow IPC без сжатия: file size ≈ in-memory nbytes. Проверка stat()
    до list(reader) закрывает класс «materialize-before-check» (§11.4).
    """
    size = path.stat().st_size
    limit = int(memory_guard.available_bytes() * safety_ratio)
    if size > limit:
        return None

    batches = []
    for b in ArrowIPCBatchReader(str(path), None):
        keep = [n for n in b.schema.names if n != "row_id"]
        batches.append(b.select(keep))
    if not batches:
        return pa.Table.from_batches([], schema=pa.schema([]))
    table = pa.Table.from_batches(batches)
    memory_guard.check_state(
        table.nbytes, name=f"join_build({path.name})"
    )
    return table'''

new = '''def _safe_materialize_partition(
    path: Path,
    memory_guard: MemoryGuard,
    *,
    safety_ratio: float = 0.5,
) -> "pa.Table | None":
    """Прочитать партицию в память, только если она влезает в бюджет.

    Returns None если file size > available * safety_ratio — caller должен
    рекурсивно разбить партицию (см. _process_partition_pair).

    Arrow IPC без сжатия: file size ≈ in-memory nbytes. Проверка stat()
    до list(reader) закрывает класс «materialize-before-check» (§11.4).

    ВАЖНО: пустой файл (без record batches) НЕ 0 байт — Arrow IPC всегда
    пишет schema-заголовок (~200-500 B). Возвращаем schema-correct
    empty_table() с колонками, а не pa.schema([]) — иначе pd.merge
    получит DataFrame без on-колонок → KeyError.
    """
    size = path.stat().st_size
    limit = int(memory_guard.available_bytes() * safety_ratio)
    if size > limit:
        return None

    import pyarrow.ipc as ipc
    ipc_reader = ipc.open_file(str(path))
    full_schema = ipc_reader.schema
    keep_cols = [n for n in full_schema.names if n != "row_id"]
    out_schema = pa.schema([full_schema.field(c) for c in keep_cols])

    n_batches = ipc_reader.num_record_batches
    if n_batches == 0:
        return out_schema.empty_table()

    batches = [
        ipc_reader.get_batch(i).select(keep_cols)
        for i in range(n_batches)
    ]
    table = pa.Table.from_batches(batches)
    memory_guard.check_state(
        table.nbytes, name=f"join_build({path.name})"
    )
    return table'''

if old not in s:
    raise SystemExit("_safe_materialize_partition pattern not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("join.py: _safe_materialize_partition — schema-correct для пустых файлов")
PY

# ---------- Fix 2: defensive check в _process_partition_pair ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/join.py")
s = p.read_text()

# Проверим, что текущий код содержит _drop_null_keys + if len == 0.
# Убедимся, что есть проверка на 0-columns build_df.
if "build_df.shape[1] == 0" in s:
    print("join.py: defensive check уже присутствует")
else:
    old = '''    build_df = _drop_null_keys(build_table.to_pandas(), list(on))
    if len(build_df) == 0:
        if how == "inner":
            return []
        # how='left': build = right; пустой build (0 совпадений) —
        # pd.merge требует on-колонки, поэтому создаём пустой DataFrame
        # с правильными колонками.
        right_rest = [c for c in right_schema.names if c not in on]
        build_df = pd.DataFrame(columns=list(on) + right_rest)
'''
    if old in s:
        new = '''    build_df = _drop_null_keys(build_table.to_pandas(), list(on))
    if len(build_df) == 0:
        if how == "inner":
            return []
        # how='left': build = right; пустой build (0 совпадений).
        # pd.merge требует on-колонки в обоих DataFrame. Если build_df
        # после _drop_null_keys потерял колонки (schema-empty table) —
        # восстанавливаем их явно.
        if build_df.shape[1] == 0:
            right_rest = [c for c in right_schema.names if c not in on]
            build_df = pd.DataFrame(columns=list(on) + right_rest)
'''
        s = s.replace(old, new, 1)
        print("join.py: defensive check добавлен")
    else:
        # Старый формат (fix 11b2b_fix с return[] в середине)
        old2 = '''    build_df = _drop_null_keys(build_table.to_pandas(), list(on))
    if len(build_df) == 0 and how == "inner":
        return []
    # how='left': пустой build — продолжаем (left preserved).
'''
        if old2 in s:
            new2 = '''    build_df = _drop_null_keys(build_table.to_pandas(), list(on))
    if len(build_df) == 0:
        if how == "inner":
            return []
        if build_df.shape[1] == 0:
            right_rest = [c for c in right_schema.names if c not in on]
            build_df = pd.DataFrame(columns=list(on) + right_rest)
'''
            s = s.replace(old2, new2, 1)
            print("join.py: defensive check добавлен (old2 variant)")
        else:
            raise SystemExit("build_df pattern not found in any known form")
p.write_text(s)
PY

# ---------- Fix 3: тест на schema-correct empty partition ----------
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_join_oom_guard.py")
s = p.read_text()

extra = '''

def test_safe_materialize_empty_file_has_columns(tmp_path):
    """Schema-only IPC file → таблица с колонками, 0 строк."""
    from cadw.io.batch_writer import ArrowIPCBatchWriter
    from cadw.ops.join import _safe_materialize_partition

    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])
    p = tmp_path / "empty.arrow"
    writer = ArrowIPCBatchWriter(str(p), schema)
    try:
        pass  # 0 batches — но schema-заголовок записан
    finally:
        writer.close()

    assert p.stat().st_size > 0, "Arrow IPC schema header must be > 0 B"

    guard = MemoryGuard(max_memory_bytes=10 * 1024 * 1024)
    table = _safe_materialize_partition(p, guard)
    assert table is not None
    assert set(table.schema.names) == {"k", "v"}, (
        f"empty file must have schema columns, got {table.schema.names}"
    )
    assert table.num_rows == 0

'''
s = s.rstrip() + extra
p.write_text(s)
print("test_join_oom_guard.py: +test_safe_materialize_empty_file_has_columns")
PY

# ---------- run ----------
echo "=== pytest tests/test_join.py tests/test_join_oom_guard.py ==="
set +e
uv run pytest tests/test_join.py tests/test_join_oom_guard.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "join/oom_guard fix2: rc=$RC"
  exit 1
fi

uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-B-2b-fix2: _safe_materialize_partition schema-correct для пустых IPC-файлов; defensive check на 0-columns build_df"

echo "==> 11-B-2b-fix2 applied"