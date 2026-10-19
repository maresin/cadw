#!/usr/bin/env bash
# patch_11b2b_fix3: AST guard на порядок stat->read для _safe_materialize_partition;
# учитывает и ipc.open_file, и ArrowIPCBatchReader.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/11b2b-join-oom-safety"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11b2b-fix3-${TS}"
git diff > ".backups/backup-pre-11b2b-fix3-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: 11-B-2b-fix2 (AST guard for open_file; correction follows)" || true
require_clean_tree || exit 1

# ---------- Fix: AST guard на _safe_materialize_partition ----------
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_join_oom_guard.py")
s = p.read_text()

old = '''def test_materialize_partition_has_size_check():
    """_safe_materialize_partition: stat().st_size до чтения."""
    src = inspect.getsource(join_mod._safe_materialize_partition)
    stat_pos = src.find(".stat().st_size")
    read_pos = src.find("ArrowIPCBatchReader")
    assert stat_pos >= 0, "stat().st_size not found in _safe_materialize_partition"
    assert read_pos >= 0, "ArrowIPCBatchReader not found"
    assert stat_pos < read_pos, (
        f"stat() (pos {stat_pos}) must precede reader (pos {read_pos})"
    )'''

new = '''def test_materialize_partition_has_size_check():
    """_safe_materialize_partition: stat().st_size до любого чтения IPC.

    Чтение может быть через ArrowIPCBatchReader или ipc.open_file — оба
    читают файл. AST-guard: позиция .stat().st_size < позиция первого
    вызова, читающего файл.
    """
    fn = join_mod._safe_materialize_partition
    src = inspect.getsource(fn)
    stat_pos = src.find(".stat().st_size")
    assert stat_pos >= 0, "stat().st_size not found in _safe_materialize_partition"

    # Ищем в AST все Call-узлы, читающие файл.
    import ast
    tree = ast.parse(inspect.cleandoc(src))
    read_names = {"ArrowIPCBatchReader", "ParquetBatchReader"}
    read_attrs = {"open_file", "open_stream", "memory_map"}

    def is_file_read(call: ast.Call) -> bool:
        f = call.func
        if isinstance(f, ast.Name) and f.id in read_names:
            return True
        if isinstance(f, ast.Attribute) and f.attr in read_attrs:
            return True
        return False

    # Найдём минимальную lineno среди file-read calls.
    read_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and is_file_read(node)
    ]
    assert read_lines, (
        "no file-read call found in _safe_materialize_partition "
        "(expected ipc.open_file or ArrowIPCBatchReader)"
    )
    first_read_line = min(read_lines)

    # stat() должен быть раньше первого чтения. Строки считаем по offset:
    # stat_pos — символьная позиция, read_line — номер строки. Сравниваем
    # через подсчёт строк до stat_pos.
    stat_line = src.count("\\n", 0, stat_pos) + 1
    assert stat_line < first_read_line, (
        f"stat() at line {stat_line} must precede first file-read at line "
        f"{first_read_line}. Source:\\n{src}"
    )'''

if old not in s:
    raise SystemExit("old test_materialize_partition_has_size_check not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("test_join_oom_guard.py: AST guard на stat->read (open_file-aware)")
PY

# ---------- run ----------
echo "=== pytest tests/test_join_oom_guard.py -v ==="
set +e
uv run pytest tests/test_join_oom_guard.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "oom_guard fix3: rc=$RC"
  exit 1
fi

echo "=== full fast + slow ==="
uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-B-2b-fix3: AST guard stat->read учитывает ipc.open_file; тест на schema-correct empty IPC"

echo "==> 11-B-2b-fix3 applied"