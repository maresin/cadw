#!/usr/bin/env bash
# patch_v052a_fix: sys.path insert в bench_p5_sort.py; докончить прогон.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v052a-sort-buffer"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-v052a-fix-${TS}"
git diff > ".backups/backup-pre-v052a-fix-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: v0.5.2a P5 (bench path fix needed)" || true
require_clean_tree || exit 1

# Fix: add sys.path insert before cadw imports
python3 - <<'PY'
from pathlib import Path
p = Path("scripts/bench_p5_sort.py")
s = p.read_text()

old = '''import gc, tempfile, threading, time
from pathlib import Path

import psutil
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.sort import external_sort'''

new = '''import gc, sys, tempfile, threading, time
from pathlib import Path

import psutil
import pyarrow as pa

# Ensure project root on sys.path when run as `python scripts/bench_p5_sort.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.sort import external_sort'''

if old not in s:
    raise SystemExit("import block not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("bench_p5_sort.py: sys.path.insert(0, project_root)")
PY

# Compile-check
python3 -c "import py_compile; py_compile.compile('scripts/bench_p5_sort.py', doraise=True); print('OK: bench compiles')"

# ============ Run ============
echo ""
echo "=== bench: P5 sort (baseline Pingis B3a: +31.9 MB / 30 MB budget) ==="
uv run python scripts/bench_p5_sort.py

echo ""
echo "=== pytest tests/test_sort.py -v ==="
set +e
uv run pytest tests/test_sort.py -v
RC1=$?
set -e
if [[ $RC1 -ne 0 ]]; then
  echo "sort: rc=$RC1"
  exit 1
fi

echo ""
echo "=== pytest full fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest full slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.5.2a-fix: bench_p5_sort sys.path; P5 verified"

echo "==> v0.5.2a-fix applied"