#!/usr/bin/env bash
# patch_v041_fix: две лишние ')' в _LocIndexer._slice (open endpoints).
# + compile-check на всех изменённых py-файлах до коммита (meta-fix).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v041-loc-complete"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-v041-fix-${TS}"
git diff > ".backups/backup-pre-v041-fix-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: v0.4.1-fix (syntax error in indexers.py; fix follows)" || true
require_clean_tree || exit 1

# ---------- Fix: two typo ) → ] ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/indexers.py")
s = p.read_text()

fixes = [
    (
        "            return self._parent.iloc[list(range(0, ph + 1)))",
        "            return self._parent.iloc[list(range(0, ph + 1))]",
    ),
    (
        "        return self._parent.iloc[list(range(pl, n)))",
        "        return self._parent.iloc[list(range(pl, n))]",
    ),
]

for old, new in fixes:
    if old not in s:
        raise SystemExit(f"expected line not found: {old!r}")
    s = s.replace(old, new, 1)

p.write_text(s)
print("indexers.py: 2 typos fixed ()] → ])")
PY

# ---------- Compile check (META-FIX: before commit, not at test collection) ----------
echo "=== compile check: cadw/indexers.py ==="
python3 -c "
import py_compile
py_compile.compile('cadw/indexers.py', doraise=True)
print('OK: cadw/indexers.py compiles')
"

echo "=== compile check: all cadw py (sanity) ==="
python3 -c "
import compileall, sys
if not compileall.compile_dir('cadw', quiet=1, force=True):
    print('FAIL: cadw/ has syntax errors', file=sys.stderr)
    sys.exit(1)
print('OK: cadw/ all modules compile')
"

# ---------- Run ----------
echo "=== pytest tests/test_loc.py -v ==="
set +e
uv run pytest tests/test_loc.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "loc: rc=$RC"
  exit 1
fi

echo "=== pytest tests/test_cadw_index.py tests/test_iloc.py -v ==="
set +e
uv run pytest tests/test_cadw_index.py tests/test_iloc.py -v
RC2=$?
set -e
if [[ $RC2 -ne 0 ]]; then
  echo "regression: rc=$RC2"
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.4.1-fix: two typos )] → ] in _LocIndexer._slice; compile-check before commit"

echo "==> v0.4.1-fix applied"