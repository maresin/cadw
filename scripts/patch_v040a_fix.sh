#!/usr/bin/env bash
# patch_v040a_fix: reset_index error message — English "already" (regex test).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v040a-cadw-index"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-v040a-fix-${TS}"
git diff > ".backups/backup-pre-v040a-fix-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: v0.4.0a-fix (error message language)" || true
require_clean_tree || exit 1

python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

old = '''            raise ValueError(
                "reset_index(drop=False): index уже RangeIndex — "
                "нет смысла выносить в колонку. "
                "Используйте reset_index(drop=True) или set_index(col) сначала."
            )'''
new = '''            raise ValueError(
                "reset_index(drop=False): already have a default "
                "RangeIndex — cannot insert. "
                "Use reset_index(drop=True) or set_index(col) first."
            )'''
if old not in s:
    raise SystemExit("reset_index ValueError not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("dataframe.py: reset_index error message — English")
PY

# ---------- run ----------
echo "=== pytest tests/test_cadw_index.py -v ==="
set +e
uv run pytest tests/test_cadw_index.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "cadw_index fix: rc=$RC"
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.4.0a-fix: reset_index error message English (matches pandas style; 26/26 tests)"

echo "==> v0.4.0a-fix applied"