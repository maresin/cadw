#!/usr/bin/env bash
# patch_11b2d_fix8: обновить regex в test_merge_cartesian_skew_raises
# (сообщение теперь "exact matches", а не "cartesian|expected").
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/11b2d-e2e-join"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_merge_planner.py")
s = p.read_text()

old = '''    with pytest.raises(MemoryLimitExceeded, match="cartesian|expected"):
        left.merge(right, on="k")'''

new = '''    # Сообщение: "exact matches between this probe batch ... Likely skew."
    # (с v0.2.5 cartesian check заменён на exact-match; §13.4/47).
    with pytest.raises(MemoryLimitExceeded, match="exact matches|Likely skew"):
        left.merge(right, on="k")'''

if old not in s:
    # fallback: найти любой похожий match=
    import re
    m = re.search(
        r'with pytest\.raises\(MemoryLimitExceeded,\s*match="[^"]*"\):\s*\n\s*left\.merge',
        s,
    )
    if not m:
        raise SystemExit("test_merge_cartesian_skew_raises pattern not found")
    s = s[:m.start()] + new + s[m.end():]
    print("test_merge_planner.py: regex обновлён (fallback)")
else:
    s = s.replace(old, new, 1)
    print("test_merge_planner.py: regex обновлён")

p.write_text(s)
print("saved")
PY

# ---------- run ----------
echo "=== pytest tests/test_merge_planner.py -v ==="
set +e
uv run pytest tests/test_merge_planner.py -v
RC_TARGET=$?
set -e
if [[ $RC_TARGET -ne 0 ]]; then
  echo "merge_planner: rc=$RC_TARGET"
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
git commit -m "11-B-2d-fix8: test_merge_cartesian_skew_raises — regex под 'exact matches' (сообщение сменилось в v0.2.5)"

echo "==> 11-B-2d-fix8 applied"