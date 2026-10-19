#!/usr/bin/env bash
# patch_v052b_fix8: убрать `sort_keys=by` из комментария; verify — precise.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v052b-pylist"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

git add -A
git commit -m "WIP: v0.5.2b (comment mentions sort_keys=by; over-strict assert)" || true
require_clean_tree || exit 1

uv run python - <<'PY'
from pathlib import Path
p = Path("cadw/ops/rank.py")
s = p.read_text()

# 1. Fix comment (still mentions sort_keys=by)
old_comment = '''            # v0.5.2b (P11): 'first' fast path — Arrow-native, no to_pylist.
            # `pc.rank(batch, sort_keys=by, tiebreaker='first')` — 1-based
            # ordinal rank per batch. Глобальный offset: global_pos.
            if method == "first":'''

new_comment = '''            # v0.5.2b (P11): 'first' fast path — Arrow-native, no to_pylist.
            # Input sorted (external_sort); rank(i) = i+1 + global_pos —
            # ordinal position. np.arange — C-level, no per-row Python.
            if method == "first":'''

if old_comment in s:
    s = s.replace(old_comment, new_comment, 1)
    print("rank.py: comment обновлён (no sort_keys=by in text)")
else:
    # Comment may already be fixed or different form
    if "sort_keys=by" in s.split("def _compute_streamable")[1][:2000]:
        # still present somewhere in first block, just remove via regex
        import re
        s = re.sub(
            r"            # `pc\.rank\(batch, sort_keys=by, tiebreaker='first'\)` — 1-based\n",
            "",
            s,
        )
        print("rank.py: comment removal via regex")
    else:
        print("rank.py: comment already fixed (no-op)")

p.write_text(s)
print("rank.py: saved")
PY

# Verify — precise (code-level, not substring)
echo ""
echo "=== verify (precise) ==="
uv run python - <<'PY'
from pathlib import Path
s = Path("cadw/ops/rank.py").read_text()

# Precise: no _pc.rank(... sort_keys=by) in CODE
import re
code_violations = re.findall(
    r"_pc\.rank\([^)]*sort_keys=by",
    s, flags=re.DOTALL,
)
assert not code_violations, (
    f"Code still calls _pc.rank(..., sort_keys=by): {code_violations}"
)
print("  OK: no _pc.rank(sort_keys=by) in code")

# np.arange present
assert "np.arange" in s, "np.arange not found"
print("  OK: np.arange in 'first' branch")

# Broad substring — allow in comments only
count_all = s.count("sort_keys=by")
count_code = len(code_violations)
if count_all > count_code:
    print(f"  NOTE: 'sort_keys=by' appears {count_all - count_code} time(s) in comments (OK)")

# Compile
import py_compile
py_compile.compile("cadw/ops/rank.py", doraise=True)
print("  OK: compiles")
PY

# Run tests
echo ""
echo "=== pytest tests/test_pylist_fixes.py -v ==="
set +e
uv run pytest tests/test_pylist_fixes.py -v
RC1=$?
set -e
if [[ $RC1 -ne 0 ]]; then
  echo "new tests: rc=$RC1"
  uv run pytest tests/test_pylist_fixes.py -x -q 2>&1 | tail -40
  exit 1
fi

echo ""
echo "=== pytest tests/test_rank.py tests/test_quantile.py ==="
set +e
uv run pytest tests/test_rank.py tests/test_quantile.py -q
RC2=$?
set -e
if [[ $RC2 -ne 0 ]]; then
  echo "regression: rc=$RC2"
  exit 1
fi

echo ""
echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.5.2b-fix8: comment cleanup (sort_keys=by); precise verify (code-only)"

echo ""
echo "==> v0.5.2b-fix8 applied"