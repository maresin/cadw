#!/usr/bin/env bash
# patch_v052b_fix7: 'first' — np.arange вместо pc.rank (sort_keys=by был
# неверен: sort_keys ожидает "ascending"/"descending", а не имя колонки).
# rank('first') на sorted input = ordinal position. Arrow-native, no to_pylist.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v052b-pylist"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

# Save WIP (rank.py broken от v2, memory_guard.py правильный)
git add -A
git commit -m "WIP: v0.5.2b (pc.rank sort_keys wrong; use np.arange)" || true
require_clean_tree || exit 1

# ============ Fix rank.py: 'first' branch ============
uv run python - <<'PY'
from pathlib import Path
p = Path("cadw/ops/rank.py")
s = p.read_text()

old = '''            if method == "first":
                local_rank = _pc.rank(
                    vals_arr, sort_keys=by, tiebreaker="first",
                )
                global_rank = _pc.add(
                    local_rank, pa.scalar(global_pos, type=pa.int64()),
                )
                global_pos += n'''

new = '''            if method == "first":
                # 'first' = ordinal position in sorted order. Input is
                # already sorted (external_sort), so rank(i) = i+1 within
                # batch, offset by global_pos. np.arange — Arrow-native,
                # no Python loop (P11).
                import numpy as _np
                local_rank = pa.array(
                    _np.arange(1, n + 1, dtype=_np.int64),
                    type=pa.int64(),
                )
                global_rank = _pc.add(
                    local_rank, pa.scalar(global_pos, type=pa.int64()),
                )
                global_pos += n'''

if old not in s:
    raise SystemExit("'first' branch pattern not found (current state may differ)")
s = s.replace(old, new, 1)
p.write_text(s)
print("  OK: 'first' branch → np.arange (no pc.rank)")
PY

# ============ Fix test: relax pc.rank check ============
uv run python - <<'PY'
from pathlib import Path
p = Path("tests/test_pylist_fixes.py")
s = p.read_text()

old = '''def test_rank_first_uses_pc_rank():
    """AST guard: 'first' path uses pc.rank (Arrow-native)."""
    import inspect
    from cadw.ops import rank as rank_mod
    src = inspect.getsource(rank_mod._compute_streamable)
    assert "pc.rank" in src or "_pc.rank" in src, (
        "P11 fix: rank 'first' must use pc.rank (Arrow-native)"
    )'''

new = '''def test_rank_first_uses_native():
    """AST guard: 'first' branch uses Arrow/numpy — no to_pylist.

    P11 goal: no Python loop in 'first' path. Uses np.arange + pc.add,
    equivalent to ordinal position on sorted input.
    """
    import inspect
    from cadw.ops import rank as rank_mod
    src = inspect.getsource(rank_mod._compute_streamable)
    idx_first = src.find('if method == "first"')
    assert idx_first >= 0, "first branch not found"
    idx_continue = src.find('continue', idx_first)
    assert idx_continue > idx_first, "continue not found after first"
    first_branch = src[idx_first:idx_continue]
    assert "to_pylist" not in first_branch, (
        "P11: 'first' branch must not call to_pylist"
    )
    assert ("np.arange" in first_branch
            or "_pc.rank" in first_branch
            or "pc.rank" in first_branch), (
        "P11: 'first' branch must use Arrow/numpy native"
    )'''

if old not in s:
    raise SystemExit("test_rank_first_uses_pc_rank not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("  OK: test_rank_first_uses_pc_rank → test_rank_first_uses_native")
PY

# ============ Verify + compile ============
echo ""
echo "=== verify + compile ==="
uv run python -c "
from pathlib import Path
s = Path('cadw/ops/rank.py').read_text()
assert 'np.arange' in s, 'np.arange missing'
assert 'sort_keys=by' not in s, 'sort_keys=by still present'
print('OK: rank.py fixed')
import py_compile
py_compile.compile('cadw/ops/rank.py', doraise=True)
print('OK: rank.py compiles')
"

# ============ Run tests ============
echo ""
echo "=== pytest tests/test_pylist_fixes.py -v ==="
set +e
uv run pytest tests/test_pylist_fixes.py -v
RC1=$?
set -e
if [[ $RC1 -ne 0 ]]; then
  echo "new tests: rc=$RC1"
  echo ""
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
git commit -m "v0.5.2b-fix7: 'first' branch np.arange (pc.rank sort_keys was wrong); AST test relaxed to 'no to_pylist'"

echo ""
echo "==> v0.5.2b-fix7 applied"