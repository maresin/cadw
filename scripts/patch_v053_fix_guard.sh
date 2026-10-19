#!/usr/bin/env bash
# patch_v053_fix_guard: обновить test_actual_matches_checked_before_pd_merge
# под Arrow API (P1 v0.5.3): _exact_actual_matches_arrow before Table.join.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v053-join-arrow"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

git add -A
git commit -m "WIP: v0.5.3 (guard obsolete after pandas→Arrow)" || true
require_clean_tree || exit 1

uv run python - <<'PY'
from pathlib import Path
p = Path("tests/test_join_oom_guard.py")
s = p.read_text()

old = '''def test_actual_matches_checked_before_pd_merge():
    """Exact-match guard вызывается ДО pd.merge в _process_partition_pair.

    Регрессия 11-B-2d: cartesian check (build × probe > max_cartesian_rows)
    давал ложные reject'ы для PK-FK (build уникален, реальный output =
    probe count, а не cartesian). Fix: точная оценка через _exact_actual_matches.
    """
    from cadw.ops.join import _process_partition_pair
    import ast

    src = inspect.getsource(_process_partition_pair)
    tree = ast.parse(inspect.cleandoc(src))

    exact_lines: list[int] = []
    merge_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "_exact_actual_matches":
                exact_lines.append(node.lineno)
            if isinstance(f, ast.Attribute) and f.attr == "merge" \\
                    and isinstance(f.value, ast.Name) and f.value.id == "pd":
                merge_lines.append(node.lineno)

    assert exact_lines, "_exact_actual_matches not called in _process_partition_pair"
    assert merge_lines, "pd.merge not called in _process_partition_pair"
    assert min(exact_lines) < min(merge_lines), (
        f"_exact_actual_matches (line {min(exact_lines)}) must precede "
        f"pd.merge (line {min(merge_lines)})"
    )'''

new = '''def test_actual_matches_checked_before_join():
    """Exact-match guard вызывается ДО join-операции в _process_partition_pair.

    v0.5.3 (P1): pandas → Arrow Table.join. Guard:
    - `_exact_actual_matches_arrow` must be called;
    - `Table.join` (или `.join(..., join_type=...)`) вызывается после.

    Регрессия 11-B-2d: cartesian check (build × probe > max_cartesian_rows)
    давал ложные reject'ы для PK-FK. Fix: точная оценка через
    `_exact_actual_matches_arrow` (pc.value_counts + join).
    """
    from cadw.ops.join import _process_partition_pair
    import ast

    src = inspect.getsource(_process_partition_pair)
    tree = ast.parse(inspect.cleandoc(src))

    exact_lines: list[int] = []
    join_lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        # Arrow-native exact-match check
        if isinstance(f, ast.Name) and f.id == "_exact_actual_matches_arrow":
            exact_lines.append(node.lineno)
        # Table.join — method call with `join_type=` kwarg (signature
        # `left_t.join(right_t, keys=..., join_type=...)`).
        if isinstance(f, ast.Attribute) and f.attr == "join":
            has_join_type = any(
                kw.arg == "join_type" for kw in node.keywords
            )
            if has_join_type:
                join_lines.append(node.lineno)

    assert exact_lines, (
        "_exact_actual_matches_arrow not called in _process_partition_pair "
        "(P1 v0.5.3 expects Arrow-native exact-match guard)"
    )
    assert join_lines, (
        "Table.join (with join_type= kwarg) not called in "
        "_process_partition_pair — P1 v0.5.3 expects Arrow-native join"
    )
    assert min(exact_lines) < min(join_lines), (
        f"_exact_actual_matches_arrow (line {min(exact_lines)}) must precede "
        f"Table.join (line {min(join_lines)})"
    )'''

if old not in s:
    raise SystemExit("old guard test not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("test_join_oom_guard.py: guard updated for Arrow Table.join")
PY

echo ""
echo "=== pytest tests/test_join_oom_guard.py -v ==="
set +e
uv run pytest tests/test_join_oom_guard.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "guard: rc=$RC"
  exit 1
fi

echo ""
echo "=== bench: join 2M × 400k (was: 14.1s, +2.9 MB) ==="
set +e
uv run python scripts/bench_join_v053.py 2>&1
set -e

echo ""
echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.5.3-fix: AST guard updated for Arrow Table.join (was pandas/pd.merge)"

echo ""
echo "==> v0.5.3-fix applied"