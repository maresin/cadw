#!/usr/bin/env python3
"""v0.7.9-fix: (a) _AssignLeavesOp.update → spec[:4];
                  (b) все tree[nid] = (j, thr, left_id, right_id) → 5-tuple.

Диагностика v0.7.9:
- 10 тестов падают на `j, thr, left, right = spec` в `_AssignLeavesOp.update`
  (я патчил только `_assign_leaf_from_array`).
- test_multiclass_gain_aggregates_across_classes — `assert 0.0 > 0`:
  multiclass `_fit_boosting_multiclass` не получил 5-tuple, потому что
  `s.replace(old_binary, new_binary, 1)` дважды — второй replace не сработал
  (multiclass-строка отличается). Fix: replace **all occurrences** без count.

Also:
- AST guard `test_all_spec_unpack_uses_slice`: во всех unpack'ах spec
  должно быть `spec[:4]` или `spec[:5]`, никогда `= spec` (4-unpack).

Run: uv run python scripts/patch_v079_fix.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:25]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


# ============ Fix 1 + 2: unpack + all assign sites ============

def fix_gbdt_py():
    print("\n=== cadw/ml/gbdt.py ===")
    p = Path("cadw/ml/gbdt.py")
    s = p.read_text()

    # ---- Fix 1: all `j, thr, left, right = spec` → spec[:4] ----
    count1 = s.count("j, thr, left, right = spec")
    print(f"  found {count1} × `j, thr, left, right = spec`")
    if count1 == 0:
        # Maybe already patched
        if "j, thr, left, right = spec[:4]" in s:
            print(f"  already patched — skip")
        else:
            raise SystemExit("no unpack site found — unexpected")
    else:
        s = s.replace(
            "j, thr, left, right = spec",
            "j, thr, left, right = spec[:4]",  # all
        )
        print(f"  OK: {count1} × replaced with spec[:4]")

    # ---- Fix 2: all `tree[nid] = (j, thr, left_id, right_id)` (4-tuple) ----
    old_assign = "tree[nid] = (j, thr, left_id, right_id)"
    count2 = s.count(old_assign)
    print(f"  found {count2} × `{old_assign}` (4-tuple, unpatched)")
    if count2 > 0:
        s = s.replace(
            old_assign,
            "tree[nid] = (j, thr, left_id, right_id, float(gain))",
            # no count → replace ALL
        )
        print(f"  OK: {count2} × replaced with 5-tuple")

    # ---- Verify ----
    s2 = s
    n_unpatched_unpack = s2.count("j, thr, left, right = spec\n")
    n_unpatched_unpack2 = s2.count("j, thr, left, right = spec)")
    n_unpatched_assign = s2.count("tree[nid] = (j, thr, left_id, right_id)")
    print(f"  post: unpack_remaining={n_unpatched_unpack+n_unpatched_unpack2}, "
          f"assign_remaining={n_unpatched_assign}")

    p.write_text(s)
    compile_check(p)


# ============ AST guard ============

GUARD_SRC = '''

def test_all_spec_unpack_uses_slice():
    """Все `j, thr, left, right = spec` в cadw/ml/gbdt.py должны быть
    `spec[:4]` (или `spec[:5]`).

    Регрессия v0.7.9-fix: 4-tuple unpack падает после перехода на 5-tuple
    (gain appended). `spec[:4]` — forward-compatible.
    """
    import ast as _ast
    import inspect as _inspect
    from cadw.ml import gbdt as gbdt_mod

    src = _inspect.getsource(gbdt_mod)
    tree = _ast.parse(src)

    violations: list[str] = []

    def is_target_tuple(node):
        """Check for tuple name pattern `j, thr, left, right` on left side."""
        if not isinstance(node, _ast.Tuple):
            return False
        elts = node.elts
        if len(elts) != 4:
            return False
        names = [e.id for e in elts if isinstance(e, _ast.Name)]
        return names == ["j", "thr", "left", "right"]

    def is_spec_slice(node):
        """spec[:4] — Subscript(spec, Slice(upper=4))."""
        if not isinstance(node, _ast.Subscript):
            return False
        if not (isinstance(node.value, _ast.Name) and node.value.id == "spec"):
            return False
        sl = node.slice
        if not isinstance(sl, _ast.Slice):
            return False
        if sl.lower is not None or sl.step is not None:
            return False
        if sl.upper is None:
            return True  # spec[:] — allowed but rare
        return isinstance(sl.upper, _ast.Constant) and sl.upper.value in (4, 5)

    for fn_node in _ast.walk(tree):
        if not isinstance(fn_node, _ast.FunctionDef):
            continue
        for node in _ast.walk(fn_node):
            if not isinstance(node, _ast.Assign):
                continue
            if len(node.targets) != 1:
                continue
            if not is_target_tuple(node.targets[0]):
                continue
            if not is_spec_slice(node.value):
                violations.append(
                    f"{fn_node.name}: line {node.lineno}, "
                    f"`j, thr, left, right = {_ast.unparse(node.value)}` "
                    f"(expected spec[:4] or spec[:5])"
                )
    assert not violations, (
        "4-unpack of spec must use spec[:4] (forward-compat with 5-tuple):\\n"
        + "\\n".join(violations)
    )
'''


def add_guard():
    print("\n=== tests/test_feature_importance_gain.py: +AST guard ===")
    p = Path("tests/test_feature_importance_gain.py")
    s = p.read_text()
    if "test_all_spec_unpack_uses_slice" in s:
        print("  already present — skip")
        return
    s = s.rstrip() + GUARD_SRC + "\n"
    p.write_text(s)
    compile_check(p)
    print("  OK: AST guard added")


# ============ CHANGELOG fix note ============

def patch_changelog():
    print("\n=== CHANGELOG.md: [0.7.9] fix note ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.7.9-fix: _AssignLeavesOp" in s:
        print("  already present — skip")
        return
    old = "## [0.7.9] — 2026-10-19\n"
    if old not in s:
        print("  WARNING: [0.7.9] not found — skipping")
        return
    note = (
        "\n### Исправлено в v0.7.9-fix\n"
        "- **`_AssignLeavesOp.update` 4-unpack** — после введения 5-tuple "
        "spec (`..., gain`) старый `j, thr, left, right = spec` падал "
        "`ValueError: too many values to unpack`. Fix: `spec[:4]` "
        "(forward-compatible slice). Тот же fix применён к "
        "`_assign_leaf_from_array` в изначальном v0.7.9.\n"
        "- **Multiclass `_fit_boosting_multiclass` 4-tuple** — "
        "`s.replace(old, new, 1)` дважды не сработало (второй replace "
        "не нашёл pattern из-за отличий в форматировании). Fix: replace "
        "**all occurrences** без count → multiclass тоже 5-tuple.\n"
        "- **+AST guard** `test_all_spec_unpack_uses_slice` — "
        "все `= spec` должны быть `= spec[:4]` / `= spec[:5]`. "
        "Ловит класс «4-unpack + 5-tuple» багов.\n"
    )
    s = s.replace(old, old + note, 1)
    p.write_text(s)
    print("  OK: [0.7.9] fix note")


# ============ main ============

def main():
    print("=== v0.7.9-fix ===")
    fix_gbdt_py()
    add_guard()
    patch_changelog()

    print("\n=== pytest tests/test_feature_importance_gain.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_feature_importance_gain.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.9-fix (test red)"], check=False)
        sys.exit(1)

    print("\n=== regression: GBDT + viz ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_gbdt_regression.py",
         "tests/test_gbdt_classification.py",
         "tests/test_gbdt_multiclass.py",
         "tests/test_viz_matplotlib.py",
         "tests/test_viz_plotly.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.9-fix (regression)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.9-fix: _AssignLeavesOp.update spec[:4]; multiclass 5-tuple "
        "(replace all, not count=1); +AST guard"])

    print("\n==> v0.7.9-fix complete. Готов squash → tag v0.7.9.")


if __name__ == "__main__":
    main()