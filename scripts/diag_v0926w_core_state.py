#!/usr/bin/env python3
# v0.9.26w-diagnostic: what's left in _core.py after B.1.
# READ-ONLY. AST inventory + test-grep audit.

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

CORE = "cadw/dataframe/_core.py"


def ast_inventory():
    src = Path(CORE).read_text()
    tree = ast.parse(src)
    print("\n" + "=" * 72)
    print(f"ACTUAL INVENTORY: {CORE} ({len(src.splitlines())} lines)")
    print("=" * 72)

    # Top-level classes
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            end = max(
                getattr(n, "end_lineno", node.lineno)
                for n in ast.walk(node)
            )
            classes.append({
                "name": node.name,
                "start": node.lineno,
                "end": end,
                "loc": end - node.lineno + 1,
                "methods": [
                    {
                        "name": m.name,
                        "lineno": m.lineno,
                        "loc": max(
                            getattr(x, "end_lineno", m.lineno)
                            for x in ast.walk(m)
                        ) - m.lineno + 1,
                    }
                    for m in node.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                ],
                "bases": [
                    ast.unparse(b) for b in node.bases
                ],
            })

    # Top-level functions
    top_fns = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = max(
                getattr(n, "end_lineno", node.lineno)
                for n in ast.walk(node)
            )
            top_fns.append({
                "name": node.name,
                "lineno": node.lineno,
                "loc": end - node.lineno + 1,
            })

    # Top-level assignments (logger = etc.)
    top_assigns = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    top_assigns.append({
                        "name": tgt.id,
                        "lineno": node.lineno,
                    })

    # Imports
    imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            imports.append((node.lineno, ast.unparse(node)))
        elif isinstance(node, ast.Import):
            imports.append((node.lineno, ast.unparse(node)))

    print(f"\n[classes] {len(classes)}")
    for c in classes:
        print(f"  {c['name']}({', '.join(c['bases'])})")
        print(f"    lines {c['start']}-{c['end']} "
              f"({c['loc']} LOC, {len(c['methods'])} methods)")
        for m in c["methods"]:
            print(f"      {m['name']}: {m['loc']} LOC (line {m['lineno']})")

    print(f"\n[top-level functions] {len(top_fns)}")
    total_helpers_loc = 0
    for fn in top_fns:
        print(f"  {fn['name']}: {fn['loc']} LOC (line {fn['lineno']})")
        total_helpers_loc += fn["loc"]

    print(f"\n[top-level assignments] {len(top_assigns)}")
    for a in top_assigns:
        print(f"  {a['name']}: line {a['lineno']}")

    print(f"\n[imports] {len(imports)}")
    for ln, txt in imports:
        print(f"  {ln}: {txt[:100]}")

    return {
        "src": src,
        "tree": tree,
        "classes": classes,
        "top_fns": top_fns,
        "top_assigns": top_assigns,
        "imports": imports,
        "total_lines": len(src.splitlines()),
        "helpers_loc": total_helpers_loc,
    }


def find_affected_tests(helper_names):
    """Tests that grep _core.py source for `def <helper>(`."""
    print("\n" + "=" * 72)
    print("AFFECTED TESTS (grep _core.py source for helper defs)")
    print("=" * 72)
    print(f"\nHelper names: {sorted(helper_names)}")

    affected = []
    for py in Path("tests").rglob("*.py"):
        try:
            src = py.read_text()
        except Exception:
            continue

        # Pattern 1: `inspect.getsource(..._core)` + `"def X("` check
        has_getsource_core = (
            "inspect.getsource(df_mod)" in src
            or "inspect.getsource(_core)" in src
            or "inspect.getsource(cadw.dataframe._core)" in src
            or "_inspect.getsource(df_mod)" in src
        )

        # Pattern 2: explicit `def X(` in `_core.py` assertion
        hits = []
        for name in helper_names:
            pat = f'"def {name}("'
            pat2 = f"'def {name}('"
            if pat in src or pat2 in src:
                hits.append(name)

        if has_getsource_core or hits:
            affected.append({
                "file": str(py),
                "has_getsource_core": has_getsource_core,
                "def_hits": hits,
            })
            print(f"\n{py}:")
            if has_getsource_core:
                print(f"  ⚠ uses inspect.getsource(_core)")
            if hits:
                print(f"  ⚠ greps: def {hits}")

    if not affected:
        print("\n(no affected tests — safe to extract)")

    return affected


def check_imports_redundant():
    """Which imports in _core.py are still needed after extraction."""
    src = Path(CORE).read_text()
    tree = ast.parse(src)
    imports = [
        (node.lineno, node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    # Collect all Name(Load) in _core.py after "imports"
    all_loads = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            all_loads.add(node.id)

    print("\n" + "=" * 72)
    print("IMPORT USAGE ANALYSIS")
    print("=" * 72)
    for ln, node in imports:
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                nm = a.asname or a.name
                # Remove from loads to check if used at all
                used = nm in all_loads
                print(f"  line {ln}: {nm!r} "
                      f"({'✓ used' if used else '✗ UNUSED?'})")
        elif isinstance(node, ast.Import):
            for a in node.names:
                nm = (a.asname or a.name).split(".")[0]
                used = nm in all_loads
                print(f"  line {ln}: {nm!r} "
                      f"({'✓ used' if used else '✗ UNUSED?'})")


def main():
    print("=== v0.9.26w-diagnostic: _core.py state ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    print("HEAD: " + head)

    info = ast_inventory()
    helper_names = [fn["name"] for fn in info["top_fns"]]
    affected = find_affected_tests(helper_names)
    check_imports_redundant()

    print("\n" + "=" * 72)
    print("SUMMARY + RECOMMENDATION")
    print("=" * 72)
    print(f"\n_core.py: {info['total_lines']} lines total")
    print(f"  - {sum(c['loc'] for c in info['classes'])} LOC in classes")
    print(f"  - {info['helpers_loc']} LOC in top-level helpers "
          f"({len(info['top_fns'])} functions)")
    print(f"  - {info['total_lines'] - sum(c['loc'] for c in info['classes']) - info['helpers_loc']} LOC imports/misc")

    print(f"\nAfter extraction to _helpers.py:")
    print(f"  _core.py ~{sum(c['loc'] for c in info['classes']) + 30} LOC "
          f"(class + imports)")
    print(f"  _helpers.py ~{info['helpers_loc'] + 30} LOC "
          f"(helpers + imports)")

    if not affected:
        print("\n✓ SAFE — no tests need updating")
    else:
        print(f"\n⚠ {len(affected)} test(s) need updating:")
        for a in affected:
            print(f"  {a['file']}")
            if a["has_getsource_core"]:
                print(f"    → inspect.getsource(_core) → _helpers")
            for name in a["def_hits"]:
                print(f"    → grep def {name}")

    print("\n" + "=" * 72)
    print("DECISION")
    print("=" * 72)
    print("""
If helpers_loc > 100 LOC:
  → extract to _helpers.py (worth it)

If affected tests:
  → update in SAME patch (§13.4/289/323)
  → typical fix: `inspect.getsource(_core)` → `_helpers`
  → alternative: AST-glob across package

If all clean:
  → v0.9.26w-extract-helpers.py
""")


if __name__ == "__main__":
    main()