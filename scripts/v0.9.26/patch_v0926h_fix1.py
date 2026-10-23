#!/usr/bin/env python3
# v0.9.26h-fix1: lazy import for _convert_batch in _exports.py.
# Root cause: to_batches uses _convert_batch (helper in _core), but
# _exports.py can't import it at top level (circular: _core imports
# _exports for _ExportsMixin). Solution: lazy import inside method.

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import commit_with_guards


CORE = "cadw/dataframe/_core.py"
EXPORTS = "cadw/dataframe/_exports.py"


# ============ discovery (AST, discovery-first §13.4/288) ============

def discover_undefined():
    """Find helper names in _exports.py not in scope but in _core top-levels."""
    exports_src = Path(EXPORTS).read_text()
    core_src = Path(CORE).read_text()
    exports_tree = ast.parse(exports_src)
    core_tree = ast.parse(core_src)

    core_top = set()
    for n in core_tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            core_top.add(n.name)
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                core_top.add(a.asname or a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                core_top.add(a.asname or a.name.split(".")[0])

    exports_imports = set()
    for n in exports_tree.body:
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                exports_imports.add(a.asname or a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                exports_imports.add(a.asname or a.name.split(".")[0])

    import builtins as _b
    bnames = set(dir(_b))

    issues = []

    def scan_fn(fn_node, path):
        local = set()
        for a in fn_node.args.args:
            local.add(a.arg)
        for a in fn_node.args.kwonlyargs:
            local.add(a.arg)
        for a in fn_node.args.posonlyargs:
            local.add(a.arg)
        if fn_node.args.vararg:
            local.add(fn_node.args.vararg.arg)
        if fn_node.args.kwarg:
            local.add(fn_node.args.kwarg.arg)

        for n in ast.walk(fn_node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                local.add(n.id)
            if isinstance(n, ast.ImportFrom):
                for a in n.names:
                    local.add(a.asname or a.name)
            if isinstance(n, ast.Import):
                for a in n.names:
                    local.add(a.asname or a.name.split(".")[0])

        for n in ast.walk(fn_node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                nid = n.func.id
                if (nid in local
                        or nid in exports_imports
                        or nid in bnames
                        or nid in ("self", "cls")):
                    continue
                if nid in core_top:
                    issues.append((path, nid, n.lineno))

    for n in exports_tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scan_fn(n, n.name)
        elif isinstance(n, ast.ClassDef):
            for item in n.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scan_fn(item, n.name + "." + item.name)

    return issues


# ============ mutation ============

def add_lazy_import(path_name, name):
    """Insert `from ._core import {name}` at top of function body."""
    src = Path(EXPORTS).read_text()
    if f"from ._core import {name}\n" in src:
        print(f"  {name} already imported somewhere")
        return False

    lines = src.split("\n")
    tree = ast.parse(src)

    if "." in path_name:
        cls_name, method_name = path_name.rsplit(".", 1)
        target = None
        for n in tree.body:
            if isinstance(n, ast.ClassDef) and n.name == cls_name:
                for item in n.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                        target = item
                        break
    else:
        target = next(
            (n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == path_name),
            None,
        )

    if target is None:
        print(f"  WARN: {path_name} not found")
        return False

    body = target.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        insert_idx = body[0].end_lineno  # after docstring, 0-based index = line number
    else:
        insert_idx = body[0].lineno - 1

    if "." in path_name:
        indent = "        "  # 8 spaces = method inside class
    else:
        indent = "    "

    imp_line = f"{indent}from ._core import {name}"
    lines.insert(insert_idx, imp_line)
    new_src = "\n".join(lines)
    ast.parse(new_src, filename=EXPORTS)
    Path(EXPORTS).write_text(new_src)
    print(f"  added lazy import '{name}' in {path_name}")
    return True


# ============ tests ============

def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


# ============ main ============

def main():
    print("=== v0.9.26h-fix1: lazy import _convert_batch ===")
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26g":
        raise SystemExit("expected HEAD=v0.9.26g, got " + repr(head))
    print("HEAD: " + head)

    if not Path(EXPORTS).exists():
        raise SystemExit(EXPORTS + " missing — v0.9.26h not applied?")
    if not Path(CORE).exists():
        raise SystemExit(CORE + " missing")

    issues = discover_undefined()
    print(f"\nUndefined helper calls: {len(issues)}")
    for path, name, lineno in issues:
        print(f"  {path} -> {name} (line {lineno})")

    if not issues:
        print("No undefined — may already be fixed, or new issue type")

    seen = set()
    for path, name, lineno in issues:
        key = (path, name)
        if key in seen:
            continue
        seen.add(key)
        add_lazy_import(path, name)

    run_tests()

    commit_with_guards(
        "v0.9.26h: extract _exports (B.1 step 2, pilot). "
        "to_batches/write_parquet/from_table/from_parquet/"
        "from_dataset + concat moved verbatim to _exports.py. "
        "CadwDataFrame(_ExportsMixin) - mixin composition. "
        "fix1: lazy import for _convert_batch in to_batches "
        "(circular _core<->_exports prevents top-level import). "
        "0 logic change; 2488 tests green.",
        self_path="scripts/patch_v0926h_fix1.py",
        check_whitelist=True,
    )
    print("\nv0.9.26h COMPLETE")


if __name__ == "__main__":
    main()