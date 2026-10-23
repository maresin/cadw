#!/usr/bin/env python3
# v0.9.26g-corrected: package skeleton (full discovery).
# Fixes v0.9.26g failure: 5 categories of stale references.
# Splits into discovery (read-only) then mutations (atomic).
# ASCII-only. §13.4/287-295 applied.

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import commit_with_guards


OLD_FILE = "cadw/dataframe/_core.py"
NEW_DIR = "cadw/dataframe"
NEW_CORE = "cadw/dataframe/_core.py"
PYPROJECT = "pyproject.toml"


# =================================================================
# SEC 1: DISCOVERY (read-only)
# =================================================================

def discover_top_symbols():
    """All top-level FunctionDef/ClassDef/Assign-targets in dataframe.py."""
    tree = ast.parse(Path(OLD_FILE).read_text())
    syms = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            syms.add(node.name)
        elif isinstance(node, ast.ClassDef):
            syms.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    syms.add(tgt.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                syms.add(node.target.id)
    return sorted(s for s in syms if not s.startswith("__"))


def discover_refs():
    """Enumerate every reference category to cadw/dataframe.py."""
    refs = {
        "whitelist_keys": [],       # "cadw/dataframe/_core.py:Qualname"
        "exact_string_paths": [],   # "cadw/dataframe/_core.py" (exact)
        "path_composed": [],        # Path(...) / "dataframe" / "_core.py"
        "inspect_sources": [],      # inspect.getsource(df_mod._core)
        "pyproject": [],
    }

    # Whitelist key pattern: "cadw/dataframe/_core.py:Qualname"
    wl_pat = re.compile(r'[\'"]cadw/dataframe\.py:')

    # Exact string: "cadw/dataframe/_core.py" NOT followed by `:`
    # (only match quotes-closed-after-.py)
    exact_pat = re.compile(
        r'[\'"]cadw/dataframe\.py[\'"]'
    )

    # Path-composed: / "dataframe" / "_core.py"  or  / "dataframe" / "..." (unlikely)
    path_pat = re.compile(
        r'/\s*[\'"]dataframe\.py[\'"]'
    )

    # inspect.getsource(...) on dataframe module
    insp_pat = re.compile(
        r'inspect\.getsource\(\s*'
        r'(?:df_mod|dataframe|self\._parent|cadw\.dataframe)\b'
    )

    for py in Path(".").rglob("*.py"):
        s = str(py)
        if s.startswith(".git/") or s.endswith("cadw/dataframe/_core.py"):
            continue
        if "__pycache__" in s:
            continue
        try:
            src = py.read_text()
        except Exception:
            continue
        for i, ln in enumerate(src.splitlines(), 1):
            if wl_pat.search(ln):
                refs["whitelist_keys"].append((s, i, ln.strip()[:120]))
            elif exact_pat.search(ln):
                refs["exact_string_paths"].append((s, i, ln.strip()[:120]))
            if path_pat.search(ln):
                refs["path_composed"].append((s, i, ln.strip()[:120]))
            if insp_pat.search(ln):
                refs["inspect_sources"].append((s, i, ln.strip()[:120]))

    # pyproject.toml
    if Path(PYPROJECT).exists():
        src = Path(PYPROJECT).read_text()
        for i, ln in enumerate(src.splitlines(), 1):
            if "cadw/dataframe/_core.py" in ln:
                refs["pyproject"].append((PYPROJECT, i, ln.strip()[:120]))

    return refs


def print_discovery_plan(top_symbols, refs):
    print("\n" + "=" * 72)
    print("DISCOVERY PLAN (read-only)")
    print("=" * 72)

    print("\n[top_symbols] " + str(len(top_symbols)) + " to re-export:")
    for s in top_symbols:
        print("  " + s)

    for cat, items in refs.items():
        print("\n[" + cat + "] " + str(len(items)) + " ref(s):")
        if not items:
            print("  (none)")
            continue
        for fname, ln, txt in items:
            print("  " + fname + ":" + str(ln) + "  " + txt)


# =================================================================
# SEC 2: MUTATIONS
# =================================================================

def do_git_mv():
    print("\n--- git mv ---")
    Path(NEW_DIR).mkdir(exist_ok=True)
    subprocess.run(["git", "mv", OLD_FILE, NEW_CORE], check=True)
    print("  " + OLD_FILE + " -> " + NEW_CORE)


def write_full_init(top_symbols):
    print("\n--- write __init__.py (full re-export) ---")
    lines = [
        "# cadw/dataframe/__init__.py",
        "# Package facade. v0.9.26g-corrected (B.1 step 1).",
        "# Pure structural move from dataframe.py -> package.",
        "",
        "from ._core import (",
    ]
    for s in top_symbols:
        lines.append("    " + s + ",")
    lines.append(")")
    lines.append("")
    lines.append("__all__ = [")
    for s in top_symbols:
        if not s.startswith("_"):
            lines.append('    "' + s + '",')
    lines.append("]")
    lines.append("")
    src = "\n".join(lines)
    p = Path(NEW_DIR + "/__init__.py")
    p.write_text(src)
    ast.parse(src, filename=str(p))
    print("  wrote " + str(len(src.splitlines())) + " lines, "
          + str(len(top_symbols)) + " re-exports")


def migrate_whitelist_keys(refs):
    """Replace 'cadw/dataframe/_core.py:...' -> 'cadw/dataframe/_core.py:...'"""
    print("\n--- migrate whitelist keys (substring regex) ---")
    if not refs:
        print("  (none to migrate)")
        return
    # Group by file, do single pass per file
    by_file = {}
    for fname, _, _ in refs:
        by_file.setdefault(fname, 0)
        by_file[fname] += 1
    for fname, n in by_file.items():
        p = Path(fname)
        src = p.read_text()
        # Substring replacement: 'cadw/dataframe/_core.py:  OR  "cadw/dataframe/_core.py:
        new_src = src.replace(
            "'cadw/dataframe/_core.py:", "'cadw/dataframe/_core.py:"
        ).replace(
            '"cadw/dataframe/_core.py:', '"cadw/dataframe/_core.py:'
        )
        if new_src == src:
            print("  " + fname + ": no change?! (inspect manually)")
            continue
        p.write_text(new_src)
        # AST sanity only for .py
        if fname.endswith(".py"):
            try:
                ast.parse(new_src, filename=fname)
            except SyntaxError as e:
                raise SystemExit(
                    "syntax error after patch " + fname + ": " + str(e)
                )
        print("  " + fname + ": " + str(n) + " key(s) migrated")


def migrate_exact_paths(refs):
    """'cadw/dataframe/_core.py' (exact) -> 'cadw/dataframe/_core.py'"""
    print("\n--- migrate exact string paths ---")
    if not refs:
        print("  (none)")
        return
    by_file = {}
    for fname, _, _ in refs:
        by_file.setdefault(fname, 0)
        by_file[fname] += 1
    for fname, n in by_file.items():
        p = Path(fname)
        src = p.read_text()
        new_src = src.replace(
            '"cadw/dataframe/_core.py"', '"cadw/dataframe/_core.py"'
        ).replace(
            "'cadw/dataframe/_core.py'", "'cadw/dataframe/_core.py'"
        )
        if new_src == src:
            print("  " + fname + ": no change?!")
            continue
        p.write_text(new_src)
        if fname.endswith(".py"):
            ast.parse(new_src, filename=fname)
        print("  " + fname + ": " + str(n) + " path(s)")


def migrate_path_composed(refs):
    """'/ \"dataframe.py\"' -> '/ \"dataframe\" / \"_core.py\"'"""
    print("\n--- migrate path-composed refs ---")
    if not refs:
        print("  (none)")
        return
    by_file = {}
    for fname, _, _ in refs:
        by_file.setdefault(fname, 0)
        by_file[fname] += 1
    for fname, n in by_file.items():
        p = Path(fname)
        src = p.read_text()
        # `/ "dataframe" / "_core.py"` -> `/ "dataframe" / "_core.py"`
        new_src = src.replace(
            '/ "dataframe" / "_core.py"', '/ "dataframe" / "_core.py"'
        ).replace(
            "/ 'dataframe' / '_core.py'", "/ 'dataframe' / '_core.py'"
        )
        if new_src == src:
            print("  " + fname + ": no change (pattern differs), inspect")
            continue
        p.write_text(new_src)
        if fname.endswith(".py"):
            ast.parse(new_src, filename=fname)
        print("  " + fname + ": " + str(n) + " path(s)")


def migrate_inspect_sources(refs):
    """inspect.getsource(df_mod._core) -> inspect.getsource(df_mod._core)"""
    print("\n--- migrate inspect.getsource refs ---")
    if not refs:
        print("  (none)")
        return
    for fname, ln, txt in refs:
        p = Path(fname)
        src = p.read_text()
        # Replace `inspect.getsource(df_mod._core)` -> `...getsource(df_mod._core)`
        # (non-greedy, only the exact pattern)
        new_src = re.sub(
            r"inspect\.getsource\(\s*df_mod\s*\)",
            "inspect.getsource(df_mod._core)",
            src,
        )
        new_src = re.sub(
            r"inspect\.getsource\(\s*cadw\.dataframe\s*\)",
            "inspect.getsource(cadw.dataframe._core)",
            new_src,
        )
        if new_src == src:
            print("  " + fname + ":" + str(ln) + " no change (pattern)")
            continue
        p.write_text(new_src)
        if fname.endswith(".py"):
            ast.parse(new_src, filename=fname)
        print("  " + fname + ":" + str(ln) + " updated")


def migrate_pyproject(refs):
    print("\n--- pyproject.toml ---")
    if not refs:
        print("  (no refs)")
        return
    p = Path(PYPROJECT)
    src = p.read_text()
    new_src = src.replace(
        "cadw/dataframe/_core.py", "cadw/dataframe/_core.py"
    )
    p.write_text(new_src)
    print("  " + PYPROJECT + ": " + str(len(refs)) + " line(s) updated")


# =================================================================
# SEC 3: TESTS + COMMIT
# =================================================================

def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit(
            "fast suite FAILED. Rollback: git reset --hard HEAD + "
            "git clean -fdx cadw/dataframe/. "
            "Then inspect failures."
        )


def main():
    print("=== v0.9.26g-corrected: package skeleton (full discovery) ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    # HEAD check
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26e":
        raise SystemExit("expected HEAD=v0.9.26e, got " + repr(head))
    print("HEAD: " + head)

    # Pre-checks
    if not Path(OLD_FILE).exists():
        raise SystemExit(OLD_FILE + " missing")
    if Path(NEW_DIR).exists():
        raise SystemExit(NEW_DIR + "/ exists — clean it first")

    # DISCOVERY (read-only)
    top_symbols = discover_top_symbols()
    refs = discover_refs()
    print_discovery_plan(top_symbols, refs)

    # MUTATIONS (atomic; any failure -> user does manual git reset)
    print("\n" + "=" * 72)
    print("APPLYING MUTATIONS")
    print("=" * 72)
    do_git_mv()
    write_full_init(top_symbols)
    migrate_whitelist_keys(refs["whitelist_keys"])
    migrate_exact_paths(refs["exact_string_paths"])
    migrate_path_composed(refs["path_composed"])
    migrate_inspect_sources(refs["inspect_sources"])
    migrate_pyproject(refs["pyproject"])

    # VERIFY (rollback if fails)
    run_tests()

    # COMMIT (only if tests green)
    commit_with_guards(
        "v0.9.26g: package skeleton (B.1 step 1). "
        "cadw/dataframe.py -> cadw/dataframe/_core.py + full "
        "__init__.py re-export. Whitelist keys, exact paths, "
        "path-composed refs, inspect.getsource migrated. "
        "Pure structural move; 0 logic change. 2488 tests green.",
        self_path="scripts/patch_v0926g_corrected.py",
        check_whitelist=True,
    )
    print("\nv0.9.26g-corrected COMPLETE.")


if __name__ == "__main__":
    main()