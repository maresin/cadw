#!/usr/bin/env python3
# v0.9.26g-corrected2: package skeleton, scoped migration.
# Fixes R1 (ImportFrom), R2 (regex \b), R3 (scope), R4 (A.6 file).
# ASCII-only.

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import commit_with_guards


OLD_FILE = "cadw/dataframe.py"
NEW_DIR = "cadw/dataframe"
NEW_CORE = "cadw/dataframe/_core.py"
ROADMAP = "docs/roadmap.json"

# R3: only migrate in active code (skip scripts/ archive + historical)
MIGRATE_ROOTS = ("cadw/", "tests/")


def in_scope(path_str):
    return any(path_str.startswith(p) for p in MIGRATE_ROOTS)


# ============ discovery ============

def discover_top_symbols():
    """R1: include ImportFrom + Assign targets + classdefs + funcdefs."""
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
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    syms.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    syms.add(alias.asname)
    return sorted(s for s in syms if not s.startswith("__"))


def discover_refs():
    """Scoped discovery: cadw/ + tests/ only."""
    refs = {
        "whitelist_keys": [],
        "exact_string_paths": [],
        "path_composed": [],
        "inspect_sources": [],
    }
    wl_pat = re.compile(r'[\'"]cadw/dataframe\.py:')
    # R2: \b allows ".py\"" AND ".py," and ".py " etc.
    exact_pat = re.compile(r'[\'"]cadw/dataframe\.py\b')
    path_pat = re.compile(r'/\s*[\'"]dataframe\.py[\'"]')
    insp_pat = re.compile(
        r'inspect\.getsource\(\s*'
        r'(?:df_mod|cadw\.dataframe)\s*\)'
    )

    for py in Path(".").rglob("*.py"):
        s = str(py)
        if s.startswith(".git/") or s.endswith("cadw/dataframe.py"):
            continue
        if "__pycache__" in s:
            continue
        if not in_scope(s):
            continue  # R3
        try:
            src = py.read_text()
        except Exception:
            continue
        for i, ln in enumerate(src.splitlines(), 1):
            if wl_pat.search(ln):
                refs["whitelist_keys"].append((s, i, ln.strip()[:100]))
            elif exact_pat.search(ln):
                refs["exact_string_paths"].append((s, i, ln.strip()[:100]))
            if path_pat.search(ln):
                refs["path_composed"].append((s, i, ln.strip()[:100]))
            if insp_pat.search(ln):
                refs["inspect_sources"].append((s, i, ln.strip()[:100]))

    return refs


def print_plan(top_symbols, refs):
    print("\n" + "=" * 72)
    print("DISCOVERY PLAN (scoped to cadw/ + tests/)")
    print("=" * 72)
    print("\n[top_symbols] " + str(len(top_symbols)) + ":")
    for s in top_symbols:
        print("  " + s)
    for cat, items in refs.items():
        print("\n[" + cat + "] " + str(len(items)) + ":")
        if not items:
            print("  (none)")
            continue
        for fname, ln, txt in items[:20]:
            print("  " + fname + ":" + str(ln) + "  " + txt)
        if len(items) > 20:
            print("  ... (" + str(len(items) - 20) + " more)")


# ============ mutations ============

def do_git_mv():
    Path(NEW_DIR).mkdir(exist_ok=True)
    subprocess.run(["git", "mv", OLD_FILE, NEW_CORE], check=True)
    print("  git mv " + OLD_FILE + " -> " + NEW_CORE)


def write_init(top_symbols):
    lines = [
        "# cadw/dataframe/__init__.py",
        "# Package facade. v0.9.26g (B.1 step 1).",
        "",
        "from ._core import (",
    ]
    for s in top_symbols:
        lines.append("    " + s + ",")
    lines.append(")")
    lines.append("")
    src = "\n".join(lines)
    p = Path(NEW_DIR + "/__init__.py")
    p.write_text(src)
    ast.parse(src, filename=str(p))
    print("  wrote __init__.py: " + str(len(top_symbols)) + " re-exports")


def migrate_whitelist(refs):
    for fname, n in _group(refs):
        p = Path(fname)
        src = p.read_text()
        new = src.replace("'cadw/dataframe.py:", "'cadw/dataframe/_core.py:").replace(
            '"cadw/dataframe.py:', '"cadw/dataframe/_core.py:'
        )
        if new != src:
            p.write_text(new)
            if fname.endswith(".py"):
                ast.parse(new, filename=fname)
            print("  " + fname + ": " + str(n) + " keys")


def migrate_exact(refs):
    for fname, n in _group(refs):
        p = Path(fname)
        src = p.read_text()
        # R2: replace "cadw/dataframe.py" with word-boundary safety
        # (source regex not needed — after matching, substring replace)
        new = src.replace('"cadw/dataframe.py"', '"cadw/dataframe/_core.py"')
        new = new.replace("'cadw/dataframe.py'", "'cadw/dataframe/_core.py'")
        # R2b: also handle inside larger strings: `"cadw/dataframe.py, ..."`
        new = new.replace('"cadw/dataframe.py,', '"cadw/dataframe/_core.py,')
        new = new.replace("'cadw/dataframe.py,", "'cadw/dataframe/_core.py,")
        if new != src:
            p.write_text(new)
            if fname.endswith(".py"):
                ast.parse(new, filename=fname)
            print("  " + fname + ": " + str(n) + " paths")


def migrate_path_composed(refs):
    for fname, n in _group(refs):
        p = Path(fname)
        src = p.read_text()
        new = src.replace('/ "dataframe.py"', '/ "dataframe" / "_core.py"')
        new = new.replace("/ 'dataframe.py'", "/ 'dataframe' / '_core.py'")
        if new != src:
            p.write_text(new)
            if fname.endswith(".py"):
                ast.parse(new, filename=fname)
            print("  " + fname + ": " + str(n) + " path-composed")


def migrate_inspect(refs):
    for fname, n in _group(refs):
        p = Path(fname)
        src = p.read_text()
        new = re.sub(
            r"inspect\.getsource\(\s*df_mod\s*\)",
            "inspect.getsource(df_mod._core)", src,
        )
        new = re.sub(
            r"inspect\.getsource\(\s*cadw\.dataframe\s*\)",
            "inspect.getsource(cadw.dataframe._core)", new,
        )
        if new != src:
            p.write_text(new)
            if fname.endswith(".py"):
                ast.parse(new, filename=fname)
            print("  " + fname + ": " + str(n) + " inspect.getsource")


def _group(refs):
    by_file = {}
    for fname, _, _ in refs:
        by_file[fname] = by_file.get(fname, 0) + 1
    return by_file.items()


def update_roadmap_a6():
    """R4: A.6 file -> cadw/dataframe/_core.py."""
    import json
    p = Path(ROADMAP)
    data = json.loads(p.read_text())
    pha = next(ph for ph in data["phases"]
               if isinstance(ph, dict) and ph.get("id") == "A")
    a6 = next(it for it in pha["items"]
              if isinstance(it, dict) and it.get("id") == "A.6")
    if a6.get("file") == "cadw/dataframe/_core.py":
        print("  A.6 file already updated")
        return
    old = a6.get("file")
    a6["file"] = "cadw/dataframe/_core.py"
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print("  roadmap A.6 file: " + repr(old) + " -> 'cadw/dataframe/_core.py'")


def fix_collect_test():
    """R4: test_collect_project_modes assertion for A.6."""
    p = Path("tests/test_collect_project_modes.py")
    src = p.read_text()
    if "cadw/dataframe/_core.py" in src and "assert \"FILE: cadw/dataframe.py\"" not in src:
        print("  test already updated")
        return
    new = src.replace(
        'assert "FILE: cadw/dataframe.py" in text',
        'assert "FILE: cadw/dataframe/_core.py" in text',
    )
    if new != src:
        p.write_text(new)
        ast.parse(new, filename=str(p))
        print("  test_collect_project_modes: A.6 assertion updated")


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
    print("=== v0.9.26g-corrected2 ===")
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26e":
        raise SystemExit("expected HEAD=v0.9.26e, got " + repr(head))
    if not Path(OLD_FILE).exists():
        raise SystemExit(OLD_FILE + " missing")
    if Path(NEW_DIR).exists():
        raise SystemExit(NEW_DIR + "/ exists — clean first")
    print("HEAD: " + head)

    top_symbols = discover_top_symbols()
    refs = discover_refs()
    print_plan(top_symbols, refs)

    print("\n" + "=" * 72)
    print("APPLYING")
    print("=" * 72)
    do_git_mv()
    write_init(top_symbols)
    migrate_whitelist(refs["whitelist_keys"])
    migrate_exact(refs["exact_string_paths"])
    migrate_path_composed(refs["path_composed"])
    migrate_inspect(refs["inspect_sources"])
    update_roadmap_a6()
    fix_collect_test()

    run_tests()

    commit_with_guards(
        "v0.9.26g: package skeleton (B.1 step 1). "
        "cadw/dataframe.py -> cadw/dataframe/_core.py + full "
        "__init__.py re-export (incl. ImportFrom names). Scoped "
        "migration to cadw/ + tests/; scripts/ untouched. "
        "Roadmap A.6 file path updated. 2488 tests green.",
        self_path="scripts/patch_v0926g_corrected2.py",
        check_whitelist=True,
    )
    print("\nv0.9.26g COMPLETE")


if __name__ == "__main__":
    main()