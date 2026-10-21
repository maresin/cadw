#!/usr/bin/env python3
# v0.9.6a-fix2 — remove stale import of _pandas_hash_column.
#
# Root cause: fix1 removed _pandas_hash_column / _python_hash_column
# from shuffle.py, but did not grep call sites (§13.4/152/174:
# "removal patch must grep call sites"). test_hash_v081.py line 21
# still had `from cadw.ops.shuffle import ..., _pandas_hash_column`
# → ImportError at collection.
#
# Changes:
#   1. Grep all tests/ + cadw/ for _pandas_hash_column /
#      _python_hash_column. Report. Fix any stale import lines.
#   2. Add AST guard test: no test file imports _pandas_hash_column
#      or _python_hash_column.
#   3. Re-run hash + shuffle + AST + regression + full suite.
#
# Run: uv run python scripts/patch_v096a_fix2.py

from __future__ import annotations
import os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    assert_compiles, commit_with_guards,
)


# ============ 1. Grep + fix stale imports ============

STALE_NAMES = ("_pandas_hash_column", "_python_hash_column")


def grep_and_report():
    print("\n=== grep: stale names in tests/ + cadw/ ===")
    hits: list[tuple[str, int, str]] = []
    for root in ("tests", "cadw"):
        for py in Path(root).rglob("*.py"):
            for lineno, line in enumerate(py.read_text().split("\n"), 1):
                for name in STALE_NAMES:
                    if name in line:
                        hits.append((str(py), lineno, line.rstrip()))
    if not hits:
        print("  no references")
        return hits
    print(f"  {len(hits)} reference(s):")
    for path, lineno, line in hits:
        print(f"    {path}:{lineno}: {line}")
    return hits


def fix_stale_imports():
    """Fix any `from cadw.ops.shuffle import ..., _pandas_hash_column`."""
    print("\n=== fix stale imports ===")
    fixed = 0
    for py in Path("tests").rglob("*.py"):
        src = py.read_text()
        original = src
        # Pattern: `from cadw.ops.shuffle import <names>` where names
        # include _pandas_hash_column or _python_hash_column.
        pat = re.compile(
            r"from cadw\.ops\.shuffle import ([^\n]+)",
        )
        def repl(m: "re.Match[str]") -> str:
            names = [n.strip() for n in m.group(1).split(",")]
            names = [
                n for n in names
                if n not in STALE_NAMES
            ]
            return "from cadw.ops.shuffle import " + ", ".join(names)
        new_src = pat.sub(repl, src)
        if new_src != original:
            py.write_text(new_src)
            fixed += 1
            print(f"  fixed {py}")
    if fixed == 0:
        print("  no stale imports to fix")
    else:
        print(f"  total fixed: {fixed}")


# ============ 2. AST guard: no test file imports stale names ============

def add_ast_guard():
    print("\n=== test_hash_v081.py: AST guard for stale names ===")
    p = Path("tests/test_hash_v081.py")
    src = p.read_text()
    if "def test_no_test_imports_stale_hash_names" in src:
        print("  already present — skip")
        return

    new_test = [
        "",
        "",
        "def test_no_test_imports_stale_hash_names():",
        "    # AST: no test file imports _pandas_hash_column or",
        "    # _python_hash_column (removed v0.9.6a). Any import attempt",
        "    # fails collection with cryptic ImportError; AST guard",
        "    # fails earlier with actionable message (§13.4/174).",
        "    import ast as _ast",
        "    from pathlib import Path as _P",
        "    STALE = {'_pandas_hash_column', '_python_hash_column'}",
        "    tests_root = _P(__file__).parent",
        "    violations = []",
        "    for py in tests_root.rglob('test_*.py'):",
        "        if py.name == _P(__file__).name:",
        "            continue",
        "        try:",
        "            tree = _ast.parse(py.read_text())",
        "        except SyntaxError:",
        "            continue",
        "        for node in _ast.walk(tree):",
        "            if not isinstance(node, _ast.ImportFrom):",
        "                continue",
        "            if not node.module or 'shuffle' not in node.module:",
        "                continue",
        "            for alias in node.names:",
        "                if alias.name in STALE:",
        "                    violations.append(",
        "                        f'{py.name}:{node.lineno} '",
        "                        f'imports {alias.name}'",
        "                    )",
        "    assert not violations, (",
        "        'stale hash imports (removed v0.9.6a):\\n'",
        "        + '\\n'.join(violations)",
        "    )",
        "",
    ]
    p.write_text(src + "\n".join(new_test))
    assert_compiles(p)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.6a-fix2: stale import fix ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    grep_and_report()
    fix_stale_imports()
    add_ast_guard()

    print("\n=== pytest: hash tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_hash_v081.py", "-v"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** HASH TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: shuffle tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_shuffle.py", "-v"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** SHUFFLE TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: AST guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_no_pandas_in_ops.py", "-q"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** AST GUARDS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: groupby + dedup + pingis (regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_groupby_v2.py",
         "tests/test_groupby_per_group.py",
         "tests/test_dedup_arrow_native.py",
         "tests/test_pingis_safety.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** GROUPBY/DEDUP/PINGIS REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    commit_with_guards(
        "v0.9.6a-fix2: remove stale _pandas_hash_column import from "
        "test_hash_v081.py (fix1 missed grep for call sites — "
        "§13.4/174). AST guard test_no_test_imports_stale_hash_names "
        "prevents the class: any test importing removed "
        "_pandas_hash_column/_python_hash_column fails AST check "
        "before collection ImportError.",
        self_path="scripts/patch_v096a_fix2.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6a complete (fix1 + fix2).")


if __name__ == "__main__":
    main()
