#!/usr/bin/env python3
# v0.9.28-fix4: comprehensive DECISIONS.md read scan + fix all.
# §13.4/351: AST-based scan — open() + .read_text() both.

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

DECISIONS_GLOB = (
    '((__import__("pathlib").Path(__file__)'
    '.resolve().parent.parent / "docs" / "decisions"))'
)
# Simpler: use a helper we add to conftest
HELPER_NAME = "_read_all_decisions_glob"


def scan_reads():
    """Find all tests that read DECISIONS.md."""
    print("=" * 72)
    print("COMPREHENSIVE SCAN: DECISIONS.md reads in tests")
    print("=" * 72)

    hits = []
    for py in Path("tests").rglob("*.py"):
        try:
            src = py.read_text()
            tree = ast.parse(src)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            # .read_text() on something ending in "DECISIONS.md"
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "read_text"):
                if _node_contains_str(node, "DECISIONS.md"):
                    hits.append((str(py), node.lineno, "read_text"))

            # open("DECISIONS.md", ...)
            elif (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in ("open", "_open")):
                for arg in node.args:
                    if (isinstance(arg, ast.Constant)
                            and isinstance(arg.value, str)
                            and "DECISIONS.md" in arg.value):
                        hits.append(
                            (str(py), node.lineno, "open")
                        )
                        break

    print(f"\nFound {len(hits)} reference(s):")
    for fname, lineno, kind in hits:
        print(f"  {fname}:{lineno}  ({kind})")
    return hits


def _node_contains_str(node, needle):
    """Check if any string constant under node contains needle."""
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            if needle in n.value:
                return True
    return False


def patch_test(fname):
    """Patch one test file: DECISIONS.md read → glob helper."""
    print(f"\n--- patch {fname} ---")
    p = Path(fname)
    src = p.read_text()
    orig = src

    # Pattern 1: `.read_text()` (already patched in test_changelog via helper)
    if fname.endswith("test_changelog.py"):
        # Already fixed by fix3, skip
        print("  (already has helper)")
        return 0

    # Pattern 2: `open("DECISIONS.md", encoding="utf-8").read()`
    # Replace with helper call. Need helper added to file first.

    # Check if helper already exists
    if "def _read_all_decisions" in src:
        print("  already has helper")
        return 0

    # Add helper (mirror of test_changelog version)
    helper = '''

def _read_all_decisions():
    """v0.9.28-fix4 (§13.4/351): DECISIONS.md split → glob.
    Concatenate all topic files for tests asserting specific sections."""
    from pathlib import Path as _P
    _docs = _P(__file__).resolve().parent.parent / "docs" / "decisions"
    return "\\n".join(f.read_text() for f in sorted(_docs.glob("*.md")))
'''
    # Insert before first def test_
    lines = src.split("\n")
    insert_at = None
    for i, ln in enumerate(lines):
        if ln.startswith("def test_") or ln.startswith("def _mk"):
            insert_at = i
            break
    if insert_at is None:
        print("  WARN: no def test_ found — inspect manually")
        return 0
    new_lines = lines[:insert_at] + helper.rstrip("\n").split("\n") + [""] + lines[insert_at:]
    src = "\n".join(new_lines)

    # Replace open pattern: 
    # `open("DECISIONS.md", encoding="utf-8").read()` → `_read_all_decisions()`
    import re
    # Try single-line form
    pat1 = re.compile(
        r'open\(\s*"DECISIONS\.md"\s*,\s*encoding\s*=\s*["\']utf-8["\']\s*\)\.read\(\)'
    )
    src, n1 = pat1.subn("_read_all_decisions()", src)

    # Also try `.read_text()` variants if any
    pat2 = re.compile(
        r'\(\s*_P\(__file__\)\.resolve\(\)\.parent\.parent\s*/\s*"DECISIONS\.md"\s*\)\.read_text\(\)'
    )
    src, n2 = pat2.subn("_read_all_decisions()", src)

    if src == orig:
        print("  WARN: no replacement applied — inspect manually")
        return 0

    p.write_text(src)
    ast.parse(src, filename=fname)
    print(f"  replaced {n1 + n2} open/read_text call(s)")
    return 1


def run_tests():
    print("\n--- docs tests ---")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py", "tests/test_roadmap.py",
         "tests/test_ts_api.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        raise SystemExit("docs tests FAILED")
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.28-fix4: comprehensive DECISIONS.md read scan ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"branch: {br}")

    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.28":
        raise SystemExit(f"expected HEAD=v0.9.28, got {head!r}")
    print(f"HEAD: {head}")

    hits = scan_reads()

    # Group hits by file
    files = sorted(set(h[0] for h in hits))
    print(f"\nFiles to patch: {len(files)}")
    for f in files:
        print(f"  {f}")

    n_patched = 0
    for f in files:
        n_patched += patch_test(f)
    print(f"\n  patched {n_patched} file(s)")

    run_tests()

    print("\n--- verify_whitelist ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "from cadw.patch_utils import verify_whitelist; "
         "verify_whitelist(raise_on_orphan=True); print('WL OK')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("whitelist failed")
    print(r.stdout.strip())

    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.28: B.2 DECISIONS split — fix4 (comprehensive scan). "
        "Commit e3c923b had 1 residual failure: "
        "test_ts_api.py::test_readme_and_changelog_contain_ts_terms "
        "reads DECISIONS.md via open(), asserts 'mint' (MinT-WLS). "
        "fix4 (§13.4/351): AST-based scan finds ALL DECISIONS.md "
        "reads (open() + .read_text() both, not just one form); "
        "applies _read_all_decisions() helper with sorted glob "
        "to all affected tests. Fixes test_ts_api.py. "
        "Docs-only; 0 production. Full fast suite now green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)
    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"\n{log}")
    print("\nv0.9.28 COMPLETE (fix4). B.2 done.")


if __name__ == "__main__":
    main()