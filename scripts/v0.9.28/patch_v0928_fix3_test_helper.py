#!/usr/bin/env python3
# v0.9.28-fix3: add _read_all_decisions helper, replace 2 read_text calls.
# Leaves L111 (path=) alone — currently passing, different usage.

import ast
import re
import subprocess
import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

TEST = Path("tests/test_changelog.py")


OLD_STMT = (
    'text = (_P(__file__).resolve().parent.parent / "DECISIONS.md")'
    '.read_text()'
)
NEW_STMT = 'text = _read_all_decisions()'

HELPER = '''

def _read_all_decisions():
    """v0.9.28-fix3: DECISIONS.md split into docs/decisions/*.md.
    Concatenate all topic files for tests asserting specific sections.
    §13.4/349."""
    _docs = _P(__file__).resolve().parent.parent / "docs" / "decisions"
    return "\\n".join(f.read_text() for f in _docs.glob("*.md"))
'''


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.27":
        raise SystemExit(f"expected HEAD=v0.9.27, got {head!r}")
    print(f"HEAD: {head}")

    src = TEST.read_text()

    # Count occurrences of OLD_STMT
    n = src.count(OLD_STMT)
    print(f"  {TEST.name}: {n} occurrences of `text = (...).read_text()`")

    if n == 0:
        if "_read_all_decisions" in src:
            print("  already fixed (helper present)")
            return "already"
        raise SystemExit("no pattern, no helper — manual inspect")

    if "_read_all_decisions" in src:
        raise SystemExit("helper already present but pattern still there")

    if "from pathlib import Path as _P" not in src:
        raise SystemExit("_P import missing — helper won't work")

    # Print context for L111 (leave alone)
    print("\n  L111 reference (leaving unchanged — different usage):")
    for i, ln in enumerate(src.split("\n"), 1):
        if "path = " in ln and "DECISIONS.md" in ln:
            print(f"    L{i}: {ln!r}")

    return "needs_fix"


def add_helper():
    print("\n--- add helper ---")
    src = TEST.read_text()

    # Insert helper right before the first `def test_` line
    lines = src.split("\n")
    insert_at = None
    for i, ln in enumerate(lines):
        if ln.startswith("def test_") or ln.startswith("def _mk"):
            insert_at = i
            break
    if insert_at is None:
        raise SystemExit("no def test_ found — inspect manually")

    helper_lines = HELPER.rstrip("\n").split("\n")
    new_lines = lines[:insert_at] + helper_lines + [""] + lines[insert_at:]
    new_src = "\n".join(new_lines)
    ast.parse(new_src, filename=str(TEST))
    TEST.write_text(new_src)
    print(f"  inserted helper at line {insert_at + 1}")


def replace_calls():
    print("\n--- replace calls ---")
    src = TEST.read_text()
    n = src.count(OLD_STMT)
    src = src.replace(OLD_STMT, NEW_STMT)
    TEST.write_text(src)
    ast.parse(src, filename=str(TEST))
    print(f"  replaced {n} occurrence(s) with `{NEW_STMT}`")


def run_tests():
    print("\n--- docs tests ---")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py", "tests/test_roadmap.py", "-q"],
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
    print("=== v0.9.28-fix3: test_changelog helper ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"branch: {br}")

    state = verify_state()
    if state == "already":
        print("\n  no changes needed — proceeding to tests + commit")
    else:
        add_helper()
        replace_calls()

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
        "v0.9.28: B.2 DECISIONS split (fix1+fix2+fix3). "
        "DECISIONS.md 288 KB → docs/decisions/ 24 files "
        "(≤ ~40 KB each). "
        "fix1: sub-split 2106-line §13.4 block into 4 equal "
        "chunks (15a-d, ~526 each) — no L3 boundaries (§13.4/347). "
        "fix2: test_decisions_v0922_sections anchor discovery "
        "via regex instead of guessed indentation (§13.4/348). "
        "fix3: add `_read_all_decisions()` helper in "
        "test_changelog.py — both `.read_text()` calls (L137, "
        "L157, asserting section order/content) now concatenate "
        "docs/decisions/*.md; L111 path-only ref left untouched "
        "since its test passes (§13.4/349 — same-pattern reuse). "
        "Phase B goal: remove context-transfer pain — 288 KB "
        "single-file → 24 per-topic files loaded selectively. "
        "Docs-only; 0 production. 2501 fast green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)
    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"\n{log}")
    print("\nv0.9.28 COMPLETE (fix1+fix2+fix3). B.2 done.")


if __name__ == "__main__":
    main()