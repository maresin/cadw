#!/usr/bin/env python3
# v0.9.26u-fix1: test_merge_uses_streaming targets _join, not _core.
# §13.4/323: monkeypatch target follows symbol after extraction.
# Bootstrap-safe.

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

NEW_MOD = "cadw/dataframe/_join.py"
TEST_PATH = "tests/test_join_streaming.py"
CORE = "cadw/dataframe/_core.py"


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26t":
        raise SystemExit(f"expected HEAD=v0.9.26t, got {head!r}")
    print("HEAD: " + head)

    if not Path(NEW_MOD).exists():
        raise SystemExit(f"{NEW_MOD} missing")
    mod_src = Path(NEW_MOD).read_text()
    if "class _JoinMixin" not in mod_src:
        raise SystemExit(f"_JoinMixin not in {NEW_MOD}")
    if "def merge(" not in mod_src:
        raise SystemExit(f"merge not in {NEW_MOD}")
    print(f"  {NEW_MOD}: OK")

    # Test still imports _core
    test_src = Path(TEST_PATH).read_text()
    if "from cadw.dataframe import _core as df_mod" not in test_src:
        raise SystemExit("test doesn't import _core — already fixed?")
    print(f"  {TEST_PATH}: imports _core (needs fix)")


def fix_join_imports():
    """Add hash_join to _join.py imports for test symmetry."""
    print("\n--- fix _join.py imports ---")
    p = Path(NEW_MOD)
    src = p.read_text()

    old = "from cadw.ops.join import hash_join_streaming"
    new = "from cadw.ops.join import hash_join, hash_join_streaming"

    if new in src:
        print("  already fixed")
        return 0
    if old not in src:
        print("  WARN: anchor not found — inspect manually")
        # Check if maybe hash_join already imported
        if "hash_join" in src and "hash_join_streaming" in src:
            print("  both already present")
            return 0
        raise SystemExit(f"anchor {old!r} not in {NEW_MOD}")

    src = src.replace(old, new, 1)
    p.write_text(src)
    ast.parse(src, filename=NEW_MOD)
    print(f"  {old!r} -> {new!r}")


def fix_test_import():
    """Update test to import _join (where merge lives)."""
    print("\n--- fix test import ---")
    p = Path(TEST_PATH)
    src = p.read_text()

    old = "from cadw.dataframe import _core as df_mod"
    new = "from cadw.dataframe import _join as df_mod"

    if new in src:
        print("  already fixed")
        return 0
    if old not in src:
        raise SystemExit(f"anchor {old!r} not in {TEST_PATH}")

    src = src.replace(old, new, 1)
    p.write_text(src)
    ast.parse(src, filename=TEST_PATH)
    print(f"  {old!r} -> {new!r}")


def audit_other_tests():
    """Check for other tests patching _core methods that moved to _join."""
    print("\n--- audit other tests for _core.hash_join* ---")
    moved_symbols = ["merge", "hash_join_streaming"]
    hits = []
    for py in Path("tests").rglob("*.py"):
        if py.name == Path(TEST_PATH).name:
            continue  # already fixed
        src = py.read_text()
        for sym in moved_symbols:
            pat_core = f"_core.{sym}"
            pat_dfmod = f"df_mod.{sym}"
            # Only flag if importing _core
            if (pat_core in src or (pat_dfmod in src and
                "from cadw.dataframe import _core" in src)):
                hits.append((str(py), sym))
    if not hits:
        print("  no other suspicious _core patches for moved symbols")
    else:
        for fname, sym in hits:
            print(f"  WARN: {fname} patches _core.{sym}")
        return hits
    return []


def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.26u-fix1: test import target ---")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()
    fix_join_imports()
    fix_test_import()
    audit_other_tests()
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
        "v0.9.26u: extract _join (B.1 step 15, single-method) "
        "+ fix1 (test target). merge (~292 LOC) moved verbatim "
        f"from _core.py to _join.py as _JoinMixin. "
        "CadwDataFrame(12 mixins). fix1 (§13.4/323): "
        "test_join_streaming monkeypatch targets _join.hash_join_"
        "streaming, not _core (symbols re-exported via mixin but "
        "separate binding after extraction); added hash_join to "
        "_join.py imports for test symmetry. 0 logic change; "
        "tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26u COMPLETE (fix1)")


if __name__ == "__main__":
    main()