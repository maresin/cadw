#!/usr/bin/env python3
# v0.9.5a-fix2: stale module cache in fix1 main flow.
#
# Bug: patch_v095a_fix1.py imported patch_utils at top (bind
# verify_whitelist via module cache), then modified patch_utils.py
# on disk (recursive _collect_all_qualnames). In-memory old
# verify_whitelist -> old _collect_all_qualnames -> 5 false-positive
# orphans even though file was correct (pytest fresh-import confirmed).
#
# fix2:
#   1. importlib.reload(patch_utils) — refresh in-process module dict.
#   2. verify_whitelist via subprocess (fresh interpreter, no cache).
#   3. commit_with_guards(verify_whitelist=True) — dogfood, uses
#      reloaded module (globals lookup at call time).
#
# Run: uv run python scripts/patch_v095a_fix2.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


def verify_files_state():
    print("\n=== verify: fix1 applied to files ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "def _walk_defs(" not in s:
        print("  WARNING: fix1 _walk_defs missing; abort to inspect")
        sys.exit(1)
    print("  recursive _walk_defs present")

    t = Path("tests/test_patch_utils.py")
    ts = t.read_text()
    if "def test_collect_all_qualnames_nested" not in ts:
        print("  WARNING: nested test missing; abort")
        sys.exit(1)
    print("  nested test present")


def reload_patch_utils():
    print("\n=== reload cadw.patch_utils (fresh in-memory) ===")
    import cadw.patch_utils as pu
    importlib.reload(pu)
    assert hasattr(pu, "verify_whitelist")
    print("  OK")


def verify_whitelist_subprocess():
    print("\n=== verify_whitelist (fresh interpreter) ===")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "from cadw.patch_utils import verify_whitelist; "
         "verify_whitelist()"],
        capture_output=True, text=True, timeout=120,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print("    " + line)
    if r.returncode != 0:
        print("\n*** verify_whitelist FAILED (subprocess) ***")
        if r.stderr.strip():
            print(r.stderr.strip()[-800:])
        sys.exit(1)
    print("  OK")


def main():
    print("=== v0.9.5a-fix2 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    verify_files_state()
    reload_patch_utils()
    verify_whitelist_subprocess()

    print("\n=== pytest: test_patch_utils.py ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_patch_utils.py", "-v"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** PATCH_UTILS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: AST guard ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py", "-q"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** AST GUARD FAILED ***")
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
        "v0.9.5a: verify_whitelist() — AST orphan detector for "
        "WHITELISTED_FUNCS; commit_with_guards(verify_whitelist=True) "
        "hook; _collect_all_qualnames recursive (nested Class.method."
        "nested funcs); fix2: reload + subprocess to defeat module "
        "cache",
        self_path="scripts/patch_v095a_fix2.py",
        verify_whitelist=True,
    )
    print("\n==> v0.9.5a complete (initial + fix1 + fix2).")


if __name__ == "__main__":
    main()
