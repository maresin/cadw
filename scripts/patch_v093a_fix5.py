#!/usr/bin/env python3
# v0.9.3a-fix5: make_id_df chunk= param; commit v0.9.3a.
#
# fix4 pattern failed: it used ASCII "->" but the file has unicode
# "\u2192" in the docstring. Classic test-authoring error class
# (§13.4/166): pattern written from memory, not from file.
#
# fix5 uses LINE-BASED match: finds lines that start with the exact
# signature / return statement, replaces those lines only. No docstring
# content in pattern.
#
# Run: uv run python scripts/patch_v093a_fix5.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def sh(cmd, check=True):
    print("  $ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print("    " + line)
    if check and r.returncode != 0:
        print("    STDERR: " + r.stderr.strip()[-1500:])
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import py_compile; py_compile.compile('" + str(path) +
         "', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("COMPILE FAIL: " + str(path))
        print(r.stderr[-800:])
        sys.exit(1)
    print("  compile OK: " + str(path))


def fix_make_id_df():
    print("\n=== tests/test_cv_keyed_prng.py: make_id_df chunk= ===")
    p = Path("tests/test_cv_keyed_prng.py")
    lines = p.read_text().split("\n")

    # Locate make_id_df def line and its body end (first line that
    # closes the return with `}, name=name)` after the def).
    def_idx = None
    for i, line in enumerate(lines):
        if line.startswith('def make_id_df(tmp_path, n, name="L"):'):
            def_idx = i
            break
        if line.startswith('def make_id_df(tmp_path, n, chunk=1000, name="L"):'):
            print("  already fixed - skip")
            return

    if def_idx is None:
        # Show all def lines to help debug
        print("  make_id_df def not found. Def lines in file:")
        for i, line in enumerate(lines):
            if line.startswith("def "):
                print("    " + repr(line))
        raise SystemExit("make_id_df def not found")

    # Find the closing `}, name=name)` inside the body (next 10 lines).
    close_idx = None
    for j in range(def_idx, min(def_idx + 10, len(lines))):
        if lines[j].rstrip().endswith("}, name=name)"):
            close_idx = j
            break

    if close_idx is None:
        print("  make_id_df body context:")
        for j in range(def_idx, min(def_idx + 10, len(lines))):
            print("    " + repr(lines[j]))
        raise SystemExit("make_id_df close line not found")

    # Apply edits
    old_sig = lines[def_idx]
    new_sig = 'def make_id_df(tmp_path, n, chunk=1000, name="L"):'
    old_close = lines[close_idx]
    new_close = old_close.replace("}, name=name)", "}, chunk=chunk, name=name)")

    print("  line " + str(def_idx + 1) + ": " + repr(old_sig))
    print("    -> " + repr(new_sig))
    print("  line " + str(close_idx + 1) + ": " + repr(old_close))
    print("    -> " + repr(new_close))

    lines[def_idx] = new_sig
    lines[close_idx] = new_close
    p.write_text("\n".join(lines))
    compile_check(p)
    print("  OK")


def main():
    print("=== v0.9.3a-fix5 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_make_id_df()

    print("\n=== pytest: cv keyed prng ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_cv_keyed_prng.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: existing cv tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_cv.py",
         "tests/test_cv_e2e.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** EXISTING CV FAILED ***")
        sys.exit(1)

    print("\n=== pytest: pipeline + scalers regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py",
         "tests/test_pipeline_dag.py",
         "tests/test_scalers.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
        "tests/test_patch_utils.py",
    ])

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

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.9.3a: KFold keyed PRNG - O(batch) state default; "
        "materialize=True opt-in (guarded check_state(20*N)); "
        "random_state=None->42; _attach_column_with_fn streaming helper"])

    print("\n==> v0.9.3a applied (committed).")
    print("NEXT: v0.9.3a-docs (DECISIONS 13.4/173+174, CHANGELOG)")


if __name__ == "__main__":
    main()
