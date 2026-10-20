#!/usr/bin/env python3
# v0.9.3a-fix4: make_id_df helper chunk= param; commit v0.9.3a.
#
# test_kfold_default_large_n_tight_budget calls
#   make_id_df(tmp_path, n=100_000, chunk=1000)
# but helper signature is (tmp_path, n, name="L") -- no chunk. Fix:
# forward chunk to make_df (which defaults to chunk=1000).
#
# Not a CADW bug -- test-helper signature. Same class as §13.4/166
# (test-authoring) -- manual assumption about helper API without
# verifying.
#
# Run: uv run python scripts/patch_v093a_fix4.py

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
    s = p.read_text()

    old = (
        'def make_id_df(tmp_path, n, name="L"):\n'
        '    """Identity data: y == row_id -> fold content reveals '
        'fold_id."""\n'
        '    return make_df(tmp_path, {\n'
        '        "x": list(range(n)), "y": list(range(n)),\n'
        '    }, name=name)\n'
    )

    new = (
        'def make_id_df(tmp_path, n, chunk=1000, name="L"):\n'
        '    """Identity data: y == row_id -> fold content reveals '
        'fold_id."""\n'
        '    return make_df(tmp_path, {\n'
        '        "x": list(range(n)), "y": list(range(n)),\n'
        '    }, chunk=chunk, name=name)\n'
    )

    if new in s and old not in s:
        print("  already fixed - skip")
        return
    if old not in s:
        print("  WARNING: expected block not found; showing context...")
        idx = s.find("def make_id_df")
        if idx >= 0:
            start = max(0, idx - 40)
            end = min(len(s), idx + 240)
            print("  context: " + repr(s[start:end]))
        raise SystemExit(
            "make_id_df block not found; inspect "
            "tests/test_cv_keyed_prng.py around 'def make_id_df'"
        )
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


def main():
    print("=== v0.9.3a-fix4 ===")

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
