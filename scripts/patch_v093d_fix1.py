#!/usr/bin/env python3
# v0.9.3d-fix1: 2 stale test/CHANGELOG details.
#
# 1. test_std_ast_cast_to_float64: `from cadw.ops import reduce as
#    reduce_mod` binds the FUNCTION `reduce` (via __init__.py's
#    `from .reduce import reduce`), not the module `reduce.py`.
#    `.StdOp` on function -> AttributeError. Fix: `from cadw.ops
#    import StdOp` (class exported).
#
# 2. CHANGELOG: "+8 fast" -- actual is +11 (parametrize x5).
#
# Run: uv run python scripts/patch_v093d_fix1.py

from __future__ import annotations
import os, subprocess, sys
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


# ============ Fix 1: AST test ============

def fix_ast_test():
    print("\n=== test_stdop_int_correctness.py: fix import ===")
    p = Path("tests/test_stdop_int_correctness.py")
    s = p.read_text()

    old = (
        'def test_std_ast_cast_to_float64():\n'
        '    # AST: StdOp.update casts arr to float64 before subtract.\n'
        '    import inspect\n'
        '    from cadw.ops import reduce as reduce_mod\n'
        '    src = inspect.getsource(reduce_mod.StdOp.update)\n'
        '    assert "pc.cast(arr, pa.float64())" in src\n'
        '    assert "pa.float64()" in src'
    )

    new = (
        'def test_std_ast_cast_to_float64():\n'
        '    # AST: StdOp.update casts arr to float64 before subtract.\n'
        '    # Note: `from cadw.ops import reduce as reduce_mod` would\n'
        '    # bind the FUNCTION `reduce` (via __init__.py re-export),\n'
        '    # not the module `reduce.py`. Import StdOp class directly.\n'
        '    import inspect\n'
        '    from cadw.ops import StdOp\n'
        '    src = inspect.getsource(StdOp.update)\n'
        '    assert "pc.cast(arr, pa.float64())" in src\n'
        '    assert "pa.float64()" in src'
    )

    if new in s and old not in s:
        print("  already fixed - skip")
        return
    if old not in s:
        raise SystemExit("AST test block not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 2: CHANGELOG count ============

def fix_changelog_count():
    print("\n=== CHANGELOG.md: +8 -> +11 fast ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()

    old = (
        "**Тесты (v0.9.3d):** +8 fast (`test_stdop_int_correctness.py`):\n"
        "known [1,2,3,4] test (+ sanity not buggy), int==float, 5 param cases\n"
        "vs numpy, float unchanged, nulls, int32, AST guard.\n"
        "\n"
        "- fast: **~1580** (+8). slow: **20** (unchanged)."
    )

    new = (
        "**Тесты (v0.9.3d):** +11 fast (`test_stdop_int_correctness.py`):\n"
        "known [1,2,3,4] test (+ sanity not buggy), int==float, 5 param cases\n"
        "vs numpy, float unchanged, nulls, int32, AST guard.\n"
        "\n"
        "- fast: **~1583** (+11). slow: **20** (unchanged)."
    )

    if new in s and old not in s:
        print("  already fixed - skip")
        return
    if old not in s:
        print("  WARNING: CHANGELOG block not found. Context:")
        idx = s.find("v0.9.3d - StdOp int")
        if idx >= 0:
            end = s.find("\n###", idx + 20)
            print("    " + repr(s[idx:end if end > 0 else idx + 900]))
        raise SystemExit("CHANGELOG block not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    print("  OK")


def main():
    print("=== v0.9.3d-fix1 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_ast_test()
    fix_changelog_count()

    print("\n=== pytest: stdop int correctness ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_stdop_int_correctness.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: reduce regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_reduce.py",
         "tests/test_reduce_state_bytes.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** REDUCE FAILED ***")
        sys.exit(1)

    print("\n=== pytest: numeric regression (gbdt/pca/scalers) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_gbdt_regression.py",
         "tests/test_pca.py",
         "tests/test_scalers.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NUMERIC FAILED ***")
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
        "v0.9.3d: C2 StdOp int-column truncation S2 fix — cast to "
        "float64 before subtract; int64 [1,2,3,4] buggy 1.414 -> correct "
        "1.291; semantic change flagged; 13.4/176 (int arithmetic "
        "truncation; float-only tests hide this)"])

    print("\n==> v0.9.3d applied.")
    print("NEXT: v0.9.3e (C1 dead code, trivial)")


if __name__ == "__main__":
    main()
