#!/usr/bin/env python3
# v0.9.6e2-fix1 — assert_test_imports_defined: builtins + lambda.
#
# Bug: dogfood failed on test_pivot_table.py:
#   undefined Names:
#     test_melt_mixed_dtype_raises: ['TypeError']
#     test_pivot_invalid_aggfunc: ['ValueError']
#     test_crosstab_matches_pivot_count: ['r']
#
# Two false positives:
#   (1) TypeError/ValueError/KeyError/RuntimeError etc. are BUILTINS,
#       but my manual `_BUILTINS` list missed exception types
#       (had only abs/all/.../str/sum/tuple/type/zip). Fix: use
#       dir(builtins) — complete list, no maintenance.
#   (2) lambda parameter: `sorted(..., key=lambda r: r["g"])` — `r`
#       is bound by Lambda args, but my walker only added args of
#       FunctionDef/AsyncFunctionDef, not Lambda. Fix: handle Lambda
#       (and posonlyargs / kwonlyargs / vararg / kwarg uniformly).
#
# Helpers + tests already on disk (v0.9.6e2 wrote them before
# dogfood failed). fix1 replaces `_BUILTINS` block, adds lambda
# branch, +2 regression tests. Full re-run + commit.
#
# Run: uv run python scripts/patch_v096e2_fix1.py

from __future__ import annotations
import ast
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import importlib
import cadw.patch_utils as pu
importlib.reload(pu)
from cadw.patch_utils import (
    assert_compiles, commit_with_guards,
    assert_no_call_in_source,
)


# ============ 1. replace _BUILTINS + add lambda handling ============

OLD_BUILTINS_BLOCK = '''_BUILTINS = frozenset(
    "abs all any bool bytes callable chr dict dir enumerate "
    "filter float format frozenset getattr hasattr hash hex id "
    "int isinstance issubclass iter len list map max min next "
    "object oct ord pow print range repr reversed round set "
    "slice sorted str sum tuple type zip True False None"
    .split()
)'''

NEW_BUILTINS_BLOCK = '''# Complete list from `dir(builtins)` — includes abs/all/.../zip AND
# every exception type (TypeError, ValueError, KeyError, ...), True/
# False/None, __import__, etc. No maintenance; no false positives on
# builtin exceptions (§13.4/195).
import builtins as _builtins_mod
_BUILTINS = frozenset(dir(_builtins_mod))'''


OLD_LAMBDA_SKIP = '''            elif isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                               _ast.ClassDef)):
                if n is not fn_node:
                    local.add(n.name)'''

NEW_LAMBDA_SKIP = '''            elif isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                               _ast.ClassDef)):
                if n is not fn_node:
                    local.add(n.name)
                    # Nested def: its args are bound names within
                    # the nested def's scope, NOT ours; skip. But
                    # the NAME is now in our local.
            elif isinstance(n, _ast.Lambda):
                # Lambda params bind in the lambda body, which may
                # reference them in this function's expression tree.
                # Add all arg kinds uniformly (§13.4/195).
                for a in getattr(n.args, "args", ()):
                    local.add(a.arg)
                for a in getattr(n.args, "posonlyargs", ()):
                    local.add(a.arg)
                for a in getattr(n.args, "kwonlyargs", ()):
                    local.add(a.arg)
                if getattr(n.args, "vararg", None):
                    local.add(n.args.vararg.arg)
                if getattr(n.args, "kwarg", None):
                    local.add(n.args.kwarg.arg)'''


def fix_builtins_block():
    print("\n=== patch_utils.py: replace _BUILTINS with dir(builtins) ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()

    if 'import builtins as _builtins_mod' in s:
        print("  already fixed — skip")
        return

    if OLD_BUILTINS_BLOCK not in s:
        raise SystemExit("_BUILTINS anchor not found")
    s = s.replace(OLD_BUILTINS_BLOCK, NEW_BUILTINS_BLOCK, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


def add_lambda_branch():
    print("\n=== patch_utils.py: add Lambda args handling ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()

    if "isinstance(n, _ast.Lambda)" in s:
        print("  already fixed — skip")
        return

    if OLD_LAMBDA_SKIP not in s:
        raise SystemExit("FunctionDef/ClassDef anchor not found")
    s = s.replace(OLD_LAMBDA_SKIP, NEW_LAMBDA_SKIP, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. add regression tests ============

NEW_REGRESSION_TESTS = [
    "",
    "",
    "def test_assert_test_imports_defined_builtin_exceptions_ok():",
    "    # §13.4/195: TypeError/ValueError/etc are builtins; must not",
    "    # be flagged as undefined.",
    "    from cadw.patch_utils import assert_test_imports_defined",
    "    src = (",
    "        'def test_a():\\n'",
    "        '    raise TypeError(\"x\")\\n'",
    "        'def test_b():\\n'",
    "        '    try:\\n'",
    "        '        int(\"no\")\\n'",
    "        '    except ValueError:\\n'",
    "        '        pass\\n'",
    "    )",
    "    assert_test_imports_defined(src, fn_name='builtin_exceptions')",
    "",
    "",
    "def test_assert_test_imports_defined_lambda_param_ok():",
    "    # §13.4/195: lambda parameter binds within lambda body; must not",
    "    # be flagged as undefined.",
    "    from cadw.patch_utils import assert_test_imports_defined",
    "    src = (",
    "        'def test_a():\\n'",
    "        '    rows = [{\"g\": \"x\"}]\\n'",
    "        '    out = sorted(rows, key=lambda r: r[\"g\"])\\n'",
    "        '    assert len(out) == 1\\n'",
    "    )",
    "    assert_test_imports_defined(src, fn_name='lambda_param')",
]


def add_regression_tests():
    print("\n=== test_patch_utils.py: +2 regression tests ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "def test_assert_test_imports_defined_builtin_exceptions_ok" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(NEW_REGRESSION_TESTS))
    assert_compiles(p)
    print("  OK")


# ============ 3. dogfood ============

def dogfood():
    print("\n=== dogfood: assert_test_imports_defined on real files ===")
    import importlib
    import cadw.patch_utils as pu
    importlib.reload(pu)
    from cadw.patch_utils import assert_test_imports_defined

    for fname in (
        "tests/test_pivot_table.py",
        "tests/test_pivot_arrow_native.py",
        "tests/test_apply.py",
    ):
        src = Path(fname).read_text()
        try:
            assert_test_imports_defined(src, fn_name=fname)
            print(f"  {fname}: OK")
        except AssertionError as e:
            print(f"  {fname}: FAIL — {e}")
            raise

    # Sanity: negative control still catches real missing.
    try:
        assert_test_imports_defined(
            "def test_x():\n    totally_undefined_fn(1)\n",
            fn_name="negative-control",
        )
    except AssertionError:
        print("  negative control: correctly rejected")
    else:
        raise SystemExit("negative control failed")


# ============ main ============

def main():
    print("=== v0.9.6e2-fix1 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_builtins_block()
    add_lambda_branch()
    add_regression_tests()
    dogfood()

    print("\n=== pytest: patch_utils + guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_patch_utils.py",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_no_pandas_in_ops.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PATCH_UTILS/GUARDS FAILED ***")
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
        "v0.9.6e2-fix1: assert_test_imports_defined too narrow — "
        "false positives on builtin exceptions + lambda params. "
        "(1) _BUILTINS replaced with `frozenset(dir(builtins))` "
        "— complete list, no maintenance; TypeError/ValueError/"
        "KeyError/etc now recognized (§13.4/195). "
        "(2) walker handles ast.Lambda — args (args/posonlyargs/"
        "kwonlyargs/vararg/kwarg) bind in lambda body. "
        "+2 regression tests (builtin_exceptions_ok, lambda_param_ok). "
        "Dogfood on real tests/test_pivot_table.py, test_pivot_arrow_"
        "native.py, test_apply.py. Helpers + 16 original tests "
        "already on disk from v0.9.6e2 (dogfood failed before "
        "pytest). v0.9.6e2 complete.",
        self_path="scripts/patch_v096e2_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6e2 complete (fix1).")


if __name__ == "__main__":
    main()
