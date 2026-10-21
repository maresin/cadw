#!/usr/bin/env python3
# v0.9.6c2-fix3 — fix2 dogfood ast.parse missing textwrap.dedent.
#
# Bug: dogfood() did
#     pt_src = inspect.getsource(CadwDataFrame.pivot_table)
#     pt_tree = ast.parse(pt_src)   # IndentationError
# inspect.getsource on a method returns the body INCLUDING its
# class-level indent (4 spaces before `def`). ast.parse requires
# zero indent for top-level node. Same issue in fix2's new test
# WAS handled (textwrap.dedent used); dogfood was not.
#
# Note: patch_utils.assert_call_order / assert_no_call_in_source
# internally call textwrap.dedent — no change needed there. Only
# raw `ast.parse(getsource(...))` needs the wrap.
#
# Rule (§13.4/191): every `ast.parse(inspect.getsource(method))`
# must be `ast.parse(textwrap.dedent(inspect.getsource(method)))`.
# Consider adding `patch_utils.parse_method_source(fn)` in v0.9.7.
#
# Both code changes from fix2 are already on disk (test rewrite +
# match broaden). fix3 only fixes dogfood's ast.parse, then runs
# full validation and commits v0.9.6c2.
#
# Run: uv run python scripts/patch_v096c2_fix3.py

from __future__ import annotations
import ast
import inspect
import os, subprocess, sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import importlib
import cadw.patch_utils as pu
importlib.reload(pu)
from cadw.patch_utils import assert_call_order, commit_with_guards


def dogfood():
    print("\n=== dogfood (fixed): contract invariants on current code ===")
    from cadw.dataframe import CadwDataFrame
    from cadw.ops.pivot import pivot_to_disk

    pt_src = textwrap.dedent(
        inspect.getsource(CadwDataFrame.pivot_table)
    )
    pt_tree = ast.parse(pt_src)
    call_names = {
        n.func.id for n in ast.walk(pt_tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "pivot_to_disk" in call_names, (
        "pivot_table must delegate to pivot_to_disk"
    )
    print("  pivot_table calls pivot_to_disk")

    # Bonus: assert also that no to_pandas / pd.pivot in facade.
    attr_names = {
        n.func.attr for n in ast.walk(pt_tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "to_pandas" not in attr_names
    assert "pivot" not in attr_names
    print("  pivot_table: no to_pandas/pd.pivot")

    pv_src = inspect.getsource(pivot_to_disk)
    assert_call_order(
        pv_src, "check_state", "ArrowIPCBatchWriter",
        fn_name="pivot_to_disk",
    )
    print("  pivot_to_disk: check_state before ArrowIPCBatchWriter")
    print("  OK")


def main():
    print("=== v0.9.6c2-fix3 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    # Code changes (fix2) already on disk. Verify and dogfood.
    dogfood()

    print("\n=== pytest: pivot regression + new tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pivot_table.py",
         "tests/test_pivot_table_e2e.py",
         "tests/test_pivot_arrow_native.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PIVOT REGRESSION FAILED ***")
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
        "v0.9.6c2-fix3: dogfood ast.parse(inspect.getsource(method)) "
        "missing textwrap.dedent — IndentationError (method source "
        "includes class-level 4-space indent; ast.parse requires "
        "zero-indent top-level). fix2 handled this in the new test "
        "(textwrap.dedent used) but dogfood() did raw ast.parse. "
        "§13.4/191: every `ast.parse(getsource(method))` must be "
        "`ast.parse(textwrap.dedent(getsource(method)))`. "
        "patch_utils.assert_call_order / assert_no_call_in_source "
        "already dedent internally — no change needed there. "
        "Candidate helper for v0.9.7: "
        "patch_utils.parse_method_source(fn). "
        "Both code changes from fix2 already on disk; fix3 only "
        "fixes dogfood. v0.9.6c2 complete.",
        self_path="scripts/patch_v096c2_fix3.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6c2 complete (fix1 + fix2 + fix3).")


if __name__ == "__main__":
    main()