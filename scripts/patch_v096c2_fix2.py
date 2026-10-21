#!/usr/bin/env python3
# v0.9.6c2-fix2 — 2 stale tests in test_pivot_table.py after v0.9.6c2.
#
# (1) test_pivot_too_large_raises:
#     match="pivot_table" — old dense-design fail-fast raised from
#     facade after estimate. New streaming: first check is
#     `pivot.filter` (check_batch before groupby_aggregate), which
#     is CORRECT contract (fail-fast before materialize). Error
#     name changed → match broadens to "pivot" (matches
#     pivot.filter / pivot.columns_est / pivot.output_batch).
#
# (2) test_pivot_table_has_check_state_before_to_pandas:
#     old contract: pivot_table has check_state Call before to_pandas.
#     New: pivot_table = thin facade delegating to pivot_to_disk; no
#     to_pandas at all; guard-by-delegation. Rewrite as two tests:
#       - test_pivot_table_delegates_to_streaming (AST: calls
#         pivot_to_disk; no to_pandas / pd.pivot attrs)
#       - test_pivot_to_disk_guarded_before_output (uses
#         patch_utils.assert_call_order — dogfood scaffold).
#
# Note: test file has duplicated basic-pivot tests also in
# test_pivot_arrow_native.py — both pass, regression safety, keep.
#
# Run: uv run python scripts/patch_v096c2_fix2.py

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
    assert_compiles, commit_with_guards, assert_call_order,
)


NEW_CONTRACT_TESTS = '''def test_pivot_table_delegates_to_streaming():
    """v0.9.6c2: pivot_table delegates to streaming pivot_to_disk.

    Old contract (v0.9.1f): dense pivot_table had check_state before
    to_pandas. New streaming design has no `to_pandas` at all —
    facade delegates to `pivot_to_disk`, which guards per-step
    (`check_state` before external_sort, before output writer).
    Guard-by-delegation: pivot_table is thin validation + call.
    """
    import ast as _ast
    import inspect
    import textwrap

    from cadw.dataframe import CadwDataFrame

    src = textwrap.dedent(inspect.getsource(CadwDataFrame.pivot_table))
    tree = _ast.parse(src)

    # 1. Delegates to pivot_to_disk (bare Name Call).
    call_names = {
        n.func.id for n in _ast.walk(tree)
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)
    }
    assert "pivot_to_disk" in call_names, (
        "pivot_table must delegate to cadw.ops.pivot.pivot_to_disk "
        "(v0.9.6c2 streaming)"
    )

    # 2. No pandas-pivot (attr Calls).
    attr_names = {
        n.func.attr for n in _ast.walk(tree)
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
    }
    assert "to_pandas" not in attr_names, (
        "pivot_table must not call to_pandas (v0.9.6c2 streaming)"
    )
    assert "pivot" not in attr_names, (
        "pivot_table must not call pd.pivot (v0.9.6c2 streaming)"
    )


def test_pivot_to_disk_guarded_before_output():
    """v0.9.6c2: pivot_to_disk has check_state Call before
    ArrowIPCBatchWriter Call (guard before output allocation).

    Uses patch_utils.assert_call_order (AST-based, v0.9.6d
    scaffold) — the helper is dogfooded here on the new streaming
    implementation (§13.4/183, /185).
    """
    import inspect

    from cadw.ops.pivot import pivot_to_disk

    src = inspect.getsource(pivot_to_disk)
    assert_call_order(
        src, "check_state", "ArrowIPCBatchWriter",
        fn_name="pivot_to_disk",
    )
'''


def fix_too_large_match():
    print("\n=== test_pivot_too_large_raises: broaden match ===")
    p = Path("tests/test_pivot_table.py")
    s = p.read_text()
    old = 'with pytest.raises(MemoryLimitExceeded, match="pivot_table"):'
    new = 'with pytest.raises(MemoryLimitExceeded, match="pivot"):'
    if old not in s:
        if new in s:
            print("  already broadened — skip")
            return
        raise SystemExit("test_pivot_too_large_raises match anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  match=\"pivot_table\" -> \"pivot\" (matches pivot.filter etc.)")


def fix_contract_test():
    print("\n=== test_pivot_table_has_check_state_before_to_pandas: "
          "rewrite to streaming contract ===")
    p = Path("tests/test_pivot_table.py")
    s = p.read_text()

    if "def test_pivot_table_delegates_to_streaming" in s:
        print("  already rewritten — skip")
        return

    tree = ast.parse(s)
    start = end = None
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name ==
                "test_pivot_table_has_check_state_before_to_pandas"):
            start, end = node.lineno, node.end_lineno
            break
    if start is None:
        raise SystemExit(
            "test_pivot_table_has_check_state_before_to_pandas not found"
        )

    lines = s.split("\n")
    p.write_text("\n".join(
        lines[:start - 1] + [NEW_CONTRACT_TESTS] + lines[end:]
    ))
    assert_compiles(p)
    print(f"  replaced lines {start}..{end} with 2 new contract tests")


def dogfood():
    print("\n=== dogfood: contract invariants on current code ===")
    import inspect
    from cadw.dataframe import CadwDataFrame
    from cadw.ops.pivot import pivot_to_disk

    # Same invariants as the rewritten tests — sanity from main.
    pt_src = inspect.getsource(CadwDataFrame.pivot_table)
    pt_tree = ast.parse(pt_src)
    call_names = {
        n.func.id for n in ast.walk(pt_tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "pivot_to_disk" in call_names
    print("  pivot_table calls pivot_to_disk")

    pv_src = inspect.getsource(pivot_to_disk)
    assert_call_order(
        pv_src, "check_state", "ArrowIPCBatchWriter",
        fn_name="pivot_to_disk",
    )
    print("  pivot_to_disk: check_state before ArrowIPCBatchWriter")
    print("  OK")


def main():
    print("=== v0.9.6c2-fix2 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_too_large_match()
    fix_contract_test()
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
        "v0.9.6c2-fix2: 2 stale tests in test_pivot_table.py after "
        "v0.9.6c2 streaming rewrite. "
        "(1) test_pivot_too_large_raises: match=\"pivot_table\" -> "
        "\"pivot\" — old dense-design fail-fast raised from facade; "
        "new streaming fails at pivot.filter (check_batch before "
        "groupby_aggregate, correct contract). match=\"pivot\" catches "
        "all pivot.* names. "
        "(2) test_pivot_table_has_check_state_before_to_pandas -> "
        "2 new contract tests: test_pivot_table_delegates_to_streaming "
        "(AST: pivot_table calls pivot_to_disk; no to_pandas/pd.pivot "
        "attrs) + test_pivot_to_disk_guarded_before_output (uses "
        "patch_utils.assert_call_order — dogfood scaffold v0.9.6d). "
        "v0.9.6c2 complete.",
        self_path="scripts/patch_v096c2_fix2.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6c2 complete (fix1 + fix2).")


if __name__ == "__main__":
    main()