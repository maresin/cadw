#!/usr/bin/env python3
# v0.9.7a-fix2 — _collect: polymorphic CadwDataFrame | pa.Table.
#
# Bug: 3 tests failed on `rt = _collect(ref, ...)` where
#     ref = df.sort_values(...).head(k)
# `CadwDataFrame.head(n)` returns pa.Table (terminal API, not
# CadwDataFrame). _collect called `ref.write_parquet` —
# AttributeError: 'pyarrow.lib.Table' object has no attribute
# 'write_parquet'.
#
# §13.4/128 (test-authoring: understand API contract). `head()` is
# terminal and returns pa.Table (unlike most CadwDataFrame methods
# that chain). The helper should accept both.
#
# Fix: _collect is polymorphic — if isinstance(pa.Table), return as-is;
# else write_parquet + read.
#
# State: code (topk.py, dataframe.py, whitelist, test_topk_arrow_native
# .py) already on disk from v0.9.7a-fix1's subprocess run. Only the
# test file's _collect needs the fix + full re-run + commit v0.9.7a.
#
# Run: uv run python scripts/patch_v097a_fix2.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import importlib
import cadw.patch_utils as pu
importlib.reload(pu)
from cadw.patch_utils import (
    assert_compiles, commit_with_guards, parse_method_source,
)


# ============ 1. fix _collect in test file ============

NEW_COLLECT = '''def _collect(df, tmp_path, name="o.parquet"):
    """Materialize CadwDataFrame OR pass-through pa.Table.

    v0.9.7a-fix2: CadwDataFrame.head() returns pa.Table (terminal
    API). Accept both for test convenience — CadwDataFrame written
    to parquet and read back (round-trip), pa.Table returned as-is.
    """
    if isinstance(df, pa.Table):
        return df
    out = tmp_path / name
    df.write_parquet(out)
    return pq.read_table(out)'''


def fix_collect():
    print("\n=== test_topk_arrow_native.py: polymorphic _collect ===")
    import ast as _ast
    p = Path("tests/test_topk_arrow_native.py")
    s = p.read_text()

    if '"""Materialize CadwDataFrame OR pass-through pa.Table.' in s:
        print("  already polymorphic — skip")
        return

    tree = _ast.parse(s)
    start = end = None
    for node in tree.body:
        if (isinstance(node, _ast.FunctionDef)
                and node.name == "_collect"):
            start, end = node.lineno, node.end_lineno
            break
    if start is None:
        raise SystemExit("_collect not found")

    lines = s.split("\n")
    p.write_text("\n".join(
        lines[:start - 1] + [NEW_COLLECT] + lines[end:]
    ))
    assert_compiles(p)
    print(f"  replaced lines {start}..{end}")


# ============ 2. main ============

def main():
    print("=== v0.9.7a-fix2 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_collect()

    print("\n=== pytest: topk + regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_topk_arrow_native.py",
         "tests/test_dataframe_v094b.py",
         "tests/test_no_pandas_in_ops.py",
         "tests/test_arrow_call_memory_guard.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** TOPK REGRESSION STILL FAILING ***")
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
        "v0.9.7a-fix2: _collect polymorphic (CadwDataFrame | pa.Table). "
        "CadwDataFrame.head() returns pa.Table (terminal API); "
        "test called _collect(ref) where ref = sort_values().head(k) "
        "→ AttributeError on pa.Table.write_parquet. "
        "§13.4/128 (test-authoring: understand API contract). "
        "Helper accepts both: pa.Table → as-is; CadwDataFrame → "
        "write + read round-trip. Code (topk.py, dataframe.py, "
        "whitelist, test_topk_arrow_native.py) already on disk from "
        "fix1's subprocess run. v0.9.7a complete.",
        self_path="scripts/patch_v097a_fix2.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.7a complete (fix1 + fix2).")


if __name__ == "__main__":
    main()
