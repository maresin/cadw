#!/usr/bin/env python3
# v0.9.6b1-fix1 — fix 3 test assertions using pa.Table.column on CadwDataFrame.
#
# Bug: test_from_table.py sort tests used `sorted_df.column("a")` —
# CadwDataFrame has no `.column()` method (that's pa.Table). Existing
# tests in test_dataframe.py write to parquet and reread, or use
# .head() → pa.Table. We can now use `df["a"].to_list()` — the API
# we added in v0.9.6b1 itself.
#
# Also demonstrates CADW-native pattern: assert on values via
# _ColumnProxy.to_list() rather than materializing a Table.

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


REPLACEMENTS = [
    # (old, new)
    (
        '    assert sorted_df.column("a").to_pylist() == [1, 1, 2, 3, 4, 5, 6, 9]\n',
        '    assert sorted_df["a"].to_list() == [1, 1, 2, 3, 4, 5, 6, 9]\n',
    ),
    (
        '    assert sorted_df.column("a").to_pylist() == [5, 4, 3, 1, 1]\n',
        '    assert sorted_df["a"].to_list() == [5, 4, 3, 1, 1]\n',
    ),
    (
        '    assert sorted_df.column("a").to_pylist() == [1, 1, 2, 2]\n'
        '    assert sorted_df.column("b").to_pylist() == [5, 10, 15, 20]\n',
        '    assert sorted_df["a"].to_list() == [1, 1, 2, 2]\n'
        '    assert sorted_df["b"].to_list() == [5, 10, 15, 20]\n',
    ),
]


def fix_test_assertions():
    print("\n=== test_from_table.py: fix .column() → [\"col\"].to_list() ===")
    p = Path("tests/test_from_table.py")
    s = p.read_text()
    fixed = 0
    for old, new in REPLACEMENTS:
        if old in s:
            s = s.replace(old, new, 1)
            fixed += 1
        else:
            print(f"  WARNING: anchor not found: {old[:60]!r}...")
    p.write_text(s)
    assert_compiles(p)
    print(f"  fixed {fixed}/{len(REPLACEMENTS)} replacements")


def main():
    print("=== v0.9.6b1-fix1 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_test_assertions()

    print("\n=== pytest: new tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_from_table.py",
         "tests/test_column_proxy_aggregators.py", "-v"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS STILL FAILING ***")
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
        "v0.9.6b1-fix1: fix 3 test assertions in test_from_table.py "
        "(used pa.Table.column on CadwDataFrame — no such method). "
        "Replace with df['col'].to_list() — the _ColumnProxy.to_list() "
        "API added in v0.9.6b1 itself. Demonstrates CADW-native "
        "test-authoring pattern (assert on values, not materialized "
        "Tables).",
        self_path="scripts/patch_v096b1_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6b1 complete (fix1).")


if __name__ == "__main__":
    main()
