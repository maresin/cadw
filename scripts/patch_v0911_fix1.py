#!/usr/bin/env python3
# v0.9.11-fix1 — dedent before ast.parse in dogfood. v0.9.11 COMPLETE.
#
# v0.9.11 dogfood died at _ast.parse(src):
#   src = inspect.getsource(CadwDataFrame._per_group_apply)
#   tree = _ast.parse(src)
#   → IndentationError: unexpected indent at line 1
#     "def _per_group_apply("
#
# Root: inspect.getsource on a class method returns source WITH the
# class-level indent (4 spaces before `def`). ast.parse requires
# top-level zero-indent. Same class as v0.9.10-fix3 dedent in
# run_verify_in_subprocess (§13.4/211 — dedent helper source before
# writing to temp file). Here — inside the dogfood, dedent before
# ast.parse (§13.4/215 new). §13.4/216: patch_utils.parse_method_source
# (§13.4/191) exists and does exactly this — must use it, not
# reinvent inline. For substring checks (which need the source string,
# not the tree), textwrap.dedent directly is simpler.
#
# State on disk (from v0.9.11, all applied + verify_state green):
#   PF4 _maybe_upgrade_schema + _flush_buf call + post-flush raise
#   U1 test pa.date64 + from_table
#   commit never reached (dogfood died)
#
# fix1: textwrap.dedent inside dogfood before ast.parse. Re-run
# verify + dogfood + pytest + commit. v0.9.11 COMPLETE.
#
# Run: uv run python scripts/patch_v0911_fix1.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


def verify_state():
    print("\n=== verify: v0.9.11 + regression ===")
    pu.verify_anchors([
        ("cadw/dataframe.py", "def _maybe_upgrade_schema("),
        ("cadw/dataframe.py", "v0.9.11 (PF4)"),
        ("cadw/dataframe.py", "post-flush widen needed"),
        ("tests/test_dt_accessor.py", "v0.9.11 (U1): actual pa.date64()"),
        # v0.9.10 regression
        ("cadw/dataframe.py", "_emit_row(row)"),
        ("cadw/dataframe.py", "v0.9.10 (U2)"),
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "compile(new_s, str(p), 'exec')"),
        ("cadw/pipeline.py", "raw_columns. Fix (backlog v0.9.11)"),
        # v0.9.9 regression
        ("cadw/patch_utils.py", "_tw.dedent(_inspect.getsource"),
        ("cadw/pipeline.py", "v0.9.9 (N1): use __serialization_name__"),
        ("cadw/accessors/_dt.py", "def _validate_timestamp_only("),
        ("cadw/dataframe.py", "Per-group apply (v0.9.6b2: CADW-native"),
    ])
    print("  all anchors present")


def dogfood():
    print("\n=== dogfood: v0.9.11 via subprocess (dedent fix) ===")

    def _verify_all():
        import ast as _ast
        import inspect
        import textwrap as _tw

        from cadw.dataframe import CadwDataFrame

        # v0.9.11-fix1 (§13.4/215): dedent before ast.parse.
        # inspect.getsource on class method returns 4-space-indented
        # source (class-level indent). ast.parse requires zero-indent.
        src = _tw.dedent(inspect.getsource(CadwDataFrame._per_group_apply))
        tree = _ast.parse(src)

        has_new = any(
            isinstance(n, _ast.FunctionDef)
            and n.name == "_maybe_upgrade_schema"
            for n in _ast.walk(tree)
        )
        assert has_new, "no _maybe_upgrade_schema def"

        has_old = any(
            isinstance(n, _ast.FunctionDef)
            and n.name == "_maybe_upgrade_for_row"
            for n in _ast.walk(tree)
        )
        assert not has_old, "_maybe_upgrade_for_row still present"

        flush_calls = False
        for n in _ast.walk(tree):
            if not (isinstance(n, _ast.FunctionDef) and n.name == "_flush_buf"):
                continue
            for sub in _ast.walk(n):
                if (isinstance(sub, _ast.Call)
                        and isinstance(sub.func, _ast.Name)
                        and sub.func.id == "_maybe_upgrade_schema"):
                    flush_calls = True
                    break
        assert flush_calls, "_flush_buf does not call _maybe_upgrade_schema"

        assert "post-flush widen needed" in src, (
            "no clear RuntimeError for post-flush widen"
        )

        # U1: module-level test fn, 0-indent — no dedent needed
        from tests.test_dt_accessor import test_dt_ceil_date64_raises_init
        u1_src = inspect.getsource(test_dt_ceil_date64_raises_init)
        assert "pa.date64()" in u1_src, "U1 test still uses date32"
        assert "from_table" in u1_src, "U1 test should use from_table"

        print("  PF4 (deferred widen, clear post-flush error) OK")
        print("  U1 (pa.date64 + from_table) OK")

    pu.run_verify_in_subprocess(_verify_all)


def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_apply.py",
         "tests/test_apply_e2e.py",
         "tests/test_dt_accessor.py",
         "tests/test_patch_utils.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGET FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=900,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


def main():
    print("=== v0.9.11-fix1 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.11-fix1: complete v0.9.11 (dedent before ast.parse in "
        "dogfood). v0.9.11 dogfood died at _ast.parse(src): "
        "src = inspect.getsource(CadwDataFrame._per_group_apply) "
        "returns 4-space-indented source (class-level indent), "
        "ast.parse requires zero-indent. §13.4/215 (new): dedent "
        "before ast.parse — same class as v0.9.10-fix3 dedent in "
        "run_verify_in_subprocess (§13.4/211), but inside dogfood. "
        "§13.4/216 (new): patch_utils.parse_method_source "
        "(§13.4/191) exists and does exactly this — must use it, "
        "not reinvent inline. Fix1: textwrap.dedent before ast.parse "
        "(for substring checks that need source string, and for "
        "ast.parse that needs zero-indent). PF4 (from v0.9.11): "
        "_maybe_upgrade_for_row → _maybe_upgrade_schema deferred to "
        "_flush_buf — per-row upgraded list alloc was O(N*K) (10M "
        "groups × K=10 ~100M allocs + ~1B attr checks), now one "
        "scan per flush (~1000x fewer allocs, ~10x fewer attr "
        "checks). Post-flush widen raises clear RuntimeError (was "
        "silent truncation or cryptic ArrowInvalid at write_batch). "
        "§11.5 BATCH_ROWS limitation now explicit. U1 (from "
        "v0.9.11): test_dt_ceil_date64_raises_init → actual "
        "pa.date64() (was pa.date32() copy-paste), via from_table "
        "(parquet date64 → date32 roundtrip, §11.5). v0.9.11 "
        "COMPLETE: PF4 + U1 applied, dogfood green (dedent fix), "
        "tests green, fast+slow green.",
        self_path="scripts/patch_v0911_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.11-fix1 applied. v0.9.11 COMPLETE.")


if __name__ == "__main__":
    main()
