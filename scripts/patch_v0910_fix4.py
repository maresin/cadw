#!/usr/bin/env python3
# v0.9.10-fix4 — dogfood negative-check via AST. v0.9.10 COMPLETE.
#
# fix3 died at dogfood: `assert "combine_chunks" not in src_concat`.
# U2 replaced `from_batches().combine_chunks().to_batches()[0]` with
# `pa.concat_arrays(...)` — but the U2 comment explains the change:
#   "# vs from_batches + combine_chunks + to_batches (3x alloc)".
# Substring `combine_chunks` matches the COMMENT, not a Call node.
#
# §13.4/161/213: negative substring check must be AST, not `not in`.
# Third occurrence this iteration:
#   - v0.9.9 F9: anchor had wrong line break (silent skip).
#   - v0.9.10-fix1/2/3: anchor indent off-by-4.
#   - v0.9.10-fix3: dogfood `not in src` matched comment.
# Same fix class as test_facade_has_no_as_py (docstring matched).
#
# State on disk (from fix3, all committed as uncommitted worktree):
#   v0.9.9 F1-F9                — intact
#   T2.1 apply_or_fail/verify   — compile-before-write (§13.4/211)
#   C1 _maybe_upgrade_for_row   — present
#   C1 tests                    — present
#   U2 _concat_batches          — pa.concat_arrays (but comment has
#                                 "combine_chunks")
#   T2.4 raw_columns docstring  — present
#   commit                      — never reached (^C after dogfood fail)
#
# fix4: replace dogfood substring with AST walk; pytest; commit.
#
# Run: uv run python scripts/patch_v0910_fix4.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ verify: all anchors present, no re-apply ============

def verify_state():
    print("\n=== verify: v0.9.10 + v0.9.9 on disk ===")
    pu.verify_anchors([
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "compile(new_s, str(p), 'exec')"),
        ("cadw/patch_utils.py", "def verify_anchors("),
        ("cadw/dataframe.py", "def _maybe_upgrade_for_row("),
        ("cadw/dataframe.py", "def _emit_row("),
        ("cadw/dataframe.py", "v0.9.10 (U2)"),
        ("cadw/pipeline.py", "raw_columns. Fix (backlog v0.9.11)"),
        # v0.9.9 regression
        ("cadw/patch_utils.py", "_tw.dedent(_inspect.getsource"),
        ("cadw/pipeline.py", "v0.9.9 (N1): use __serialization_name__"),
        ("cadw/accessors/_dt.py", "def _validate_timestamp_only("),
        ("cadw/ops/groupby.py", "v0.9.9 (N3)"),
        ("cadw/pipeline.py", "v0.9.9 (N5)"),
        ("cadw/ml/pca.py", "if TYPE_CHECKING:  # v0.9.9 (B2)"),
        ("cadw/pipeline.py", "# v0.9.9 (B3)"),
        ("cadw/ml/pca.py", "v0.9.9 (B4)"),
        ("cadw/dataframe.py", "Per-group apply (v0.9.6b2: CADW-native"),
        # v0.9.10 tests
        ("tests/test_patch_utils.py", "test_apply_or_fail_no_write_on_compile_fail"),
        ("tests/test_apply.py", "test_apply_mixed_int_float_widens_schema"),
    ])
    print("  all anchors present")


# ============ dogfood (AST, no substring) ============

def dogfood():
    print("\n=== dogfood: v0.9.10 via subprocess (AST negative check) ===")

    def _verify_all():
        import ast as _ast
        import inspect

        # T2.1
        from cadw.patch_utils import apply_or_fail, verify_anchors
        assert callable(apply_or_fail)
        assert callable(verify_anchors)

        # apply_or_fail compile-before-write (§13.4/211):
        #   `new_s = s.replace(old, new, 1)` appears BEFORE
        #   `p.write_text(new_s)`. Use ast to locate exact Call nodes.
        src_aof = inspect.getsource(apply_or_fail)
        tree_aof = _ast.parse(src_aof)
        # Find `p.write_text(...)` Call line and `compile(...)` Call line
        compile_lines = [
            n.lineno for n in _ast.walk(tree_aof)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Name)
            and n.func.id == "compile"
        ]
        write_lines = [
            n.lineno for n in _ast.walk(tree_aof)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "write_text"
            and isinstance(n.func.value, _ast.Name)
            and n.func.value.id == "p"
        ]
        assert compile_lines, "apply_or_fail has no compile() Call"
        assert write_lines, "apply_or_fail has no p.write_text() Call"
        assert min(compile_lines) < min(write_lines), (
            f"§13.4/211 regression: compile() at line {min(compile_lines)} "
            f"must precede p.write_text() at line {min(write_lines)}"
        )

        # C1: _per_group_apply has _emit_row + _maybe_upgrade_for_row
        from cadw.dataframe import CadwDataFrame
        src = inspect.getsource(CadwDataFrame._per_group_apply)
        assert "_maybe_upgrade_for_row" in src
        n = src.count("_emit_row(row)")
        assert n == 3, f"expected 3 _emit_row calls, got {n}"

        # U2: AST negative check — no combine_chunks() Call
        # (fix3: substring matched comment; now walk AST Call nodes)
        from cadw.dataframe import _concat_batches
        src_concat = inspect.getsource(_concat_batches)
        tree_concat = _ast.parse(src_concat)
        combine_calls = [
            n for n in _ast.walk(tree_concat)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "combine_chunks"
        ]
        assert not combine_calls, (
            f"U2 regression: _concat_batches calls combine_chunks() "
            f"at line(s) {[c.lineno for c in combine_calls]}"
        )
        # Positive: concat_arrays Call present
        concat_calls = [
            n for n in _ast.walk(tree_concat)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "concat_arrays"
        ]
        assert concat_calls, "_concat_batches has no pa.concat_arrays() Call"

        # T2.4
        from cadw.pipeline import _merge_fanout
        src_mf = inspect.getsource(_merge_fanout)
        assert "raw_columns" in src_mf

        print("  T2.1 (compile-before-write) OK")
        print("  C1 (_emit_row=3) OK")
        print("  U2 (concat_arrays yes, combine_chunks no) OK")
        print("  T2.4 (raw_columns) OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_patch_utils.py",
         "tests/test_apply.py",
         "tests/test_apply_e2e.py",
         "tests/test_pipeline_dag.py", "-q"],
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


# ============ main ============

def main():
    print("=== v0.9.10-fix4 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.10-fix4: complete v0.9.10 (dogfood AST negative check). "
        "fix3 applied all fixes (T2.1, C1, U2, T2.4) but dogfood "
        "failed on `assert 'combine_chunks' not in src_concat` — "
        "U2 comment explains the change ('# vs from_batches + "
        "combine_chunks + to_batches'), substring matches comment. "
        "§13.4/213 (new): negative substring check must be AST, not "
        "`not in src` — comments and docstrings are invisible to "
        "ast.Call nodes. Third §13.4/161 recurrence this iteration "
        "(anchor miss v0.9.9 F9; anchor indent fix1/2; dogfood "
        "comment fix3). Same class as test_facade_has_no_as_py "
        "(docstring matched). fix4: dogfood walks ast.Call nodes — "
        "no combine_chunks() Call in _concat_batches, concat_arrays "
        "Call present; also AST-verifies apply_or_fail "
        "compile()-before-write_text() ordering (§13.4/211 pinned). "
        "v0.9.10 worktree intact from fix3: T2.1 (apply_or_fail "
        "compile-before-write + verify_anchors + tests), C1 "
        "(_maybe_upgrade_for_row + _emit_row + 2 tests, silent "
        "int→float truncation fixed, BATCH_ROWS limitation "
        "documented), U2 (_concat_batches per-column concat_arrays), "
        "T2.4 (_merge_fanout raw_columns §11.5). Deferred: U1, U3, "
        "PF1-3, A1, N6, R3.",
        self_path="scripts/patch_v0910_fix4.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.10-fix4 applied. v0.9.10 COMPLETE.")


if __name__ == "__main__":
    main()
