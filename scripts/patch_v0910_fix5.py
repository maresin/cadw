#!/usr/bin/env python3
# v0.9.10-fix5 — rollback test data bug + v0.9.10 COMPLETE.
#
# fix4 dogfood green (AST negative check works). pytest failed one
# test: test_apply_or_fail_no_write_on_compile_fail. Root is TEST
# DATA, not apply_or_fail: old = '    x = 1\n', new = 'x = 1\n'.
# `new in s` is True (new is substring of old which is in original),
# so idempotency path returns False before compile — no SystemExit.
#
# §13.4/214 (new): test data for `apply_or_fail` rollback must have
# `new` NOT be substring of `old` (and vice versa). Otherwise
# idempotency check `if new in s` misfires. For real patches this
# is safe (`new` = long marker-tagged block, `old` = short source),
# but the rollback test needs clean non-substring data.
#
# Fix: old = 'x = 1', new = 'def broken(' — not substrings of each
# other. Apply → replace old with new → new_s = 'def broken(\n' →
# compile fails (unclosed paren) → SystemExit before write. File
# unchanged.
#
# State on disk (all v0.9.10 worktree from fix3+fix4, uncommitted):
#   T2.1 apply_or_fail (compile-before-write) + verify_anchors
#   T2.1 tests (5) + rollback test (broken data)
#   C1 _maybe_upgrade_for_row + _emit_row + 2 tests
#   U2 _concat_batches per-column concat_arrays
#   T2.4 _merge_fanout raw_columns §11.5
#   fix4 dogfood AST (green)
#
# Run: uv run python scripts/patch_v0910_fix5.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ Fix rollback test: non-substring old/new ============

OLD_TEST = '''def test_apply_or_fail_no_write_on_compile_fail(tmp_path):
    """v0.9.10-fix3 (§13.4/211): no write on compile fail.

    Prior: apply_or_fail wrote file, then assert_compiles → corrupt
    file on bad anchor. Fix: compile in memory, only write on success.
    """
    import pytest
    from cadw.patch_utils import apply_or_fail
    p = tmp_path / 'f.py'
    original = 'def f():\\n    x = 1\\n'
    p.write_text(original)
    with pytest.raises(SystemExit):
        apply_or_fail(str(p), '    x = 1\\n', 'x = 1\\n', label='t')
    assert p.read_text() == original'''

NEW_TEST = '''def test_apply_or_fail_no_write_on_compile_fail(tmp_path):
    """v0.9.10-fix3/fix5 (§13.4/211/214): no write on compile fail.

    Prior (fix3): apply_or_fail wrote file, then assert_compiles →
    corrupt file on bad anchor. Fix (fix3): compile in memory, only
    write on success.

    Test data (fix5): `old` and `new` MUST NOT be substrings of each
    other (§13.4/214). Prior test used old='    x = 1\\n',
    new='x = 1\\n' — new is substring of old (inside original
    'def f():\\n    x = 1\\n'). apply_or_fail's idempotency check
    `if new in s` returned False (assumed already applied) → no
    SystemExit. Now: old='x = 1', new='def broken(' — no substring
    relation; replacement is syntactically broken → compile fail →
    SystemExit before write.
    """
    import pytest
    from cadw.patch_utils import apply_or_fail
    p = tmp_path / 'f.py'
    original = 'x = 1\\n'
    p.write_text(original)
    with pytest.raises(SystemExit):
        apply_or_fail(str(p), 'x = 1', 'def broken(', label='t')
    assert p.read_text() == original'''


def fix_rollback_test():
    print("\n=== fix rollback test data (§13.4/214) ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if NEW_TEST in s:
        print("  already fixed — skip")
        return
    if OLD_TEST not in s:
        raise SystemExit("rollback test anchor not found")
    new_s = s.replace(OLD_TEST, NEW_TEST, 1)
    try:
        compile(new_s, str(p), "exec")
    except SyntaxError as e:
        raise SystemExit(f"fix_rollback_test compile fail: {e}") from e
    p.write_text(new_s)
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.10 + v0.9.9 on disk ===")
    pu.verify_anchors([
        # T2.1
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "compile(new_s, str(p), 'exec')"),
        ("cadw/patch_utils.py", "def verify_anchors("),
        # C1
        ("cadw/dataframe.py", "def _maybe_upgrade_for_row("),
        ("cadw/dataframe.py", "def _emit_row("),
        # U2
        ("cadw/dataframe.py", "v0.9.10 (U2)"),
        # T2.4
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
        # Tests
        ("tests/test_patch_utils.py", "test_apply_or_fail_no_write_on_compile_fail"),
        ("tests/test_apply.py", "test_apply_mixed_int_float_widens_schema"),
    ])
    print("  all anchors present")


# ============ dogfood (same as fix4, AST-based) ============

def dogfood():
    print("\n=== dogfood: v0.9.10 via subprocess (AST) ===")

    def _verify_all():
        import ast as _ast
        import inspect
        from cadw.patch_utils import apply_or_fail, verify_anchors
        assert callable(apply_or_fail) and callable(verify_anchors)

        src_aof = inspect.getsource(apply_or_fail)
        tree_aof = _ast.parse(src_aof)
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
        assert compile_lines and write_lines
        assert min(compile_lines) < min(write_lines), (
            "§13.4/211: compile must precede write_text"
        )

        from cadw.dataframe import CadwDataFrame, _concat_batches
        src = inspect.getsource(CadwDataFrame._per_group_apply)
        assert "_maybe_upgrade_for_row" in src
        assert src.count("_emit_row(row)") == 3

        src_concat = inspect.getsource(_concat_batches)
        tree_concat = _ast.parse(src_concat)
        combine_calls = [
            n for n in _ast.walk(tree_concat)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "combine_chunks"
        ]
        assert not combine_calls
        concat_calls = [
            n for n in _ast.walk(tree_concat)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "concat_arrays"
        ]
        assert concat_calls

        from cadw.pipeline import _merge_fanout
        assert "raw_columns" in inspect.getsource(_merge_fanout)

        print("  T2.1 + C1 + U2 + T2.4 verified")

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
    print("=== v0.9.10-fix5 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    verify_state()
    fix_rollback_test()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.10-fix5: complete v0.9.10 (rollback test data bug). "
        "fix4 dogfood green (AST negative check works). pytest "
        "failed test_apply_or_fail_no_write_on_compile_fail. Root "
        "is TEST DATA not apply_or_fail: old='    x = 1\\n', "
        "new='x = 1\\n' — new is substring of old which is in "
        "original 'def f():\\n    x = 1\\n'. apply_or_fail "
        "idempotency check `if new in s` returned False (assumed "
        "already applied) → no SystemExit. §13.4/214 (new): test "
        "data for apply_or_fail rollback must not have new as "
        "substring of old (or vice versa) — idempotency check "
        "misfires. Real patches safe (long marker-tagged new, "
        "short old). Fix: old='x = 1', new='def broken(' — no "
        "substring relation; apply → replace → compile fails → "
        "SystemExit before write. File unchanged. v0.9.10 COMPLETE: "
        "T2.1 apply_or_fail (compile-before-write §13.4/211) + "
        "verify_anchors + 5 tests + rollback; C1 "
        "_maybe_upgrade_for_row + _emit_row (silent int→float widen, "
        "BATCH_ROWS limitation documented) + 2 tests; U2 "
        "_concat_batches per-column concat_arrays; T2.4 "
        "_merge_fanout raw_columns §11.5 docstring. Deferred: U1 "
        "(test_dt_ceil_date64 date32 vs date64 cosmetic), U3 "
        "(pyproject version), PF1-3 (Arrow-native heavy keys/pivot/"
        "sample perf), PF4 (C1 hot-path alloc in _emit_row, new), "
        "A1 (dataframe.py split), N6 (Dataset.iter_batches guard), "
        "R3 (planner spill stub).",
        self_path="scripts/patch_v0910_fix5.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.10-fix5 applied. v0.9.10 COMPLETE.")


if __name__ == "__main__":
    main()
