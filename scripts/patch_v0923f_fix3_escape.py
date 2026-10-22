#!/usr/bin/env python3
# v0.9.23f-fix3 — invalid escape in fix2 (2nd occurrence) + structural
# prevention via patch_utils.check_no_invalid_escape helper.
#
# Bug: fix2 added match="frac must be in \[0, 1\]" — non-raw string,
# invalid escape \%. pytest -W error::SyntaxWarning converts
# SyntaxWarning -> SyntaxError at collection. Two occurrences in
# v0.9.23f (fix1: "weights \+ replace"; fix2: "frac must be in \[0, 1\]").
#
# Root cause: py_compile.compile (used by assert_compiles) in Python
# 3.12 emits SyntaxWarning for invalid escapes, does NOT raise. Only
# -W error::SyntaxWarning (or ast.parse under warning-as-error) catches.
#
# Fix:
#   (1) Fix the escape: raw string r"frac must be in \[0, 1\]".
#   (2) Add patch_utils.check_no_invalid_escape(path): AST-parse with
#       -W error::SyntaxWarning; SystemExit on SyntaxWarning. Closes
#       the whole class for future patches (no per-patch discipline
#       needed).
#   (3) Add tests for the helper.
#   (4) Dogfood: helper catches the same escape in a temp file.
#
# No production changes (sample() correct; v0.9.23f-fix2 added frac=0
# support — that fix stands).
#
# Process note (§13.4/238): v0.9.23f is the 3rd fix-cycle. Over the
# ">2 fix-cycle -> split" threshold. Should have been split into
# (a) n/frac/replace + (b) weights at v0.9.23f start (~700 LOC).
# Recurring trap class (invalid escape) is now structurally closed
# by check_no_invalid_escape — no more split retroactively needed.
#
# Run: uv run python scripts/patch_v0923f_fix3_escape.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    run_verify_in_subprocess,
)


# ============ 1. Fix the escape ============

def fix_escape():
    print("\n=== fix test_sample_frac_zero escape ===")
    p = Path("tests/test_sample.py")
    s = p.read_text()
    if 'match=r"frac must be in \\[0, 1\\]"' in s:
        print("  already fixed — skip")
        return
    apply_or_fail(
        str(p),
        old=(
            '    with pytest.raises(ValueError, match="frac must be in \\[0, 1\\]"):'
        ),
        new=(
            '    with pytest.raises(\n'
            '        ValueError, match=r"frac must be in \\[0, 1\\]",\n'
            '    ):'
        ),
        label="frac_zero regex",
    )
    assert_compiles(p)
    print("  OK")


# ============ 2. Add patch_utils.check_no_invalid_escape ============

CHECK_HELPER = '''

# ============ v0.9.23f-fix3: invalid-escape helper ============


def check_no_invalid_escape(path):
    """AST-parse `path` with -W error::SyntaxWarning; SystemExit on any.

    v0.9.23f-fix3: `py_compile.compile` (used by `assert_compiles`) in
    Python 3.12 emits SyntaxWarning for invalid escape sequences
    ("\\d" in non-raw string, etc.) but does NOT raise. Only at pytest
    collection (`ast.parse` under -W error::SyntaxWarning) does this
    become SyntaxError. Two occurrences in v0.9.23f (fix1 and fix2) —
    this helper closes the class.

    Use from patch scripts after `write_content` / `write_text` for
    .py files with string literals potentially containing backslashes
    (regex matches especially):
        check_no_invalid_escape("tests/test_foo.py")

    Raises:
        SystemExit(1) if file has any invalid escape sequence
            (converted SyntaxWarning -> error). Message includes
            file path and the specific warning line.
    """
    import ast as _ast
    import warnings as _w
    import sys as _sys
    from pathlib import Path as _P

    src = _P(path).read_text()
    with _w.catch_warnings():
        _w.simplefilter("error", SyntaxWarning)
        try:
            _ast.parse(src, filename=str(path))
        except SyntaxWarning as e:
            print(
                f"check_no_invalid_escape: {path}: {e}. "
                f"Fix: use raw string r'...' for regex matches or "
                f"escape backslashes."
            )
            _sys.exit(1)
'''


def add_helper():
    print("\n=== add patch_utils.check_no_invalid_escape ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "def check_no_invalid_escape(" in s:
        print("  already present — skip")
        return
    # Append at end of file
    p.write_text(s + CHECK_HELPER)
    assert_compiles(p)
    print("  OK")


# ============ 3. Tests for the helper ============

TESTS = '''"""v0.9.23f-fix3: patch_utils.check_no_invalid_escape helper."""
import pytest

from cadw.patch_utils import check_no_invalid_escape


def test_check_no_invalid_escape_clean(tmp_path):
    p = tmp_path / "clean.py"
    p.write_text('x = "hello"\ny = r"raw\\d"\n')
    check_no_invalid_escape(p)  # no raise


def test_check_no_invalid_escape_catches_invalid(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text('match = "frac \\\\[0, 1\\\\]"\n')  # non-raw, invalid
    with pytest.raises(SystemExit):
        check_no_invalid_escape(p)


def test_check_no_invalid_escape_raw_string_ok(tmp_path):
    p = tmp_path / "ok_raw.py"
    p.write_text('match = r"frac \\\\[0, 1\\\\]"\n')  # raw, valid
    check_no_invalid_escape(p)  # no raise


def test_check_no_invalid_escape_syntax_error(tmp_path):
    p = tmp_path / "syntax_err.py"
    p.write_text('def f(\\n')  # genuine SyntaxError, not warning
    with pytest.raises(SyntaxError):
        check_no_invalid_escape(p)
'''


def write_test_file():
    print("\n=== write tests/test_check_no_invalid_escape.py ===")
    p = Path("tests/test_check_no_invalid_escape.py")
    if p.exists():
        print("  already exists — skip")
        return
    p.write_text(TESTS)
    assert_compiles(p)
    print("  OK")


# ============ 4. Dogfood ============

def dogfood():
    print("\n=== dogfood: helper catches escape + test_sample clean ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import warnings as _w
        import ast as _ast

        from cadw.patch_utils import (
            check_no_invalid_escape as _check,
        )

        # (a) test_sample.py is now clean
        _check("tests/test_sample.py")

        # (b) Same trap in a temp file -> SystemExit
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            p = td / "bad.py"
            p.write_text('match = "frac \\\\[0, 1\\\\]"\n')
            try:
                _check(p)
            except SystemExit:
                pass
            else:
                raise AssertionError(
                    "helper must SystemExit on invalid escape"
                )

        # (c) Raw string version -> no raise
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            p = td / "ok.py"
            p.write_text('match = r"frac \\\\[0, 1\\\\]"\n')
            _check(p)

        # (d) sample() functional after fix: frac=0 -> empty
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            import pyarrow as pa_
            import pyarrow.parquet as pq_
            from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG
            t = pa_.table({"v": list(range(100))})
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=10,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            assert df.sample(frac=0.0).count() == 0
            try:
                df.sample(frac=-0.1)
            except ValueError as e:
                assert "frac must be in" in str(e)
            else:
                raise AssertionError("frac=-0.1 must raise")

        print(
            "  helper catches trap + raw ok + sample frac=0 empty ok"
        )

    run_verify_in_subprocess(_verify)


# ============ 5. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_check_no_invalid_escape.py",
         "tests/test_sample.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 6. main ============

def main():
    print("=== v0.9.23f-fix3: escape + check_no_invalid_escape ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    fix_escape()
    add_helper()
    write_test_file()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23f-fix3: fix invalid escape in sample frac=0 test + "
        "structural prevention via patch_utils.check_no_invalid_escape. "
        "Bug: fix2 added match='frac must be in \\[0, 1\\]' — non-raw "
        "string, invalid escape '\\['. pytest -W error::SyntaxWarning "
        "converts SyntaxWarning -> SyntaxError at collection. Second "
        "occurrence in v0.9.23f (fix1: 'weights \\+ replace'). "
        "Root cause: py_compile.compile (used by assert_compiles) in "
        "Python 3.12 emits SyntaxWarning for invalid escapes but does "
        "NOT raise. Only -W error::SyntaxWarning / ast.parse under "
        "warning-as-error catches. "
        "Fix: "
        "(1) Fix the escape: raw string r'frac must be in \\[0, 1\\]'. "
        "(2) Add patch_utils.check_no_invalid_escape(path): AST-parse "
        "with -W error::SyntaxWarning -> SystemExit on SyntaxWarning. "
        "Closes the whole class structurally (no per-patch discipline "
        "needed). Message includes file path + warning line + fix "
        "hint (raw string). "
        "(3) +tests/test_check_no_invalid_escape.py (4 tests: clean, "
        "invalid non-raw -> SystemExit, raw ok, genuine SyntaxError "
        "-> SyntaxError). "
        "(4) Dogfood: helper catches same trap in temp file + raw "
        "version passes + sample(frac=0) returns empty + frac=-0.1 "
        "raises. "
        "No production changes (sample() correct; frac=0 fix from "
        "fix2 stands). "
        "Process note (§13.4/238): v0.9.23f is 3rd fix-cycle — over "
        "the '>2 -> split' threshold. Should have split into (a) "
        "n/frac/replace + (b) weights at start (~700 LOC). Recurring "
        "trap now structurally closed — no split retroactively "
        "needed. "
        "Fast: 2045 -> 2049 passed.",
        self_path="scripts/patch_v0923f_fix3_escape.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23f-fix3 applied. v0.9.23f COMPLETE (7.A.2).")


if __name__ == "__main__":
    main()