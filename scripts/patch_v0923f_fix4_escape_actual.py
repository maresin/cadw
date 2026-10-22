#!/usr/bin/env python3
# v0.9.23f-fix4 — complete fix3 (partial-fail recovery).
#
# fix3 partially applied:
#   - tests/test_sample.py: escape fixed (OK, unchanged in fix4)
#   - cadw/patch_utils.py: check_no_invalid_escape added, but docstring
#     contains "\d" (invalid escape, 3rd occurrence in v0.9.23f)
#   - tests/test_check_no_invalid_escape.py: NOT written (compile fail)
#
# Fix4 does:
#   (1) Reword patch_utils.py docstring: "\d" -> "backslash-d"
#       (no backslash in that phrase, no escape to be invalid)
#   (2) Rewrite test_check_no_invalid_escape.py using chr(92)
#       (backslash) and chr(10) (newline) to avoid escape
#       ambiguity in the test file itself. Target files written via
#       path.write_text use constructed strings, not raw backslashes.
#   (3) Add check_no_invalid_escape call to write_test_file
#       (structural — py_compile.compile in Python 3.12 does NOT
#       raise on SyntaxWarning, so assert_compiles alone is
#       insufficient).
#   (4) Dogfood: verify patch_utils.py, test_sample.py,
#       test_check_no_invalid_escape.py all clean; helper catches
#       invalid escape; raw string passes.
#
# Run: uv run python scripts/patch_v0923f_fix4_escape_actual.py

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


# ============ 1. Reword patch_utils.py docstring ============

def fix_patch_utils_docstring():
    print("\n=== fix patch_utils.py docstring ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "backslash-d in non-raw" in s:
        print("  already fixed — skip")
        return
    # Reword: replace the illegal-escape example with plain phrase.
    apply_or_fail(
        str(p),
        old='("\\d" in non-raw string, etc.) but does NOT raise.',
        new='(backslash-d in non-raw string, etc.) but does NOT raise.',
        label="patch_utils_docstring",
    )
    assert_compiles(p)
    # Structural check: this file must now be free of invalid escapes
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 2. Rewrite test file with chr(92) ============

TESTS = '''"""v0.9.23f-fix4: check_no_invalid_escape helper tests.

Uses chr(92) (single backslash) and chr(10) (newline) as constants
to avoid escape ambiguity in this file. Prior version failed with
SyntaxError: unterminated string literal because \\n in the test
source became a real newline inside a string literal (via triple-
quote processing in the patch script).
"""
import pytest

from cadw.patch_utils import check_no_invalid_escape


BS = chr(92)  # single backslash character
NL = chr(10)  # newline character


def test_check_no_invalid_escape_clean(tmp_path):
    p = tmp_path / "clean.py"
    p.write_text("x = 1" + NL)
    check_no_invalid_escape(p)


def test_check_no_invalid_escape_catches_invalid(tmp_path):
    p = tmp_path / "bad.py"
    # Invalid escape: backslash + d in non-raw string
    p.write_text('match = "' + BS + 'd"' + NL)
    with pytest.raises(SystemExit):
        check_no_invalid_escape(p)


def test_check_no_invalid_escape_raw_string_ok(tmp_path):
    p = tmp_path / "ok_raw.py"
    # Raw string: backslash + d is literal, valid
    p.write_text('match = r"' + BS + 'd"' + NL)
    check_no_invalid_escape(p)


def test_check_no_invalid_escape_escaped_ok(tmp_path):
    p = tmp_path / "ok_escaped.py"
    # Escaped backslash: BS + BS in source = valid (parses as single)
    p.write_text('match = "' + BS + BS + 'd"' + NL)
    check_no_invalid_escape(p)


def test_check_no_invalid_escape_syntax_error(tmp_path):
    p = tmp_path / "syntax_err.py"
    p.write_text("def f(")
    with pytest.raises(SyntaxError):
        check_no_invalid_escape(p)
'''


def write_test_file():
    print("\n=== rewrite tests/test_check_no_invalid_escape.py ===")
    p = Path("tests/test_check_no_invalid_escape.py")
    p.write_text(TESTS)
    assert_compiles(p)
    # Structural check: no invalid escape in this file
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 3. Structural: check_no_invalid_escape in write_test_file ============

def fix_patch_utils_write_test_helpers():
    print("\n=== add check_no_invalid_escape to write_content_checked ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "def write_content_checked(path, lines):\n    \"\"\"write_content + assert_compiles + check_no_invalid_escape" in s:
        print("  already fixed — skip")
        return
    apply_or_fail(
        str(p),
        old=(
            'def write_content_checked(path, lines):\n'
            '    """write_content + assert_compiles (§13.4/193).\n'
            '\n'
            '    Use for any file write where syntax errors from escape/quote\n'
            '    traps (multi-line templates, embedded backslashes) could slip\n'
            '    through. assert_compiles raises SystemExit on syntax error\n'
            '    BEFORE the file is imported or run by anything else.\n'
            '    """\n'
            '    write_content(path, lines)\n'
            '    assert_compiles(path)'
        ),
        new=(
            'def write_content_checked(path, lines):\n'
            '    """write_content + assert_compiles + check_no_invalid_escape.\n'
            '\n'
            '    Use for any file write where syntax errors from escape/quote\n'
            '    traps (multi-line templates, embedded backslashes) could slip\n'
            '    through. assert_compiles raises SystemExit on syntax error\n'
            '    BEFORE the file is imported or run by anything else.\n'
            '\n'
            '    v0.9.23f-fix4: also check_no_invalid_escape — py_compile\n'
            '    (used by assert_compiles) in Python 3.12 emits\n'
            '    SyntaxWarning for invalid escapes but does NOT raise.\n'
            '    Only pytest with -W error::SyntaxWarning catches.\n'
            '    check_no_invalid_escape makes this a hard failure at\n'
            '    write time.\n'
            '    """\n'
            '    write_content(path, lines)\n'
            '    assert_compiles(path)\n'
            '    if str(path).endswith(".py"):\n'
            '        check_no_invalid_escape(path)'
        ),
        label="write_content_checked_extension",
    )
    assert_compiles(p)
    print("  OK")


# ============ 4. Dogfood ============

def dogfood():
    print("\n=== dogfood: 3 real files clean + helper catches trap ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        from cadw.patch_utils import check_no_invalid_escape as _check

        # 3 real files clean
        _check("cadw/patch_utils.py")
        _check("tests/test_sample.py")
        _check("tests/test_check_no_invalid_escape.py")

        # Helper catches invalid escape in temp file
        BS = chr(92)
        NL = chr(10)
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            bad = td / "bad.py"
            bad.write_text('x = "' + BS + 'd"' + NL)
            try:
                _check(bad)
            except SystemExit:
                pass
            else:
                raise AssertionError(
                    "helper must SystemExit on invalid escape"
                )

        # Raw string version passes
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            ok = td / "ok.py"
            ok.write_text('x = r"' + BS + 'd"' + NL)
            _check(ok)

        # sample() functional post-fix
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
            "  helper works + 3 files clean + sample frac=0/negative ok"
        )

    run_verify_in_subprocess(_verify)


# ============ 5. Tests ============

def run_pytest():
    print("\n=== pytest: targeted (with -W error::SyntaxWarning) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_check_no_invalid_escape.py",
         "tests/test_sample.py",
         "tests/test_patch_utils.py",
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
    print("=== v0.9.23f-fix4: escape (actual) ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    fix_patch_utils_docstring()
    write_test_file()
    fix_patch_utils_write_test_helpers()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23f-fix4: complete fix3 (partial-fail recovery + "
        "structural prevention). "
        "fix3 partially applied: test_sample.py escape fixed; then "
        "patch_utils.py warning + test file compile-fail stopped it. "
        "This patch: "
        "(1) Reword patch_utils.py docstring: '\\d' -> 'backslash-d' "
        "(invalid escape in the helper's OWN docstring — 3rd "
        "occurrence of the class in v0.9.23f: fix1 '\\+', fix2 "
        "'\\[', fix3 '\\d'). "
        "(2) Rewrite tests/test_check_no_invalid_escape.py with "
        "chr(92) (backslash) + chr(10) (newline) constants instead "
        "of literal escapes. Prior version failed compile with "
        "'unterminated string literal' because \\n in TESTS became "
        "real newline inside a string literal (triple-quote "
        "processing in patch script split it). "
        "(3) Structural: write_content_checked() now calls "
        "check_no_invalid_escape() for .py files after assert_compiles. "
        "py_compile.compile in Python 3.12 does NOT raise on "
        "SyntaxWarning (invalid escapes) — only pytest with -W "
        "error::SyntaxWarning catches. This makes the trap a hard "
        "failure at write time. "
        "(4) Dogfood: 3 real files clean (patch_utils, test_sample, "
        "test_check_no_invalid_escape) + helper catches trap in "
        "temp file + raw string passes + sample frac=0/negative. "
        "Process note (§13.4/238): v0.9.23f is 4th fix-cycle. "
        "Recurring trap class now structurally closed (write_content_"
        "checked + check_no_invalid_escape). No split retroactively "
        "needed — but v0.9.23g+ will use write_content_checked from "
        "the start for any test file with strings. "
        "Fast: 2045 -> ~2055 passed.",
        self_path="scripts/patch_v0923f_fix4_escape_actual.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23f-fix4 applied. v0.9.23f COMPLETE (7.A.2).")


if __name__ == "__main__":
    main()