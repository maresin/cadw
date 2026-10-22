#!/usr/bin/env python3
# v0.9.23f-fix5 — Python 3.12 quirk: ast.parse under -W error::
# SyntaxWarning raises SyntaxError (not SyntaxWarning) for invalid
# escapes. The helper caught only SyntaxWarning -> SyntaxError
# propagated -> dogfood failed.
#
# Fix: except (SyntaxWarning, SyntaxError), check message for
# "invalid escape sequence". Genuine SyntaxErrors re-raise (test
# test_check_no_invalid_escape_syntax_error asserts this).
#
# This is 5th fix-cycle on v0.9.23f (over §13.4/238 threshold of 2).
# Structural fixes now in place:
#   - write_content_checked calls check_no_invalid_escape
#   - check_no_invalid_escape handles both warning types
#   - all test files written via BS = chr(92) pattern
# v0.9.23g+ should not hit this class.
#
# Run: uv run python scripts/patch_v0923f_fix5_syntax_error_quirk.py

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


# ============ 1. Fix check_no_invalid_escape exception types ============

def fix_check_helper():
    print("\n=== fix check_no_invalid_escape: catch SyntaxError too ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "v0.9.23f-fix5: Python 3.12 ast.parse" in s:
        print("  already fixed — skip")
        return

    old = (
        '    src = _P(path).read_text()\n'
        '    with _w.catch_warnings():\n'
        '        _w.simplefilter("error", SyntaxWarning)\n'
        '        try:\n'
        '            _ast.parse(src, filename=str(path))\n'
        '        except SyntaxWarning as e:\n'
        '            print(\n'
        '                f"check_no_invalid_escape: {path}: {e}. "\n'
        '                f"Fix: use raw string r\'...\' for regex matches or "\n'
        '                f"escape backslashes."\n'
        '            )\n'
        '            _sys.exit(1)\n'
    )

    new = (
        '    src = _P(path).read_text()\n'
        '    with _w.catch_warnings():\n'
        '        _w.simplefilter("error", SyntaxWarning)\n'
        '        try:\n'
        '            _ast.parse(src, filename=str(path))\n'
        '        except (SyntaxWarning, SyntaxError) as e:\n'
        '            # v0.9.23f-fix5: Python 3.12 quirk — ast.parse with\n'
        '            # warnings.simplefilter("error", SyntaxWarning) raises\n'
        '            # SyntaxError (NOT SyntaxWarning) for invalid escapes.\n'
        '            # The C compiler elevates the warning directly to\n'
        '            # SyntaxError with message "invalid escape sequence".\n'
        '            # Genuine syntax errors (e.g. "def f(") also raise\n'
        '            # SyntaxError — distinguish by message.\n'
        '            msg = str(e)\n'
        '            is_invalid_escape = (\n'
        '                isinstance(e, SyntaxWarning)\n'
        '                or "invalid escape sequence" in msg\n'
        '            )\n'
        '            if is_invalid_escape:\n'
        '                print(\n'
        '                    f"check_no_invalid_escape: {path}: {e}. "\n'
        '                    f"Fix: use raw string r\'...\' for regex "\n'
        '                    f"matches or escape backslashes."\n'
        '                )\n'
        '                _sys.exit(1)\n'
        '            # Genuine syntax error — re-raise (preserves\n'
        '            # test_check_no_invalid_escape_syntax_error contract).\n'
        '            raise\n'
    )

    if old not in s:
        raise SystemExit(
            "check_no_invalid_escape body anchor not found — source "
            "may have drifted"
        )
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    # Self-check: helper file itself has no invalid escapes
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 2. Add explicit warning-vs-error test ============

def add_quirk_test():
    print("\n=== add test: invalid escape -> SystemExit, not SyntaxError ===")
    p = Path("tests/test_check_no_invalid_escape.py")
    s = p.read_text()
    if "test_check_no_invalid_escape_quirk_py312" in s:
        print("  already present — skip")
        return

    extra = '''

def test_check_no_invalid_escape_quirk_py312(tmp_path):
    """v0.9.23f-fix5: invalid escape -> SystemExit, NOT SyntaxError.

    Python 3.12 quirk: ast.parse under warnings.simplefilter("error",
    SyntaxWarning) raises SyntaxError (not SyntaxWarning) for invalid
    escapes. Helper must catch both and map invalid-escape case to
    SystemExit (like the old SyntaxWarning path).
    """
    p = tmp_path / "bad_escape.py"
    p.write_text('x = "' + BS + 'd"' + NL)
    try:
        check_no_invalid_escape(p)
    except SystemExit:
        pass  # expected
    except SyntaxError:
        pytest.fail(
            "check_no_invalid_escape must SystemExit (not SyntaxError) "
            "for invalid escape; Python 3.12 raises SyntaxError from "
            "ast.parse under -W error::SyntaxWarning"
        )
'''

    p.write_text(s + extra)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 3. Dogfood ============

def dogfood():
    print("\n=== dogfood: invalid escape -> SystemExit, genuine syntax -> SyntaxError ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        from cadw.patch_utils import check_no_invalid_escape as _check

        BS = chr(92)
        NL = chr(10)

        # (a) invalid escape -> SystemExit (NOT SyntaxError)
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            bad = td / "bad.py"
            bad.write_text('x = "' + BS + 'd"' + NL)
            try:
                _check(bad)
            except SystemExit:
                pass  # expected (v0.9.23f-fix5)
            except SyntaxError as e:
                raise AssertionError(
                    f"check_no_invalid_escape must SystemExit for "
                    f"invalid escape, but raised SyntaxError: {e}"
                )
            else:
                raise AssertionError("must raise SystemExit")

        # (b) genuine syntax error -> SyntaxError (re-raise)
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            syntax_err = td / "syntax_err.py"
            syntax_err.write_text("def f(" + NL)
            try:
                _check(syntax_err)
            except SystemExit:
                raise AssertionError(
                    "genuine syntax error must re-raise SyntaxError, "
                    "not SystemExit"
                )
            except SyntaxError:
                pass  # expected

        # (c) raw string with backslash-d -> OK
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            ok = td / "ok.py"
            ok.write_text('x = r"' + BS + 'd"' + NL)
            _check(ok)

        # (d) 3 real files clean
        _check("cadw/patch_utils.py")
        _check("tests/test_sample.py")
        _check("tests/test_check_no_invalid_escape.py")

        # (e) write_content_checked now enforces for .py
        # (attempt writing invalid escape -> SystemExit at write time)
        from cadw.patch_utils import write_content_checked as _wcc
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            target = td / "test_bad.py"
            try:
                _wcc(target, ['x = "' + BS + 'd"'])
            except SystemExit:
                pass
            else:
                raise AssertionError(
                    "write_content_checked must catch invalid escape "
                    "in .py files at write time"
                )

        print(
            "  invalid->SystemExit + genuine->SyntaxError + raw ok + "
            "3 files clean + write_content_checked enforces"
        )

    run_verify_in_subprocess(_verify)


# ============ 4. Tests ============

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


# ============ 5. main ============

def main():
    print("=== v0.9.23f-fix5: Python 3.12 SyntaxError quirk ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    fix_check_helper()
    add_quirk_test()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23f-fix5: Python 3.12 SyntaxError quirk in "
        "check_no_invalid_escape. "
        "Bug (dogfood-caught): ast.parse under "
        "warnings.simplefilter('error', SyntaxWarning) raises "
        "SyntaxError (NOT SyntaxWarning) for invalid escapes in "
        "Python 3.12 — the C compiler elevates the warning directly "
        "to SyntaxError with message 'invalid escape sequence'. "
        "Helper caught only SyntaxWarning -> SyntaxError propagated "
        "-> dogfood fail (and would mask real fix). "
        "Fix: except (SyntaxWarning, SyntaxError), check message for "
        "'invalid escape sequence'. Genuine syntax errors (e.g. "
        "'def f(') re-raise SyntaxError (preserves "
        "test_check_no_invalid_escape_syntax_error contract). "
        "+1 test test_check_no_invalid_escape_quirk_py312 (invalid "
        "escape -> SystemExit not SyntaxError). "
        "Dogfood: invalid->SystemExit + genuine->SyntaxError + raw "
        "ok + 3 real files clean + write_content_checked enforces "
        "at write time. "
        "Process note (§13.4/238): 5th fix-cycle on v0.9.23f. "
        "Way over the '>2 -> split' threshold. Recurring trap "
        "classes now all structurally closed: "
        "  - write_content_checked auto-calls check_no_invalid_escape "
        "for .py files; "
        "  - check_no_invalid_escape handles both warning types; "
        "  - test files use BS=chr(92) pattern (no literal \\); "
        "  - _write_run schema= explicit; "
        "  - AST helpers (assert_no_call_in_source, "
        "assert_call_order) for call-presence/order. "
        "v0.9.23g+ should not hit this class. "
        "Fast: ~2045 -> ~2046 passed (+1 explicit test).",
        self_path="scripts/patch_v0923f_fix5_syntax_error_quirk.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23f-fix5 applied. v0.9.23f COMPLETE.")


if __name__ == "__main__":
    main()