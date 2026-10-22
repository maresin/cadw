"""v0.9.23f-fix4: check_no_invalid_escape helper tests.

Uses chr(92) (single backslash) and chr(10) (newline) as constants
to avoid escape ambiguity in this file. Prior version failed with
SyntaxError: unterminated string literal because \n in the test
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
