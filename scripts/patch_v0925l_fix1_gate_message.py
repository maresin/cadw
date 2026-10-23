#!/usr/bin/env python3
# v0.9.25l-fix1 — test-only: .dt gate message, not tz_convert message.
#
# STATE: HEAD = v0.9.25k. v0.9.25l uncommitted (production applied:
# _DtAccessor.strftime + tz_convert, _StrAccessor.split; 92 tests
# pass, 1 fails in test_v0925l.py).
#
# BUG: test_tz_convert_non_temporal_raises:
#   with pytest.raises(TypeError, match="timestamp required"):
#       df["v"].dt.tz_convert("UTC")
#
# Actual raised:
#   TypeError: .dt: column 'v' has type int64; expected timestamp or date
# raised by _dt.py::_validate_temporal at __init__, BEFORE tz_convert
# body runs.
#
# Test assumed tz_convert's own validation message ("timestamp
# required"), but .dt accessor's __init__ gate hits first with a
# different message. Production CORRECT — gate at __init__ is the
# right place (fail-fast, prevents accidental non-temporal .dt
# method calls). Test asserted wrong error layer.
#
# Fix scope: 1 line in tests/test_v0925l.py. Change regex
# "timestamp required" → "expected timestamp or date". Test-only.
#
# §13.4/277 (NEW): when testing error paths that pass through an
# accessor __init__ gate, match the ACTUAL raising layer. .dt
# validation gate at __init__ (from v0.9.4b) fires before tz_convert
# body — its message "expected timestamp or date" is authoritative
# for public API users. Writing test with the inner method's
# intended message (e.g. "timestamp required") skips the gate in
# the mental model.
#
# Fix-cycle count v0.9.25l: initial(1) → fix1(1) = 2 attempts,
# 0 production bugs. 92 of 93 passed first attempt — strftime
# (naive + tz-aware), tz_convert, split all validated by 22 tests.
# 1 failure = test-overreach, not production defect.
#
# §13.4/264 (REPL-first) applied during initial v0.9.25l (12 REPL
# checks: pc.strftime exists, naive+tz-aware, cast UTC→NY for
# tz_convert, assume_timezone, split_pattern basic+null,
# list<string> round-trip through Dataset, anchors). No production
# assumption mismatches. Test-overreach was NOT a PyArrow behavior
# assumption but a code-layering assumption (which layer raises) —
# REPL couldn't catch it; static reasoning through the raising
# function chain could have.
#
# §13.4/260 preserved: no subprocess dogfood. §13.4/261 preserved:
# no symbol moves → no whitelist path updates. §13.4/262 preserved:
# pa.scalar typed scalar. §13.4/265: no copy-pasted asserts from
# sibling tests.
#
# Run: uv run python scripts/patch_v0925l_fix1_gate_message.py

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    check_no_invalid_escape,
)


TESTS_PATH = "tests/test_v0925l.py"


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25k; v0.9.25l uncommitted; "
          "broken test regex present ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25k" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25k; got {log[:200]!r}"
        )
    if "v0.9.25l" in log:
        raise SystemExit("v0.9.25l already committed — fix1 misplaced")

    src = Path(TESTS_PATH).read_text()
    if "def test_tz_convert_non_temporal_raises" not in src:
        raise SystemExit(
            "test_tz_convert_non_temporal_raises not found — "
            "v0.9.25l didn't apply?"
        )
    if 'match="timestamp required"' not in src:
        raise SystemExit(
            'expected match="timestamp required" (bug marker); '
            "already fixed?"
        )

    # Production methods present
    dt_src = Path("cadw/accessors/_dt.py").read_text()
    for marker in ("def strftime(self", "def tz_convert(self"):
        if marker not in dt_src:
            raise SystemExit(
                f"production marker {marker!r} missing in _dt.py — "
                f"v0.9.25l production didn't apply?"
            )
    str_src = Path("cadw/accessors/_str.py").read_text()
    if "def split(self" not in str_src:
        raise SystemExit("_StrAccessor.split missing in _str.py")

    # Verify actual error message matches what fix asserts
    import pyarrow as pa
    from cadw import CadwDataFrame, MemoryGuard
    df = CadwDataFrame.from_table(
        pa.table({"v": [1]}),
        memory_guard=MemoryGuard(max_memory_mb=20),
    )
    try:
        df["v"].dt.tz_convert("UTC")
    except TypeError as e:
        msg = str(e)
        if "expected timestamp or date" not in msg:
            raise SystemExit(
                f"actual error message differs from fix assertion: "
                f"{msg!r}. Expected 'expected timestamp or date' — "
                f"would need another look."
            )
    else:
        raise SystemExit(
            "expected TypeError from .dt gate, got none. "
            "Production behavior drift?"
        )

    print("  state OK: production intact, test regex broken, "
          "actual error verified")


# ============ 1. fix broken test ============

OLD_TEST = '''def test_tz_convert_non_temporal_raises():
    df = _mk({"v": [1]})
    with pytest.raises(TypeError, match="timestamp required"):
        df["v"].dt.tz_convert("UTC")'''


NEW_TEST = '''def test_tz_convert_non_temporal_raises():
    """v0.9.25l-fix1 (§13.4/277): .dt accessor gate at __init__
    rejects non-temporal columns BEFORE tz_convert body runs. The
    gate's message ('expected timestamp or date') is authoritative,
    not tz_convert's own intended message ('timestamp required').
    """
    df = _mk({"v": [1]})
    with pytest.raises(TypeError, match="expected timestamp or date"):
        df["v"].dt.tz_convert("UTC")'''


def fix_test():
    print("\n=== fix tz_convert_non_temporal_raises regex ===")
    applied = apply_or_fail(
        TESTS_PATH, OLD_TEST, NEW_TEST,
        label="tz_convert_non_temporal_regex",
    )
    if applied:
        assert_compiles(TESTS_PATH)
        check_no_invalid_escape(TESTS_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. pytest ============

def run_pytest():
    print("\n=== pre-flight: facade AST guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py",
         "tests/test_dataframe.py::test_facade_has_no_row_loops",
         "tests/test_dataframe_filters.py::test_facade_has_no_as_py_ast",
         "tests/test_dataframe_filters.py::test_facade_has_no_row_loop_ast",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** FACADE AST GUARD FAILED — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: v0925l + dt/str accessor suites ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925l.py",
         "tests/test_dt_accessor.py",
         "tests/test_str_accessor.py",
         "tests/test_v0925k.py",
         "tests/test_dataframe.py",
         "tests/test_dataframe_filters.py",
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


# ============ 3. main ============

def main():
    print("=== v0.9.25l-fix1: fix .dt gate error regex ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    fix_test()
    run_pytest()

    commit_with_guards(
        "v0.9.25l: Phase C batch C-4b — dt.strftime + dt.tz_convert "
        "+ str.split + fix1 (gate message regex). "
        "3 of 20 Pandas P0 features. All pure compositions of "
        "existing accessors + Arrow kernels. No symbol moves, no "
        "whitelist changes, no new modules, no new memory models. "
        "Features: "
        "(1) `_DtAccessor.strftime(fmt)` — format timestamps to "
        "strings via pc.strftime. Naive + tz-aware timestamps. "
        "Returns new column `{col}_strftime` (string). "
        "(2) `_DtAccessor.tz_convert(tz)` — convert tz-aware "
        "timestamps between timezones via pc.cast (naive → UTC "
        "first via pc.assume_timezone; see docstring). Returns "
        "new column `{col}_tz_{tz}` (timestamp with target tz). "
        "(3) `_StrAccessor.split(pat, max_splits=-1)` — split "
        "strings to list<string> via pc.split_pattern. Nulls "
        "propagate. Returns new column `{col}_split` "
        "(type=list<item: string>). "
        "12 REPL checks in verify_state (pc.strftime exists/naive/"
        "tz-aware; pc.cast UTC→NY; pc.assume_timezone; "
        "pc.split_pattern exists/basic/null; list<string> round-trip "
        "through Dataset.from_batches; _DtAccessor._strftime/"
        "tz_convert + _StrAccessor.split absent; both accessor "
        "anchors present). All PASSED. "
        "fix1: test_tz_convert_non_temporal_raises asserted wrong "
        "error layer. Test expected tz_convert's own 'timestamp "
        "required' message, but .dt accessor __init__ gate "
        "(_validate_temporal in _dt.py) fires FIRST with "
        "'.dt: column 'v' has type int64; expected timestamp or "
        "date'. Production CORRECT — gate at __init__ is right "
        "place (fail-fast, prevents accidental non-temporal .dt "
        "calls). Test regex updated: 'timestamp required' → "
        "'expected timestamp or date'. Test-only fix. 92 of 93 "
        "tests passed first attempt — all 3 features validated; "
        "only test-overreach failed. "
        "§13.4/277 (NEW): when testing error paths through an "
        "accessor __init__ gate, match the ACTUAL raising layer. "
        ".dt gate at __init__ fires before method bodies — its "
        "message is authoritative for public API users. Writing "
        "test with inner method's intended message skips the gate "
        "in the mental model. Static reasoning through the raising "
        "function chain (accessor __init__ → _validate_temporal) "
        "would have caught it where REPL could not (no PyArrow "
        "behavior assumption involved — the mismatch was "
        "code-layering, not library behavior). "
        "§13.4/264 (REPL-first): no production-side assumption "
        "mismatches. Test-overreach was NOT a PyArrow behavior "
        "assumption. "
        "§13.4/260 preserved: no subprocess dogfood. §13.4/261 "
        "preserved: no symbol moves → no whitelist path updates. "
        "§13.4/262 preserved: pa.scalar typed scalar. §13.4/265: "
        "no copy-pasted asserts from sibling tests (checked "
        "test_dt_accessor.py — no tz_convert gate test to copy "
        "from; fix is genuinely new pattern). "
        "Tests (tests/test_v0925l.py): 22 total — strftime (7), "
        "tz_convert (7), split (8). Plus fix1 regex correction. "
        "Fast suite: previous → +22 (net from v0.9.25l) with fix1 "
        "bringing full suite to green. "
        "Next: Phase C remaining features (see roadmap).",
        self_path="scripts/patch_v0925l_fix1_gate_message.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25l COMPLETE (3 features + fix1). "
          "Next: Phase C remaining.")


if __name__ == "__main__":
    main()