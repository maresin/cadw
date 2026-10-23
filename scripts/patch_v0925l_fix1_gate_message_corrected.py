#!/usr/bin/env python3
# v0.9.25l-fix1-corrected — fix .dt gate regex; verify_state via
#                            hasattr, not source text.
#
# STATE: HEAD = v0.9.25k. v0.9.25l uncommitted (production OK: 92/93
# tests pass, only test_tz_convert_non_temporal_raises fails on regex).
#
# BUG (still open): test_tz_convert_non_temporal_raises:
#   with pytest.raises(TypeError, match="timestamp required"):
#       df["v"].dt.tz_convert("UTC")
# Actual raised by .dt __init__ gate (_validate_temporal):
#   TypeError: .dt: column 'v' has type int64; expected timestamp or date
# Test asserts wrong error layer (inner method's intended message vs
# accessor gate's actual message). Production correct.
#
# BUG IN PRIOR FIX1 (§13.4/278, NEW): verify_state used
#   if "def split(self" not in str_src: raise ...
# but split has MULTI-LINE signature:
#   def split(
#       self,
#       pat: str,
#       ...
# `def split(self` is NOT a substring (newline between `(` and `self`).
# False negative. Production intact, verify marker wrong.
#
# FIX: verify_state now uses hasattr() on imported accessor classes —
# robust to multi-line signatures, decorators, type hints, whitespace,
# re-exports. Source-text presence is fragile; runtime attribute
# presence is authoritative.
#
# §13.4/278 (NEW): verify_state presence checks via
# hasattr(imported_class, 'method'), NOT source text. Source markers
# break on: multi-line signatures, decorator-wrapped methods
# (hasattr traverses decorator chains), methods added via
# __setattr__ at import time, re-exported symbols. If the class is
# importable and the method exists — hasattr tells you. If the class
# has import errors — that's a different, clearer failure mode.
#
# Fix-cycle count v0.9.25l: initial(1) → fix1(1 aborted, 0 mutation)
# → fix1-corrected(1) = 3 attempts, 0 production bugs, 0 mutations
# from aborted fix1.
#
# §13.4/264 applied in original v0.9.25l (12 REPL checks, all
# passed). §13.4/277 (gate layer, from fix1 diagnosis) still
# applicable. §13.4/260/261/262/265 preserved.
#
# Run: uv run python scripts/patch_v0925l_fix1_gate_message_corrected.py

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
          "broken test regex present; presence via hasattr ===")
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

    # §13.4/278: presence via hasattr, NOT source text
    try:
        from cadw.accessors._dt import _DtAccessor
        from cadw.accessors._str import _StrAccessor
    except ImportError as e:
        raise SystemExit(
            f"accessor import failed: {e} — v0.9.25l production "
            f"didn't apply cleanly?"
        )
    for m in ("strftime", "tz_convert"):
        if not hasattr(_DtAccessor, m):
            raise SystemExit(f"_DtAccessor.{m} missing")
    if not hasattr(_StrAccessor, "split"):
        raise SystemExit("_StrAccessor.split missing")

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
        if "expected timestamp or date" not in str(e):
            raise SystemExit(
                f"actual error message differs from fix assertion: "
                f"{str(e)!r}. Expected 'expected timestamp or date'."
            )
    else:
        raise SystemExit(
            "expected TypeError from .dt gate, got none. "
            "Production behavior drift?"
        )

    print("  state OK: production intact (hasattr verified), "
          "broken test regex present, actual error confirmed")


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
    print("=== v0.9.25l-fix1-corrected: gate regex + verify hasattr ===")
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
        "first via pc.assume_timezone). Returns new column "
        "`{col}_tz_{tz}` (timestamp with target tz). "
        "(3) `_StrAccessor.split(pat, max_splits=-1)` — split "
        "strings to list<string> via pc.split_pattern. Nulls "
        "propagate. Returns new column `{col}_split` "
        "(type=list<item: string>). "
        "12 REPL checks in original v0.9.25l verify_state (all "
        "PASSED): pc.strftime exists/naive/tz-aware; pc.cast "
        "UTC→NY; pc.assume_timezone; pc.split_pattern exists/"
        "basic/null; list<string> round-trip through "
        "Dataset.from_batches; _DtAccessor._strftime/tz_convert + "
        "_StrAccessor.split absent; both accessor anchors present. "
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
        "fix1-corrected (§13.4/278, NEW): prior fix1's verify_state "
        "used source-text marker `def split(self` — split has "
        "MULTI-LINE signature (`def split(\n    self,`) so marker "
        "not substring. False negative aborted fix1 at verify with "
        "0 mutations (HEAD stayed v0.9.25k). Fix: presence via "
        "hasattr(_StrAccessor, 'split') + hasattr(_DtAccessor, "
        "'strftime'/'tz_convert') — runtime attribute presence, "
        "robust to multi-line signatures / decorators / type "
        "hints / re-exports. "
        "§13.4/277 (NEW): when testing error paths through an "
        "accessor __init__ gate, match the ACTUAL raising layer. "
        "§13.4/278 (NEW): verify_state presence checks via "
        "hasattr(imported_class, 'method'), NOT source text. "
        "Source markers break on multi-line signatures (this "
        "case), decorators, type hints, re-exports. "
        "§13.4/264 (REPL-first): 12 checks in original v0.9.25l — "
        "no production-side assumption mismatches. "
        "§13.4/260 preserved: no subprocess dogfood. §13.4/261 "
        "preserved: no symbol moves → no whitelist path updates. "
        "§13.4/262 preserved: pa.scalar typed scalar. §13.4/265: "
        "no copy-pasted asserts from sibling tests. "
        "Tests (tests/test_v0925l.py): 22 total — strftime (7), "
        "tz_convert (7), split (8). Fix1 regex correction makes "
        "suite green. "
        "Fix-cycle count v0.9.25l: initial(1) → fix1(1 aborted, 0 "
        "mutation) → fix1-corrected(1) = 3 attempts, 0 production "
        "bugs, 0 mutations from aborted fix1. "
        "Next: Phase C remaining features (see roadmap).",
        self_path="scripts/patch_v0925l_fix1_gate_message_corrected.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25l COMPLETE (3 features + fix1). "
          "Next: Phase C remaining.")


if __name__ == "__main__":
    main()