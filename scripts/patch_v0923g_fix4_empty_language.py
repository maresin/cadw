#!/usr/bin/env python3
# v0.9.23g-fix4 — exception message language mismatch + commit fix3+fix4.
#
# Bug: value_counts / _nunique_one catch ValueError and check
# `"empty" in str(e).lower()`, but the message from hash_shuffle is
# Russian: "hash_shuffle: пустой вход". So the check returns False
# and the ValueError re-raises for all-null + dropna=True case.
#
# "empty" in "пустой вход" -> False (different language).
#
# Fix: patch_utils.is_empty_input_error(exc) helper — checks both
# English "empty" and Russian "пустой" (case-insensitive). Documented
# at the helper. Use in value_counts and _nunique_one.
#
# Also: fix3 was NOT committed (targeted pytest failed in fix3's
# commit_with_guards — blocked correctly). So this commit includes
# fix3 + fix4 changes together. Commit message reflects both.
#
# Run: uv run python scripts/patch_v0923g_fix4_empty_language.py

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


# ============ 1. patch_utils.is_empty_input_error helper ============

HELPER = '''

# ============ v0.9.23g-fix4: is_empty_input_error helper ============


def is_empty_input_error(exc):
    """True iff `exc` looks like the "empty input" ValueError from
    hash_shuffle / groupby_aggregate.

    v0.9.23g-fix4: prior code checked `"empty" in str(exc).lower()`,
    but hash_shuffle emits Russian "пустой вход" -> check returned
    False -> exception re-raised instead of caught.

    Checks both English "empty" and Russian "пустой" (substring,
    case-insensitive for English; Russian lowercased). Both strings
    are hardcoded in cadw (not user-provided), so language coupling
    is deliberate: if either message is reworded, update this helper.
    """
    msg = str(exc)
    msg_lower = msg.lower()
    return "empty" in msg_lower or "пустой" in msg_lower
'''


def add_helper():
    print("\n=== add patch_utils.is_empty_input_error ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "def is_empty_input_error(" in s:
        print("  already present — skip")
        return
    p.write_text(s + HELPER)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 2. value_counts: use helper ============

def fix_value_counts():
    print("\n=== value_counts: use is_empty_input_error ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "is_empty_input_error(e)" in s and "value_counts.input" in s:
        # Check if the specific value_counts except block is updated
        if "v0.9.23g-fix4: is_empty_input_error" in s:
            print("  already fixed — skip")
            return

    old = (
        '        except ValueError as e:\n'
        '            # groupby_aggregate raises ValueError on empty input. With\n'
        '            # dropna=True and all-null column, filtered is empty ->\n'
        '            # gba sees empty -> ValueError. Return empty table.\n'
        '            if "empty" in str(e).lower():\n'
        '                return _empty_value_counts_table(\n'
        '                    column, col_type, normalize=normalize,\n'
        '                )\n'
        '            raise'
    )
    new = (
        '        except ValueError as e:\n'
        '            # groupby_aggregate raises ValueError on empty input. With\n'
        '            # dropna=True and all-null column, filtered is empty ->\n'
        '            # gba sees empty -> ValueError. Return empty table.\n'
        '            # v0.9.23g-fix4: is_empty_input_error checks both\n'
        '            # languages ("empty" / "пустой") — prior was English-only\n'
        '            # and missed Russian message.\n'
        '            from cadw.patch_utils import is_empty_input_error\n'
        '            if is_empty_input_error(e):\n'
        '                return _empty_value_counts_table(\n'
        '                    column, col_type, normalize=normalize,\n'
        '                )\n'
        '            raise'
    )
    if old not in s:
        raise SystemExit("value_counts except block anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 3. _nunique_one: use helper ============

def fix_nunique_one():
    print("\n=== _nunique_one: use is_empty_input_error ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "_nunique_one" in s and "is_empty_input_error" in s:
        if "v0.9.23g-fix4: is_empty_input_error" in s:
            print("  already fixed — skip")
            return

    old = (
        '            except ValueError as e:\n'
        '                # groupby_aggregate raises on empty input. dropna=True\n'
        '                # + all-null column -> filtered is empty -> 0 uniques.\n'
        '                if "empty" in str(e).lower():\n'
        '                    return 0\n'
        '                raise'
    )
    new = (
        '            except ValueError as e:\n'
        '                # groupby_aggregate raises on empty input. dropna=True\n'
        '                # + all-null column -> filtered is empty -> 0 uniques.\n'
        '                # v0.9.23g-fix4: is_empty_input_error (both languages).\n'
        '                from cadw.patch_utils import is_empty_input_error\n'
        '                if is_empty_input_error(e):\n'
        '                    return 0\n'
        '                raise'
    )
    if old not in s:
        raise SystemExit("_nunique_one except block anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 4. Tests for helper ============

def add_helper_tests():
    print("\n=== add is_empty_input_error tests ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "test_is_empty_input_error" in s:
        print("  already present — skip")
        return

    extra = '''

# ============ v0.9.23g-fix4: is_empty_input_error ============

def test_is_empty_input_error_english():
    from cadw.patch_utils import is_empty_input_error
    exc = ValueError("hash_shuffle: empty input")
    assert is_empty_input_error(exc) is True


def test_is_empty_input_error_russian():
    """Prior bug: 'empty' check missed Russian 'пустой вход'."""
    from cadw.patch_utils import is_empty_input_error
    exc = ValueError("hash_shuffle: пустой вход. ...")
    assert is_empty_input_error(exc) is True


def test_is_empty_input_error_case_insensitive():
    from cadw.patch_utils import is_empty_input_error
    assert is_empty_input_error(ValueError("EMPTY INPUT")) is True


def test_is_empty_input_error_other_error():
    from cadw.patch_utils import is_empty_input_error
    exc = ValueError("some other error")
    assert is_empty_input_error(exc) is False
'''

    p.write_text(s + extra)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 5. Dogfood ============

def dogfood():
    print("\n=== dogfood: all-null + dropna=True works ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG
        from cadw.patch_utils import is_empty_input_error as _ie

        # Helper sanity
        assert _ie(ValueError("hash_shuffle: пустой вход")) is True
        assert _ie(ValueError("empty input")) is True

        # (a) value_counts all-null + dropna=True -> empty table
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "k": pa_.array([None, None], type=pa_.string()),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            out = df.value_counts("k", dropna=True)
            assert out.num_rows == 0
            assert out.schema.names == ["k", "count"]

        # (b) nunique all-null + dropna=True -> 0
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "k": pa_.array([None, None], type=pa_.string()),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            assert df.nunique("k", dropna=True) == 0

        # (c) Sanity: normal cases still work
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "k": pa_.array(
                    ["a", None, "a", None, "b"], type=pa_.string(),
                ),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            out = df.value_counts("k")
            rows = {(r["k"], r["count"]) for r in out.to_pylist()}
            assert (None, 2) in rows
            assert df.nunique("k") == 3
            assert df.nunique("k", dropna=True) == 2

        print("  all-null+dropna all-null cases + sanity ok")

    run_verify_in_subprocess(_verify)


# ============ 6. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_eda_basics.py",
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


# ============ 7. main ============

def main():
    print("=== v0.9.23g-fix4: empty input language + commit fix3+fix4 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    add_helper()
    fix_value_counts()
    fix_nunique_one()
    add_helper_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23g-fix3+fix4: value_counts/nunique edge cases + empty-"
        "input language + idempotency-by-def. "
        "fix3 (uncommitted — targeted pytest gate blocked commit) "
        "adds _empty_value_counts_table helper (real) + "
        "patch_utils.idempotency_check_by_def (def-line vs bare-name "
        "idempotency guard — §13.4/182). fix2's add_empty_helper had "
        "the trap: `if \"_empty_value_counts_table\" in s` — the name "
        "already appeared in value_counts body (call site) -> guard "
        "fired -> helper never added -> dogfood NameError. "
        "fix4 (this patch, fixes the pytest-gate failure): "
        "value_counts / _nunique_one caught ValueError and checked "
        "`\"empty\" in str(e).lower()`, but hash_shuffle emits "
        "Russian 'пустой вход' — English check returned False -> "
        "ValueError re-raised for all-null + dropna=True case. "
        "Fix: patch_utils.is_empty_input_error(exc) checks both "
        "English 'empty' and Russian 'пустой' (deliberate coupling — "
        "both messages hardcoded in cadw; if reworded, update helper). "
        "+tests/test_eda_basics.py: 2 existing regression tests now "
        "pass (value_counts/nunique all-null dropna=True) from fix2's "
        "regression set. "
        "+tests/test_patch_utils.py: 4 new tests for "
        "is_empty_input_error (English, Russian, case-insensitive, "
        "negative case) from fix3+fix4. "
        "Process note: 4th fix-cycle on v0.9.23g (over threshold). "
        "Classes now structurally closed: "
        "  - multi-batch tests via chunk=1 (fix1); "
        "  - set_column vs set_column+.drop (fix2); "
        "  - idempotency_check_by_def (def-line, not name) (fix3); "
        "  - is_empty_input_error (both languages) (fix4). "
        "Fast: ~2090 -> ~2098 passed.",
        self_path="scripts/patch_v0923g_fix4_empty_language.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23g-fix4 applied. v0.9.23g + fix3 + fix4 all committed.")


if __name__ == "__main__":
    main()