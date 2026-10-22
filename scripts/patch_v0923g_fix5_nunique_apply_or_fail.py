#!/usr/bin/env python3
# v0.9.23g-fix5 — _nunique_one not fixed (custom idempotency guard
# false-positive, §13.4/182 3rd occurrence in v0.9.23g).
#
# fix4's fix_nunique_one had:
#   if "v0.9.23g-fix4: is_empty_input_error" in s: skip
# But fix_value_counts (ran first) already inserted that marker in
# value_counts. Whole-file substring found it -> _nunique_one never
# updated -> all-null + dropna=True still raises ValueError.
#
# Root cause: my custom guard redundant over apply_or_fail. apply_or_
# fail is ALREADY idempotent (returns False if new in s, scoped to
# exact old/new block). No custom guard needed.
#
# Fix:
#   (1) apply_or_fail directly for _nunique_one (no custom guard).
#   (2) AST tests: both value_counts AND nunique reference
#       is_empty_input_error (would've caught this).
#   (3) apply_or_fail docstring note: do NOT wrap in `if marker in s:
#       skip` — whole-file substring traps (§13.4/182, 3rd occurrence
#       in v0.9.23g).
#   (4) Commit fix3+fix4+fix5 together (all uncommitted due to
#       fix4's dogfood failure).
#
# Run: uv run python scripts/patch_v0923g_fix5_nunique_apply_or_fail.py

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


# ============ 1. _nunique_one: apply_or_fail (no custom guard) ============

OLD = '''            except ValueError as e:
                # groupby_aggregate raises on empty input. dropna=True
                # + all-null column -> filtered is empty -> 0 uniques.
                if "empty" in str(e).lower():
                    return 0
                raise'''

NEW = '''            except ValueError as e:
                # groupby_aggregate raises on empty input. dropna=True
                # + all-null column -> filtered is empty -> 0 uniques.
                # v0.9.23g-fix5: is_empty_input_error (both languages).
                from cadw.patch_utils import is_empty_input_error
                if is_empty_input_error(e):
                    return 0
                raise'''


def fix_nunique_one():
    print("\n=== _nunique_one: apply_or_fail (no custom guard) ===")
    p = Path("cadw/dataframe.py")
    # apply_or_fail IS the idempotency guard. No wrapping needed.
    applied = apply_or_fail(
        str(p), OLD, NEW, label="_nunique_one except block",
    )
    if applied:
        assert_compiles(p)
        print("  applied")
    else:
        print("  already fixed — skip (apply_or_fail returned False)")


# ============ 2. apply_or_fail docstring note ============

def add_apply_or_fail_note():
    print("\n=== apply_or_fail docstring note (no custom guards) ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "v0.9.23g-fix5: do NOT wrap apply_or_fail" in s:
        print("  already present — skip")
        return
    apply_or_fail(
        str(p),
        old=(
            '    Fail-fast (SystemExit) if `old` anchor missing — no silent no-op.\n'
            '    Prints `context` lines around partial-match zone on miss (§13.4/160).\n'
            '    Idempotent: returns False if `new` already present (no write).\n'
        ),
        new=(
            '    Fail-fast (SystemExit) if `old` anchor missing — no silent no-op.\n'
            '    Prints `context` lines around partial-match zone on miss (§13.4/160).\n'
            '    Idempotent: returns False if `new` already present (no write).\n'
            '\n'
            '    v0.9.23g-fix5: do NOT wrap apply_or_fail in\n'
            '    `if "<marker>" in src: return` — whole-file substring\n'
            '    idempotency is broken (§13.4/182, 3rd occurrence in\n'
            '    v0.9.23g: value_counts marker bled into _nunique_one\n'
            '    check). apply_or_fail already scopes to the exact old/new\n'
            '    block; wrapping in custom guards is redundant + bug-prone.\n'
            '    Just call apply_or_fail directly; use its return value\n'
            '    (True=applied, False=already-present) as the signal.\n'
        ),
        label="apply_or_fail_docstring",
    )
    assert_compiles(p)
    print("  OK")


# ============ 3. AST tests ============

def add_ast_tests():
    print("\n=== AST tests: both methods use is_empty_input_error ===")
    p = Path("tests/test_eda_basics.py")
    s = p.read_text()
    if "test_eda_value_counts_uses_is_empty_input_error" in s:
        print("  already present — skip")
        return

    extra = '''

# ============ v0.9.23g-fix5: is_empty_input_error AST guards ============

def test_eda_value_counts_uses_is_empty_input_error():
    """value_counts catches empty-input via helper (both languages).

    Prior bug: `"empty" in str(e).lower()` missed Russian
    "пустой вход" -> ValueError re-raised on all-null + dropna=True.
    """
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.value_counts)
    assert "is_empty_input_error" in src, (
        "value_counts must use patch_utils.is_empty_input_error "
        "for empty-input detection (English+Russian)"
    )


def test_eda_nunique_uses_is_empty_input_error():
    """nunique (_nunique_one) uses helper (would've caught fix4 bug).

    fix4's custom guard false-positived (marker in value_counts bled
    into _nunique_one check). Fix5 fixed it; this test prevents
    regression.
    """
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.nunique)
    assert "is_empty_input_error" in src, (
        "nunique must use patch_utils.is_empty_input_error "
        "inside _nunique_one (fix5 bug: custom guard skipped the "
        "actual fix)"
    )
'''

    p.write_text(s + extra)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 4. Dogfood ============

def dogfood():
    print("\n=== dogfood: all-null+dropna now works for BOTH ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG

        # (a) value_counts all-null + dropna=True
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
            assert out.num_rows == 0, f"expected 0 rows, got {out.num_rows}"

        # (b) nunique all-null + dropna=True  ← THIS WAS BROKEN
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
            assert df.nunique("k", dropna=True) == 0, (
                f"nunique all-null dropna=True: expected 0, "
                f"got {df.nunique('k', dropna=True)}"
            )

        # (c) sanity: normal cases
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

        # (d) AST guards via inspect (dogfood-level)
        import inspect
        from cadw.dataframe import CadwDataFrame as _CDF2
        assert "is_empty_input_error" in inspect.getsource(
            _CDF2.value_counts
        )
        assert "is_empty_input_error" in inspect.getsource(
            _CDF2.nunique
        )

        print(
            "  value_counts+dropna all-null ok + nunique+dropna "
            "all-null ok (was broken) + AST both methods ok"
        )

    run_verify_in_subprocess(_verify)


# ============ 5. Tests ============

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


# ============ 6. main ============

def main():
    print("=== v0.9.23g-fix5: _nunique_one + apply_or_fail idempotency note ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    fix_nunique_one()
    add_apply_or_fail_note()
    add_ast_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23g-fix3+fix4+fix5: value_counts/nunique edges + "
        "empty-input language + idempotency-by-def + apply_or_fail "
        "note. "
        "All three fix-cycles were caught by pytest/dogfood gates "
        "(commit blocked each time); no partial commits. "
        "fix3: _empty_value_counts_table helper (fix2's idempotency "
        "guard had the whole-file substring trap: name already in "
        "value_counts body -> guard fired -> helper never added). "
        "patch_utils.idempotency_check_by_def (def-line vs bare-name "
        "idempotency guard, §13.4/182). "
        "fix4: is_empty_input_error helper checks both English "
        "'empty' and Russian 'пустой' — prior `\"empty\" in str(e)."
        "lower()` missed Russian 'пустой вход' from hash_shuffle -> "
        "all-null + dropna=True re-raised. Applied to value_counts. "
        "fix5: _nunique_one STILL not fixed after fix4 — its custom "
        "guard `if 'v0.9.23g-fix4: is_empty_input_error' in s: skip` "
        "false-positived (marker in value_counts already; whole-file "
        "substring, 3rd occurrence in v0.9.23g). Fix: call "
        "apply_or_fail directly — it IS the idempotency guard "
        "(returns False if new in s, scoped to exact old/new block). "
        "No custom wrapping. "
        "Structural fixes closed in this 3-fix chain: "
        "  - idempotency_check_by_def(path, name) — for helper "
        "presence checks; "
        "  - is_empty_input_error(exc) — both languages; "
        "  - apply_or_fail docstring note — do NOT wrap in "
        "'if marker in s: skip' (§13.4/182, 3rd occurrence); "
        "  - 2 AST tests in test_eda_basics.py (value_counts + "
        "nunique both reference is_empty_input_error); "
        "  - 4 patch_utils tests (is_empty_input_error EN/RU/case/"
        "negative); "
        "  - 4 patch_utils tests (idempotency_check_by_def def/call/"
        "docstring/def+call); "
        "  - all regression tests use chunk=1 (fix1 lesson). "
        "Process note: 5th fix-cycle on v0.9.23g (way over §13.4/238 "
        "threshold of 2). Root cause: v0.9.23g bundled 5 methods "
        "with cross-cutting edge-case semantics (dropna, normalize, "
        "null, empty) — should have split into 3 patches (fillna+"
        "isna, value_counts, nunique), each 1-2 cycles. Lesson: "
        "split by method when cross-cutting edge-case semantics, "
        "not just by LOC. "
        "Fast: 2051 -> ~2100 passed.",
        self_path="scripts/patch_v0923g_fix5_nunique_apply_or_fail.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23g-fix5 applied. v0.9.23g + fix3+4+5 all committed.")


if __name__ == "__main__":
    main()