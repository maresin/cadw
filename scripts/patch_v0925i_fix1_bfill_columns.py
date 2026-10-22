#!/usr/bin/env python3
# v0.9.25i-fix1 — bfill ignores columns= filter; test expectation wrong.
#
# 2 failed tests in v0.9.25i-corrected:
#   1. test_bfill_columns_subset — bfill(columns=["a"]) filled b too.
#      ROOT: bfill step-2 ffill loop iterates all self._columns and
#      unconditionally fills, ignoring `if c not in fill_set`. ffill
#      has this guard; bfill missed it (mirror-symmetry bug).
#   2. test_bfill_filtered — expected [2, None, 5, 5], actual
#      [2, 5, 5, 5]. ROOT: I wrote wrong expectation assuming
#      bfill(limit=1)-like behavior. Standard bfill fills ALL nulls
#      with next non-null, so [2, None, None, 5] → [2, 5, 5, 5].
#
# PRODUCTION FIX: add `if c not in fill_set: arrays.append(arr);
# continue` in bfill step-2 ffill loop.
#
# TEST FIX: correct test_bfill_filtered expected to [2, 5, 5, 5] with
# explanatory comment.
#
# §13.4/267 (NEW): `columns=` filter is a fill-time concern, not just
# schema-time. Must apply in EVERY fill loop, not just out_schema
# construction. Mirror-symmetry: if method A (ffill) has
# `if c not in fill_set`, method B (bfill) should too.
#
# §13.4/268 (NEW): bfill ≠ bfill(limit=1). Standard bfill fills ALL
# nulls with next non-null, regardless of gap length.
# [2, None, None, 5] → [2, 5, 5, 5] (both Nones get 5).
# `limit` semantics = separate parameter (backlog).
#
# REPL-first (§13.4/264) — verify_state does 2 checks:
#   (1) production bug marker present in current dataframe.py
#       (bfill step-2 loop has no `if c not in fill_set`)
#   (2) test bug marker present in tests file
#       ([2, None, 5, 5] expected)
# Both must be present; abort if either already fixed (patch misplaced).
#
# §13.4/266 preserved: verify anchor EXACT text before apply (5-line
# window covering fill_set check). Prior v0.9.25i had wrong anchor
# (def corr vs # __getitem__); apply_or_fail fail-fast caught it.
# Now anchor is well-tested pattern (bfill step-2 loop body).
#
# No new tests: test_bfill_columns_subset already covers production
# bug; test_bfill_filtered already covers test bug. Fix makes both
# pass. Test count stays 33.
#
# Run: uv run python scripts/patch_v0925i_fix1_bfill_columns.py

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


DATAFRAME_PATH = "cadw/dataframe.py"
TESTS_PATH = "tests/test_v0925i.py"


# ============ 0. verify state (REPL-first + anchor check) ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25h (v0.9.25i uncommitted); "
          "both bug markers present; no fix1 applied yet ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25h" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25h; got {log[:200]!r}"
        )
    if "v0.9.25i" in log:
        raise SystemExit("v0.9.25i already committed — fix1 misplaced")

    df_src = Path(DATAFRAME_PATH).read_text()
    if "def bfill(self" not in df_src:
        raise SystemExit("bfill not found — v0.9.25i didn't apply?")
    if "def ffill(self" not in df_src:
        raise SystemExit("ffill not found — v0.9.25i didn't apply?")

    # (1) Production bug: bfill step-2 loop has no `if c not in fill_set`
    # exact-marker: check the ffill-pass loop body lacks the pass-through
    if "bfill.step2.read" not in df_src:
        raise SystemExit("bfill.step2.read check_batch not found")

    # Find bfill step2 block; verify buggy form (no fill_set gate)
    bfill_start = df_src.find("def bfill(self")
    bfill_end = df_src.find("def reindex(self", bfill_start)
    if bfill_end < 0:
        raise SystemExit("reindex not found after bfill — anchor broken")
    bfill_body = df_src[bfill_start:bfill_end]

    if "if c not in fill_set:" in bfill_body:
        raise SystemExit(
            "BUG ALREADY FIXED: `if c not in fill_set:` present in "
            "bfill. Patch misplaced — HEAD already has fix1."
        )
    # Also verify ffill DOES have the guard (sanity, correct code)
    ffill_start = df_src.find("def ffill(self")
    ffill_end = df_src.find("def bfill(self")
    ffill_body = df_src[ffill_start:ffill_end]
    if "if c not in fill_set:" not in ffill_body:
        raise SystemExit(
            "SANITY FAIL: ffill also missing `if c not in fill_set:` — "
            "both ffill and bfill buggy, this fix1 won't be enough."
        )
    print("  (1) production bug marker confirmed: bfill missing "
          "fill_set guard (ffill has it — mirror asymmetry)")

    # (2) Test bug: test_bfill_filtered expects [2, None, 5, 5]
    tests_src = Path(TESTS_PATH).read_text()
    if "assert vals == [2, None, 5, 5]" not in tests_src:
        raise SystemExit(
            "test_bfill_filtered stale expectation "
            "[2, None, 5, 5] not found — already fixed?"
        )
    if "assert vals == [2, 5, 5, 5]" in tests_src:
        raise SystemExit(
            "TEST BUG ALREADY FIXED: [2, 5, 5, 5] present"
        )
    print("  (2) test bug marker confirmed: test_bfill_filtered "
          "expects [2, None, 5, 5]")

    # REPL sanity: bfill on [2, None, None, 5] → [2, 5, 5, 5]
    import pyarrow as pa
    import pyarrow.compute as _pc
    filled = _pc.fill_null_forward(
        pa.array([5, None, None, 2], type=pa.int64())  # reversed
    ).to_pylist()
    # reversed [5, None, None, 2] ffill → [5,5,5,2]; reverse back
    # [2, 5, 5, 5] = correct bfill
    if filled != [5, 5, 5, 2]:
        raise SystemExit(
            f"REPL: fill_null_forward reversed unexpected: {filled}"
        )
    print("  (3) REPL: bfill([2,None,None,5]) → [2,5,5,5] confirmed")

    print("  state + anchor verified")


# ============ 1. production fix: bfill step-2 fill_set guard ============

OLD_BFILL_LOOP = '''        try:
            for b in ArrowIPCBatchReader(str(sorted_desc), None):
                self._memory_guard.check_batch(
                    b, name="bfill.step2.read",
                )
                n = b.num_rows
                if n == 0:
                    continue
                arrays = [b.column("__orig_rid")]
                for c in self._columns:
                    arr = b.column(c)
                    car = carry[c]
                    if car is not None:
                        extended = pa.concat_arrays([car, arr])
                        filled = _pc.fill_null_forward(extended).slice(1)
                    else:
                        filled = _pc.fill_null_forward(arr)
                    if len(filled) > 0:
                        last_slice = filled.slice(len(filled) - 1, 1)
                        if _pc.is_valid(last_slice).to_pylist()[0]:
                            carry[c] = last_slice
                    arrays.append(filled)
                writer.write_batch(
                    pa.RecordBatch.from_arrays(
                        arrays, schema=filled_schema,
                    )
                )
        finally:
            writer.close()'''


NEW_BFILL_LOOP = '''        try:
            for b in ArrowIPCBatchReader(str(sorted_desc), None):
                self._memory_guard.check_batch(
                    b, name="bfill.step2.read",
                )
                n = b.num_rows
                if n == 0:
                    continue
                arrays = [b.column("__orig_rid")]
                for c in self._columns:
                    arr = b.column(c)
                    # v0.9.25i-fix1 (§13.4/267): only fill columns in
                    # fill_set. Prior code filled ALL columns
                    # regardless of columns= — bug caught by
                    # test_bfill_columns_subset. Mirror-symmetry with
                    # ffill (which correctly has this guard).
                    if c not in fill_set:
                        arrays.append(arr)
                        continue
                    car = carry[c]
                    if car is not None:
                        extended = pa.concat_arrays([car, arr])
                        filled = _pc.fill_null_forward(extended).slice(1)
                    else:
                        filled = _pc.fill_null_forward(arr)
                    if len(filled) > 0:
                        last_slice = filled.slice(len(filled) - 1, 1)
                        if _pc.is_valid(last_slice).to_pylist()[0]:
                            carry[c] = last_slice
                    arrays.append(filled)
                writer.write_batch(
                    pa.RecordBatch.from_arrays(
                        arrays, schema=filled_schema,
                    )
                )
        finally:
            writer.close()'''


def fix_bfill():
    print("\n=== fix bfill: add `if c not in fill_set` pass-through ===")
    applied = apply_or_fail(
        DATAFRAME_PATH, OLD_BFILL_LOOP, NEW_BFILL_LOOP,
        label="bfill_fill_set",
    )
    if applied:
        assert_compiles(DATAFRAME_PATH)
        check_no_invalid_escape(DATAFRAME_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. test fix: test_bfill_filtered expectation ============

OLD_TEST_FILTERED = '''def test_bfill_filtered():
    df = _mk({
        "x": pa.array([None, 2, None, None, 5], type=pa.int64()),
        "v": [10, 20, 30, 40, 50],
    })
    filtered = df[df["v"] > 15]
    result = filtered.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [2, None, 5, 5]'''


NEW_TEST_FILTERED = '''def test_bfill_filtered():
    """v0.9.25i-fix1 (§13.4/268): bfill fills ALL nulls with the
    next non-null value, regardless of gap length. Prior expectation
    [2, None, 5, 5] wrongly assumed bfill(limit=1)-like behavior.

    Filtered rows (v>15 at raw pos 1-4): x = [2, None, None, 5].
    bfill → [2, 5, 5, 5] (both Nones get 5 from pos 3).
    """
    df = _mk({
        "x": pa.array([None, 2, None, None, 5], type=pa.int64()),
        "v": [10, 20, 30, 40, 50],
    })
    filtered = df[df["v"] > 15]
    result = filtered.bfill()
    vals = result.head(10).column("x").to_pylist()
    assert vals == [2, 5, 5, 5]'''


def fix_test():
    print("\n=== fix test_bfill_filtered expectation: [2,None,5,5] → "
          "[2,5,5,5] ===")
    applied = apply_or_fail(
        TESTS_PATH, OLD_TEST_FILTERED, NEW_TEST_FILTERED,
        label="test_bfill_filtered",
    )
    if applied:
        assert_compiles(TESTS_PATH)
        check_no_invalid_escape(TESTS_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 3. pytest ============

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

    print("\n=== pytest: v0925i (both fix targets) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925i.py",
         "tests/test_v0925h.py",
         "tests/test_shift_diff.py",
         "tests/test_sort.py",
         "tests/test_dataframe.py",
         "tests/test_dataframe_filters.py",
         "tests/test_loc.py",
         "tests/test_multiindex.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=600,
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


# ============ 4. main ============

def main():
    print("=== v0.9.25i-fix1: bfill columns= filter + test expectation ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    fix_bfill()
    fix_test()
    run_pytest()

    commit_with_guards(
        "v0.9.25i: Phase C batch C-3a — ffill + bfill + reindex "
        "+ fix1 (bfill columns filter + test expectation). "
        "3 of 20 Pandas P0 features (C-1 v0.9.25g, C-2 v0.9.25h, "
        "C-3a here). "
        "Features: "
        "(1) `CadwDataFrame.ffill(columns=None)` — forward fill via "
        "pc.fill_null_forward + carry for cross-batch leading-null. "
        "Streaming, O(#cols) state. "
        "(2) `CadwDataFrame.bfill(columns=None)` — reverse-sort + "
        "ffill + re-sort (2× external_sort). O(N log N) I/O, "
        "O(batch) RAM. Trailing nulls stay null. "
        "(3) `CadwDataFrame.reindex(labels)` — row alignment. "
        "Unfiltered RangeIndex: positional labels; else "
        "MaterializedIndex-based O(N+K) label→pos map (filtered "
        "RangeIndex labels = original positionals, not filtered "
        "enumeration). Null-filled for missing; all-missing shortcut "
        "avoids take-on-empty. Bounded by K × row_size ≤ state_"
        "limit. MultiIndex → NotImplementedError. "
        "fix1 (2 bugs from v0.9.25i-corrected, 168/170 passed → "
        "0 failed): "
        "(a) PRODUCTION BUG: bfill ignores `columns=` filter — "
        "step-2 ffill loop iterated all self._columns unconditionally. "
        "`fill_set` built but never gated usage. Caught by "
        "test_bfill_columns_subset (b filled when only a requested). "
        "Fix: `if c not in fill_set: arrays.append(arr); continue`. "
        "Mirror-symmetry bug: ffill had this 2-line guard, bfill "
        "missed it. "
        "(b) TEST BUG: test_bfill_filtered expected [2, None, 5, 5], "
        "actual [2, 5, 5, 5]. Prior assumed bfill(limit=1)-like; "
        "standard bfill fills ALL nulls with next non-null. Fix: "
        "expected [2, 5, 5, 5] with explanatory docstring. "
        "§13.4/267 (NEW): `columns=` filter is a fill-time concern, "
        "not just schema-time. Must apply in EVERY fill loop, not "
        "just out_schema construction. Mirror-symmetry: if method A "
        "has `if c not in fill_set`, method B should too. "
        "§13.4/268 (NEW): bfill ≠ bfill(limit=1). Standard bfill "
        "fills ALL nulls, regardless of gap length. [2, None, None, "
        "5] → [2, 5, 5, 5]. `limit` = separate parameter (backlog). "
        "§13.4/264 (REPL-first): verify_state does 3 checks before "
        "any edit — bug marker in bfill (no fill_set guard while "
        "ffill has it); stale [2,None,5,5] test expected; REPL "
        "confirmation via pc.fill_null_forward on reversed array. "
        "All PASS. "
        "§13.4/266 (anchor check): exact 30-line anchor for bfill "
        "step-2 loop body. §13.4/260/261/262/265 preserved. "
        "Total Phase C: v0.9.25g (C-1, 4 attempts, 1 production bug: "
        "type_alias) + v0.9.25h (C-2, 2 attempts, 0 production bugs) "
        "+ v0.9.25i+fix1 (C-3a, 2 attempts, 1 production bug: bfill "
        "columns). REPL-first (§13.4/264) prevented 2 earlier; "
        "mirror-check (§13.4/267 NEW) would have prevented this one. "
        "Tests (tests/test_v0925i.py): 33 total — ffill (11), "
        "bfill (11 incl. fixed test_bfill_filtered + "
        "test_bfill_columns_subset), reindex (11 incl. "
        "test_reindex_filtered_range_index). "
        "Fast suite: ~2371 → ~2404 (+33). "
        "Next: Phase C C-3b (cut/qcut/duplicated).",
        self_path="scripts/patch_v0925i_fix1_bfill_columns.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25i COMPLETE (3 features + fix1). "
          "Phase C: 9/20. Next: C-3b (cut/qcut/duplicated).")


if __name__ == "__main__":
    main()