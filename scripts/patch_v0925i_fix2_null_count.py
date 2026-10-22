#!/usr/bin/env python3
# v0.9.25i-fix2 — `is_valid(x).to_pylist()[0]` → `x.null_count == 0`.
#
# CONTEXT: v0.9.25i-fix1 (bfill columns filter) passed targeted
# (221/221), but FULL FAST suite surfaced 2 failures in
# tests/test_v0925d_a.py:
#   LOW_FACTOR cadw/dataframe.py:1211 CadwDataFrame.ffill._iter
#   LOW_FACTOR cadw/dataframe.py:1343 CadwDataFrame.bfill
#
# These are DIAGNOSTIC meta-guard tests from v0.9.25d-b2 that assert
# `unguarded == 0 AND low_factor == 0` across cadw/. My ffill/bfill
# introduce 2 new `to_pylist` sites inside functions whose check_batch
# has default extra_factor=1.0 (< 3.0). The diagnostic tool doesn't
# look at array length — it flags the SITE (to_pylist Call + guard
# factor).
#
# ROOT: in ffill and bfill I wrote the last-slice validity check as:
#   last_slice = filled.slice(len(filled) - 1, 1)
#   if _pc.is_valid(last_slice).to_pylist()[0]:
#       carry[c] = last_slice
#
# The to_pylist on 1-element array is O(1) memory — no true low_factor.
# But it uses a Python-object pattern (list + [0] index) where Arrow
# metadata suffices.
#
# FIX: `last_slice.null_count == 0` — semantically identical for
# 1-element slice (null_count > 0 iff null). Arrow null_count is O(1)
# (validity bitmap popcount), no Python object.
#
# §13.4/269 (NEW): 1-element .to_pylist()[0] is an anti-pattern even
# when memory-bounded. Arrow metadata (null_count, length,
# is_valid().as_py()) is O(1) without Python object. Diagnostic tool
# flags any to_pylist inside check_batch(factor<3.0) scope — correct
# discipline: don't accumulate Python-object patterns for convenience.
# If a to_pylist is genuinely O(1), use Arrow metadata instead of
# whitelisting. Whitelisting = no Arrow-native equivalent exists.
#
# ALTERNATIVE REJECTED: whitelist the 2 sites (masks inefficiency);
# bump extra_factor to 3.0 (lies about peak RAM — fill_null_forward
# creates new array = batch size, so peak = batch, not 3× batch; the
# to_pylist on 1 element adds ~28 B per batch per column, negligible
# but still Python-object noise).
#
# §13.4/264 (REPL-first): verify_state does 3 checks before edits —
#   (1) both sites present (bug marker: .to_pylist()[0] after is_valid)
#   (2) diagnostic currently reports low_factor=2 (via subprocess or
#       direct scan)
#   (3) REPL: null_count semantics on 1-element slice matches
#       is_valid semantics ([v].null_count == 0 iff is_valid([v])==True)
#
# §13.4/266 (anchor check): 2 distinct anchors — ffill block has
# `yield b.select` + `yield pa.RecordBatch`, bfill has `writer.
# write_batch`. Different contexts, disjoint matches.
#
# §13.4/267 (mirror-check): both ffill and bfill get the SAME fix.
# Enforced by design — this script does both in one patch.
#
# No test changes: all 33 v0.9.25i tests unchanged; diagnostic meta-
# guards self-recover (2402→2404 passed).
#
# Run: uv run python scripts/patch_v0925i_fix2_null_count.py

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


# ============ 0. verify state (REPL-first, §13.4/264) ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25h (v0.9.25i uncommitted); "
          "both to_pylist bug markers present; diagnostic reports "
          "low_factor=2 ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25h" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25h; got {log[:200]!r}"
        )
    if "v0.9.25i" in log:
        raise SystemExit(
            "v0.9.25i already committed — fix2 misplaced"
        )

    src = Path(DATAFRAME_PATH).read_text()

    # (1) Bug markers: at least 2 occurrences of the to_pylist anti-
    # pattern in ffill/bfill blocks.
    bad_marker = "if _pc.is_valid(last_slice).to_pylist()[0]:"
    n_bad = src.count(bad_marker)
    if n_bad < 2:
        if n_bad == 0:
            raise SystemExit(
                "BUG ALREADY FIXED: no `_pc.is_valid(last_slice)."
                "to_pylist()[0]` present — fix2 misplaced"
            )
        raise SystemExit(
            f"expected 2 occurrences of to_pylist anti-pattern, "
            f"found {n_bad} — partial fix2?"
        )

    # Sanity: fixed marker NOT present yet
    if "if last_slice.null_count == 0:" in src:
        raise SystemExit(
            "PARTIAL FIX2: `last_slice.null_count == 0` already "
            "present alongside old marker — inconsistent state"
        )

    print(f"  (1) production bug markers: {n_bad} × "
          f"`is_valid(last_slice).to_pylist()[0]` confirmed")

    # (2) Diagnostic scan reports low_factor=2
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "diag_mod",
        "scripts/service/diagnose_unguarded_to_pylist.py",
    )
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)
    results = diag.scan()
    lf = results.get("low_factor", [])
    if len(lf) < 2:
        raise SystemExit(
            f"expected low_factor >= 2 (ffill + bfill), got "
            f"{len(lf)}. Diagnostic may be stale."
        )
    qualnames = {s["qualname"] for s in lf}
    expected = {"CadwDataFrame.ffill._iter", "CadwDataFrame.bfill"}
    if not expected.issubset(qualnames):
        raise SystemExit(
            f"expected qualnames {expected} in low_factor, got "
            f"{qualnames}. Line numbers or scope may have shifted."
        )
    print(f"  (2) diagnostic: low_factor={len(lf)} with qualnames "
          f"{sorted(qualnames)} confirmed")

    # (3) REPL: null_count semantics on 1-element slice
    import pyarrow as pa
    # Valid 1-element slice
    arr_valid = pa.array([42], type=pa.int64())
    if arr_valid.null_count != 0:
        raise SystemExit(
            f"[42].null_count = {arr_valid.null_count}, expected 0"
        )
    # Null 1-element slice
    arr_null = pa.array([None], type=pa.int64())
    if arr_null.null_count != 1:
        raise SystemExit(
            f"[None].null_count = {arr_null.null_count}, expected 1"
        )
    # is_valid equivalence
    import pyarrow.compute as _pc
    if not _pc.is_valid(arr_valid).to_pylist()[0]:
        raise SystemExit("is_valid([42])[0] != True")
    if _pc.is_valid(arr_null).to_pylist()[0]:
        raise SystemExit("is_valid([None])[0] != False")
    print("  (3) REPL: null_count==0 ⇔ is_valid[0]==True "
          "(1-element slice) confirmed")

    print("  state + markers + diagnostic + REPL verified")


# ============ 1. fix ffill ============

FFILL_OLD = '''                if n == 0:
                    yield b.select(self._columns)
                    continue
                arrays = []
                for c in self._columns:
                    arr = b.column(c)
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
                    arrays.append(filled)'''


FFILL_NEW = '''                if n == 0:
                    yield b.select(self._columns)
                    continue
                arrays = []
                for c in self._columns:
                    arr = b.column(c)
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
                        # v0.9.25i-fix2 (§13.4/269): null_count is
                        # O(1) Arrow metadata — replaces
                        # `is_valid(last_slice).to_pylist()[0]`
                        # (1-element to_pylist still flagged by
                        # diagnostic as low_factor since check_batch
                        # factor<3.0). Semantically identical for
                        # 1-element slice.
                        if last_slice.null_count == 0:
                            carry[c] = last_slice
                    arrays.append(filled)'''


def fix_ffill():
    print("\n=== fix ffill: null_count == 0 ===")
    applied = apply_or_fail(
        DATAFRAME_PATH, FFILL_OLD, FFILL_NEW, label="ffill_null_count",
    )
    if applied:
        assert_compiles(DATAFRAME_PATH)
        check_no_invalid_escape(DATAFRAME_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 2. fix bfill ============

BFILL_OLD = '''                arrays = [b.column("__orig_rid")]
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
                    arrays.append(filled)'''


BFILL_NEW = '''                arrays = [b.column("__orig_rid")]
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
                        # v0.9.25i-fix2 (§13.4/269): null_count O(1).
                        if last_slice.null_count == 0:
                            carry[c] = last_slice
                    arrays.append(filled)'''


def fix_bfill():
    print("\n=== fix bfill: null_count == 0 ===")
    applied = apply_or_fail(
        DATAFRAME_PATH, BFILL_OLD, BFILL_NEW, label="bfill_null_count",
    )
    if applied:
        assert_compiles(DATAFRAME_PATH)
        check_no_invalid_escape(DATAFRAME_PATH)
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

    print("\n=== pre-flight: diagnostic meta-guards (target of fix) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925d_a.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** DIAGNOSTIC META-GUARDS FAILED — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: v0925i + v0925h + related ===")
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
    print("=== v0.9.25i-fix2: to_pylist → null_count ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    fix_ffill()
    fix_bfill()
    run_pytest()

    commit_with_guards(
        "v0.9.25i: Phase C batch C-3a — ffill + bfill + reindex "
        "+ fix1 (bfill columns) + fix2 (null_count). "
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
        "MaterializedIndex-based O(N+K) label→pos map. Null-filled "
        "for missing; all-missing shortcut avoids take-on-empty. "
        "Bounded by K × row_size ≤ state_limit. "
        "fix1 (bfill columns filter + test expectation): "
        "(a) PRODUCTION BUG: bfill ignored `columns=` — step-2 loop "
        "iterated all self._columns unconditionally. `fill_set` "
        "built but never gated. Caught by test_bfill_columns_subset. "
        "Fix: `if c not in fill_set: arrays.append(arr); continue`. "
        "Mirror-symmetry bug (ffill had guard, bfill missed it). "
        "(b) TEST BUG: test_bfill_filtered expected [2,None,5,5], "
        "actual [2,5,5,5] — standard bfill fills ALL nulls with next "
        "non-null, not limit=1 semantics. Fix: expected [2,5,5,5] "
        "with explanatory docstring. "
        "fix2 (to_pylist → null_count): "
        "v0.9.25i-fix1 passed targeted (221/221) but full suite "
        "surfaced 2 failures in tests/test_v0925d_a.py (diagnostic "
        "meta-guards from v0.9.25d-b2 asserting low_factor=0 across "
        "cadw/). ffill/bfill introduced 2 new to_pylist sites: "
        "`is_valid(last_slice).to_pylist()[0]` (1-element array "
        "validity check). Diagnostic doesn't look at array length — "
        "flags SITE (to_pylist Call + check_batch factor<3.0). Fix: "
        "`last_slice.null_count == 0` (O(1) Arrow metadata, "
        "semantically identical for 1-element slice). 2 identical "
        "substitutions, 1 per method. "
        "§13.4/267 (NEW): `columns=` filter is fill-time concern, "
        "not just schema-time. Must apply in EVERY fill loop. "
        "Mirror-symmetry: if A has `if c not in fill_set`, B should "
        "too. "
        "§13.4/268 (NEW): bfill ≠ bfill(limit=1). Standard bfill "
        "fills ALL nulls with next non-null, regardless of gap "
        "length. `limit` = separate parameter (backlog). "
        "§13.4/269 (NEW): 1-element `.to_pylist()[0]` is anti-"
        "pattern even when memory-bounded. Arrow metadata "
        "(null_count, length, is_valid().as_py()) is O(1) without "
        "Python object. Diagnostic tool flags any to_pylist inside "
        "check_batch(factor<3.0) scope — correct discipline: use "
        "Arrow metadata instead of whitelisting. Whitelisting = no "
        "Arrow-native equivalent exists. "
        "§13.4/264 (REPL-first): verify_state does 3 checks — bug "
        "markers × 2; diagnostic reports low_factor=2 with expected "
        "qualnames; REPL null_count ⇔ is_valid equivalence on "
        "1-element slice. All PASS. "
        "§13.4/266 (anchor check): 2 disjoint anchors — ffill has "
        "`yield b.select` + `yield pa.RecordBatch`, bfill has "
        "`writer.write_batch`. No ambiguity. "
        "§13.4/260/261/262/265 preserved. "
        "DEFERRED: ffill/bfill limit= → v0.9.25j; reindex MultiIndex "
        "→ NotImplementedError; bfill pending-buffer op → backlog. "
        "Tests (tests/test_v0925i.py): 33 total — ffill (11), "
        "bfill (11), reindex (11). "
        "Fast suite: 2371 baseline + 33 = 2404 passed, 4 skipped. "
        "Next: Phase C C-3b (cut/qcut/duplicated).",
        self_path="scripts/patch_v0925i_fix2_null_count.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25i COMPLETE (3 features + fix1 + fix2). "
          "Phase C: 9/20. Next: C-3b (cut/qcut/duplicated).")


if __name__ == "__main__":
    main()