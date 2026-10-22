#!/usr/bin/env python3
# v0.9.23i-fix5 — stale test after `how` enum extension.
#
# test_merge_unsupported_how (v0.2.x): tested how in ("right","outer",
# "cross") -> NotImplementedError with match="only 'inner', 'left'".
# After v0.9.23i:
#   - 'outer' is SUPPORTED (no raise)
#   - 'right'/'cross' raise NEW message
#     "supported: inner, left, outer, anti, semi. 'right' — v0.9.23j."
#
# Class: §13.4/144 (stale-test after behavior change). Pattern: extend
# enum -> must update ALL tests that test enum boundary. I extended
# how but forgot this test.
#
# Fix:
#   (1) bad_how = ("right", "cross") — removed 'outer'.
#   (2) match=updated regex.
#   (3) Add explicit semi check with match="semi".
#
# Run: uv run python scripts/patch_v0923i_fix5_stale_how_test.py

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


# ============ 1. Fix stale test ============

OLD_TEST = '''def test_merge_unsupported_how(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1], "a": [10]})
    right = make_df(tmp_path, "R", {"k": [1], "b": [100]})
    for bad_how in ("right", "outer", "cross"):
        with pytest.raises(NotImplementedError, match="only 'inner', 'left'"):
            left.merge(right, on="k", how=bad_how)'''

NEW_TEST = '''def test_merge_unsupported_how(tmp_path):
    """v0.9.23i-fix5: 'outer' now supported; 'right'/'cross' still raise.

    Prior test (v0.2.x) expected only inner/left. Enum extended in
    v0.9.23i (outer/anti/semi) and v0.9.23j ('right'). Update:
    - outer -> supported (no longer in bad_how)
    - right, cross -> raise with updated message
    - semi -> raises NotImplementedError (v0.9.23j placeholder)
    """
    left = make_df(tmp_path, "L", {"k": [1], "a": [10]})
    right = make_df(tmp_path, "R", {"k": [1], "b": [100]})
    for bad_how in ("right", "cross"):
        with pytest.raises(
            NotImplementedError,
            match="supported: inner, left, outer, anti, semi",
        ):
            left.merge(right, on="k", how=bad_how)
    # semi: explicit placeholder message
    with pytest.raises(NotImplementedError, match="semi"):
        left.merge(right, on="k", how="semi")'''


def fix_stale_test():
    print("\n=== fix test_merge_unsupported_how ===")
    p = Path("tests/test_join.py")
    applied = apply_or_fail(
        str(p), OLD_TEST, NEW_TEST, label="merge_unsupported_how",
    )
    if applied:
        assert_compiles(p)
        print("  applied")
    else:
        print("  already fixed — skip")


# ============ 2. Dogfood ============

def dogfood():
    print("\n=== dogfood: how enum boundaries ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG

        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t_l = pa_.table({"k": [1], "a": [10]})
            t_r = pa_.table({"k": [1], "b": [100]})
            src_l = td / "L.parquet"
            src_r = td / "R.parquet"
            pq_.write_table(t_l, src_l)
            pq_.write_table(t_r, src_r)
            left = _CDF.from_parquet(
                src_l, td / "dsL", chunk_size=1,
                memory_guard=_MG(max_memory_mb=100),
                spill_dir=td / "spillL",
            )
            right = _CDF.from_parquet(
                src_r, td / "dsR", chunk_size=1,
                memory_guard=_MG(max_memory_mb=100),
                spill_dir=td / "spillR",
            )
            # (a) supported hows: inner, left, outer, anti — all work
            for supported in ("inner", "left", "outer", "anti"):
                _ = left.merge(right, on="k", how=supported)
            # (b) unsupported: right, cross -> NotImplementedError
            for bad in ("right", "cross"):
                try:
                    left.merge(right, on="k", how=bad)
                except NotImplementedError as e:
                    assert "supported: inner, left, outer, anti, semi" in str(e)
                else:
                    raise AssertionError(
                        f"how={bad!r} must raise NotImplementedError"
                    )
            # (c) semi: NotImplementedError (v0.9.23j placeholder)
            try:
                left.merge(right, on="k", how="semi")
            except NotImplementedError as e:
                assert "semi" in str(e)
            else:
                raise AssertionError("semi must raise NotImplementedError")

        print(
            "  inner/left/outer/anti work; right/cross raise; "
            "semi raises with 'semi' message ok"
        )

    run_verify_in_subprocess(_verify)


# ============ 3. Tests ============

def run_pytest():
    print("\n=== pre-flight: test_facade_has_no_as_py ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py",
         "-q"],
        timeout=60,
    )
    if r.returncode != 0:
        print("\n*** FACADE AS_PY VIOLATION — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_join.py",
         "tests/test_anti_outer_join.py",
         "tests/test_join_streaming.py",
         "tests/test_join_strict.py",
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
    print("=== v0.9.23i-fix5: stale merge_unsupported_how ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    fix_stale_test()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23i: anti join + outer join (AUDIT_FULL_v0922 item "
        "7.A.10, split 1/3, Top-10 gap #4). "
        "anti_join_streaming + outer (via merge dispatch). "
        "merge(how=...) extended: inner | left | outer | anti | semi "
        "(semi -> NotImplementedError, v0.9.23j). "
        "fix1: cadw/__init__.py anchor miss recovery; only "
        "cadw/ops/__init__.py extended. "
        "fix2: planner how map (anti/outer/semi -> 'left'); "
        "predicted_output per how (anti/semi -> 0). "
        "fix3 (2 production bugs): (1) NameError '_pc' — missing "
        "import in _process_partition_pair_anti; (2) SILENT multi-"
        "column anti data loss — AND-of-per-column is_in is cross-"
        "product false positive; fix: composite key via "
        "_combine_join_keys. Dead code removed. "
        "fix4 (patch-script bug): fix3's write_extra_tests "
        "referenced closure-local `extra` -> NameError. Fix4: "
        "module-level MULTIKEY_TESTS constant + verify_state at "
        "start. "
        "fix5 (stale test, §13.4/144): test_merge_unsupported_how "
        "(v0.2.x) expected how in ('right','outer','cross') all "
        "raise match=\"only 'inner', 'left'\". After v0.9.23i: "
        "outer supported; message updated. Class: extend enum -> "
        "must update all tests that check enum boundary. "
        "Fix: bad_how=('right','cross'); match=updated; explicit "
        "semi check. "
        "+tests/test_anti_outer_join.py (26 tests: anti/outer full "
        "matrix including multi-key crossed, null-left, empty, "
        "multi-key; 7 AST guards). "
        "+tests/test_join.py (stale test updated: right/cross raise "
        "with new message; semi explicit check). "
        "Dogfood invariants only: inner/left/outer/anti work; "
        "right/cross raise; semi raises with 'semi' message. "
        "Process note: 5th fix-cycle on v0.9.23i (way over "
        "§13.4/238 threshold of 2). Cumulative: 2 production bugs "
        "(planner-how, multi-key anti silent data loss) + 1 "
        "missing import + 1 dead code + 1 anchor miss + 1 patch-"
        "script bug + 1 stale test. All lessons appended to "
        "§13.4 patterns (multi-column join composite key, patch-"
        "script module-level content, verify predecessor state, "
        "stale-test on enum extension). "
        "Fast: 2125 -> ~2150 passed.",
        self_path="scripts/patch_v0923i_fix5_stale_how_test.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23i-fix5 applied. 7.A.10 partial (anti + outer).")


if __name__ == "__main__":
    main()