#!/usr/bin/env python3
# v0.9.23i-fix4 — fix3 had dead-code / NameError in test file write.
#
# Bug (in fix3's patch script): `write_extra_tests()` referenced `extra`
# which was only defined locally in the dead-code `add_multikey_tests()`.
# NameError before test file write. Production fix in cadw/ops/join.py
# DID apply (fix3's first call succeeded); only test file write failed.
#
# This is 4th fix-cycle on v0.9.23i. Production code now correct (verified
# below). fix4:
#   (1) verify _process_partition_pair_anti has fix3 changes (AST)
#   (2) write tests via module-level MULTIKEY_TESTS constant (not closure)
#   (3) dogfood (multi-key crossed + null)
#   (4) pytest, commit (fix3 + fix4 together — fix3 uncommitted)
#
# Lessons:
#   - Patch script helper functions: no closures over `extra` — use
#     module-level constants for content strings.
#   - Dead code in patch scripts (add_multikey_tests) = code smell;
#     if a function isn't called, delete it before commit.
#   - Test-content strings that need to be written must be at module
#     level of the patch script (visible from all helpers).
#
# Run: uv run python scripts/patch_v0923i_fix4_fix_extra.py

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


# ============ 1. Verify fix3 production change applied ============

def verify_state():
    print("\n=== verify fix3 applied to _process_partition_pair_anti ===")
    import inspect
    from cadw.ops import join as j

    src = inspect.getsource(j._process_partition_pair_anti)

    if "import pyarrow.compute as _pc" not in src:
        raise SystemExit(
            "_process_partition_pair_anti missing `import pyarrow.compute "
            "as _pc` — fix3 did not apply. Re-run fix3."
        )
    if "_combine_join_keys" not in src:
        raise SystemExit(
            "_process_partition_pair_anti missing _combine_join_keys — "
            "multi-column composite key fix did not apply."
        )
    if "probe_batch.filter(unmatched)" not in src:
        raise SystemExit(
            "_process_partition_pair_anti missing filter(unmatched) — "
            "unexpected body."
        )
    # Dead code removed: no `.to_batches() if isinstance(... pa.Table)`.
    if "isinstance(\n            filtered, pa.Table\n        )" in src:
        raise SystemExit(
            "_process_partition_pair_anti still has dead code "
            "(isinstance(filtered, pa.Table)). Fix3 should have removed."
        )

    print("  import _pc + _combine_join_keys + no dead code — OK")


# ============ 2. Write multi-key tests (module-level constant) ============

MULTIKEY_TESTS = '''

# ============ v0.9.23i-fix3: multi-column crossed-tuple bug ============
#
# Bug (silent data loss): prior code used AND of per-column is_in.
# For left=(1,"a"), right={(1,"b"),(2,"a")}:
#   is_in(k1:[1,2])=True, is_in(k2:["b","a"])=True -> AND=True ->
#   treated as matched -> NOT emitted. But no right row == (1,"a")
#   -> SHOULD emit. This is the class fixed by fix3 (composite key).

def test_anti_multi_key_crossed(tmp_path):
    """left-tuple appears in cross-product of right columns but not as
    a right row -> must be EMITTED (anti).

    Canonical detector for the AND-of-per-column is_in bug.
    """
    left = make_df(tmp_path, {
        "k1": [1], "k2": ["a"], "v": [100],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1, 2], "k2": ["b", "a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="anti")
    rows = _rows(result)
    assert len(rows) == 1, (
        f"multi-key anti cross-product bug: expected 1 row, got {rows}"
    )
    assert rows[0]["k1"] == 1 and rows[0]["k2"] == "a"


def test_anti_multi_key_actual_match_not_emitted(tmp_path):
    """left-tuple IS a right row -> excluded from anti."""
    left = make_df(tmp_path, {
        "k1": [1, 1], "k2": ["a", "b"], "v": [10, 20],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1], "k2": ["a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="anti")
    rows = _rows(result)
    # (1,"a") matched -> excluded; (1,"b") unmatched -> emitted.
    assert len(rows) == 1
    assert rows[0]["k1"] == 1 and rows[0]["k2"] == "b"


def test_outer_multi_key_crossed(tmp_path):
    """Cross-product case: outer must include both sides properly."""
    left = make_df(tmp_path, {
        "k1": [1], "k2": ["a"], "v": [100],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1, 2], "k2": ["b", "a"], "w": [200, 300],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="outer")
    rows = _rows(result)
    # left(1,"a"): no match. right(1,"b"), right(2,"a"): no match.
    # Outer = 1 + 2 = 3 rows total, both sides padded where needed.
    assert len(rows) == 3
    by_key = {(r["k1"], r["k2"]): r for r in rows}
    assert by_key[(1, "a")]["v"] == 100
    assert by_key[(1, "a")]["w"] is None
    assert by_key[(1, "b")]["v"] is None
    assert by_key[(1, "b")]["w"] == 200
    assert by_key[(2, "a")]["v"] is None
    assert by_key[(2, "a")]["w"] == 300


def test_anti_multi_key_null_emitted(tmp_path):
    """Multi-key with null in one col -> never matched -> emitted."""
    left = make_df(tmp_path, {
        "k1": [1, None], "k2": ["a", "b"], "v": [10, 20],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1], "k2": ["a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="anti")
    rows = _rows(result)
    # (1, "a") matched -> excluded. (None, "b") unmatched -> emitted.
    assert len(rows) == 1
    assert rows[0]["v"] == 20
'''


def write_extra_tests():
    print("\n=== write multi-key tests ===")
    p = Path("tests/test_anti_outer_join.py")
    s = p.read_text()
    if "test_anti_multi_key_crossed" in s:
        print("  already present — skip")
        return
    p.write_text(s + MULTIKEY_TESTS)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 3. Dogfood ============

def dogfood():
    print("\n=== dogfood: multi-key crossed anti + outer + null ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG

        def _rows(df_):
            out = []
            for b in df_._iter_filtered_batches():
                out.extend(b.to_pylist())
            return out

        # (a) SINGLE-column anti (import _pc was missing — the crash)
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            left_t = pa_.table({
                "k": pa_.array([1, None, 3], type=pa_.int64()),
            })
            right_t = pa_.table({"k": [1, 2]})
            src_l = td / "L.parquet"
            src_r = td / "R.parquet"
            pq_.write_table(left_t, src_l)
            pq_.write_table(right_t, src_r)
            left = _CDF.from_parquet(
                src_l, td / "dsL", chunk_size=2,
                memory_guard=_MG(max_memory_mb=100),
                spill_dir=td / "spillL",
            )
            right = _CDF.from_parquet(
                src_r, td / "dsR", chunk_size=1,
                memory_guard=_MG(max_memory_mb=100),
                spill_dir=td / "spillR",
            )
            anti = left.merge(right, on="k", how="anti")
            rows = _rows(anti)
            keys = {r["k"] for r in rows}
            assert 3 in keys and None in keys

        # (b) MULTI-KEY CROSSED (silent data loss regression)
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            left_t = pa_.table({
                "k1": [1], "k2": ["a"], "v": [100],
            })
            right_t = pa_.table({
                "k1": [1, 2], "k2": ["b", "a"], "w": [200, 300],
            })
            src_l = td / "L.parquet"
            src_r = td / "R.parquet"
            pq_.write_table(left_t, src_l)
            pq_.write_table(right_t, src_r)
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
            anti = left.merge(right, on=["k1", "k2"], how="anti")
            rows = _rows(anti)
            assert len(rows) == 1, (
                f"crossed-tuple anti bug: expected 1 row, got {rows}"
            )
            assert rows[0]["k1"] == 1 and rows[0]["k2"] == "a"

        # (c) MULTI-KEY with actual match
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            left_t = pa_.table({
                "k1": [1, 1], "k2": ["a", "b"], "v": [10, 20],
            })
            right_t = pa_.table({"k1": [1], "k2": ["a"]})
            src_l = td / "L.parquet"
            src_r = td / "R.parquet"
            pq_.write_table(left_t, src_l)
            pq_.write_table(right_t, src_r)
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
            anti = left.merge(right, on=["k1", "k2"], how="anti")
            rows = _rows(anti)
            assert len(rows) == 1
            assert rows[0]["k2"] == "b"

        # (d) OUTER multi-key crossed
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            left_t = pa_.table({
                "k1": [1], "k2": ["a"], "v": [100],
            })
            right_t = pa_.table({
                "k1": [1, 2], "k2": ["b", "a"], "w": [200, 300],
            })
            src_l = td / "L.parquet"
            src_r = td / "R.parquet"
            pq_.write_table(left_t, src_l)
            pq_.write_table(right_t, src_r)
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
            outer = left.merge(right, on=["k1", "k2"], how="outer")
            rows = _rows(outer)
            assert len(rows) == 3
            by_key = {(r["k1"], r["k2"]): r for r in rows}
            assert by_key[(1, "a")]["v"] == 100
            assert by_key[(1, "a")]["w"] is None
            assert by_key[(1, "b")]["v"] is None
            assert by_key[(1, "b")]["w"] == 200
            assert by_key[(2, "a")]["v"] is None
            assert by_key[(2, "a")]["w"] == 300

        print(
            "  single-col anti + MULTI-KEY crossed (fixed) + "
            "multi-key match + outer crossed ok"
        )

    run_verify_in_subprocess(_verify)


# ============ 4. Tests ============

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
         "tests/test_anti_outer_join.py",
         "tests/test_join.py",
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


# ============ 5. main ============

def main():
    print("=== v0.9.23i-fix4: test file write (fix3 partial) ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    verify_state()
    write_extra_tests()
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
        "_combine_join_keys. Dead code (isinstance(filtered, pa.Table)"
        " + .to_batches()) removed. "
        "fix4 (patch-script bug): fix3's write_extra_tests "
        "referenced `extra` from a dead-code closure -> NameError "
        "before test file write. Production change in join.py "
        "DID apply; only test file failed to write. Fix4: module-"
        "level MULTIKEY_TESTS constant (no closure); verify_state() "
        "checks fix3 applied before touching test file. "
        "Patch-script lessons (permanent): "
        "  - content strings written to files must be module-level "
        "constants, not closure locals; "
        "  - dead-code helper functions (defined but uncalled) = "
        "code smell; delete before commit; "
        "  - fix scripts verify predecessor state at start "
        "(verify_state) — never assume previous fix applied. "
        "+tests/test_anti_outer_join.py (26 tests: anti basic, "
        "all-match, none-match, empty-right, null-left, multi-key; "
        "outer basic, all-match, disjoint, null-left, empty-left, "
        "empty-right, multi-key; 4 NEW cross-product tests "
        "(anti-multi-key-crossed, anti-multi-key-actual-match, "
        "outer-multi-key-crossed, anti-multi-key-null); "
        "unsupported how raises, semi NotImplementedError; 7 AST). "
        "Dogfood invariants only (no distributions, v0.9.23h "
        "lesson): single-col + MULTI-KEY CROSSED (fixed) + multi-"
        "key match + outer crossed. "
        "Process note: 4th fix-cycle on v0.9.23i (over §13.4/238 "
        "threshold of 2). 2 production bugs (planner-how, multi-"
        "key anti silent data loss), 1 missing import, 1 dead code, "
        "1 anchor miss, 1 patch-script bug. Structural lesson: "
        "multi-key join/anti/outer MUST use composite key, never "
        "AND-of-per-column predicates; crossed-tuple test is the "
        "canonical detector. "
        "Fast: 2125 -> ~2150 passed.",
        self_path="scripts/patch_v0923i_fix4_fix_extra.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23i-fix4 applied. 7.A.10 partial (anti + outer).")


if __name__ == "__main__":
    main()