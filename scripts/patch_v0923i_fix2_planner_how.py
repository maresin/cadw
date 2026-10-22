#!/usr/bin/env python3
# v0.9.23i-fix2 — planner how mapping + warning skip for anti/semi.
#
# Bug (dogfood-caught): merge() passes `how` verbatim into
# JoinMemoryModel.solve op_context. Model supports only 'inner'|'left'
# -> ValueError for how='anti'/'outer'/'semi'.
#
# Fix:
#   1. Map `how` -> `_planner_how` before plan(): anti/semi/outer ->
#      'left' (planner build=right, probe=left; same class as left).
#   2. predicted_output for whole-op warning:
#      - anti/semi -> 0 (output ≤ left_n, not left_n*avg_matches;
#        per-probe-batch guard sufficient);
#      - outer -> left_join_pred + right_n*row_size (left + anti_right
#        parts concatenate to at most left+right rows);
#      - inner/left -> left_n * avg_matches * row_size (unchanged).
#   3. Import fix: remove `semi_join_streaming` from lazy import in
#      merge() dispatch — function doesn't exist yet (v0.9.23j).
#
# v0.9.23j will:
#   - add semi_join_streaming (v0.9.23j)
#   - re-add semi to the dispatch import
#   - extend semi branch
#
# Run: uv run python scripts/patch_v0923i_fix2_planner_how.py

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


# ============ 1. Planner how mapping ============

OLD_PLAN = '''                params = self._memory_guard.plan(
                    "join", left_chars,
                    op_context={"right_chars": right_chars, "how": how},
                )'''

NEW_PLAN = '''                # v0.9.23i-fix2: JoinMemoryModel.solve supports only
                # 'inner'|'left'. Map anti/semi/outer -> 'left' for
                # planning (planner build=right, probe=left; same class).
                # Actual execution dispatches to anti_join_streaming /
                # hash_join_streaming(how='left') appropriately.
                _planner_how = (
                    "inner" if how == "inner" else "left"
                )
                params = self._memory_guard.plan(
                    "join", left_chars,
                    op_context={
                        "right_chars": right_chars,
                        "how": _planner_how,
                    },
                )'''


def fix_planner_how():
    print("\n=== map planner how (anti/semi/outer -> left) ===")
    p = Path("cadw/dataframe.py")
    applied = apply_or_fail(
        str(p), OLD_PLAN, NEW_PLAN, label="planner_how_map",
    )
    if applied:
        assert_compiles(p)
        print("  applied")
    else:
        print("  already applied — skip")


# ============ 2. predicted_output for anti/semi/outer ============

OLD_PREDICT = '''                predicted_output = (
                    left_chars.n_rows
                    * params.avg_matches
                    * params.output_row_size
                )'''

NEW_PREDICT = '''                # v0.9.23i-fix2: predicted_output per how.
                # anti/semi: ≤ left_n (per-probe-batch guard sufficient,
                # whole-op overhead only — 40 MB fixed).
                # outer: left_join (left_n*avg_matches) + anti_right
                # (≤ right_n) rows.
                # inner/left: left_n*avg_matches (unchanged).
                if how in ("anti", "semi"):
                    predicted_output = 0
                elif how == "outer":
                    predicted_output = (
                        left_chars.n_rows
                        * params.avg_matches
                        * params.output_row_size
                        + right_chars.n_rows * params.output_row_size
                    )
                else:
                    predicted_output = (
                        left_chars.n_rows
                        * params.avg_matches
                        * params.output_row_size
                    )'''


def fix_predicted_output():
    print("\n=== predicted_output per how ===")
    p = Path("cadw/dataframe.py")
    applied = apply_or_fail(
        str(p), OLD_PREDICT, NEW_PREDICT, label="predicted_output_how",
    )
    if applied:
        assert_compiles(p)
        print("  applied")
    else:
        print("  already applied — skip")


# ============ 3. Remove non-existent semi_join_streaming import ============

OLD_IMPORT = '''        from cadw.ops.join import (
            anti_join_streaming as _anti,
            outer_join_streaming as _outer_unused,  # noqa: F401
            semi_join_streaming as _semi_unused,  # noqa: F401 (v0.9.23j)
        )'''

NEW_IMPORT = '''        # v0.9.23i-fix2: only anti (semi in v0.9.23j; outer via
        # left+anti composition inside dispatch, no direct call).
        from cadw.ops.join import anti_join_streaming as _anti'''


def fix_import():
    print("\n=== remove semi_join_streaming import (v0.9.23j) ===")
    p = Path("cadw/dataframe.py")
    applied = apply_or_fail(
        str(p), OLD_IMPORT, NEW_IMPORT, label="dispatch_import",
    )
    if applied:
        assert_compiles(p)
        print("  applied")
    else:
        print("  already applied — skip")


# ============ 4. Tests for planner + predicted_output ============

def add_tests():
    print("\n=== add planner-how tests to test_anti_outer_join.py ===")
    p = Path("tests/test_anti_outer_join.py")
    s = p.read_text()
    if "test_anti_planner_how_mapped" in s:
        print("  already present — skip")
        return
    extra = '''

# ============ v0.9.23i-fix2: planner how mapping ============

def test_anti_planner_how_mapped():
    """AST: merge() maps anti -> 'left' for planner (not 'anti')."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.merge)
    assert '_planner_how = (' in src, (
        "merge must compute _planner_how before plan() call"
    )
    # Right next to plan() call (avoid false positives from other code).
    idx_plan = src.find("self._memory_guard.plan(")
    idx_map = src.find('_planner_how = (')
    assert idx_map >= 0 and idx_map < idx_plan, (
        "_planner_how must be computed before plan() call"
    )


def test_anti_whole_op_warning_skip():
    """AST: predicted_output=0 for anti/semi (output ≤ left_n)."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.merge)
    assert (
        'if how in ("anti", "semi"):' in src
        and "predicted_output = 0" in src
    ), "anti/semi should skip predicted_output (per-batch guard sufficient)"


def test_no_semi_join_streaming_import_yet():
    """v0.9.23i-fix2: semi_join_streaming import removed (v0.9.23j)."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.merge)
    assert "semi_join_streaming" not in src, (
        "semi_join_streaming doesn't exist yet (v0.9.23j); "
        "remove from dispatch import"
    )
'''

    p.write_text(s + extra)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 5. Dogfood ============

def dogfood():
    print("\n=== dogfood: anti + outer via merge (planner fixed) ===")

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

        # (a) Anti via merge
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            left_t = pa_.table({
                "k": [1, 2, 3, 4, 5],
                "a": ["w", "x", "y", "z", "u"],
            })
            right_t = pa_.table({"k": [2, 4], "b": ["p", "q"]})
            src_l = td / "L.parquet"
            src_r = td / "R.parquet"
            pq_.write_table(left_t, src_l)
            pq_.write_table(right_t, src_r)
            left = _CDF.from_parquet(
                src_l, td / "dsL", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spillL",
            )
            right = _CDF.from_parquet(
                src_r, td / "dsR", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spillR",
            )
            anti = left.merge(right, on="k", how="anti")
            anti_rows = _rows(anti)
            right_keys = set(right_t.column("k").to_pylist())
            left_keys = set(left_t.column("k").to_pylist())
            for r in anti_rows:
                assert r["k"] not in right_keys
            assert {r["k"] for r in anti_rows} == left_keys - right_keys

        # (b) Outer via merge
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            left_t = pa_.table({
                "k": [1, 2, 3], "a": ["w", "x", "y"],
            })
            right_t = pa_.table({
                "k": [2, 3, 4], "b": ["p", "q", "r"],
            })
            src_l = td / "L.parquet"
            src_r = td / "R.parquet"
            pq_.write_table(left_t, src_l)
            pq_.write_table(right_t, src_r)
            left = _CDF.from_parquet(
                src_l, td / "dsL", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spillL",
            )
            right = _CDF.from_parquet(
                src_r, td / "dsR", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spillR",
            )
            outer = left.merge(right, on="k", how="outer")
            rows = _rows(outer)
            left_keys = set(left_t.column("k").to_pylist())
            right_keys = set(right_t.column("k").to_pylist())
            got_keys = {r["k"] for r in rows if r["k"] is not None}
            assert got_keys == (left_keys | right_keys)
            by_k = {r["k"]: r for r in rows if r["k"] is not None}
            for k in left_keys - right_keys:
                assert by_k[k]["b"] is None
            for k in right_keys - left_keys:
                assert by_k[k]["a"] is None

        # (c) Semi raises (v0.9.23j placeholder)
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            left_t = pa_.table({"k": [1, 2]})
            right_t = pa_.table({"k": [1]})
            src_l = td / "L.parquet"
            src_r = td / "R.parquet"
            pq_.write_table(left_t, src_l)
            pq_.write_table(right_t, src_r)
            left = _CDF.from_parquet(
                src_l, td / "dsL", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spillL",
            )
            right = _CDF.from_parquet(
                src_r, td / "dsR", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spillR",
            )
            try:
                left.merge(right, on="k", how="semi")
            except NotImplementedError as e:
                assert "semi" in str(e)
            else:
                raise AssertionError("semi must raise NotImplementedError")

        # (d) Inner/left unaffected (regression)
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            left_t = pa_.table({"k": [1, 2, 3], "a": ["w", "x", "y"]})
            right_t = pa_.table({"k": [1, 3], "b": ["p", "q"]})
            src_l = td / "L.parquet"
            src_r = td / "R.parquet"
            pq_.write_table(left_t, src_l)
            pq_.write_table(right_t, src_r)
            left = _CDF.from_parquet(
                src_l, td / "dsL", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spillL",
            )
            right = _CDF.from_parquet(
                src_r, td / "dsR", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spillR",
            )
            assert left.merge(right, on="k").count() == 2  # inner
            assert left.merge(right, on="k", how="left").count() == 3

        print(
            "  anti + outer (planner fixed) + semi raises + "
            "inner/left regression ok"
        )

    run_verify_in_subprocess(_verify)


# ============ 6. Tests ============

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


# ============ 7. main ============

def main():
    print("=== v0.9.23i-fix2: planner how mapping ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    fix_planner_how()
    fix_predicted_output()
    fix_import()
    add_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23i: anti join + outer join (AUDIT_FULL_v0922 item "
        "7.A.10, split 1/3, Top-10 gap #4). "
        "anti_join_streaming(left, right, on): left rows whose key "
        "absent from right. HashShuffle + _safe_materialize_partition. "
        "NULL semantics: null left key never matches -> emitted "
        "(SQL semantics); null right key never matches -> doesn't "
        "exclude any left. fill_null(False) on is_in mask. "
        "outer_join_streaming: via merge dispatch (re-iterable right "
        "needed). Composition: left_join + anti_join(right, left) "
        "padded with null left; concat via IPC writer. "
        "merge(how=...) extended: inner | left | outer | anti | semi "
        "(semi -> NotImplementedError in v0.9.23i, v0.9.23j). "
        "'right' still NotImplementedError (v0.9.23j). "
        "Export anti_join_streaming, outer_join_streaming from "
        "cadw.ops (public API; not from cadw top-level — "
        "cadw/__init__.py has no `from .ops.join import ...` line, "
        "pre-existing project state; documented as known issue, not "
        "touched here). "
        "fix1: cadw/__init__.py anchor miss recovery. Fix: only "
        "extend cadw/ops/__init__.py (safe single-line anchor); "
        "cadw/__init__.py untouched — anti/outer via cadw.ops.join "
        "or `from cadw.ops import ...`; dataframe.py and tests use "
        "cadw.ops.join path directly. "
        "fix2 (dogfood-caught): merge() passed `how` verbatim into "
        "JoinMemoryModel.solve -> ValueError for anti/outer/semi "
        "(model supports only 'inner'|'left'). Fix: _planner_how map "
        "(anti/semi/outer -> 'left'; planner build=right, probe=left; "
        "same class). Predicted_output per how: anti/semi -> 0 "
        "(output ≤ left_n, per-batch guard sufficient); outer -> "
        "left_join + right_n; inner/left unchanged. "
        "Also removes non-existent semi_join_streaming from dispatch "
        "import (lazy inside merge, never triggered before planner "
        "failure — removed proactively; v0.9.23j re-adds). "
        "+tests/test_anti_outer_join.py (22 tests: anti basic, "
        "all-match, none-match, empty-right, null-left, multi-key; "
        "outer basic, all-match, disjoint, null-left, empty-left, "
        "empty-right, multi-key; unsupported how raises, semi "
        "NotImplementedError; 7 AST — hash_shuffle, is_in, fill_null "
        "(False), merge dispatch, planner how map, warning skip for "
        "anti/semi, no semi_join_streaming import yet; 1 export "
        "test). "
        "Dogfood invariants only (no distributions, v0.9.23h lesson): "
        "anti not-in + set; outer union + padding; null-left anti "
        "emitted; semi raises; inner/left regression. "
        "Split rationale (v0.9.23h rule >200 LOC): anti+outer ~150 "
        "LOC (this patch), semi ~40 LOC (v0.9.23j), categorical FE "
        "3 patches (v0.9.23k-l). "
        "Fast: 2125 -> ~2144 passed.",
        self_path="scripts/patch_v0923i_fix2_planner_how.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23i-fix2 applied. 7.A.10 partial (anti + outer).")


if __name__ == "__main__":
    main()