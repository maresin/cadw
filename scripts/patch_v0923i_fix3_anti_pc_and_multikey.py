#!/usr/bin/env python3
# v0.9.23i-fix3 — anti_join: _pc import + multi-column correctness.
#
# Bug 1 (dogfood-caught): NameError '_pc' not defined in
# _process_partition_pair_anti. Missing `import pyarrow.compute as _pc`.
#
# Bug 2 (silent data loss, HIGH): multi-column anti was WRONG.
# Prior code: matched = AND_c is_in(probe[c], right_set[c]).
# This is true iff EACH left-column value appears SOMEWHERE in the
# corresponding right column — NOT "a right row matches on ALL cols".
# Example: left=[(1,"a")], right=[(1,"b"),(2,"a")].
#   is_in("k1",[1,2])=True, is_in("k2",["b","a"])=True, AND=True
#   -> "matched" -> NOT emitted. But no right row == (1,"a") ->
#   SHOULD emit. Silent data loss in anti (multi-key).
# Fix: composite key via _combine_join_keys (already in join.py);
# single-column path unchanged (correct); multi-column path uses
# is_in on combined string keys.
#
# Bug 3 (minor dead code): filtered.to_batches() if isinstance(
# filtered, pa.Table) else [filtered] — probe_batch.filter() returns
# RecordBatch always; .to_batches() doesn't exist. Dead branch.
# Simplify to single batch emission.
#
# Also: dogfood uses mem_mb=64 (not 20) to avoid warning spam from
# _JOIN_FIXED_OVERHEAD_BYTES (40 MB) > state_limit (10 MB) for
# anti/semi (existing §9.2.6 behavior, not a bug).
#
# +tests/test_anti_outer_join.py: 4 new multi-key tests
# (test_anti_multi_key_disjoint_per_col, test_anti_multi_key_crossed,
# test_outer_multi_key_crossed) — specifically catch Bug 2 class:
# left-tuple appears in cross-product of right-columns but not as a
# right row.
#
# Run: uv run python scripts/patch_v0923i_fix3_anti_pc_and_multikey.py

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


# ============ 1. Replace _process_partition_pair_anti ============

OLD_BODY = '''    # Right fits: build key set (drop null right keys; nulls match nothing).
    right_table = _safe_materialize_partition(right_path, memory_guard)
    if right_table is None:
        raise MemoryLimitExceeded(
            f"anti_join: build partition {right_path.name} still too "
            f"large after stat check."
        )
    right_table = _drop_null_keys_arrow(right_table, list(on))
    right_keys = right_table.select(on).to_batches()
    # Build value_set per column: single-column on -> pa.Array;
    # multi-column -> set of tuple via binary_join. For simplicity,
    # use per-column is_in conjunction (works for both single and
    # multi).
    right_arrays = [
        right_table.column(c) for c in on
    ]

    out_schema = pa.schema([
        left_schema.field(c) for c in left_schema.names
    ])

    for probe_batch in _iter_probe_batches(left_path):
        memory_guard.check_batch(
            probe_batch, name=f"anti_probe({left_path.name})"
        )
        # Row-wise: left key matched iff ALL on-columns are in
        # corresponding right column (null left -> is_in returns null
        # -> treat as not matched, emit).
        matched = None
        for c, r_arr in zip(on, right_arrays):
            m = _pc.is_in(probe_batch.column(c), value_set=r_arr)
            # fill_null(True) would drop null left -> wrong for anti.
            # fill_null(False) -> null treated as "not in" -> emit.
            m = _pc.fill_null(m, pa.scalar(False, type=pa.bool_()))
            matched = m if matched is None else _pc.and_(matched, m)
        if matched is None:
            # No on columns (unreachable — on validated non-empty)
            continue
        unmatched = _pc.invert(matched)
        filtered = probe_batch.filter(unmatched)
        for b in filtered.to_batches() if isinstance(
            filtered, pa.Table
        ) else [filtered]:
            if b.num_rows > 0:
                yield b'''

NEW_BODY = '''    import pyarrow.compute as _pc  # v0.9.23i-fix3: was missing

    # Right fits: build key set (drop null right keys; nulls match nothing).
    right_table = _safe_materialize_partition(right_path, memory_guard)
    if right_table is None:
        raise MemoryLimitExceeded(
            f"anti_join: build partition {right_path.name} still too "
            f"large after stat check."
        )
    right_table = _drop_null_keys_arrow(right_table, list(on))

    out_schema = pa.schema([
        left_schema.field(c) for c in left_schema.names
    ])

    for probe_batch in _iter_probe_batches(left_path):
        memory_guard.check_batch(
            probe_batch, name=f"anti_probe({left_path.name})"
        )
        # v0.9.23i-fix3: composite key for multi-column (prior code
        # used AND of per-column is_in — WRONG: true iff each left
        # value appears in its right column separately, not "a right
        # row matches all cols"). Single-column: direct is_in.
        if len(on) == 1:
            m = _pc.is_in(
                probe_batch.column(on[0]),
                value_set=right_table.column(on[0]).combine_chunks(),
            )
        else:
            # Composite: binary_join probe + right into single key.
            probe_tbl = pa.Table.from_batches([probe_batch])
            probe_key = _combine_join_keys(probe_tbl, list(on))
            right_key = _combine_join_keys(right_table, list(on))
            m = _pc.is_in(probe_key, value_set=right_key)
        # fill_null(True) would silently drop null left (matched) —
        # anti should EMIT null-key left rows (SQL semantics: null
        # never matches). fill_null(False) -> null -> unmatched -> emit.
        m = _pc.fill_null(m, pa.scalar(False, type=pa.bool_()))
        unmatched = _pc.invert(m)
        filtered = probe_batch.filter(unmatched)
        if filtered.num_rows > 0:
            yield filtered'''


def fix_anti_body():
    print("\n=== fix _process_partition_pair_anti (import + multikey) ===")
    p = Path("cadw/ops/join.py")
    applied = apply_or_fail(
        str(p), OLD_BODY, NEW_BODY, label="anti_body_v2",
    )
    if applied:
        assert_compiles(p)
        print("  applied")
    else:
        print("  already fixed — skip")


# ============ 2. Strengthen tests: multi-column crossed tuples ============

def add_multikey_tests():
    print("\n=== add multi-column crossed-tuple tests ===")
    p = Path("tests/test_anti_outer_join.py")
    s = p.read_text()
    if "test_anti_multi_key_crossed" in s:
        print("  already present — skip")
        return
    extra = '''

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
    """
    left = make_df(tmp_path, {
        "k1": [1], "k2": ["a"], "v": [100],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1, 2], "k2": ["b", "a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="anti")
    rows = _rows(result)
    # (1, "a") crossed: (1,*) yes, (*, "a") yes, but no (1, "a") row.
    # Anti must EMIT.
    assert len(rows) == 1, (
        f"multi-key anti cross-product bug: expected 1 row, got {rows}"
    )
    assert rows[0]["k1"] == 1 and rows[0]["k2"] == "a"


def test_anti_multi_key_match_emitted_not(tmp_path):
    """left-tuple IS a right row -> not emitted."""
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
    # Outer = 1 + 2 = 3 rows total, all padded.
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
    p.write_text(s + extra)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 3. Dogfood with bigger budget (avoid warning spam) ============

def dogfood():
    print("\n=== dogfood: multi-key crossed anti + outer ===")

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

        # (a) Single-column anti (with null)
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

        # (b) MULTI-KEY CROSSED (Bug 2 regression)
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
            # (1, "a") is NOT a right row -> must EMIT despite being in
            # cross-product (bug: prior AND-of-is_in excluded it)
            assert len(rows) == 1, (
                f"crossed-tuple anti bug: expected 1 row, got {rows}"
            )
            assert rows[0]["k1"] == 1 and rows[0]["k2"] == "a"
            assert rows[0]["v"] == 100

        # (c) Multi-key with actual match
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

        # (d) Outer multi-key crossed
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
            # 1 (left) + 2 (anti_right) = 3 rows total
            assert len(rows) == 3
            by_key = {(r["k1"], r["k2"]): r for r in rows}
            assert by_key[(1, "a")]["v"] == 100
            assert by_key[(1, "a")]["w"] is None
            assert by_key[(1, "b")]["v"] is None
            assert by_key[(1, "b")]["w"] == 200
            assert by_key[(2, "a")]["v"] is None
            assert by_key[(2, "a")]["w"] == 300

        print(
            "  single-col anti + MULTI-KEY crossed (bug fixed) + "
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
    print("=== v0.9.23i-fix3: anti _pc + multikey ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    fix_anti_body()
    write_extra_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23i: anti join + outer join (AUDIT_FULL_v0922 item "
        "7.A.10, split 1/3, Top-10 gap #4). "
        "anti_join_streaming(left, right, on): left rows whose key "
        "absent from right. HashShuffle + _safe_materialize_partition. "
        "NULL semantics: null left key never matches -> emitted; null "
        "right key never matches -> doesn't exclude any left. "
        "outer_join_streaming: via merge dispatch (left_join + "
        "anti_join(right, left) padded + concat IPC). "
        "merge(how=...) extended: inner | left | outer | anti | semi "
        "(semi -> NotImplementedError, v0.9.23j). "
        "Export anti_join_streaming, outer_join_streaming from "
        "cadw.ops. "
        "fix1: cadw/__init__.py anchor miss recovery; extend only "
        "cadw/ops/__init__.py. "
        "fix2 (dogfood-caught): merge() passed how verbatim to "
        "JoinMemoryModel.solve -> ValueError for anti/outer/semi; "
        "fix: _planner_how map (-> 'left'). predicted_output per "
        "how: anti/semi -> 0, outer -> left+right_n, inner/left "
        "unchanged. Remove non-existent semi_join_streaming import. "
        "fix3 (dogfood-caught 1 production bug + 1 SILENT DATA LOSS): "
        "(1) NameError '_pc' not defined — missing `import "
        "pyarrow.compute as _pc` in _process_partition_pair_anti. "
        "(2) SILENT multi-column anti bug: prior `matched = AND_c "
        "is_in(probe[c], right_set[c])` is true iff each left value "
        "appears SOMEWHERE in corresponding right column, NOT 'a "
        "right row matches on ALL cols'. left=(1,'a'), right={(1,"
        "'b'),(2,'a')}: cross-product match -> excluded, but no "
        "right row == (1,'a') -> SHOULD emit. Fix: composite key "
        "via _combine_join_keys -> is_in. Single-column unchanged "
        "(correct). "
        "(3) Dead code removed: filtered.to_batches() if isinstance("
        "filtered, pa.Table) — probe_batch.filter() returns "
        "RecordBatch, no .to_batches(). "
        "+tests/test_anti_outer_join.py (26 tests: anti basic, "
        "all-match, none-match, empty-right, null-left, multi-key; "
        "outer basic, all-match, disjoint, null-left, empty-left, "
        "empty-right, multi-key; **4 NEW**: anti-multi-key-crossed "
        "(catches Bug 2), anti-multi-key-actual-match, outer-"
        "multi-key-crossed, anti-multi-key-null; unsupported how "
        "raises, semi NotImplementedError; 7 AST — hash_shuffle, "
        "is_in, fill_null(False), merge dispatch, planner how map, "
        "warning skip, no semi_join_streaming import; export test). "
        "Dogfood invariants only (v0.9.23h lesson): anti not-in + "
        "set; outer union + padding; null-left anti emitted; semi "
        "raises; inner/left regression; **multi-key crossed bug "
        "fixed**. "
        "Split rationale (v0.9.23h rule >200 LOC): anti+outer ~150 "
        "LOC (this patch), semi ~40 LOC (v0.9.23j), categorical FE "
        "3 patches (v0.9.23k-l). "
        "Process note: 3rd fix-cycle on v0.9.23i (over §13.4/238 "
        "threshold). 2 production bugs (planner-how, multikey anti) "
        "+ 1 hidden NameError + 1 dead code + 1 anchor miss. "
        "Structural lesson: multi-key join/anti/outer MUST use "
        "composite key, never AND-of-per-column predicates "
        "(cross-product false positive). Test design: crossed-tuple "
        "case is the canonical detector. "
        "Fast: 2125 -> ~2150 passed.",
        self_path="scripts/patch_v0923i_fix3_anti_pc_and_multikey.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23i-fix3 applied. 7.A.10 partial (anti + outer).")


if __name__ == "__main__":
    main()