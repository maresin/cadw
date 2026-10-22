#!/usr/bin/env python3
# v0.9.23g-fix1 — 2 production bugs in v0.9.23g:
#
# (1) isna() — "Tried to write record batch with different schema".
#     Root cause: out_fields used nullable=False, but
#     Dataset.from_batches reconstructs per-column batches via
#     pa.RecordBatch.from_arrays([col], names=[field_name]) which
#     defaults to nullable=True. Schema mismatch at write_batch.
#     Violates §13.4/169 (all output fields nullable=True).
#     Fix: nullable=False -> nullable=True.
#
# (2) nunique() — WRONG result on multi-batch data. Sum of per-batch
#     count_distinct = sum of per-batch uniques, NOT global nunique.
#     Example: values [a, a] split into batch1=[a], batch2=[a]:
#     per-batch count_distinct = 1 + 1 = 2, but global nunique = 1.
#     SILENT DATA CORRUPTION. Dogfood did not catch this because
#     chunk=2 and test values do not overlap across batches.
#     Fix: use groupby_aggregate (HashShuffle + count) — the
#     correct path for multi-batch. Number of rows in output = K =
#     global unique count. Slower (HashShuffle per column) but
#     correct. Same pattern as value_counts.
#
# Added regression tests: isna_multibatch (chunk=1), nunique_multibatch
# (chunk=1). These would have caught both bugs. Structural lesson:
# multi-batch tests must use chunk=1 (worst case) to exercise the
# per-batch boundary.
#
# Run: uv run python scripts/patch_v0923g_fix1_isna_nunique.py

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


# ============ 1. isna(): nullable=False -> nullable=True ============

def fix_isna_nullable():
    print("\n=== fix isna() nullable=False -> True ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if (
        'out_fields = [\n'
        '            pa.field(c, pa.bool_(), nullable=True) for c in self._columns\n'
        '        ]'
    ) in s:
        print("  already fixed — skip")
        return
    apply_or_fail(
        str(p),
        old=(
            '        out_fields = [\n'
            '            pa.field(c, pa.bool_(), nullable=False) for c in self._columns\n'
            '        ]'
        ),
        new=(
            '        # v0.9.23g-fix1 (§13.4/169): nullable=True. Dataset.\n'
            '        # from_batches reconstructs per-column batches via\n'
            '        # RecordBatch.from_arrays([col]) which always sets\n'
            '        # nullable=True. nullable=False here would trigger\n'
            '        # ArrowInvalid: Tried to write record batch with\n'
            '        # different schema at write_batch.\n'
            '        out_fields = [\n'
            '            pa.field(c, pa.bool_(), nullable=True) for c in self._columns\n'
            '        ]'
        ),
        label="isna_nullable",
    )
    assert_compiles(p)
    print("  OK")


# ============ 2. nunique(): rewrite via groupby_aggregate ============

OLD_NUNIQUE = '''        import pyarrow.compute as _pc

        if column is not None:
            if column not in self._columns:
                raise KeyError(
                    f"nunique: column {column!r} not in {self._columns}"
                )
            total = 0
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(
                    b, name=f"nunique[{column}]",
                )
                total += int(_pc.count_distinct(
                    b.column(column),
                ).as_py() or 0)
            return total

        # All columns: per-column count_distinct passes.
        out: dict = {}
        for c in self._columns:
            total = 0
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(
                    b, name=f"nunique[{c}]",
                )
                total += int(_pc.count_distinct(b.column(c)).as_py() or 0)
            out[c] = total
        return out'''

NEW_NUNIQUE = '''        from cadw.ops.groupby import groupby_aggregate as _gba

        # v0.9.23g-fix1: global nunique must use groupby_aggregate
        # (HashShuffle + count). Prior implementation summed per-batch
        # count_distinct -> WRONG on multi-batch (SILENT DATA
        # CORRUPTION: sum of per-batch uniques != global unique).
        # Example: [a, a] -> batch1=[a], batch2=[a] -> 1+1=2, actual=1.
        def _nunique_one(col_name: str) -> int:
            # groupby_aggregate over [col] → one row per unique value.
            # Number of output rows = global unique count.
            result = _gba(
                self._iter_filtered_batches(),
                [col_name],
                [col_name],
                ["count"],
                self._memory_guard,
                self._spill_dir / f"nunique_{col_name}",
                log=False,
            )
            return result.num_rows

        if column is not None:
            if column not in self._columns:
                raise KeyError(
                    f"nunique: column {column!r} not in {self._columns}"
                )
            return _nunique_one(column)

        # All columns: per-column groupby_aggregate.
        out: dict = {}
        for c in self._columns:
            out[c] = _nunique_one(c)
        return out'''

def fix_nunique_multibatch():
    print("\n=== fix nunique() multi-batch bug ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "v0.9.23g-fix1: global nunique must use groupby_aggregate" in s:
        print("  already fixed — skip")
        return
    if OLD_NUNIQUE not in s:
        raise SystemExit("nunique body anchor not found")
    s = s.replace(OLD_NUNIQUE, NEW_NUNIQUE, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 3. Regression tests ============

def add_regression_tests():
    print("\n=== add multibatch regression tests ===")
    p = Path("tests/test_eda_basics.py")
    s = p.read_text()
    if "test_nunique_multibatch_overlap" in s:
        print("  already present — skip")
        return

    extra = '''

# ============ v0.9.23g-fix1: multi-batch regression ============

def test_isna_multibatch(tmp_path):
    """isna works when chunk=1 (schema nullable=True per §13.4/169).

    Prior v0.9.23g had nullable=False in out_fields, which crashed
    with ArrowInvalid on write_batch (Dataset.from_batches
    reconstructs columns with nullable=True).
    """
    df = make_df(tmp_path, {
        "a": pa.array([1, None, 3, None], type=pa.int64()),
    }, chunk=1)  # 4 batches of 1 row
    out = df.isna()
    t = tmp_path / "o.parquet"
    out.write_parquet(t)
    assert pq.read_table(t).column("a").to_pylist() == [
        False, True, False, True,
    ]


def test_nunique_multibatch_overlap(tmp_path):
    """SILENT DATA CORRUPTION regression: multi-batch overlap.

    Prior v0.9.23g summed per-batch count_distinct. Values [a, a]
    with chunk=1: batch1=[a] -> count_distinct=1, batch2=[a] ->
    count_distinct=1, sum=2. But global nunique=1.

    Fix: groupby_aggregate (HashShuffle + count) — always correct.
    """
    df = make_df(tmp_path, {
        "k": ["a", "a", "a", "a"],
    }, chunk=1)  # 4 batches, all "a"
    assert df.nunique("k") == 1


def test_nunique_multibatch_partial_overlap(tmp_path):
    """Values spread across batches with partial overlap."""
    df = make_df(tmp_path, {
        "k": ["a", "b", "a", "c", "b", "d"],
    }, chunk=2)  # batches: [a,b], [a,c], [b,d]
    # Per-batch sum would be 2+2+2=6, actual=4 (a,b,c,d)
    assert df.nunique("k") == 4


def test_nunique_all_columns_multibatch(tmp_path):
    """All-columns nunique with chunk=1."""
    df = make_df(tmp_path, {
        "a": ["x", "x", "y"],
        "b": [1, 1, 1],
    }, chunk=1)
    assert df.nunique() == {"a": 2, "b": 1}


def test_value_counts_multibatch_overlap(tmp_path):
    """value_counts is already correct (HashShuffle), but pin it."""
    df = make_df(tmp_path, {
        "k": ["a", "a", "b"],
    }, chunk=1)
    out = df.value_counts("k")
    rows = {(r["k"], r["count"]) for r in out.to_pylist()}
    assert rows == {("a", 2), ("b", 1)}
'''

    p.write_text(s + extra)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 4. Dogfood ============

def dogfood():
    print("\n=== dogfood: isna + nunique multi-batch overlap ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG

        # (a) isna with chunk=1 (the crash case)
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "a": pa_.array([1, None, 3, None], type=pa_.int64()),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            out = df.isna()
            o = td / "o.parquet"
            out.write_parquet(o)
            got = pq_.read_table(o).column("a").to_pylist()
            assert got == [False, True, False, True], got

        # (b) nunique MULTI-BATCH OVERLAP (the silent corruption case)
        # All values "a", 4 batches of 1
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({"k": ["a", "a", "a", "a"]})
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            assert df.nunique("k") == 1, (
                f"nunique single value across 4 batches must be 1, "
                f"got {df.nunique('k')}"
            )

        # (c) nunique multi-batch partial overlap
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({"k": ["a", "b", "a", "c", "b", "d"]})
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            # Actual unique: a,b,c,d = 4. Prior bug would give 6.
            assert df.nunique("k") == 4, (
                f"nunique must be 4 (a,b,c,d), got {df.nunique('k')}"
            )

        # (d) sanity: other methods still work
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "k": ["a", "a", "b"],
                "v": pa_.array([1, None, 3], type=pa_.int64()),
            })
            src = td / "s.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            assert df.isna_sum() == {"k": 0, "v": 1}
            assert df.nunique("k") == 2
            vc = df.value_counts("k")
            rows = {(r["k"], r["count"]) for r in vc.to_pylist()}
            assert rows == {("a", 2), ("b", 1)}
            filled = df.fillna(0, columns=["v"])
            o = td / "f.parquet"
            filled.write_parquet(o)
            assert pq_.read_table(o).column("v").to_pylist() == [1, 0, 3]

        print(
            "  isna chunk=1 + nunique overlap (4 batches) + partial "
            "overlap + sanity ok"
        )

    run_verify_in_subprocess(_verify)


# ============ 5. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_eda_basics.py",
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
    print("=== v0.9.23g-fix1: isna nullable + nunique multi-batch ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    fix_isna_nullable()
    fix_nunique_multibatch()
    add_regression_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23g: EDA basics — isna / isna_sum / fillna / "
        "value_counts / nunique (AUDIT_FULL_v0922 items 7.A.5 + "
        "7.A.3; closes row 9 + Top-10 gap #7). "
        "Five streaming methods: isna (bool mask, pc.is_null); "
        "isna_sum (per-col CountOp via pc.is_null+sum); "
        "fillna (per-batch pc.fill_null, fill scalar pre-constructed "
        "at entry -> fail-fast on type mismatch); value_counts "
        "(HashShuffle+groupby_aggregate, sort=True via in-memory "
        "sort_indices on K rows, nulls own group); nunique "
        "(groupby_aggregate per column -> K = global unique count). "
        "fix1 (2 production bugs): "
        "(1) isna() CRASH: ArrowInvalid 'different schema' at "
        "write_batch. out_fields used nullable=False, but "
        "Dataset.from_batches reconstructs per-column via "
        "RecordBatch.from_arrays([col]) which is always "
        "nullable=True (§13.4/169). Fix: nullable=True. "
        "(2) nunique() SILENT DATA CORRUPTION: summed per-batch "
        "count_distinct (NOT global nunique). [a,a] chunk=1 -> "
        "1+1=2, actual=1. Dogfood didn't catch because chunk=2 on "
        "non-overlapping test data. Fix: groupby_aggregate "
        "(HashShuffle + count) per column -> correct for any batch "
        "split. Slower (HashShuffle per col) but correct. "
        "+tests/test_eda_basics.py (31 tests total: 5 new regression "
        "- isna_multibatch chunk=1, nunique_multibatch_overlap "
        "all-a/chunk=1, nunique_multibatch_partial_overlap "
        "chunk=2, nunique_all_columns_multibatch, "
        "value_counts_multibatch_overlap). "
        "Structural lesson: multi-batch tests MUST use chunk=1 "
        "(worst case) so per-batch boundary is exercised. "
        "Fast: 2051 -> ~2082 passed.",
        self_path="scripts/patch_v0923g_fix1_isna_nunique.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23g-fix1 applied. v0.9.23g complete (7.A.5 + 7.A.3).")


if __name__ == "__main__":
    main()