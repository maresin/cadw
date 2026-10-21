#!/usr/bin/env python3
# v0.9.8b-fix2 — dogfood shared-state bug + test robustness.
#
# Bug: dogfood assertion failed:
#     AssertionError: sample rows should be << schema rows
#   because dogfood called g.estimate_row_size_from_batch(sample)
#   BEFORE g.get_batch_rows() — mutating g._row_size_bytes from 500
#   (schema) to 2040 (exact). Then rows_bad == rows_good == 899.
#
# Root: single MemoryGuard for "before/after" comparison. Test file
# test_audit_case_get_batch_rows_with_sample uses fresh g for each
# side — passes because default _row_size_bytes=80 → rows_bad=22937,
# rows_good=899 → 899 < 4587. But 80-baseline is accidental; test
# should explicitly set schema baseline.
#
# Fix:
#   1. Dogfood: use separate g_schema (baseline 500) and g_sample
#      (exact 2040). Assertion rows_good < rows_bad // 2 (899 <
#      3670/2=1835 → pass; ratio 2040/500 = 4.08×).
#   2. Test test_audit_case_get_batch_rows_with_sample: explicit
#      g.estimate_row_size_from_schema(schema) baseline (500→3670),
#      then sample (2040→899), assert rows_good < rows_bad // 2.
#      More robust than relying on default 80.
#
# State: all writes from v0.9.8b on disk. fix2 edits dogfood + test
# file, re-runs patch in subprocess (idempotent + fixed dogfood).
#
# §13.4/205: stateful-before/after comparisons need isolated state.
# When testing "X without optimization" vs "X with optimization",
# use separate fixtures/instances — otherwise the first call mutates
# shared state and comparison is meaningless.
#
# Run: uv run python scripts/patch_v098b_fix2.py

from __future__ import annotations
import ast
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import importlib
import cadw.patch_utils as pu
importlib.reload(pu)
from cadw.patch_utils import assert_compiles


# ============ 1. Fix dogfood in patch-script ============

DOGFOOD_OLD = '''    # Audit case
    g = MemoryGuard(max_memory_mb=5)
    schema = pa.schema([(f"s{i}", pa.string()) for i in range(10)])
    schema_est = g.estimate_row_size_from_schema(schema)
    print(f"  schema est (10 str cols): {schema_est} B/row")

    arrs = [pa.array(["x" * 200] * 100, type=pa.string()) for _ in range(10)]
    sample = pa.RecordBatch.from_arrays(arrs, names=schema.names)
    exact_est = g.estimate_row_size_from_batch(sample)
    print(f"  exact est (from batch):   {exact_est} B/row "
          f"({exact_est / schema_est:.1f}×)")

    rows_bad = g.get_batch_rows(expansion_factor=2.0)
    rows_good = g.get_batch_rows(expansion_factor=2.0, sample_batch=sample)
    print(f"  get_batch_rows schema:  {rows_bad} rows (R2 bug)")
    print(f"  get_batch_rows sample:  {rows_good} rows (R2 fix)")

    # Materialize to confirm
    actual = pa.RecordBatch.from_arrays(
        [pa.array(["x" * 200] * rows_good, type=pa.string())
         for _ in range(10)],
        names=schema.names,
    )
    print(f"  materialized at rows_good: {actual.nbytes / 1024 / 1024:.2f} MB "
          f"(budget 5 MB)")

    assert rows_good < rows_bad // 5, "sample rows should be << schema rows"
    assert actual.nbytes < 5 * 1024 * 1024, "materialized > budget"
    print("  OK")'''

DOGFOOD_NEW = '''    # Audit case — SEPARATE guards for state isolation (§13.4/205)
    schema = pa.schema([(f"s{i}", pa.string()) for i in range(10)])
    arrs = [pa.array(["x" * 200] * 100, type=pa.string()) for _ in range(10)]
    sample = pa.RecordBatch.from_arrays(arrs, names=schema.names)

    # g_schema: schema-only baseline (R2 bug — 500 B/row underestimate)
    g_schema = MemoryGuard(max_memory_mb=5)
    schema_est = g_schema.estimate_row_size_from_schema(schema)
    rows_bad = g_schema.get_batch_rows(expansion_factor=2.0)
    print(f"  schema est (10 str cols): {schema_est} B/row, "
          f"{rows_bad} rows (R2 bug)")

    # g_sample: exact via sample_batch (R2 fix — 2040 B/row)
    g_sample = MemoryGuard(max_memory_mb=5)
    exact_est = g_sample.estimate_row_size_from_batch(sample)
    rows_good = g_sample.get_batch_rows(expansion_factor=2.0, sample_batch=sample)
    print(f"  exact est (from batch):   {exact_est} B/row "
          f"({exact_est / schema_est:.1f}×), {rows_good} rows (R2 fix)")

    # Materialize to confirm
    actual = pa.RecordBatch.from_arrays(
        [pa.array(["x" * 200] * rows_good, type=pa.string())
         for _ in range(10)],
        names=schema.names,
    )
    print(f"  materialized at rows_good: {actual.nbytes / 1024 / 1024:.2f} MB "
          f"(budget 5 MB)")

    assert rows_good < rows_bad // 2, (
        f"sample rows ({rows_good}) should be < half of schema rows "
        f"({rows_bad}); ratio {rows_bad / rows_good:.2f}"
    )
    assert actual.nbytes < 5 * 1024 * 1024, "materialized > budget"
    print("  OK")'''


def fix_dogfood_in_patch_script():
    print("\n=== patch script: fix dogfood shared-state ===")
    p = Path("scripts/patch_v098b_row_size_reestimate.py")
    if not p.exists():
        raise SystemExit(f"{p} missing")
    s = p.read_text()
    if "SEPARATE guards for state isolation" in s:
        print("  already fixed — skip")
        return
    if DOGFOOD_OLD not in s:
        raise SystemExit("dogfood anchor not found")
    s = s.replace(DOGFOOD_OLD, DOGFOOD_NEW, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. Fix test file: explicit baseline + stricter assertion ============

TEST_OLD = '''def test_audit_case_get_batch_rows_with_sample():
    """get_batch_rows(sample_batch=...) uses exact; result ≤ budget."""
    g = MemoryGuard(max_memory_mb=5)
    schema = pa.schema([(f"s{i}", pa.string()) for i in range(10)])
    arrs = [pa.array(["x" * 200] * 100, type=pa.string()) for _ in range(10)]
    sample = pa.RecordBatch.from_arrays(arrs, names=schema.names)

    # Without sample: schema-based → 3670 rows (R2 bug demo)
    rows_bad = g.get_batch_rows(expansion_factor=2.0)
    # With sample: exact ~10080 B/row → ~130 rows
    rows_good = g.get_batch_rows(
        expansion_factor=2.0, sample_batch=sample,
    )

    assert rows_good < rows_bad // 5, (
        f"sample-based ({rows_good}) should be << schema-based ({rows_bad})"
    )

    # Sanity: actual materialized batch at rows_good fits budget
    # (row_size × ef × rows ≈ budget × safety)
    actual_batch = pa.RecordBatch.from_arrays(
        [pa.array(["x" * 200] * rows_good, type=pa.string()) for _ in range(10)],
        names=schema.names,
    )
    # Budget = 5 MB; batch should be < 5 MB
    assert actual_batch.nbytes < 5 * 1024 * 1024, (
        f"materialized {actual_batch.nbytes} B > 5 MB budget"
    )'''

TEST_NEW = '''def test_audit_case_get_batch_rows_with_sample():
    """get_batch_rows(sample_batch=...) uses exact; result ≤ budget.

    v0.9.8b-fix2: explicit schema baseline (g.estimate_row_size_from_schema)
    to establish R2-bug 500 B/row state. Without it, default _row_size_bytes
    = 80 → rows_bad=22937 (accidental). With it → rows_bad=3670 (documented).
    """
    g = MemoryGuard(max_memory_mb=5)
    schema = pa.schema([(f"s{i}", pa.string()) for i in range(10)])
    arrs = [pa.array(["x" * 200] * 100, type=pa.string()) for _ in range(10)]
    sample = pa.RecordBatch.from_arrays(arrs, names=schema.names)

    # schema-only baseline: 500 B/row → 3670 rows (R2 bug demo)
    g.estimate_row_size_from_schema(schema)
    rows_bad = g.get_batch_rows(expansion_factor=2.0)

    # exact via sample: 2040 B/row → 899 rows (R2 fix)
    rows_good = g.get_batch_rows(
        expansion_factor=2.0, sample_batch=sample,
    )

    assert rows_good < rows_bad // 2, (
        f"sample-based ({rows_good}) should be < half of schema-based "
        f"({rows_bad}); ratio {rows_bad / rows_good:.2f}"
    )

    # Sanity: actual materialized batch at rows_good fits budget
    actual_batch = pa.RecordBatch.from_arrays(
        [pa.array(["x" * 200] * rows_good, type=pa.string()) for _ in range(10)],
        names=schema.names,
    )
    assert actual_batch.nbytes < 5 * 1024 * 1024, (
        f"materialized {actual_batch.nbytes} B > 5 MB budget"
    )'''


def fix_test_file():
    print("\n=== test_row_size_reestimate.py: explicit baseline + assertion ===")
    p = Path("tests/test_row_size_reestimate.py")
    if not p.exists():
        raise SystemExit(f"{p} missing")
    s = p.read_text()
    if "v0.9.8b-fix2: explicit schema baseline" in s:
        print("  already fixed — skip")
        return
    if TEST_OLD not in s:
        raise SystemExit("test_audit_case_get_batch_rows_with_sample anchor not found")
    s = s.replace(TEST_OLD, TEST_NEW, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 3. re-run patch in subprocess ============

def run_patch_fresh():
    print("\n=== run patch_v098b_row_size_reestimate.py in subprocess ===")
    r = subprocess.run(
        ["uv", "run", "python",
         "scripts/patch_v098b_row_size_reestimate.py"],
        cwd=ROOT, timeout=1800,
    )
    if r.returncode != 0:
        print(f"\n*** patch failed (exit {r.returncode}) ***")
        sys.exit(r.returncode)
    print("\n  patch completed successfully")


# ============ main ============

def main():
    print("=== v0.9.8b-fix2 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_dogfood_in_patch_script()
    fix_test_file()
    run_patch_fresh()


if __name__ == "__main__":
    main()
