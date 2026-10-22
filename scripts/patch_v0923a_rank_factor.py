#!/usr/bin/env python3
# v0.9.23a — rank extra_factor fix (AUDIT_FULL_v0922 §3.7, item 7.A.11)
#
# Three derived factors (per prior-audit 3.5·K pylist factor —
# 28 B/value vs Arrow 8 B/value):
#   _compute_streamable  : 5.0 → 7.0   (2 pylist: by + row_id)
#   _pass1_groups_to_disk: 3.0 → 3.5   (1 pylist: by)
#   _pass2_ranks_to_disk : 5.0 → 7.0   (2 pylist: by + row_id)
#
# Guard-only fix. No behavior change — check_batch still guards,
# but with tighter factor. No output identity change.
#
# Run: uv run python scripts/patch_v0923a_rank_factor.py

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


# ============ 1. _compute_streamable: 5.0 → 7.0 ============

def fix_streamable():
    print("\n=== fix _compute_streamable extra_factor 5.0 → 7.0 ===")
    p = Path("cadw/ops/rank.py")
    s = p.read_text()
    if 'name="rank.streamable.input", extra_factor=7.0' in s:
        print("  already fixed — skip")
        return
    applied = apply_or_fail(
        str(p),
        old=(
            '            memory_guard.check_batch(\n'
            '                batch, name="rank.streamable.input", extra_factor=5.0,\n'
            '            )'
        ),
        new=(
            '            memory_guard.check_batch(\n'
            '                batch, name="rank.streamable.input", extra_factor=7.0,\n'
            '            )'
        ),
        label="rank.streamable.input",
    )
    if applied:
        assert_compiles(p)
        print("  OK")


# ============ 2. _pass1_groups_to_disk: 3.0 → 3.5 ============

def fix_pass1():
    print("\n=== fix _pass1_groups_to_disk extra_factor 3.0 → 3.5 ===")
    p = Path("cadw/ops/rank.py")
    s = p.read_text()
    if 'name="rank.pass1.input", extra_factor=3.5' in s:
        print("  already fixed — skip")
        return
    applied = apply_or_fail(
        str(p),
        old=(
            '            memory_guard.check_batch(\n'
            '                batch, name="rank.pass1.input", extra_factor=3.0,\n'
            '            )'
        ),
        new=(
            '            memory_guard.check_batch(\n'
            '                batch, name="rank.pass1.input", extra_factor=3.5,\n'
            '            )'
        ),
        label="rank.pass1.input",
    )
    if applied:
        assert_compiles(p)
        print("  OK")


# ============ 3. _pass2_ranks_to_disk: 5.0 → 7.0 ============

def fix_pass2():
    print("\n=== fix _pass2_ranks_to_disk extra_factor 5.0 → 7.0 ===")
    p = Path("cadw/ops/rank.py")
    s = p.read_text()
    if 'name="rank.pass2.input", extra_factor=7.0' in s:
        print("  already fixed — skip")
        return
    applied = apply_or_fail(
        str(p),
        old=(
            '            memory_guard.check_batch(\n'
            '                batch, name="rank.pass2.input", extra_factor=5.0,\n'
            '            )'
        ),
        new=(
            '            memory_guard.check_batch(\n'
            '                batch, name="rank.pass2.input", extra_factor=7.0,\n'
            '            )'
        ),
        label="rank.pass2.input",
    )
    if applied:
        assert_compiles(p)
        print("  OK")


# ============ 4. Tests ============

TESTS = '''"""v0.9.23a: rank extra_factor fix (AUDIT_FULL_v0922 §3.7, item 7.A.11).

Derived factor per prior-audit 3.5*K pylist (28 B/value vs 8 B/value):
  - 1 pylist column: 3.5  (pass1)
  - 2 pylist columns: 7.0 (streamable, pass2)
"""
import inspect

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=5, name="L"):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / f"spill_{name}",
    )


# ============ AST checks: factors ============

def test_streamable_extra_factor_is_7():
    """_compute_streamable: 2 pylist (by + row_id) -> 3.5*2 = 7.0."""
    from cadw.ops.rank import _compute_streamable
    src = inspect.getsource(_compute_streamable)
    assert 'name="rank.streamable.input", extra_factor=7.0' in src, (
        "rank.streamable.input extra_factor must be 7.0 "
        "(AUDIT_FULL_v0922 §3.7)"
    )


def test_pass1_extra_factor_is_3_5():
    """_pass1_groups_to_disk: 1 pylist (by) -> 3.5."""
    from cadw.ops.rank import _pass1_groups_to_disk
    src = inspect.getsource(_pass1_groups_to_disk)
    assert 'name="rank.pass1.input", extra_factor=3.5' in src, (
        "rank.pass1.input extra_factor must be 3.5"
    )


def test_pass2_extra_factor_is_7():
    """_pass2_ranks_to_disk: 2 pylist (by + row_id) -> 3.5*2 = 7.0."""
    from cadw.ops.rank import _pass2_ranks_to_disk
    src = inspect.getsource(_pass2_ranks_to_disk)
    assert 'name="rank.pass2.input", extra_factor=7.0' in src, (
        "rank.pass2.input extra_factor must be 7.0"
    )


# ============ Regression: rank still correct ============

def test_rank_nullfree_still_correct(tmp_path):
    """Null-free rank(by='k', method='first') unchanged by factor bump."""
    df = make_df(tmp_path, {
        "k": [1.0, 3.0, 2.0, 4.0, 5.0],
        "v": [10, 30, 20, 40, 50],
    })
    result = df.rank(by="k", method="first")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # sort by k: row_id 0(v=10)->1, 2(v=20)->2, 1(v=30)->3, 3(v=40)->4, 4(v=50)->5
    # original order: [1, 3, 2, 4, 5]
    assert [r["k_rank"] for r in rows] == [1, 3, 2, 4, 5]


def test_rank_with_null_still_correct(tmp_path):
    """Null-path rank(by='k', method='first') still correct."""
    df = make_df(tmp_path, {
        "k": pa.array([1.0, None, 3.0, 2.0, None, 4.0], type=pa.float64()),
        "v": list(range(6)),
    })
    result = df.rank(by="k", method="first")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    # Nulls sort last. Non-null ranks: 1.0->1, 2.0->2, 3.0->3, 4.0->4;
    # nulls: 5, 6 (by row_id tie-break).
    # Original order (k=[1,None,3,2,None,4]):
    # row 0(k=1)->1, row 1(None)->5, row 2(k=3)->3,
    # row 3(k=2)->2, row 4(None)->6, row 5(k=4)->4
    assert [r["k_rank"] for r in rows] == [1, 5, 3, 2, 6, 4]
'''


def write_test_file():
    print("\n=== write tests/test_rank_extra_factor.py ===")
    p = Path("tests/test_rank_extra_factor.py")
    if p.exists():
        print("  already exists — skip")
        return
    p.write_text(TESTS)
    assert_compiles(p)
    print("  OK")


# ============ 5. Dogfood ============

def dogfood():
    print("\n=== dogfood: rank on null-free + null path ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import inspect

        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG
        from cadw.ops.rank import (
            _compute_streamable, _pass1_groups_to_disk, _pass2_ranks_to_disk,
        )

        # AST checks
        assert 'extra_factor=7.0' in inspect.getsource(_compute_streamable)
        assert 'extra_factor=3.5' in inspect.getsource(_pass1_groups_to_disk)
        assert 'extra_factor=7.0' in inspect.getsource(_pass2_ranks_to_disk)

        # Functional: null-free
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({"k": [1.0, 3.0, 2.0, 4.0, 5.0], "v": list(range(5))})
            src = td / "src.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            res = df.rank(by="k", method="first")
            out = td / "o.parquet"
            res.write_parquet(out)
            assert [r["k_rank"] for r in pq_.read_table(out).to_pylist()] == [
                1, 3, 2, 4, 5,
            ]

        # Functional: null path
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "k": pa_.array([1.0, None, 3.0, 2.0, None, 4.0],
                               type=pa_.float64()),
                "v": list(range(6)),
            })
            src = td / "src.parquet"
            pq_.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            res = df.rank(by="k", method="first")
            out = td / "o.parquet"
            res.write_parquet(out)
            assert [r["k_rank"] for r in pq_.read_table(out).to_pylist()] == [
                1, 5, 3, 2, 6, 4,
            ]

        print("  AST + functional (null-free + null path) ok")

    run_verify_in_subprocess(_verify)


# ============ 6. Tests ============

def run_pytest():
    print("\n=== pytest: targeted + fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_rank_extra_factor.py",
         "tests/test_rank.py",
         "tests/test_pylist_fixes.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** RANK REGRESSION FAILED ***")
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
    print("=== v0.9.23a: rank extra_factor fix ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_streamable()
    fix_pass1()
    fix_pass2()
    write_test_file()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23a: rank extra_factor 5.0->7.0 (streamable, pass2), "
        "3.0->3.5 (pass1). AUDIT_FULL_v0922 §3.7, item 7.A.11. "
        "Derived factor = 3.5*K (28 B pylist / 8 B Arrow): "
        "1 pylist col -> 3.5 (pass1 by); 2 pylist cols -> 7.0 "
        "(streamable/pass2: by + row_id). Guard-only fix, no "
        "output change. Prior 5.0 under by 29% on derivation, "
        "~4x on measured pylist delta (1.61 MB / 80 KB batch = "
        "20.2x at N=100k with 10% NaN). Not an OOM today "
        "(state_limit generous) but at tight budget (100 MB "
        "state_limit, batch=10k, 2 cols) check fires ~1.6 MB late. "
        "See _hotspot.md §3.7, _v0922_verify.md §4.4.",
        self_path="scripts/patch_v0923a_rank_factor.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23a applied.")


if __name__ == "__main__":
    main()