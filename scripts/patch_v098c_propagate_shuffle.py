#!/usr/bin/env python3
# v0.9.8c — B1: propagate_chars supports hash_shuffle (audit crash).
#
# Audit reference (bench_planner.log TEST P1):
#   plan_pipeline([OpSpec(name="hash_shuffle")], dc, budget)
#   → ValueError: propagate_chars: unsupported op 'hash_shuffle'.
#     Supported: filter, with_column, groupby, sort, reduce, join.
#   plan_pipeline([filter, hash_shuffle, groupby]) → same crash.
#
# Root: HashShuffleMemoryModel IS registered (get_model("hash_shuffle")
# works; solve() returns HashShuffleParams). But propagate_chars has no
# branch for op.name == "hash_shuffle" → final `raise ValueError`.
# Same for dedup/rank/quantile/resample — no branch AND no memory model.
#
# Fix in cadw/core/planner.py::propagate_chars:
#   - hash_shuffle (precise, audit case):
#       N unchanged, row_size unchanged, order = "random".
#       (row_id appended internally but not modeled; conservative on N.)
#   - dedup (conservative, no model yet):
#       N = k_est or n_rows (best estimate; over-estimate safe), row_size
#       unchanged, k_est unchanged, order = "random".
#   - rank (precise, no model yet):
#       N unchanged, row_size + 8 (rank int64), order = "random" (sort
#       → back to orig).
#   - quantile (precise, no model yet):
#       N = 1, row_size = 8, k_est = 1.
#   - resample (conservative, no model yet):
#       N = k_est or n_rows (buckets), row_size unchanged, order =
#       "sorted".
#
# Update final raise message: Supported list includes new ops.
#
# Tests (two-level, §13.4/190):
#   - propagate_chars per op: N/row_size/order expected.
#   - plan_pipeline chain [filter, hash_shuffle, groupby] no crash
#     (audit case), params set for shuffle, groupby works downstream.
#   - plan_pipeline chain [filter, hash_shuffle, sort] no crash.
#   - dedup/rank/quantile/resample in chain: no crash (model=None).
#   - Supported list message contains all 11 ops.
#
# Scaffold v2: write_content_checked, is_patched.
# Dogfood self-contained imports (§13.4/200). Subprocess re-run if
# stale (not needed here — planner.py only; no in-process class
# mutation before dogfood).
#
# Run: uv run python scripts/patch_v098c_propagate_shuffle.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import importlib
import cadw.patch_utils as pu
importlib.reload(pu)
from cadw.patch_utils import (
    write_content_checked, assert_compiles, commit_with_guards,
    is_patched,
)


# ============ 1. Add branches to propagate_chars ============

# Anchor: the existing `raise ValueError` at the end.
OLD_RAISE = '''    # Для hash_shuffle и прочих пропагатор не определён
    raise ValueError(
        f"propagate_chars: unsupported op {op.name!r}. "
        f"Supported: filter, with_column, groupby, sort, reduce, join."
    )'''

NEW_BLOCK = '''    # v0.9.8c (B1): hash_shuffle — audit case.
    # N unchanged (partition does not change row count), row_size
    # unchanged (key + value + row_id; row_id uint64 not modeled —
    # conservative on N, not row_size). Order becomes "random".
    if op.name == "hash_shuffle":
        return _copy_chars(input_chars, order="random")

    # v0.9.8c: dedup — conservative N = k_est or n_rows (no model yet).
    # Best estimate k_est; if None → n_rows (safe). row_size unchanged.
    if op.name == "dedup":
        k = input_chars.k_est or input_chars.n_rows
        return _copy_chars(
            input_chars,
            n_rows=k,
            k_est=k,
            order="random",
        )

    # v0.9.8c: rank — N unchanged, row_size + 8 (rank int64).
    # order = "random" (sort by key then re-sort by row_id).
    if op.name == "rank":
        return _copy_chars(
            input_chars,
            row_size_bytes=input_chars.row_size_bytes + 8,
            order="random",
        )

    # v0.9.8c: quantile — N = 1, row_size = 8 (scalar float64).
    if op.name == "quantile":
        return _copy_chars(
            input_chars,
            n_rows=1,
            k_est=1,
            row_size_bytes=8,
            order="random",
        )

    # v0.9.8c: resample — conservative N = k_est or n_rows (buckets),
    # row_size unchanged (on + agg cols; conservative upper bound).
    if op.name == "resample":
        k = input_chars.k_est or input_chars.n_rows
        return _copy_chars(
            input_chars,
            n_rows=k,
            k_est=k,
            order="sorted",
        )

    # Unsupported op.
    raise ValueError(
        f"propagate_chars: unsupported op {op.name!r}. "
        f"Supported: filter, with_column, groupby, sort, reduce, join, "
        f"hash_shuffle, dedup, rank, quantile, resample."
    )'''


def add_propagate_branches():
    print("\n=== planner.py: add propagate_chars branches ===")
    p = Path("cadw/core/planner.py")
    s = p.read_text()
    if "if op.name == \"hash_shuffle\":" in s:
        print("  already present — skip")
        return
    if OLD_RAISE not in s:
        raise SystemExit("propagate_chars final raise anchor not found")
    s = s.replace(OLD_RAISE, NEW_BLOCK, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. Tests ============

TEST_LINES = [
    '"""v0.9.8c: B1 — propagate_chars supports hash_shuffle (+4 ops)."""',
    'from __future__ import annotations',
    '',
    'import inspect',
    '',
    'import pyarrow as pa',
    'import pytest',
    '',
    'from cadw.core.memory_guard import MemoryGuard',
    'from cadw.core.models import DataChars',
    'from cadw.core.models_join import JoinParams',
    'from cadw.core.models_shuffle import HashShuffleParams',
    'from cadw.core.planner import (',
    '    OpSpec,',
    '    plan_pipeline,',
    '    propagate_chars,',
    ')',
    '',
    '',
    'def _dc(n_rows=10_000, k_est=1000, row_size=16, size_of_key_entry=80):',
    '    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])',
    '    return DataChars(',
    '        schema=schema,',
    '        n_rows=n_rows,',
    '        row_size_bytes=row_size,',
    '        k_est=k_est,',
    '        avg_key_size_bytes=8,',
    '        size_of_key_entry_bytes=size_of_key_entry,',
    '    )',
    '',
    '',
    '# ============ propagate_chars per op ============',
    '',
    '',
    'def test_propagate_chars_hash_shuffle():',
    '    dc = _dc(n_rows=100_000)',
    '    out = propagate_chars(OpSpec(name="hash_shuffle"), dc, None)',
    '    assert out.n_rows == 100_000',
    '    assert out.row_size_bytes == dc.row_size_bytes',
    '    assert out.order == "random"',
    '',
    '',
    'def test_propagate_chars_dedup():',
    '    dc = _dc(n_rows=100_000, k_est=1000)',
    '    out = propagate_chars(OpSpec(name="dedup"), dc, None)',
    '    assert out.n_rows == 1000',
    '    assert out.k_est == 1000',
    '    assert out.order == "random"',
    '',
    '',
    'def test_propagate_chars_dedup_no_k_est():',
    '    dc = _dc(n_rows=100_000, k_est=None)',
    '    out = propagate_chars(OpSpec(name="dedup"), dc, None)',
    '    # k_est=None → conservative n_rows',
    '    assert out.n_rows == 100_000',
    '',
    '',
    'def test_propagate_chars_rank():',
    '    dc = _dc(n_rows=10_000, row_size=16)',
    '    out = propagate_chars(OpSpec(name="rank"), dc, None)',
    '    assert out.n_rows == 10_000',
    '    assert out.row_size_bytes == 16 + 8  # +rank int64',
    '    assert out.order == "random"',
    '',
    '',
    'def test_propagate_chars_quantile():',
    '    dc = _dc(n_rows=10_000, row_size=16)',
    '    out = propagate_chars(OpSpec(name="quantile"), dc, None)',
    '    assert out.n_rows == 1',
    '    assert out.k_est == 1',
    '    assert out.row_size_bytes == 8',
    '',
    '',
    'def test_propagate_chars_resample():',
    '    dc = _dc(n_rows=100_000, k_est=1000)',
    '    out = propagate_chars(OpSpec(name="resample"), dc, None)',
    '    assert out.n_rows == 1000',
    '    assert out.order == "sorted"',
    '',
    '',
    '# ============ audit case: chain no crash ============',
    '',
    '',
    'def test_plan_pipeline_filter_hash_shuffle_groupby():',
    '    """Audit B1: [filter, hash_shuffle, groupby] must plan, not crash."""',
    '    dc = _dc(n_rows=100_000, k_est=1000)',
    '    ops = [',
    '        OpSpec(name="filter"),',
    '        OpSpec(name="hash_shuffle"),',
    '        OpSpec(name="groupby"),',
    '    ]',
    '    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)',
    '    assert plan.predicted_outcome == "fits"',
    '    assert len(plan.steps) == 3',
    '    # Step 1 (filter): propagator-only, params None.',
    '    assert plan.steps[0].params is None',
    '    # Step 2 (hash_shuffle): model exists, params set.',
    '    assert isinstance(plan.steps[1].params, HashShuffleParams)',
    '    # Step 3 (groupby): model exists, params set.',
    '    assert plan.steps[2].params is not None',
    '',
    '',
    'def test_plan_pipeline_hash_shuffle_alone():',
    '    """Audit B1: hash_shuffle alone — must not crash."""',
    '    dc = _dc(n_rows=100_000, k_est=1000)',
    '    plan = plan_pipeline(',
    '        [OpSpec(name="hash_shuffle")], dc, 10 * 1024 * 1024,',
    '    )',
    '    assert plan.predicted_outcome == "fits"',
    '    assert isinstance(plan.steps[0].params, HashShuffleParams)',
    '',
    '',
    'def test_plan_pipeline_filter_hash_shuffle_sort():',
    '    """Variation: [filter, hash_shuffle, sort] no crash."""',
    '    dc = _dc(n_rows=100_000, k_est=1000)',
    '    ops = [',
    '        OpSpec(name="filter"),',
    '        OpSpec(name="hash_shuffle"),',
    '        OpSpec(name="sort"),',
    '    ]',
    '    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)',
    '    assert plan.predicted_outcome == "fits"',
    '    assert plan.steps[2].output_chars.order == "sorted"',
    '',
    '',
    'def test_plan_pipeline_dedup_no_crash():',
    '    """dedup in chain: no model → propagator-only, no crash."""',
    '    dc = _dc(n_rows=100_000, k_est=1000)',
    '    ops = [OpSpec(name="dedup"), OpSpec(name="groupby")]',
    '    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)',
    '    assert plan.predicted_outcome == "fits"',
    '    # After dedup, N = k_est = 1000; groupby sees 1000.',
    '    assert plan.steps[1].input_chars.n_rows == 1000',
    '',
    '',
    'def test_plan_pipeline_rank_no_crash():',
    '    dc = _dc(n_rows=10_000)',
    '    ops = [OpSpec(name="rank"), OpSpec(name="sort")]',
    '    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)',
    '    assert plan.predicted_outcome == "fits"',
    '',
    '',
    'def test_plan_pipeline_quantile_no_crash():',
    '    dc = _dc(n_rows=10_000)',
    '    plan = plan_pipeline(',
    '        [OpSpec(name="quantile")], dc, 1 * 1024 * 1024,',
    '    )',
    '    assert plan.predicted_outcome == "fits"',
    '    assert plan.steps[0].output_chars.n_rows == 1',
    '',
    '',
    'def test_plan_pipeline_resample_no_crash():',
    '    dc = _dc(n_rows=100_000, k_est=1000)',
    '    plan = plan_pipeline(',
    '        [OpSpec(name="resample"), OpSpec(name="sort")],',
    '        dc, 10 * 1024 * 1024,',
    '    )',
    '    assert plan.predicted_outcome == "fits"',
    '    assert plan.steps[0].output_chars.order == "sorted"',
    '',
    '',
    '# ============ error message includes new ops ============',
    '',
    '',
    'def test_propagate_chars_supported_list_mentions_new_ops():',
    '    """Error message for unsupported op lists hash_shuffle + others."""',
    '    dc = _dc()',
    '    with pytest.raises(ValueError, match="Supported:"):',
    '        propagate_chars(OpSpec(name="nonexistent_op"), dc, None)',
    '    with pytest.raises(ValueError) as excinfo:',
    '        propagate_chars(OpSpec(name="nonexistent_op"), dc, None)',
    '    msg = str(excinfo.value)',
    '    for name in (',
    '        "hash_shuffle", "dedup", "rank", "quantile", "resample",',
    '        "filter", "with_column", "groupby", "sort", "reduce",',
    '        "join",',
    '    ):',
    '        assert name in msg, f"expected {name!r} in Supported list"',
    '',
    '',
    '# ============ AST guard: branches present (§13.4/190) ============',
    '',
    '',
    'def test_propagate_chars_has_all_new_branches():',
    '    from cadw.core.planner import propagate_chars',
    '    src = inspect.getsource(propagate_chars)',
    '    for op_name in (',
    '        "hash_shuffle", "dedup", "rank", "quantile", "resample",',
    '    ):',
    '        assert f\'op.name == "{op_name}"\' in src, (',
    '            f"propagate_chars missing branch for {op_name!r}"',
    '        )',
]


def create_tests():
    print("\n=== create tests/test_propagate_chars_ops.py ===")
    p = Path("tests/test_propagate_chars_ops.py")
    if p.exists():
        print("  already exists — skip")
        return
    write_content_checked(p, TEST_LINES)
    print("  written + compiled")


# ============ 3. Dogfood ============

def dogfood():
    print("\n=== dogfood: B1 audit case no crash ===")
    import pyarrow as pa
    from cadw.core.models import DataChars
    from cadw.core.models_shuffle import HashShuffleParams
    from cadw.core.planner import OpSpec, plan_pipeline, propagate_chars

    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])
    dc = DataChars(
        schema=schema, n_rows=100_000, row_size_bytes=16,
        k_est=1000, avg_key_size_bytes=8,
        size_of_key_entry_bytes=80,
    )

    # propagate_chars hash_shuffle directly
    out = propagate_chars(OpSpec(name="hash_shuffle"), dc, None)
    print(f"  propagate hash_shuffle: n={out.n_rows}, order={out.order}")
    assert out.n_rows == dc.n_rows
    assert out.order == "random"

    # Audit chain: [filter, hash_shuffle, groupby]
    ops = [
        OpSpec(name="filter"),
        OpSpec(name="hash_shuffle"),
        OpSpec(name="groupby"),
    ]
    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)
    print(f"  plan_pipeline chain: outcome={plan.predicted_outcome}, "
          f"steps={len(plan.steps)}")
    assert plan.predicted_outcome == "fits"
    assert isinstance(plan.steps[1].params, HashShuffleParams)

    # Chain with sort: [hash_shuffle, sort]
    plan2 = plan_pipeline(
        [OpSpec(name="hash_shuffle"), OpSpec(name="sort")],
        dc, 10 * 1024 * 1024,
    )
    assert plan2.predicted_outcome == "fits"
    assert plan2.steps[-1].output_chars.order == "sorted"
    print("  hash_shuffle → sort: no crash, order=sorted")

    # dedup / rank / quantile / resample in chain (propagator-only)
    for op_name in ("dedup", "rank", "quantile", "resample"):
        p = plan_pipeline(
            [OpSpec(name=op_name)], dc, 10 * 1024 * 1024,
        )
        assert p.predicted_outcome == "fits"
        print(f"  {op_name}: no crash")


# ============ main ============

def main():
    print("=== v0.9.8c: B1 propagate_chars hash_shuffle ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    add_propagate_branches()
    create_tests()
    dogfood()

    print("\n=== pytest: propagate + planner regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_propagate_chars_ops.py",
         "tests/test_planner.py",
         "tests/test_planner_join.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** B1 REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    commit_with_guards(
        "v0.9.8c: B1 — propagate_chars supports hash_shuffle (+4 ops). "
        "Audit bench_planner.log: plan_pipeline([filter, hash_shuffle, "
        "groupby]) → ValueError: propagate_chars: unsupported op "
        "'hash_shuffle' (hard crash, not PipelineMemoryError). "
        "HashShuffleMemoryModel IS registered (solve returns "
        "HashShuffleParams), but propagate_chars had no branch → final "
        "raise. Fix: add branches in cadw/core/planner.py::"
        "propagate_chars — hash_shuffle (N unchanged, row_size "
        "unchanged, order=random, precise); dedup (N=k_est or n_rows, "
        "row_size unchanged, order=random, conservative); rank "
        "(N unchanged, row_size +8 rank int64, order=random); quantile "
        "(N=1, row_size=8, k_est=1); resample (N=k_est or n_rows, "
        "row_size unchanged, order=sorted, conservative). Final raise "
        "message updated with all 11 ops. Chain [filter, hash_shuffle, "
        "groupby] plans (B1 audit case). dedup/rank/quantile/resample "
        "no crash (propagator-only, model=None — full memory models "
        "backlog v0.9.9). Tests: 15 (per-op N/row_size/order; audit "
        "chain [filter, hash_shuffle, groupby]; [filter, hash_shuffle, "
        "sort]; dedup/rank/quantile/resample in chains; error message "
        "lists all ops; AST guard branches present). B1 closed. v0.9.8 "
        "(R1 + R2 + B1) complete.",
        self_path="scripts/patch_v098c_propagate_shuffle.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.8c applied. v0.9.8 (R1+R2+B1) complete.")


if __name__ == "__main__":
    main()
