#!/usr/bin/env python3
# v0.9.21-fix1 — per-group as_py → per-batch slice to_pylist.
# v0.9.21 COMPLETE.
#
# v0.9.21 introduced 4 .as_py() calls (2 in _per_group_shift, 2 in
# _per_group_rolling):
#   1. first_gk = tuple(b.column(gc)[0].as_py() ...) — batch boundary
#   2. prev_group = tuple(b.column(gc)[start].as_py() ...) in gi>0
#      — PER GROUP, O(#groups). Worst case N unique = N as_py calls.
#
# Failures:
#   test_facade_has_no_as_py: "calls .as_py() 4 time(s)"
#   test_facade_has_no_as_py_ast: "facade calls .as_py on data"
#
# Also real perf bug: per-group prev_group update is O(#groups)
# (not cross-batch, only per-group within batch) — defeats
# Arrow-native goal.
#
# Fix: batch-level key extraction via 1-row to_pylist
# (guard-friendly, bounded #batches × K_cols):
#   - batch_first_key = tuple(b.column(gc).slice(0,1).to_pylist()[0] ...)
#   - prev_group = tuple(b.column(gc).slice(n-1,1).to_pylist()[0] ...)
#     (at end of batch, for NEXT batch's cross-batch check)
# Per-group prev_group update REMOVED (only gi>0 reset, no key track).
#
# Semantics preserved: consecutive rows same key → share state; data
# sorted by group_cols → same key is contiguous → batch-end key ==
# last group's key → next batch's first_key comparison correct.
#
# 1-fix cycle for atomic v0.9.21 (target per §13.4/238; not cascade
# like v0.9.20's 5 fix-cycles for 4-feature patch).
#
# Run: uv run python scripts/patch_v0921_fix1.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ fix 1: _per_group_shift ============

def fix_shift_as_py():
    print("\n=== fix shift: per-group as_py → per-batch to_pylist ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if "batch_first_key = tuple(" in s:
        print("  already fixed — skip")
        return

    # Anchor 1: first_gk extraction
    old1 = '''                first_gk = tuple(
                    b.column(gc)[0].as_py() for gc in group_cols
                )
                if prev_group is _SENTINEL or first_gk != prev_group:
                    if prev_group is not _SENTINEL:
                        for dq in bufs.values():
                            dq.clear()
                    prev_group = first_gk'''

    new1 = '''                # v0.9.21-fix1: batch-level key via 1-row to_pylist
                # (bounded #batches × K_cols, NOT per-group as_py).
                batch_first_key = tuple(
                    b.column(gc).slice(0, 1).to_pylist()[0]
                    for gc in group_cols
                )
                if (prev_group is _SENTINEL
                        or batch_first_key != prev_group):
                    if prev_group is not _SENTINEL:
                        for dq in bufs.values():
                            dq.clear()'''

    if old1 not in s:
        raise SystemExit("shift first_gk anchor not found")
    s = s.replace(old1, new1, 1)

    # Anchor 2: per-group prev_group update (remove)
    old2 = '''                    if gi > 0:
                        for dq in bufs.values():
                            dq.clear()
                        prev_group = tuple(
                            b.column(gc)[start].as_py()
                            for gc in group_cols
                        )
                    group_len = end - start
                    gvals = {
                        c: b.column(c).slice(start, group_len).to_pylist()
                        for c in cols
                    }
                    group_shifted = {c: [None] * group_len for c in cols}'''

    new2 = '''                    if gi > 0:
                        # New group within batch — reset state (no key
                        # tracking; batch-boundary key handles carryover).
                        for dq in bufs.values():
                            dq.clear()
                    group_len = end - start
                    gvals = {
                        c: b.column(c).slice(start, group_len).to_pylist()
                        for c in cols
                    }
                    group_shifted = {c: [None] * group_len for c in cols}'''

    if old2 not in s:
        raise SystemExit("shift per-group prev_group anchor not found")
    s = s.replace(old2, new2, 1)

    # Anchor 3: end-of-batch prev_group tracking
    old3 = '''                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
        finally:
            writer.close()'''

    new3 = '''                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
                # v0.9.21-fix1: track last row's key for next batch's
                # cross-batch continuity (bounded 1-row to_pylist).
                prev_group = tuple(
                    b.column(gc).slice(n - 1, 1).to_pylist()[0]
                    for gc in group_cols
                )
        finally:
            writer.close()'''

    if old3 not in s:
        raise SystemExit("shift writer.write_batch anchor not found")
    s = s.replace(old3, new3, 1)

    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK (3 replacements)")


# ============ fix 2: _per_group_rolling ============

def fix_rolling_as_py():
    print("\n=== fix rolling: same pattern ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    # Already fixed if batch_first_key appears twice (once per function)
    if s.count("batch_first_key = tuple(") >= 2:
        print("  already fixed — skip")
        return

    old1 = '''                first_gk = tuple(
                    b.column(gc)[0].as_py() for gc in group_cols
                )
                if prev_group is _SENTINEL or first_gk != prev_group:
                    if prev_group is not _SENTINEL:
                        for st in states.values():
                            st.reset()
                    prev_group = first_gk'''

    new1 = '''                # v0.9.21-fix1: batch-level key via 1-row to_pylist.
                batch_first_key = tuple(
                    b.column(gc).slice(0, 1).to_pylist()[0]
                    for gc in group_cols
                )
                if (prev_group is _SENTINEL
                        or batch_first_key != prev_group):
                    if prev_group is not _SENTINEL:
                        for st in states.values():
                            st.reset()'''

    if old1 not in s:
        raise SystemExit("rolling first_gk anchor not found")
    s = s.replace(old1, new1, 1)

    old2 = '''                    if gi > 0:
                        for st in states.values():
                            st.reset()
                        prev_group = tuple(
                            b.column(gc)[start].as_py()
                            for gc in group_cols
                        )
                    group_len = end - start
                    gvals = {
                        c: b.column(c).slice(start, group_len).to_pylist()
                        for c in cols
                    }
                    group_rolled = {c: [None] * group_len for c in cols}'''

    new2 = '''                    if gi > 0:
                        # New group within batch — reset state.
                        for st in states.values():
                            st.reset()
                    group_len = end - start
                    gvals = {
                        c: b.column(c).slice(start, group_len).to_pylist()
                        for c in cols
                    }
                    group_rolled = {c: [None] * group_len for c in cols}'''

    if old2 not in s:
        raise SystemExit("rolling per-group anchor not found")
    s = s.replace(old2, new2, 1)

    # rolling-specific: uses all_rolled_arrow (different anchor from shift)
    old3 = '''                for c in cols:
                    arrays.append(pa.concat_arrays(all_rolled_arrow[c]))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
        finally:
            writer.close()'''

    new3 = '''                for c in cols:
                    arrays.append(pa.concat_arrays(all_rolled_arrow[c]))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
                # v0.9.21-fix1: track last row's key for next batch.
                prev_group = tuple(
                    b.column(gc).slice(n - 1, 1).to_pylist()[0]
                    for gc in group_cols
                )
        finally:
            writer.close()'''

    if old3 not in s:
        raise SystemExit("rolling writer.write_batch anchor not found")
    s = s.replace(old3, new3, 1)

    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK (3 replacements)")


# ============ test: AST no as_py in shift/rolling ============

def add_regression_test():
    print("\n=== test: no as_py in shift/rolling ===")
    p = Path("tests/test_groupby_per_group.py")
    s = p.read_text()
    if "def test_per_group_shift_no_as_py_batch_level_only" in s:
        print("  already present — skip")
        return

    extra = '''

# ============ v0.9.21-fix1: no as_py in per-group shift/rolling ============


def test_per_group_shift_no_as_py_batch_level_only():
    """v0.9.21-fix1: cross-batch key via 1-row to_pylist, no as_py.

    Prior v0.9.21 (initial) used per-group .as_py() for prev_group
    tracking: O(#groups), triggered test_facade_has_no_as_py. Fix:
    batch-level only via .slice().to_pylist()[0] on 1-row slices.
    """
    import ast as _ast
    import inspect as _inspect
    import textwrap as _tw
    from cadw.dataframe import CadwDataFrame

    for fn_name in ("_per_group_shift", "_per_group_rolling"):
        fn = getattr(CadwDataFrame, fn_name)
        src = _tw.dedent(_inspect.getsource(fn))
        tree = _ast.parse(src)
        as_py_lines = [
            n.lineno for n in _ast.walk(tree)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "as_py"
        ]
        assert not as_py_lines, (
            f"{fn_name}: {len(as_py_lines)} as_py calls at "
            f"{as_py_lines}; use .slice().to_pylist()[0] for bounded "
            f"batch key. See §13.4/235-240."
        )
        assert ".slice(0, 1).to_pylist()" in src, (
            f"{fn_name}: expected bounded batch_first_key extraction"
        )
        assert ".slice(n - 1, 1).to_pylist()" in src, (
            f"{fn_name}: expected bounded prev_last_key extraction"
        )
'''

    p.write_text(s + extra, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.21 + fix1 + regression ===")
    pu.verify_anchors([
        # v0.9.21-fix1
        ("cadw/dataframe.py", "batch_first_key = tuple("),
        ("cadw/dataframe.py", ".slice(0, 1).to_pylist()[0]"),
        ("cadw/dataframe.py", ".slice(n - 1, 1).to_pylist()[0]"),
        ("tests/test_groupby_per_group.py",
         "def test_per_group_shift_no_as_py_batch_level_only"),
        # v0.9.21
        ("cadw/dataframe.py", "def _group_boundaries_arrow("),
        ("cadw/dataframe.py",
         "# v0.9.21 (§1.4): Arrow-native group boundaries in shift"),
        ("cadw/dataframe.py",
         "# v0.9.21 (§1.4): Arrow-native group boundaries in rolling"),
        ("DECISIONS.md", "**Эпизод 233 (append with fallback anchor"),
        ("DECISIONS.md", "**Эпизод 240 (formula-as-sizing"),
        ("tests/test_groupby_per_group.py",
         "def test_group_boundaries_arrow_correctness"),
        ("tests/test_changelog.py", "def test_decisions_233_240_order"),
        # v0.9.20 regression
        ("cadw/core/memory_guard.py", "sample_peak: bool = False"),
        ("cadw/core/memory_guard.py", "# v0.9.20-fix3"),
        ("tests/test_arrow_call_memory_guard.py",
         '"cadw/core/memory_guard.py:MemoryGuard.calibrate"'),
        ("DECISIONS.md", "**Эпизод 226 (nullable int to_numpy"),
        ("DECISIONS.md", "**Эпизод 232 (metric unit must match"),
        ("scripts/bench_v0919_calibration.py", "pred_batch"),
        # v0.9.19/18/17/16
        ("cadw/core/memory_guard.py", "def calibrate("),
        ("cadw/ops/reduce.py", "with memory_guard.calibrate("),
        ("cadw/ops/shuffle.py", "§13.4/227"),
        ("cadw/dataset/index.py", "v0.9.18 (AUDIT §1.6)"),
        ("DECISIONS.md", "**Эпизод 178 (guard, вакуумный"),
        ("DECISIONS.md", "**Эпизод 225 (static guard scope"),
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
    ])
    # Verify absence of per-group as_py pattern
    s = Path("cadw/dataframe.py").read_text()
    assert "first_gk = tuple(" not in s, (
        "first_gk = tuple(...) should be replaced by batch_first_key"
    )
    assert "prev_group = tuple(\n" not in s or (
        "slice(n - 1, 1).to_pylist()[0]" in s
    ), "prev_group should only be set to batch last row's key"
    print("  all anchors present; no stale first_gk / per-group prev_group")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.21-fix1 (happy path only) ===")

    def _verify_all():
        import pyarrow as pa
        from cadw import CadwDataFrame, MemoryGuard
        from cadw.dataframe import CadwDataFrame as _CDF

        # 1. AST: no as_py in shift/rolling
        import ast as _ast
        import inspect as _inspect
        import textwrap as _tw
        for fn_name in ("_per_group_shift", "_per_group_rolling"):
            fn = getattr(_CDF, fn_name)
            src = _tw.dedent(_inspect.getsource(fn))
            tree = _ast.parse(src)
            as_py_count = sum(
                1 for n in _ast.walk(tree)
                if isinstance(n, _ast.Call)
                and isinstance(n.func, _ast.Attribute)
                and n.func.attr == "as_py"
            )
            assert as_py_count == 0, (
                f"{fn_name}: {as_py_count} as_py (expected 0)"
            )

        # 2. End-to-end: shift correct
        t = pa.table({
            "sensor": ["a", "a", "a", "b", "b", "b"],
            "ts": [1, 2, 3, 1, 2, 3],
            "v": [10, 20, 30, 100, 200, 300],
        })
        df = CadwDataFrame.from_table(
            t, memory_guard=MemoryGuard(max_memory_mb=20),
        )
        res = df.groupby("sensor").shift(1, on="ts", columns=["v"])
        got_v = []
        for b in res._iter_filtered_batches():
            got_v.extend(b.column("v").to_pylist())
        assert got_v == [None, 10, 20, None, 100, 200], (
            f"shift result: {got_v}"
        )

        # 3. Cross-batch continuity (chunk=1: group spans batches)
        t2 = pa.table({
            "sensor": ["a"] * 5,
            "ts": list(range(5)),
            "v": [10, 20, 30, 40, 50],
        })
        df2 = CadwDataFrame.from_table(
            t2, memory_guard=MemoryGuard(max_memory_mb=20),
        )
        res2 = df2.groupby("sensor").shift(1, on="ts", columns=["v"])
        got2 = []
        for b in res2._iter_filtered_batches():
            got2.extend(b.column("v").to_pylist())
        assert got2 == [None, 10, 20, 30, 40], (
            f"cross-batch continuity broken: {got2}"
        )

        # 4. Rolling correct
        res3 = df.groupby("sensor").rolling(
            2, "sum", on="ts", columns=["v"],
        )
        got_s = []
        for b in res3._iter_filtered_batches():
            got_s.extend(b.column("v_rolling_sum").to_pylist())
        assert got_s == [None, 30, 50, None, 300, 500], (
            f"rolling result: {got_s}"
        )

        print("  AST: 0 as_py in shift/rolling OK")
        print("  shift end-to-end: [None,10,20,None,100,200] OK")
        print("  cross-batch continuity: [None,10,20,30,40] OK")
        print("  rolling end-to-end: [None,30,50,None,300,500] OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_groupby_per_group.py",
         "tests/test_dataframe.py",
         "tests/test_dataframe_filters.py",
         "tests/test_shift_diff.py",
         "tests/test_shift_on_col.py",
         "tests/test_changelog.py", "-q"],
        timeout=900,
    )
    if r.returncode != 0:
        print("\n*** TARGET FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


# ============ main ============

def main():
    print("=== v0.9.21-fix1 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_shift_as_py()
    fix_rolling_as_py()
    add_regression_test()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.21: §1.4 Arrow-native group boundaries + DECISIONS "
        "§13.4/233-240 append (initial + fix1, atomic — initial never "
        "committed). (§1.4 fix) Prior: per-batch "
        "`group_arrs = {gc: b.column(gc).to_pylist()}` (n × K_cols "
        "Python objects) + per-row tuple(group_arrs[gc][i]). New: "
        "`_group_boundaries_arrow` (pc.equal + pc.and_ across cols) "
        "computes boundaries — O(n) C++ ops + transient bool masks. "
        "Per-group slice to_pylist bounded by max_group_size. Output "
        "Arrow-native via pa.concat_arrays. Null==null → same group "
        "(SQL-style). Cross-batch continuity preserved. (DECISIONS "
        "append §13.4/233-240) 8 meta-lessons from v0.9.20 chain + "
        "bench analysis: 233 fallback anchor → LATEST; 234 "
        "contiguous anchor vs multi-line; 235 stale test names; "
        "236 peak_delta vs peak_abs; 237 generic method names; 238 "
        "multi-feature patch → fix cascade; 239 sampler miss ≠ no "
        "peak; 240 formula-as-sizing ≠ formula-as-validation metric. "
        "(Fix1) initial v0.9.21 introduced 4 .as_py() calls in "
        "_per_group_shift/rolling: 1 per batch (batch_first_key) + 1 "
        "PER GROUP (prev_group update in gi>0). Triggered "
        "test_facade_has_no_as_py (4 as_py) + test_facade_has_no_as_py_ast. "
        "ALSO real perf bug: per-group prev_group = O(#groups) as_py "
        "calls — worst case N unique groups → N as_py, defeats "
        "Arrow-native goal. Fix: (a) batch_first_key via "
        "`.slice(0,1).to_pylist()[0]` (bounded #batches × K_cols, "
        "guard-friendly); (b) remove per-group prev_group update "
        "(only gi>0 reset, no key track); (c) track prev_group = "
        "batch last row's key via `.slice(n-1,1).to_pylist()[0]` at "
        "end of batch. Same fix for shift and rolling. (Backward "
        "compat) end-to-end verified via dogfood: shift "
        "[None,10,20,None,100,200], rolling [None,30,50,None,300,500], "
        "cross-batch continuity [None,10,20,30,40] (chunk=1, group "
        "spans batches). 1 new test: "
        "test_per_group_shift_no_as_py_batch_level_only (AST no "
        "as_py in shift/rolling, bounded slice.to_pylist present). "
        "1-fix cycle for atomic v0.9.21 (target per §13.4/238; not "
        "cascade like v0.9.20's 5 cycles for 4-feature patch). "
        "Deferred to v0.9.22: per-op predicted wiring в modules; "
        "bench metric redesign; Pingis round 3; A1 split.",
        self_path="scripts/patch_v0921_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.21-fix1 applied. v0.9.21 COMPLETE.")


if __name__ == "__main__":
    main()
