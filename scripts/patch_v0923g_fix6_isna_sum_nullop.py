#!/usr/bin/env python3
# v0.9.23g-fix6 — isna_sum uses .as_py() in facade -> NullCountOp.
#
# test_facade_has_no_as_py (AST guard) forbids ANY .as_py() in
# cadw/dataframe.py. isna_sum used `_pc.sum(...).as_py() or 0` — 1
# occurrence. Guard fires (correctly — pattern encourages per-row
# loops).
#
# Fix: NullCountOp ReduceOp in cadw/ops/reduce.py. `batch.column(c).
# null_count` is O(1) metadata (IPC-read batches have it precomputed).
# isna_sum uses reduce() — no .as_py(), no cast, no per-batch sum.
# Same d-pass cost, no facade violation.
#
# +tests/test_reduce.py: NullCountOp direct (init/update/merge/
# estimate/state_nbytes).
# +tests/test_eda_basics.py: isna_sum still works (existing 3 tests);
# new AST guard test: no .as_py in isna_sum body.
#
# Run: uv run python scripts/patch_v0923g_fix6_isna_sum_nullop.py

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


# ============ 1. NullCountOp in reduce.py ============

NULL_OP_ANCHOR = '''# ---------------------------------------------------------------------------
# min / max
# ---------------------------------------------------------------------------'''

NULL_COUNT_OP = '''# ---------------------------------------------------------------------------
# null count (v0.9.23g-fix6)
# ---------------------------------------------------------------------------

class NullCountOp(ReduceOp):
    """Count nulls in a single column.

    `batch.column(c).null_count` — O(1) metadata (IPC-read batches
    have it precomputed; pyarrow caches after first touch). No per-row
    iteration, no pc.is_null, no cast, no sum. State = 1 int64.

    Used by `CadwDataFrame.isna_sum()` — façade stays free of
    `.as_py()` (test_facade_has_no_as_py AST guard).
    """

    def __init__(self, column: str) -> None:
        self.column = column

    def estimate_state_nbytes(self) -> int:
        return 8

    def init(self) -> State:
        return 0

    def update(self, state: State, batch: pa.RecordBatch) -> State:
        return state + int(batch.column(self.column).null_count)

    def merge(self, a: State, b: State) -> State:
        return a + b

    def finalize(self, state: State) -> int:
        return int(state)

    def state_nbytes(self, state: State) -> int:
        return 8


# ---------------------------------------------------------------------------
# min / max
# ---------------------------------------------------------------------------'''


def add_null_count_op():
    print("\n=== add NullCountOp to reduce.py ===")
    p = Path("cadw/ops/reduce.py")
    s = p.read_text()
    if "class NullCountOp(ReduceOp):" in s:
        print("  already present — skip")
        return
    if NULL_OP_ANCHOR not in s:
        raise SystemExit("reduce.py anchor not found")
    s = s.replace(NULL_OP_ANCHOR, NULL_COUNT_OP, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. Export NullCountOp ============

def add_export():
    print("\n=== export NullCountOp in ops/__init__.py + cadw/__init__.py ===")
    # ops/__init__.py
    p = Path("cadw/ops/__init__.py")
    s = p.read_text()
    if "NullCountOp" in s:
        print("  ops/__init__ — already present")
    else:
        apply_or_fail(
            str(p),
            old=(
                'from .reduce import (\n'
                '    reduce, ReduceOp,\n'
                '    CountOp, SumOp, MinMaxOp, MeanOp, StdOp, CovarianceOp, HistogramOp,\n'
                ')'
            ),
            new=(
                'from .reduce import (\n'
                '    reduce, ReduceOp,\n'
                '    CountOp, SumOp, MinMaxOp, MeanOp, StdOp, CovarianceOp, HistogramOp,\n'
                '    NullCountOp,\n'
                ')'
            ),
            label="ops_init_import",
        )
        apply_or_fail(
            str(p),
            old=(
                '    "CountOp", "SumOp", "MinMaxOp", "MeanOp", "StdOp",\n'
                '    "CovarianceOp", "HistogramOp",'
            ),
            new=(
                '    "CountOp", "SumOp", "MinMaxOp", "MeanOp", "StdOp",\n'
                '    "CovarianceOp", "HistogramOp", "NullCountOp",'
            ),
            label="ops_init_all",
        )
        p.write_text(p.read_text())  # no-op, apply_or_fail уже записал
        print("  ops/__init__ — OK")
    # cadw/__init__.py
    p = Path("cadw/__init__.py")
    s = p.read_text()
    if "NullCountOp" in s:
        print("  cadw/__init__ — already present")
    else:
        apply_or_fail(
            str(p),
            old=(
                'from .ops.reduce import (\n'
                '    reduce, ReduceOp,\n'
                '    CountOp, SumOp, MinMaxOp, MeanOp, StdOp, CovarianceOp, HistogramOp,\n'
                ')'
            ),
            new=(
                'from .ops.reduce import (\n'
                '    reduce, ReduceOp,\n'
                '    CountOp, SumOp, MinMaxOp, MeanOp, StdOp, CovarianceOp, HistogramOp,\n'
                '    NullCountOp,\n'
                ')'
            ),
            label="cadw_init_import",
        )
        apply_or_fail(
            str(p),
            old=(
                '    "CountOp", "SumOp", "MinMaxOp", "MeanOp", "StdOp",\n'
                '    "CovarianceOp", "HistogramOp",'
            ),
            new=(
                '    "CountOp", "SumOp", "MinMaxOp", "MeanOp", "StdOp",\n'
                '    "CovarianceOp", "HistogramOp", "NullCountOp",'
            ),
            label="cadw_init_all",
        )
        print("  cadw/__init__ — OK")
    assert_compiles(Path("cadw/__init__.py"))
    assert_compiles(Path("cadw/ops/__init__.py"))


# ============ 3. isna_sum: use NullCountOp ============

OLD_ISNA_SUM = '''        import pyarrow.compute as _pc

        out: dict = {}
        for c in self._columns:
            total = 0
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(
                    b, name=f"isna_sum[{c}]",
                )
                total += int(_pc.sum(
                    _pc.cast(_pc.is_null(b.column(c)), pa.int64()),
                ).as_py() or 0)
            out[c] = total
        return out'''

NEW_ISNA_SUM = '''        from cadw.ops.reduce import NullCountOp, reduce

        # v0.9.23g-fix6: NullCountOp (no .as_py in facade —
        # test_facade_has_no_as_py AST guard). batch.column(c).
        # null_count is O(1) metadata; d Reduce passes, same cost as
        # prior per-batch sum approach.
        out: dict = {}
        for c in self._columns:
            out[c] = reduce(
                self._iter_filtered_batches(),
                NullCountOp(c),
                self._memory_guard,
                log=False,
            )
        return out'''

def fix_isna_sum():
    print("\n=== isna_sum: use NullCountOp ===")
    p = Path("cadw/dataframe.py")
    applied = apply_or_fail(
        str(p), OLD_ISNA_SUM, NEW_ISNA_SUM,
        label="isna_sum NullCountOp",
    )
    if applied:
        assert_compiles(p)
        print("  applied")
    else:
        print("  already fixed — skip")


# ============ 4. Tests ============

REDUCE_TESTS = '''

# ============ v0.9.23g-fix6: NullCountOp ============

def test_null_count_op_init_state_nbytes():
    from cadw.ops.reduce import NullCountOp
    op = NullCountOp("x")
    st = op.init()
    assert st == 0
    assert op.state_nbytes(st) == 8
    assert op.estimate_state_nbytes() == 8


def test_null_count_op_update_merge():
    from cadw.ops.reduce import NullCountOp
    op = NullCountOp("x")
    b1 = pa.RecordBatch.from_arrays(
        [pa.array([1, None, 3, None], type=pa.int64())],
        names=["x"],
    )
    b2 = pa.RecordBatch.from_arrays(
        [pa.array([None, None], type=pa.int64())],
        names=["x"],
    )
    s1 = op.update(op.init(), b1)
    s2 = op.update(op.init(), b2)
    assert s1 == 2
    assert s2 == 2
    assert op.merge(s1, s2) == 4
    assert op.finalize(op.merge(s1, s2)) == 4


def test_null_count_op_no_nulls():
    from cadw.ops.reduce import NullCountOp
    op = NullCountOp("x")
    b = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3], type=pa.int64())],
        names=["x"],
    )
    assert op.update(op.init(), b) == 0
'''

EDA_TESTS = '''

# ============ v0.9.23g-fix6: no .as_py in isna_sum ============

def test_isna_sum_no_as_py_ast():
    """AST: isna_sum body has no .as_py() Call.

    test_facade_has_no_as_py catches facade-wide. This pins the
    specific fix (isna_sum uses NullCountOp, not pc.sum().as_py()).
    """
    import ast as _ast
    import inspect as _inspect
    import textwrap as _tw
    from cadw.dataframe import CadwDataFrame
    src = _tw.dedent(_inspect.getsource(CadwDataFrame.isna_sum))
    tree = _ast.parse(src)
    as_py_calls = [
        n for n in _ast.walk(tree)
        if isinstance(n, _ast.Call)
        and isinstance(n.func, _ast.Attribute)
        and n.func.attr == "as_py"
    ]
    assert not as_py_calls, (
        f"isna_sum calls .as_py() {len(as_py_calls)} time(s); "
        f"use NullCountOp (Reduce)"
    )
    assert "NullCountOp" in src, (
        "isna_sum must use NullCountOp"
    )
'''


def add_reduce_tests():
    print("\n=== add NullCountOp tests to test_reduce.py ===")
    p = Path("tests/test_reduce.py")
    s = p.read_text()
    if "test_null_count_op_init_state_nbytes" in s:
        print("  already present — skip")
        return
    p.write_text(s + REDUCE_TESTS)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


def add_eda_tests():
    print("\n=== add isna_sum no-as_py AST test to test_eda_basics.py ===")
    p = Path("tests/test_eda_basics.py")
    s = p.read_text()
    if "test_isna_sum_no_as_py_ast" in s:
        print("  already present — skip")
        return
    p.write_text(s + EDA_TESTS)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 5. Dogfood ============

def dogfood():
    print("\n=== dogfood: isna_sum via NullCountOp, no .as_py ===")

    def _verify():
        import inspect
        import ast as _ast
        import textwrap as _tw
        import tempfile
        from pathlib import Path as _P
        import pyarrow as pa_
        import pyarrow.parquet as pq_
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG
        from cadw.dataframe import CadwDataFrame as _DF

        # (a) no .as_py() in isna_sum body
        src = _tw.dedent(inspect.getsource(_DF.isna_sum))
        tree = _ast.parse(src)
        as_py = [
            n for n in _ast.walk(tree)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "as_py"
        ]
        assert not as_py, f"isna_sum has {len(as_py)} .as_py() calls"
        assert "NullCountOp" in src

        # (b) facade-wide no .as_py
        import cadw.dataframe as _dm
        facade_src = inspect.getsource(_dm)
        facade_tree = _ast.parse(facade_src)
        all_as_py = [
            n for n in _ast.walk(facade_tree)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "as_py"
        ]
        assert not all_as_py, (
            f"cadw/dataframe.py has {len(all_as_py)} .as_py() call(s) — "
            f"test_facade_has_no_as_py will fail"
        )

        # (c) isna_sum functional: same values as before
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            t = pa_.table({
                "a": pa_.array([1, None, 3, None, 5], type=pa_.int64()),
                "b": pa_.array(
                    ["x", "y", None, None, "z"], type=pa_.string(),
                ),
                "v": [10, 20, 30, 40, 50],
            })
            src_p = td / "s.parquet"
            pq_.write_table(t, src_p)
            df = _CDF.from_parquet(
                src_p, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spill",
            )
            assert df.isna_sum() == {"a": 2, "b": 2, "v": 0}

            # filtered
            filtered = df[df["v"] > 15]
            assert filtered.isna_sum()["a"] == 2
            assert filtered.isna_sum()["b"] == 2

        # (d) NullCountOp direct
        from cadw.ops.reduce import NullCountOp
        op = NullCountOp("x")
        b = pa_.RecordBatch.from_arrays(
            [pa_.array([1, None, 3, None], type=pa_.int64())],
            names=["x"],
        )
        assert op.update(op.init(), b) == 2
        assert op.finalize(op.update(op.init(), b)) == 2
        assert op.state_nbytes(op.init()) == 8

        print(
            "  isna_sum no .as_py + facade-wide clean + functional "
            "+ NullCountOp ok"
        )

    run_verify_in_subprocess(_verify)


# ============ 6. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_reduce.py",
         "tests/test_eda_basics.py",
         "tests/test_dataframe.py",
         "tests/test_dataframe_filters.py",
         "tests/test_api_availability.py",
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


# ============ 7. main ============

def main():
    print("=== v0.9.23g-fix6: NullCountOp for isna_sum ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    add_null_count_op()
    add_export()
    fix_isna_sum()
    add_reduce_tests()
    add_eda_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23g-fix6: isna_sum uses NullCountOp (no .as_py in "
        "facade). "
        "test_facade_has_no_as_py (AST guard, §3.8) forbids ANY "
        ".as_py() in cadw/dataframe.py. isna_sum used "
        "`_pc.sum(_pc.cast(_pc.is_null(...), int64)).as_py() or 0` "
        "-> 1 occurrence -> guard fires (correctly — pattern "
        "encourages per-row loops). "
        "Fix: NullCountOp ReduceOp in cadw/ops/reduce.py. "
        "batch.column(c).null_count is O(1) metadata (IPC-read "
        "batches precompute; pyarrow caches). No per-row iteration, "
        "no pc.is_null, no cast, no sum. State = 1 int64. "
        "isna_sum uses reduce(NullCountOp) per column — d passes, "
        "same cost as prior per-batch sum. No .as_py() in facade. "
        "Export NullCountOp from cadw.ops + cadw (public API). "
        "+tests/test_reduce.py (3 tests: NullCountOp init/state_"
        "nbytes, update/merge/finalize, no-nulls). "
        "+tests/test_eda_basics.py (1 AST test: isna_sum no "
        ".as_py + uses NullCountOp). "
        "Process note: 6th fix-cycle on v0.9.23g — highest for "
        "any single patch in the project. Cumulative root cause: "
        "v0.9.23g bundled 5 methods with 4 cross-cutting edge-case "
        "semantics (dropna/normalize/nulls/empty). Structural "
        "lessons fully enumerated: "
        "  - split by method, not by LOC, when edge-case semantics "
        "cross-cut (should have been 3 patches: fillna+isna, "
        "value_counts, nunique); "
        "  - apply_or_fail IS the idempotency guard — no custom "
        "`if marker in s: skip`; "
        "  - AST test for EVERY edge case: if code branches on "
        "`if 'marker' in s`, AST must verify; "
        "  - facade .as_py() is a hard NO — delegate to ReduceOp "
        "even for scalar reads (O(1) is irrelevant, the pattern is "
        "the vector); "
        "  - dogfood must exercise ALL parameters, not just default; "
        "  - multi-batch regression via chunk=1. "
        "Fast: ~2098 -> ~2102 passed.",
        self_path="scripts/patch_v0923g_fix6_isna_sum_nullop.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23g-fix6 applied. v0.9.23g COMPLETE (7.A.5 + 7.A.3).")


if __name__ == "__main__":
    main()