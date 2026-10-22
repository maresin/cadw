#!/usr/bin/env python3
# v0.9.23i-fix1 — extend cadw/ops/__init__.py (skipping cadw/__init__.py).
#
# v0.9.23i anchor miss: cadw/__init__.py has no
# `from .ops.join import ...` line (existing project state — hash_join
# is in __all__ but not imported; possibly transitively reachable via
# dataframe module). Fix: only extend cadw/ops/__init__.py (safe
# single-line import anchor). cadw/__init__.py untouched — anti/outer
# accessible via cadw.ops.join.anti_join_streaming or `from cadw.ops
# import anti_join_streaming`; internal users (dataframe.py, tests)
# already use cadw.ops.join path directly.
#
# Completes v0.9.23i (anti + outer join):
#   - verify cadw/ops/join.py has anti/outer (applied in v0.9.23i)
#   - verify dataframe.py merge dispatch (applied in v0.9.23i)
#   - extend cadw/ops/__init__.py imports + __all__
#   - write test file (v0.9.23i attempted, failed before write)
#   - dogfood, pytest, commit
#
# Run: uv run python scripts/patch_v0923i_fix1_ops_init.py

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


# ============ 1. Verify v0.9.23i partial state ============

def verify_state():
    print("\n=== verify v0.9.23i partial state ===")
    join_src = Path("cadw/ops/join.py").read_text()
    if "def anti_join_streaming(" not in join_src:
        raise SystemExit(
            "cadw/ops/join.py missing anti_join_streaming — "
            "v0.9.23i add_anti_outer() did not apply. Re-run v0.9.23i."
        )
    if "def outer_join_streaming(" not in join_src:
        raise SystemExit(
            "cadw/ops/join.py missing outer_join_streaming"
        )
    df_src = Path("cadw/dataframe.py").read_text()
    if '"outer", "anti", "semi"' not in df_src:
        raise SystemExit(
            "cadw/dataframe.py missing merge how check extension"
        )
    if "_anti(" not in df_src:
        raise SystemExit(
            "cadw/dataframe.py missing merge dispatch to anti"
        )
    print("  join.py has anti/outer, dataframe.py has merge dispatch — OK")


# ============ 2. Extend cadw/ops/__init__.py ============

def fix_ops_init():
    print("\n=== extend cadw/ops/__init__.py ===")
    p = Path("cadw/ops/__init__.py")
    apply_or_fail(
        str(p),
        old=(
            'from .join import hash_join, hash_join_streaming, inner_join, left_join'
        ),
        new=(
            'from .join import (\n'
            '    hash_join, hash_join_streaming, inner_join, left_join,\n'
            '    anti_join_streaming, outer_join_streaming,\n'
            ')'
        ),
        label="ops_init_join_import",
    )
    apply_or_fail(
        str(p),
        old=(
            '    "external_sort", "read_sorted", "compute_fan_in", "hash_join", "hash_join_streaming", "inner_join", "left_join",'
        ),
        new=(
            '    "external_sort", "read_sorted", "compute_fan_in", "hash_join", "hash_join_streaming", "inner_join", "left_join",\n'
            '    "anti_join_streaming", "outer_join_streaming",'
        ),
        label="ops_init_join_all",
    )
    assert_compiles(p)
    print("  OK")


# ============ 3. Write test file (v0.9.23i attempted, was not written) ============

TESTS = '''"""v0.9.23i: anti join + outer join (7.A.10 split 1/3)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=20):
    t = pa.table(data)
    p = tmp_path / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=tmp_path / f"spill_{name}",
    )


def _rows(df):
    out = []
    for b in df._iter_filtered_batches():
        out.extend(b.to_pylist())
    return out


# ============ anti join ============

def test_anti_basic(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2, 3, 4], "a": ["w", "x", "y", "z"]}, name="L")
    right = make_df(tmp_path, {"k": [1, 3], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="anti")
    rows = sorted(_rows(result), key=lambda r: r["k"])
    # Keys 2, 4 absent from right
    assert [r["k"] for r in rows] == [2, 4]
    assert [r["a"] for r in rows] == ["x", "z"]
    assert "b" not in rows[0]


def test_anti_all_match(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2, 3]}, name="L")
    right = make_df(tmp_path, {"k": [1, 2, 3]}, name="R")
    result = left.merge(right, on="k", how="anti")
    assert result.count() == 0


def test_anti_none_match(tmp_path):
    left = make_df(tmp_path, {"k": [10, 20]}, name="L")
    right = make_df(tmp_path, {"k": [1, 2, 3]}, name="R")
    result = left.merge(right, on="k", how="anti")
    assert result.count() == 2


def test_anti_empty_right(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2]}, name="L")
    right_data = {"k": pa.array([], type=pa.int64())}
    right = make_df(tmp_path, right_data, name="R")
    result = left.merge(right, on="k", how="anti")
    assert result.count() == 2


def test_anti_null_left_key_emitted(tmp_path):
    """SQL semantics: null left key never matches -> emitted in anti."""
    left = make_df(tmp_path, {
        "k": pa.array([1, None, 3], type=pa.int64()),
    }, name="L")
    right = make_df(tmp_path, {"k": [1, 2]}, name="R")
    result = left.merge(right, on="k", how="anti")
    rows = _rows(result)
    # 1 matched; None and 3 unmatched
    assert len(rows) == 2
    keys = {r["k"] for r in rows}
    assert 3 in keys
    assert None in keys


def test_anti_multi_key(tmp_path):
    left = make_df(tmp_path, {
        "k1": [1, 1, 2, 2], "k2": ["a", "b", "a", "b"],
        "v": [10, 20, 30, 40],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1, 2], "k2": ["a", "a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="anti")
    rows = sorted(_rows(result), key=lambda r: (r["k1"], r["k2"]))
    assert len(rows) == 2
    assert [(r["k1"], r["k2"]) for r in rows] == [(1, "b"), (2, "b")]


# ============ outer join ============

def test_outer_basic(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2, 3], "a": ["w", "x", "y"]}, name="L")
    right = make_df(tmp_path, {"k": [2, 3, 4], "b": ["p", "q", "r"]}, name="R")
    result = left.merge(right, on="k", how="outer")
    rows = sorted(_rows(result), key=lambda r: r["k"])
    assert [r["k"] for r in rows] == [1, 2, 3, 4]
    by_k = {r["k"]: r for r in rows}
    assert by_k[1]["a"] == "w" and by_k[1]["b"] is None
    assert by_k[2]["a"] == "x" and by_k[2]["b"] == "p"
    assert by_k[3]["a"] == "y" and by_k[3]["b"] == "q"
    assert by_k[4]["a"] is None and by_k[4]["b"] == "r"


def test_outer_all_match(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2], "a": ["x", "y"]}, name="L")
    right = make_df(tmp_path, {"k": [1, 2], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="outer")
    assert result.count() == 2


def test_outer_disjoint(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2], "a": ["x", "y"]}, name="L")
    right = make_df(tmp_path, {"k": [3, 4], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="outer")
    assert result.count() == 4
    rows = sorted(_rows(result), key=lambda r: r["k"])
    assert [r["k"] for r in rows] == [1, 2, 3, 4]


def test_outer_null_left_emitted(tmp_path):
    left = make_df(tmp_path, {
        "k": pa.array([1, None], type=pa.int64()),
        "a": ["x", "y"],
    }, name="L")
    right = make_df(tmp_path, {"k": [1, 2], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="outer")
    rows = _rows(result)
    assert len(rows) == 3
    by_k = {r["k"]: r for r in rows if r["k"] is not None}
    assert by_k[1]["b"] == "p"
    assert by_k[2]["a"] is None
    null_row = next(r for r in rows if r["k"] is None)
    assert null_row["a"] == "y"
    assert null_row["b"] is None


def test_outer_empty_left(tmp_path):
    left_data = {
        "k": pa.array([], type=pa.int64()),
        "a": pa.array([], type=pa.string()),
    }
    left = make_df(tmp_path, left_data, name="L")
    right = make_df(tmp_path, {"k": [1, 2], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="outer")
    rows = sorted(_rows(result), key=lambda r: r["k"])
    assert [r["k"] for r in rows] == [1, 2]
    assert all(r["a"] is None for r in rows)


def test_outer_empty_right(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2], "a": ["x", "y"]}, name="L")
    right_data = {
        "k": pa.array([], type=pa.int64()),
        "b": pa.array([], type=pa.string()),
    }
    right = make_df(tmp_path, right_data, name="R")
    result = left.merge(right, on="k", how="outer")
    rows = sorted(_rows(result), key=lambda r: r["k"])
    assert [r["k"] for r in rows] == [1, 2]
    assert all(r["b"] is None for r in rows)


def test_outer_multi_key(tmp_path):
    left = make_df(tmp_path, {
        "k1": [1, 1, 2], "k2": ["a", "b", "a"], "v": [10, 20, 30],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1, 2, 3], "k2": ["a", "a", "a"], "w": [100, 200, 300],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="outer")
    rows = sorted(_rows(result), key=lambda r: (r["k1"], r["k2"]))
    assert len(rows) == 4
    assert rows[0]["v"] == 10 and rows[0]["w"] == 100
    assert rows[1]["v"] == 20 and rows[1]["w"] is None
    assert rows[2]["v"] == 30 and rows[2]["w"] == 200
    assert rows[3]["v"] is None and rows[3]["w"] == 300


# ============ Dispatch / errors ============

def test_merge_how_unsupported_raises(tmp_path):
    left = make_df(tmp_path, {"k": [1]}, name="L")
    right = make_df(tmp_path, {"k": [1]}, name="R")
    with pytest.raises(NotImplementedError, match="right"):
        left.merge(right, on="k", how="right")


def test_merge_how_semi_raises(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2]}, name="L")
    right = make_df(tmp_path, {"k": [1]}, name="R")
    with pytest.raises(NotImplementedError, match="semi"):
        left.merge(right, on="k", how="semi")


# ============ AST guards ============

def test_anti_uses_hash_shuffle():
    import inspect
    from cadw.ops import join as j
    src = inspect.getsource(j.anti_join_streaming)
    assert "hash_shuffle" in src


def test_outer_dispatches_via_merge():
    """AST: merge() dispatches how='outer' to left+anti composition."""
    import inspect
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame.merge)
    assert '"outer"' in src
    assert '"anti"' in src


def test_anti_uses_is_in():
    import inspect
    from cadw.ops import join as j
    src = inspect.getsource(j._process_partition_pair_anti)
    assert "is_in" in src


def test_anti_fill_null_false():
    """Anti: null left key must NOT be silently matched (fill_null False)."""
    import inspect
    from cadw.ops import join as j
    src = inspect.getsource(j._process_partition_pair_anti)
    assert "fill_null" in src
    assert "False" in src


# ============ Export from cadw.ops ============

def test_anti_outer_exported_from_cadw_ops():
    """Public API: cadw.ops.anti_join_streaming / outer_join_streaming."""
    from cadw.ops import (
        anti_join_streaming, outer_join_streaming,
    )
    assert callable(anti_join_streaming)
    assert callable(outer_join_streaming)
'''


def write_test_file():
    print("\n=== write tests/test_anti_outer_join.py ===")
    p = Path("tests/test_anti_outer_join.py")
    if p.exists():
        print("  already exists — skip")
        return
    p.write_text(TESTS)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 4. Dogfood ============

def dogfood():
    print("\n=== dogfood: anti + outer invariants ===")

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

        # (a) Anti: invariant — emitted keys NOT in right; all unmatched.
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
            for r in anti_rows:
                assert r["k"] not in right_keys, (
                    f"anti emitted matched key {r['k']}"
                )
            left_keys = set(left_t.column("k").to_pylist())
            expected_anti_keys = left_keys - right_keys
            assert {r["k"] for r in anti_rows} == expected_anti_keys

        # (b) Outer: union + padding.
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

        # (c) Anti with null-left key: emitted.
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
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spillL",
            )
            right = _CDF.from_parquet(
                src_r, td / "dsR", chunk_size=1,
                memory_guard=_MG(max_memory_mb=20),
                spill_dir=td / "spillR",
            )
            anti = left.merge(right, on="k", how="anti")
            rows = _rows(anti)
            keys = {r["k"] for r in rows}
            assert 3 in keys
            assert None in keys

        # (d) Semi placeholder raises
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

        # (e) Export from cadw.ops
        from cadw.ops import (
            anti_join_streaming as _aj, outer_join_streaming as _oj,
        )
        assert callable(_aj)
        assert callable(_oj)

        print(
            "  anti (not-in invariant) + outer (union+padding) + "
            "null-left anti + semi placeholder + cadw.ops export ok"
        )

    run_verify_in_subprocess(_verify)


# ============ 5. Tests ============

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


# ============ 6. main ============

def main():
    print("=== v0.9.23i-fix1: ops/__init__ + test file ===")
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
    fix_ops_init()
    write_test_file()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23i: anti join + outer join (AUDIT_FULL_v0922 item "
        "7.A.10, split 1/3, Top-10 gap #4). "
        "anti_join_streaming(left, right, on): left rows whose key "
        "absent from right. HashShuffle + _safe_materialize_partition. "
        "NULL semantics: null left key never matches -> emitted "
        "(SQL semantics); null right key never matches -> doesn't "
        "exclude any left. fill_null(False) on is_in mask (null "
        "left -> treated as 'not in' -> emit). "
        "outer_join_streaming: implemented via merge dispatch "
        "(needs re-iterable right, so not for direct iteration). "
        "Composition in merge(how='outer'): left_join(left, right) "
        "-> all left + right padded; anti_join(right, left) -> right-"
        "only rows; pad with null left cols; concat via IPC writer. "
        "merge(how=...) extended: inner | left | outer | anti | semi "
        "(semi -> NotImplementedError in v0.9.23i, v0.9.23j). "
        "'right' still NotImplementedError (v0.9.23j). "
        "fix1 (v0.9.23i anchor miss recovery): "
        "cadw/__init__.py has no `from .ops.join import ...` line "
        "(existing project state — hash_join in __all__ but not "
        "imported). Fix: only extend cadw/ops/__init__.py (safe "
        "single-line anchor). cadw/__init__.py untouched — anti/"
        "outer accessible via cadw.ops.join.anti_join_streaming or "
        "`from cadw.ops import ...`; dataframe.py and tests use "
        "cadw.ops.join path directly. "
        "+tests/test_anti_outer_join.py (19 tests: anti basic, "
        "all-match, none-match, empty-right, null-left-key, multi-"
        "key; outer basic (both+padding), all-match, disjoint, "
        "null-left, empty-left, empty-right, multi-key; unsupported "
        "how raises, semi NotImplementedError; 4 AST guards - "
        "hash_shuffle, is_in, fill_null(False), merge dispatch; 1 "
        "export test — cadw.ops.anti_join_streaming/outer_join_"
        "streaming callable). "
        "Dogfood invariants only (NO distributions, v0.9.23h "
        "lesson): anti = not-in invariant per emitted key + "
        "expected-anti-set == left-right; outer = union of keys + "
        "null-padding correctness; null-left anti emitted; semi "
        "raises; cadw.ops export callable. "
        "Split rationale (v0.9.23h rule > 200 LOC): anti+outer "
        "together ~150 LOC (this patch), semi ~40 LOC (v0.9.23j), "
        "categorical FE 3 patches (v0.9.23k-l). "
        "Fast: 2125 -> ~2144 passed.",
        self_path="scripts/patch_v0923i_fix1_ops_init.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23i-fix1 applied. 7.A.10 partial (anti + outer).")


if __name__ == "__main__":
    main()