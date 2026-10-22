#!/usr/bin/env python3
# v0.9.23c-fix2 — 4 test fixes for v0.9.23c (Arrow batched merge).
#
# Bug 1: test_merge_multi_key_mixed_order uses _write_run with default
# schema (k, v, row_id), but rows have (k1, k2, v, row_id). _write_run
# wrote a file with schema (k, v, row_id), _merge_keys failed on
# streams[0].reader.schema.field("k1"). Fix: pass explicit schema.
#
# Bug 2: test_merge_keys_no_heapq uses substring `"heapq" not in src`.
# Docstring mentions "heapq" ("Replaces per-row Python heapq...") ->
# false positive. §13.4/161 (substring instead of AST). Fix:
# patch_utils.assert_no_call_in_source(src, 'heappush', 'heappop').
#
# Bug 3: test_merge_keys_has_guard uses `src.find(...)`. Docstring
# mentions "pc.sort_indices" (comment "# Sort spec for pc.sort_indices"
# at line 605) BEFORE actual check_state Call (line 1313). §13.4/185
# (substring ordering). Fix: patch_utils.assert_call_order(
# src, 'check_state', 'sort_indices').
#
# Bug 4: test_merge_keys_uses_arrow_builder (test_sort.py, v0.9.23c)
# still has `"heapq" not in src` -> same false positive as Bug 2. Fix:
# remove heapq assertion (test_merge_keys_no_heapq in test_merge_arrow.py
# covers it, AST-based after fix2). Keep pc.sort_indices + _RunStream.
#
# Run: uv run python scripts/patch_v0923c_fix2_test_drift.py

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


# ============ Bug 1: test_merge_multi_key_mixed_order explicit schema ==

def fix_multi_key_schema():
    print("\n=== Bug 1: test_merge_multi_key_mixed_order schema ===")
    p = Path("tests/test_merge_arrow.py")
    s = p.read_text()
    if "# v0.9.23c-fix2: explicit schema (k1, k2, v, row_id)" in s:
        print("  already fixed — skip")
        return

    old = '''def test_merge_multi_key_mixed_order(tmp_path):
    """Two keys, mixed asc/desc: [k1 asc, k2 desc]."""
    run1 = _write_run(tmp_path / "r1.arrow", [
        {"k1": 1, "k2": 10, "v": 1, "row_id": 0},
        {"k1": 1, "k2": 5, "v": 2, "row_id": 1},
        {"k1": 2, "k2": 100, "v": 3, "row_id": 2},
    ])
    run2 = _write_run(tmp_path / "r2.arrow", [
        {"k1": 1, "k2": 7, "v": 4, "row_id": 3},
        {"k1": 2, "k2": 50, "v": 5, "row_id": 4},
    ])'''

    new = '''def test_merge_multi_key_mixed_order(tmp_path):
    """Two keys, mixed asc/desc: [k1 asc, k2 desc]."""
    # v0.9.23c-fix2: explicit schema (k1, k2, v, row_id). Prior used
    # _write_run default schema (k, v, row_id), which does not contain
    # k1/k2 -> file was written with wrong schema -> _merge_keys KeyError.
    schema = pa.schema([
        ("k1", pa.int64()),
        ("k2", pa.int64()),
        ("v", pa.int64()),
        ("row_id", pa.uint64()),
    ])
    run1 = _write_run(tmp_path / "r1.arrow", [
        {"k1": 1, "k2": 10, "v": 1, "row_id": 0},
        {"k1": 1, "k2": 5, "v": 2, "row_id": 1},
        {"k1": 2, "k2": 100, "v": 3, "row_id": 2},
    ], schema=schema)
    run2 = _write_run(tmp_path / "r2.arrow", [
        {"k1": 1, "k2": 7, "v": 4, "row_id": 3},
        {"k1": 2, "k2": 50, "v": 5, "row_id": 4},
    ], schema=schema)'''

    if old not in s:
        raise SystemExit("test_merge_multi_key_mixed_order anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ Bug 2: test_merge_keys_no_heapq AST-based ============

def fix_no_heapq_test():
    print("\n=== Bug 2: test_merge_keys_no_heapq -> AST ===")
    p = Path("tests/test_merge_arrow.py")
    s = p.read_text()
    if "assert_no_call_in_source" in s and "heappush" in s:
        print("  already fixed — skip")
        return

    old = '''def test_merge_keys_no_heapq():
    """v0.9.23c: no heapq in _merge_keys."""
    import inspect
    from cadw.ops import sort as sort_mod
    src = inspect.getsource(sort_mod._merge_keys)
    assert "heapq" not in src, (
        "v0.9.23c: _merge_keys must not use heapq (per-row Python loop)"
    )'''

    new = '''def test_merge_keys_no_heapq():
    """v0.9.23c: no heapq.* Call in _merge_keys (AST-based).

    v0.9.23c-fix2: prior used `"heapq" not in src` — docstring says
    "Replaces per-row Python heapq with batched Arrow sort" -> substring
    false positive (§13.4/161). Now AST-based via
    patch_utils.assert_no_call_in_source, which walks Call nodes only
    (docstring/comments invisible).
    """
    import inspect
    from cadw.ops import sort as sort_mod
    from cadw.patch_utils import assert_no_call_in_source
    src = inspect.getsource(sort_mod._merge_keys)
    assert_no_call_in_source(
        src, "heappush", "heappop",
        fn_name="sort._merge_keys",
    )'''

    if old not in s:
        raise SystemExit("test_merge_keys_no_heapq anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ Bug 3: test_merge_keys_has_guard AST-based ============

def fix_has_guard_test():
    print("\n=== Bug 3: test_merge_keys_has_guard -> AST ===")
    p = Path("tests/test_merge_arrow.py")
    s = p.read_text()
    if "assert_call_order" in s and "test_merge_keys_has_guard" in s:
        print("  already fixed — skip")
        return

    old = '''def test_merge_keys_has_guard():
    """check_state called before pc.sort_indices (AST guard §3.11)."""
    import inspect
    from cadw.ops import sort as sort_mod
    src = inspect.getsource(sort_mod._merge_keys)
    cs = src.find("check_state")
    si = src.find("pc.sort_indices")
    assert cs >= 0 and si >= 0
    assert cs < si, (
        "check_state must precede pc.sort_indices (guards allocate)"
    )'''

    new = '''def test_merge_keys_has_guard():
    """check_state Call precedes pc.sort_indices Call (AST guard §3.11).

    v0.9.23c-fix2: prior used src.find(). Docstring mentions
    "pc.sort_indices" (in "# Sort spec for pc.sort_indices" comment,
    line ~605) BEFORE actual check_state Call (line ~1313) ->
    substring order wrong (§13.4/185). Now AST-based via
    patch_utils.assert_call_order, which compares Call node linenos
    (docstring/comments invisible).
    """
    import inspect
    from cadw.ops import sort as sort_mod
    from cadw.patch_utils import assert_call_order
    src = inspect.getsource(sort_mod._merge_keys)
    assert_call_order(
        src, "check_state", "sort_indices",
        fn_name="sort._merge_keys",
    )'''

    if old not in s:
        raise SystemExit("test_merge_keys_has_guard anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ Bug 4: test_sort.py substring heapq ============

def fix_sort_test_heapq():
    print("\n=== Bug 4: test_sort.py remove heapq substring ===")
    p = Path("tests/test_sort.py")
    s = p.read_text()
    if "# v0.9.23c-fix2: heapq assertion removed" in s:
        print("  already fixed — skip")
        return

    old = '''    assert "pc.sort_indices" in src, (
        "v0.9.23c: _merge_keys must use pc.sort_indices (Arrow-native "
        "batched merge, AUDIT_FULL_v0922 7.A.1)"
    )
    assert "heapq" not in src, (
        "v0.9.23c: _merge_keys must NOT use heapq (per-row Python loop "
        "removed in favor of batched Arrow sort)"
    )
    assert "_RunStream" in src, (
        "v0.9.23c: _merge_keys must use _RunStream (buffered reader)"
    )'''

    new = '''    assert "pc.sort_indices" in src, (
        "v0.9.23c: _merge_keys must use pc.sort_indices (Arrow-native "
        "batched merge, AUDIT_FULL_v0922 7.A.1)"
    )
    # v0.9.23c-fix2: heapq assertion removed. Substring `"heapq" not in
    # src` false-positived on docstring ("Replaces per-row Python heapq
    # ..."). AST-based check lives in test_merge_arrow.py::
    # test_merge_keys_no_heapq (patch_utils.assert_no_call_in_source).
    assert "_RunStream" in src, (
        "v0.9.23c: _merge_keys must use _RunStream (buffered reader)"
    )'''

    if old not in s:
        raise SystemExit("test_sort.py heapq assertion anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ Dogfood ============

def dogfood():
    print("\n=== dogfood: 4 fixed tests pass + merge identity ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import pyarrow as pa_
        import pyarrow.parquet as pq_

        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG
        from cadw.io.batch_writer import ArrowIPCBatchWriter as _W
        from cadw.ops.sort import _merge_keys as _mk

        # ---- multi-key mixed order (Bug 1 fixed) ----
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            schema = pa_.schema([
                ("k1", pa_.int64()),
                ("k2", pa_.int64()),
                ("v", pa_.int64()),
                ("row_id", pa_.uint64()),
            ])

            def _write(name, rows):
                p = td / name
                t = pa_.Table.from_pylist(rows, schema=schema)
                w = _W(str(p), schema)
                for b in t.to_batches():
                    w.write_batch(b)
                w.close()
                return p

            run1 = _write("r1.arrow", [
                {"k1": 1, "k2": 10, "v": 1, "row_id": 0},
                {"k1": 1, "k2": 5, "v": 2, "row_id": 1},
                {"k1": 2, "k2": 100, "v": 3, "row_id": 2},
            ])
            run2 = _write("r2.arrow", [
                {"k1": 1, "k2": 7, "v": 4, "row_id": 3},
                {"k1": 2, "k2": 50, "v": 5, "row_id": 4},
            ])
            out = td / "out.arrow"
            _mk(
                [run1, run2], ["k1", "k2", "v", "row_id"],
                ["k1", "k2"], ["k1", "k2", "v"],
                [True, False], False, out, read_batch_size=10,
                memory_guard=_MG(max_memory_mb=20),
            )
            r = pa_.ipc.open_file(str(out))
            got = []
            for i in range(r.num_record_batches):
                got.extend(r.get_batch(i).to_pylist())
            expected = [(1, 10), (1, 7), (1, 5), (2, 100), (2, 50)]
            assert [(x["k1"], x["k2"]) for x in got] == expected, got

        # ---- AST tests (Bug 2, 3) via helpers ----
        import inspect
        from cadw.ops import sort as _sort_mod
        from cadw.patch_utils import (
            assert_no_call_in_source as _ancs,
            assert_call_order as _aco,
        )
        _src = inspect.getsource(_sort_mod._merge_keys)
        _ancs(_src, "heappush", "heappop", fn_name="_merge_keys")
        _aco(_src, "check_state", "sort_indices", fn_name="_merge_keys")

        # ---- full external_sort correctness (integration) ----
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            import numpy as _np_
            rng = _np_.random.default_rng(3)
            n = 500
            ks = rng.integers(0, 30, size=n).astype(_np_.int64)
            vs = _np_.arange(n).astype(_np_.int64)
            tbl = pa_.table({"k": pa_.array(ks), "v": pa_.array(vs)})
            src = td / "big.parquet"
            pq_.write_table(tbl, src, row_group_size=50)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=50,
                memory_guard=_MG(max_memory_mb=10),
                spill_dir=td / "spill",
            )
            res = df.sort_values("k")
            outp = td / "out.parquet"
            res.write_parquet(outp)
            assert pq_.read_table(outp).column("k").to_pylist() == sorted(
                ks.tolist()
            )

        print("  Bug 1 (multi-key) + Bug 2/3 (AST) + integration ok")

    run_verify_in_subprocess(_verify)


# ============ Tests ============

def run_pytest():
    print("\n=== pytest: targeted (4 previously-failing tests) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_merge_arrow.py",
         "tests/test_sort.py",
         "tests/test_sort_merge_p_s1.py",
         "tests/test_shift_on_col.py",
         "tests/test_rolling_on_col.py",
         "tests/test_groupby_per_group.py",
         "tests/test_rank.py",
         "tests/test_quantile.py",
         "tests/test_resample.py",
         "-q"],
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


# ============ main ============

def main():
    print("=== v0.9.23c-fix2: 4 test fixes ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    fix_multi_key_schema()
    fix_no_heapq_test()
    fix_has_guard_test()
    fix_sort_test_heapq()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23c-fix2: 4 test fixes for Arrow batched merge. "
        "All 4 failures were test bugs, production code correct "
        "(dogfood passed in v0.9.23c-fix1; 226/230 targeted tests "
        "already green). "
        "(1) test_merge_multi_key_mixed_order: _write_run default "
        "schema is (k, v, row_id) but rows are (k1, k2, v, row_id) "
        "-> file written with wrong schema -> _merge_keys KeyError "
        "'k1 not in schema'. Fix: explicit schema. "
        "(2) test_merge_keys_no_heapq: substring `'heapq' not in "
        "src` false-positived on docstring (\"Replaces per-row "
        "Python heapq...\"). §13.4/161 (substring instead of AST). "
        "Fix: patch_utils.assert_no_call_in_source(src, 'heappush', "
        "'heappop') — AST Call nodes only, docstrings invisible. "
        "(3) test_merge_keys_has_guard: src.find() order was wrong "
        "because docstring mentions pc.sort_indices (comment "
        "'# Sort spec for pc.sort_indices' at ~line 605) before "
        "actual check_state Call at ~line 1313. §13.4/185 "
        "(substring ordering). Fix: patch_utils.assert_call_order("
        "src, 'check_state', 'sort_indices') — AST Call linenos. "
        "(4) test_merge_keys_uses_arrow_builder in test_sort.py: "
        "same `'heapq' not in src` false positive (v0.9.23c update "
        "kept it); remove assertion, AST check lives in test_merge_"
        "arrow.py::test_merge_keys_no_heapq. Keep pc.sort_indices + "
        "_RunStream assertions. "
        "Fast: 1950 passed unchanged (test-only fixes, "
        "no new tests added in fix2; +5 dogfood checks not "
        "counted in pytest).",
        self_path="scripts/patch_v0923c_fix2_test_drift.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23c-fix2 applied. v0.9.23c complete (7.A.1).")


if __name__ == "__main__":
    main()