#!/usr/bin/env python3
# v0.9.23c-fix1 — three fixes for v0.9.23c (Arrow batched merge).
#
# Bug 1 (dogfood-caught): np.unique used in _merge_keys but np not
# imported at module level in cadw/ops/sort.py (numpy imported locally
# inside _NumpyBackedBuilder only). Fix: replace np.unique with
# Arrow-native pc.value_counts on __run_idx column. K = active runs
# (bounded by fan_in ≈ 8), safe per §9.2.13. No numpy import needed.
#
# Bug 2 (latent test-drift): test_merge_keys_has_check_batch in
# tests/test_sort_merge_p_s1.py asserts "check_batch" in source
# _merge_keys. In v0.9.23c, check_batch moved to _RunStream._next_batch
# (called via refill from _merge_keys). Update test to accept either
# form and assert _RunStream._next_batch has check_batch.
#
# Bug 3 (latent, proactively): _python_sorted_reference in
# tests/test_merge_arrow.py nests row_id inside each per-column tuple,
# so for multi-key sort row_id decides before second column is compared.
# Fix: flat tuple (null_rank_1, value_1, ..., row_id). Current tests
# pass with buggy version (only 1 key column) but flat version is
# correct for future multi-key tests.
#
# Run: uv run python scripts/patch_v0923c_fix1_np_import.py

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


# ============ Bug 1: np.unique -> pc.value_counts ============

def fix_np_import():
    print("\n=== Bug 1: np.unique -> pc.value_counts ===")
    p = Path("cadw/ops/sort.py")
    s = p.read_text()
    if "pc.value_counts(top.column" in s:
        print("  already fixed — skip")
        return
    apply_or_fail(
        str(p),
        old=(
            '            # Advance each run by how many of its rows were taken\n'
            '            run_idx_np = top.column("__run_idx").to_numpy(zero_copy_only=False)\n'
            '            unique, counts = np.unique(run_idx_np, return_counts=True)\n'
            '            consumed = {\n'
            '                int(u): int(c)\n'
            '                for u, c in zip(unique.tolist(), counts.tolist())\n'
            '            }'
        ),
        new=(
            '            # Advance each run by how many of its rows were taken.\n'
            '            # Arrow-native via pc.value_counts (K = active runs,\n'
            '            # bounded by fan_in <= 8; safe per §9.2.13).\n'
            '            # v0.9.23c-fix1: prior np.unique failed because numpy\n'
            '            # is not imported at module level in sort.py (numpy\n'
            '            # imported lazily inside _NumpyBackedBuilder only).\n'
            '            vc = pc.value_counts(top.column("__run_idx"))\n'
            '            consumed = {\n'
            '                int(v): int(c)\n'
            '                for v, c in zip(\n'
            '                    vc.field("values").to_pylist(),\n'
            '                    vc.field("counts").to_pylist(),\n'
            '                )\n'
            '            }'
        ),
        label="merge_keys_np_import",
    )
    assert_compiles(p)
    print("  OK")


# ============ Bug 2: test_merge_keys_has_check_batch ============

def fix_check_batch_test():
    print("\n=== Bug 2: test_merge_keys_has_check_batch ===")
    p = Path("tests/test_sort_merge_p_s1.py")
    s = p.read_text()
    if "v0.9.23c: check_batch moved into _RunStream._next_batch" in s:
        print("  already fixed — skip")
        return

    old = '''def test_merge_keys_has_check_batch():
    import inspect
    from cadw.ops import sort as sort_mod
    src = inspect.getsource(sort_mod._merge_keys)
    assert "memory_guard" in src, (
        "_merge_keys must accept memory_guard param"
    )
    assert "check_batch" in src, (
        "_merge_keys must call check_batch on read batches"
    )'''

    new = '''def test_merge_keys_has_check_batch():
    """v0.9.23c: check_batch moved into _RunStream._next_batch.

    Prior (v0.5.2a): _merge_keys directly called check_batch on each
    read batch. v0.9.23c (Arrow batched merge) delegates read + guard
    to _RunStream.refill(memory_guard) -> _RunStream._next_batch,
    which calls memory_guard.check_batch on each batch.
    """
    import inspect
    from cadw.ops import sort as sort_mod
    src = inspect.getsource(sort_mod._merge_keys)
    assert "memory_guard" in src, (
        "_merge_keys must accept memory_guard param"
    )
    # Either direct check_batch OR refill(memory_guard) delegating to
    # _RunStream._next_batch (which has check_batch).
    refill_with_guard = "refill(memory_guard)" in src
    check_batch_direct = "check_batch" in src
    assert refill_with_guard or check_batch_direct, (
        "_merge_keys must guard read batches: either direct "
        "check_batch or refill(memory_guard) (v0.9.23c: _RunStream "
        "checks internally)"
    )
    # Also verify _RunStream._next_batch has the actual check_batch call.
    stream_src = inspect.getsource(sort_mod._RunStream)
    assert "check_batch" in stream_src, (
        "_RunStream._next_batch must call check_batch on read batches"
    )
    assert "check_batch" in inspect.getsource(
        sort_mod._RunStream._next_batch
    ), "_RunStream._next_batch must call check_batch directly"'''

    if old not in s:
        raise SystemExit("test_merge_keys_has_check_batch anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ Bug 3: _python_sorted_reference flat tuple ============

def fix_python_sorted_reference():
    print("\n=== Bug 3: _python_sorted_reference flat tuple ===")
    p = Path("tests/test_merge_arrow.py")
    s = p.read_text()
    if "v0.9.23c-fix1: flat tuple" in s:
        print("  already fixed — skip")
        return

    old = '''def _python_sorted_reference(rows, key_columns, ascending, nulls_first):
    """Stable Python sorted with per-column asc/desc + nulls_first + row_id.

    Matches external_sort's contract exactly. row_id monotonic with
    input order -> stable sort by key equals heap's (key, row_id) compare.
    """
    def _cmp_key(row):
        parts = []
        for col, asc in zip(key_columns, ascending):
            v = row[col]
            # nulls_first: nulls compare less (0); else greater (2)
            if v is None:
                parts.append((0 if nulls_first else 2, 0, row["row_id"]))
            else:
                parts.append((1, v if asc else _Neg(v), row["row_id"]))
        return tuple(parts)
    return sorted(rows, key=_cmp_key)'''

    new = '''def _python_sorted_reference(rows, key_columns, ascending, nulls_first):
    """Stable Python sorted with per-column asc/desc + nulls_first + row_id.

    Matches external_sort's contract exactly. row_id monotonic with
    input order -> stable sort by key equals heap's (key, row_id) compare.

    v0.9.23c-fix1: flat tuple (not per-column sub-tuples). Prior
    version nested row_id inside each per-column tuple, causing
    row_id to decide before second key column on multi-key sorts.
    Current tests pass with buggy version (only 1 key column), but
    flat tuple is correct for any key count.
    """
    def _cmp_key(row):
        parts = []
        for col, asc in zip(key_columns, ascending):
            v = row[col]
            # nulls_first: nulls compare less (0); else greater (2)
            if v is None:
                parts.append(0 if nulls_first else 2)
                parts.append(0)  # placeholder value for null
            else:
                parts.append(1)
                parts.append(v if asc else _Neg(v))
        parts.append(row["row_id"])
        return tuple(parts)
    return sorted(rows, key=_cmp_key)'''

    if old not in s:
        raise SystemExit("_python_sorted_reference anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ Dogfood ============

def dogfood():
    print("\n=== dogfood: merge identity + integration ===")

    def _verify():
        import tempfile
        from pathlib import Path as _P
        import numpy as _np_
        import pyarrow as pa_
        import pyarrow.parquet as pq_

        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG
        from cadw.io.batch_writer import ArrowIPCBatchWriter as _W
        from cadw.ops.sort import _merge_keys as _mk

        # ---- small explicit ----
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            schema = pa_.schema([
                ("k", pa_.int64()),
                ("v", pa_.int64()),
                ("row_id", pa_.uint64()),
            ])
            runs = [
                [{"k": 1, "v": 10, "row_id": 0},
                 {"k": 5, "v": 50, "row_id": 2}],
                [{"k": 3, "v": 30, "row_id": 1},
                 {"k": 7, "v": 70, "row_id": 3}],
            ]
            paths = []
            for i, run in enumerate(runs):
                p = td / ("r%d.arrow" % i)
                t = pa_.Table.from_pylist(run, schema=schema)
                w = _W(str(p), schema)
                for b in t.to_batches():
                    w.write_batch(b)
                w.close()
                paths.append(p)

            out = td / "out.arrow"
            _mk(
                paths, ["k", "v", "row_id"], ["k"], ["k", "v"],
                [True], False, out, read_batch_size=10,
                memory_guard=_MG(max_memory_mb=20),
            )
            r = pa_.ipc.open_file(str(out))
            got = []
            for i in range(r.num_record_batches):
                got.extend(r.get_batch(i).to_pylist())
            assert [x["k"] for x in got] == [1, 3, 5, 7], got

        # ---- external_sort 1000-row ----
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            rng = _np_.random.default_rng(7)
            n = 1000
            ks = rng.integers(0, 50, size=n).astype(_np_.int64)
            vs = _np_.arange(n).astype(_np_.int64)
            tbl = pa_.table({"k": pa_.array(ks), "v": pa_.array(vs)})
            src = td / "big.parquet"
            pq_.write_table(tbl, src, row_group_size=50)
            df = _CDF.from_parquet(
                src, td / "big_ds", chunk_size=50,
                memory_guard=_MG(max_memory_mb=10),
                spill_dir=td / "big_spill",
            )
            res = df.sort_values("k")
            outp = td / "big_out.parquet"
            res.write_parquet(outp)
            got_k = pq_.read_table(outp).column("k").to_pylist()
            assert got_k == sorted(ks.tolist()), "external_sort 1000-row"

        # ---- identity: random 1000 rows, ties, multi-run ----
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            rng = _np_.random.default_rng(42)
            n = 1000
            keys = rng.integers(0, 20, size=n).astype(_np_.int64).tolist()
            vals = list(range(n))
            # Split into 4 sorted runs by key
            indices = sorted(range(n), key=lambda i: keys[i])
            runs = [[], [], [], []]
            for pos, i in enumerate(indices):
                runs[pos % 4].append({"k": keys[i], "v": vals[i],
                                       "row_id": i})
            schema = pa_.schema([
                ("k", pa_.int64()),
                ("v", pa_.int64()),
                ("row_id", pa_.uint64()),
            ])
            paths = []
            for ri, run_rows in enumerate(runs):
                run_rows.sort(key=lambda r: (r["k"], r["row_id"]))
                p = td / ("r%d.arrow" % ri)
                t = pa_.Table.from_pylist(run_rows, schema=schema)
                w = _W(str(p), schema)
                for b in t.to_batches():
                    w.write_batch(b)
                w.close()
                paths.append(p)

            out = td / "out.arrow"
            _mk(
                paths, ["k", "v", "row_id"], ["k"], ["k", "v"],
                [True], False, out, read_batch_size=10,
                memory_guard=_MG(max_memory_mb=50),
            )
            r = pa_.ipc.open_file(str(out))
            got = []
            for i in range(r.num_record_batches):
                got.extend(r.get_batch(i).to_pylist())
            expected = sorted(
                [{"k": keys[i], "v": vals[i], "row_id": i}
                 for i in range(n)],
                key=lambda r: (r["k"], r["row_id"]),
            )
            assert [(x["k"], x["v"]) for x in got] == [
                (x["k"], x["v"]) for x in expected
            ], "identity (k-tie-break)"

        print("  small explicit + ext_sort 1000-row + identity (ties) ok")

    run_verify_in_subprocess(_verify)


# ============ Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
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
    print("=== v0.9.23c-fix1: np.unique, stale check_batch test, "
          "multi-key sort helper ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    fix_np_import()
    fix_check_batch_test()
    fix_python_sorted_reference()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23c: Arrow-native batched merge in _merge_keys "
        "(AUDIT_FULL_v0922 item 7.A.1). Replaces per-row Python heapq "
        "(sort.py:325-430) with batched Arrow sort: _RunStream buffers "
        "rbs rows per run -> pa.concat_tables + __run_idx tag -> "
        "pc.sort_indices (C++) with per-column asc/desc + nulls_first + "
        "row_id tie-break -> take first rbs -> write -> advance cursors. "
        "Output identity: byte-for-byte same row order (row_id monotonic "
        "with input order so stable sorted == heap tie-break). Peak: "
        "3*K*rbs*row_bytes (buffers + merged + sorted); at K=8, rbs=1000, "
        "row=640 -> ~15 MB. Pre-fix measured 24.4s at N=500k/8 cols/32 MB "
        "(audit §6.6); post-fix target <5s. Semantics shift: "
        "read_batch_size now means buffer-per-run (was output flush "
        "size); output batch shape unchanged (<= rbs rows). Guard: "
        "check_state at top bounds 3*K*rbs*row_bytes; check_batch per "
        "refill read (in _RunStream._next_batch). "
        "fix1 (three fixes, all caught by dogfood + test-drift): "
        "(1) np.unique in _merge_keys -> pc.value_counts (numpy not "
        "imported at module level in sort.py; Arrow-native is cleaner, "
        "K bounded by fan_in <= 8 per §9.2.13); (2) test_merge_keys_"
        "has_check_batch updated (check_batch moved into _RunStream."
        "_next_batch; test accepts either direct or delegated form); "
        "(3) _python_sorted_reference flat tuple (prior per-column "
        "sub-tuples nested row_id inside first column, wrong for "
        "multi-key; current tests pass but footgun). "
        "+tests/test_merge_arrow.py (15 tests: identity vs Python "
        "sorted, mixed asc/desc, nulls_first/last, tie-break, multi-"
        "level merge, edge cases rbs=1 / rbs>total / K=1 / empty, "
        "memory guard, 3 AST); +tests/test_merge_arrow_bench.py (slow: "
        "N=500k/8 cols/32 MB <5s assertion). "
        "Fast: 1935 -> 1950 passed.",
        self_path="scripts/patch_v0923c_fix1_np_import.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23c-fix1 applied. v0.9.23c complete (7.A.1).")


if __name__ == "__main__":
    main()