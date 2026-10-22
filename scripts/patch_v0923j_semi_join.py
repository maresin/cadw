#!/usr/bin/env python3
# v0.9.23j — semi join (AUDIT_FULL_v0922 7.A.10, split 3/3 final).
#
# Semi join = mirror of anti: emit left rows whose key IS matched in
# right. One-line diff from anti (_process_partition_pair_semi:
# filter(matched) vs filter(unmatched)). Same composite-key multi-
# column correctness. Same SQL null semantics: null left key never
# matches -> NOT emitted (opposite of anti).
#
# Lessons from v0.9.23i (5 fix-cycle) applied upfront:
#   1. composite key from start (no AND-of-per-column is_in);
#   2. explicit `import pyarrow.compute as _pc`;
#   3. no dead code (isinstance(filtered, pa.Table));
#   4. module-level SEMI_TESTS constant (no closure `extra`);
#   5. verify_state at start (planner map + predicted_output);
#   6. enum-stale-test grep BEFORE commit (how="semi" in tests);
#   7. crossed-tuple test (canonical multi-key detector);
#   8. pre-flight test_facade_has_no_as_py;
#   9. dogfood invariants only (no distributions);
#  10. copy anti, don't share helper (minimal risk to tested anti).
#
# Run: uv run python scripts/patch_v0923j_semi_join.py

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


# ============ 0. Verify v0.9.23i-fix2 state (planner map, warning) ====

def verify_state():
    print("\n=== verify v0.9.23i state (planner + predicted_output) ===")
    p = Path("cadw/dataframe.py")
    src = p.read_text()
    if '_planner_how = (' not in src:
        raise SystemExit(
            "merge() missing _planner_how — v0.9.23i-fix2 did not apply. "
            "Re-run fix2."
        )
    if 'if how in ("anti", "semi"):' not in src:
        raise SystemExit(
            "merge() missing predicted_output per how — fix2 did not apply."
        )
    print("  _planner_how + predicted_output-per-how present — OK")


# ============ 1. _process_partition_pair_semi (copy of anti, filter=matched) ====

SEMI_FN = '''

def _process_partition_pair_semi(
    left_path: Path,
    right_path: Path,
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    memory_guard: MemoryGuard,
    spill_dir: Path,
    *,
    depth: int = 0,
    max_depth: int = 5,
):
    """Generator: semi-join one partition pair (v0.9.23j).

    Mirror of _process_partition_pair_anti with filter(matched) instead
    of filter(unmatched). Emits left rows whose key IS matched in right.

    SQL semantics: null left key never matches -> NOT emitted (opposite
    of anti). Null right key never matches -> doesn't make any left row
    matched.

    Multi-column: composite key via _combine_join_keys (fix3 lesson —
    never AND-of-per-column is_in, cross-product false positive).
    """
    import pyarrow.compute as _pc

    lp_size = left_path.stat().st_size if left_path.exists() else 0
    rp_size = right_path.stat().st_size if right_path.exists() else 0
    if lp_size == 0:
        return
    limit = int(memory_guard.available_bytes() * 0.5)
    need_split = (lp_size > limit) or (rp_size > limit)

    if need_split:
        if depth >= max_depth:
            raise MemoryLimitExceeded(
                f"semi_join: depth={depth} >= max_depth={max_depth}; "
                f"lp={lp_size}, rp={rp_size}, limit={limit}."
            )
        sub_left_dir = spill_dir / f"semi_l_d{depth}_{left_path.stem}"
        sub_right_dir = spill_dir / f"semi_r_d{depth}_{right_path.stem}"
        sub_left_dir.mkdir(parents=True, exist_ok=True)
        sub_right_dir.mkdir(parents=True, exist_ok=True)
        left_rest = [c for c in left_schema.names if c not in on]
        right_rest = [c for c in right_schema.names if c not in on]

        def _iter_lp():
            for b in ArrowIPCBatchReader(str(left_path), None):
                keep = [c for c in list(on) + left_rest
                        if c in b.schema.names]
                yield b.select(keep)

        def _iter_rp():
            for b in ArrowIPCBatchReader(str(right_path), None):
                keep = [c for c in list(on) + right_rest
                        if c in b.schema.names]
                yield b.select(keep)

        sub_left = hash_shuffle(
            _iter_lp(), list(on), left_rest, 4,
            memory_guard, sub_left_dir, hash_seed=depth + 1, log=False,
        )
        sub_right = hash_shuffle(
            _iter_rp(), list(on), right_rest, 4,
            memory_guard, sub_right_dir, hash_seed=depth + 1, log=False,
        )
        for sub_l, sub_r in zip(sub_left, sub_right):
            yield from _process_partition_pair_semi(
                sub_l, sub_r, left_schema, right_schema, on,
                memory_guard, spill_dir, depth=depth + 1,
                max_depth=max_depth,
            )
        return

    right_table = _safe_materialize_partition(right_path, memory_guard)
    if right_table is None:
        raise MemoryLimitExceeded(
            f"semi_join: build partition {right_path.name} still too "
            f"large after stat check."
        )
    right_table = _drop_null_keys_arrow(right_table, list(on))

    for probe_batch in _iter_probe_batches(left_path):
        memory_guard.check_batch(
            probe_batch, name=f"semi_probe({left_path.name})"
        )
        # Composite key for multi-column (fix3 lesson).
        if len(on) == 1:
            m = _pc.is_in(
                probe_batch.column(on[0]),
                value_set=right_table.column(on[0]).combine_chunks(),
            )
        else:
            probe_tbl = pa.Table.from_batches([probe_batch])
            probe_key = _combine_join_keys(probe_tbl, list(on))
            right_key = _combine_join_keys(right_table, list(on))
            m = _pc.is_in(probe_key, value_set=right_key)
        # Null left -> is_in returns null -> fill_null(False) treats as
        # "not matched" -> NOT emitted (SQL semantics for semi).
        m = _pc.fill_null(m, pa.scalar(False, type=pa.bool_()))
        filtered = probe_batch.filter(m)
        if filtered.num_rows > 0:
            yield filtered


def semi_join_streaming(
    left_batches: Iterable[pa.RecordBatch],
    right_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    memory_guard: MemoryGuard,
    spill_dir: Path,
    *,
    num_partitions: int = 8,
    hash_seed: int = 0,
) -> tuple[pa.Schema, Path]:
    """Semi join: left rows whose key IS present in right.

    v0.9.23j (AUDIT_FULL_v0922 item 7.A.10, split 3/3 final).

    Output schema = left_schema (no right columns). Streaming;
    peak = per-partition build + probe batch. Mirror of
    anti_join_streaming with filter(matched).

    SQL semantics: null left key -> NOT emitted (null never matches).
    """
    if not on:
        raise ValueError("semi_join: on must be non-empty")
    for c in on:
        if c not in left_schema.names:
            raise KeyError(f"semi_join: on {c!r} not in left schema")
        if c not in right_schema.names:
            raise KeyError(f"semi_join: on {c!r} not in right schema")

    left_rest = [c for c in left_schema.names if c not in on]
    right_rest = [c for c in right_schema.names if c not in on]

    left_it = iter(left_batches)
    right_it = iter(right_batches)
    try:
        first_left = next(left_it)
    except StopIteration:
        out_schema = pa.schema(
            [left_schema.field(c) for c in left_schema.names]
        )
        return out_schema, _write_empty_ipc(out_schema, spill_dir)
    try:
        first_right = next(right_it)
    except StopIteration:
        # No right rows -> semi = empty.
        out_schema = pa.schema(
            [left_schema.field(c) for c in left_schema.names]
        )
        return out_schema, _write_empty_ipc(out_schema, spill_dir)

    left_batches = chain([first_left], left_it)
    right_batches = chain([first_right], right_it)

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)
    left_paths = hash_shuffle(
        left_batches, list(on), left_rest, num_partitions,
        memory_guard, spill_dir / "semi_left",
        hash_seed=hash_seed, log=False,
    )
    right_paths = hash_shuffle(
        right_batches, list(on), right_rest, num_partitions,
        memory_guard, spill_dir / "semi_right",
        hash_seed=hash_seed, log=False,
    )

    out_schema = pa.schema(
        [left_schema.field(c) for c in left_schema.names]
    )
    out_path = spill_dir / "semi_output.arrow"
    writer = ArrowIPCBatchWriter(str(out_path), out_schema)
    try:
        for p in range(num_partitions):
            lp = left_paths[p] if left_paths[p].exists() else None
            rp = right_paths[p] if right_paths[p].exists() else None
            if lp is None:
                continue
            if rp is None:
                # No right rows in partition -> no match -> skip (semi).
                continue
            for rb in _process_partition_pair_semi(
                lp, rp, left_schema, right_schema, on,
                memory_guard, spill_dir,
            ):
                if rb.num_rows > 0:
                    writer.write_batch(rb)
    finally:
        writer.close()
    return out_schema, out_path

'''

def add_semi_fn():
    print("\n=== add _process_partition_pair_semi + semi_join_streaming ===")
    p = Path("cadw/ops/join.py")
    s = p.read_text()
    if "def semi_join_streaming(" in s:
        print("  already present — skip")
        return
    # Insert after inner_join definition (before left_join).
    anchor = (
        'def left_join(\n'
        '    left_batches: Iterable[pa.RecordBatch],'
    )
    if anchor not in s:
        raise SystemExit("left_join anchor not found in join.py")
    s = s.replace(anchor, SEMI_FN + anchor, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. Export semi_join_streaming from cadw.ops ============

def export_semi():
    print("\n=== export semi_join_streaming from cadw.ops ===")
    p = Path("cadw/ops/__init__.py")
    s = p.read_text()
    if "semi_join_streaming" in s:
        print("  already present — skip")
        return
    apply_or_fail(
        str(p),
        old=(
            'from .join import (\n'
            '    hash_join, hash_join_streaming, inner_join, left_join,\n'
            '    anti_join_streaming, outer_join_streaming,\n'
            ')'
        ),
        new=(
            'from .join import (\n'
            '    hash_join, hash_join_streaming, inner_join, left_join,\n'
            '    anti_join_streaming, outer_join_streaming,\n'
            '    semi_join_streaming,\n'
            ')'
        ),
        label="ops_init_semi_import",
    )
    apply_or_fail(
        str(p),
        old=(
            '    "anti_join_streaming", "outer_join_streaming",'
        ),
        new=(
            '    "anti_join_streaming", "outer_join_streaming",\n'
            '    "semi_join_streaming",'
        ),
        label="ops_init_semi_all",
    )
    assert_compiles(p)
    print("  OK")


# ============ 3. merge dispatch: how='semi' branch ============

OLD_SEMI_BRANCH = '''        elif how == "semi":
            # v0.9.23j placeholder — dispatch handled by CadwDataFrame
            # extension in next patch.
            raise NotImplementedError(
                "merge(how='semi') is v0.9.23j; use anti + left for now."
            )'''

NEW_SEMI_BRANCH = '''        elif how == "semi":
            # v0.9.23j: semi join = left rows with match in right.
            from cadw.ops.join import semi_join_streaming as _semi
            out_schema, out_path = _semi(
                self._iter_filtered_batches(),
                right._iter_filtered_batches(),
                left_schema, right_schema, on=on_list,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir / "join" / "semi",
                num_partitions=num_partitions,
            )'''

def fix_semi_branch():
    print("\n=== merge dispatch: semi branch (replace placeholder) ===")
    p = Path("cadw/dataframe.py")
    applied = apply_or_fail(
        str(p), OLD_SEMI_BRANCH, NEW_SEMI_BRANCH, label="semi_branch",
    )
    if applied:
        assert_compiles(p)
        print("  applied")
    else:
        print("  already applied — skip")


# ============ 4. merge how docstring update ============

def update_merge_docstring():
    print("\n=== update merge() how docstring ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    if 'how="semi"' in s and 'left_schema_out' in s:
        # v0.9.23i already left some; check specifically:
        pass
    # Docstring mentions only 'inner', 'left' — extend.
    apply_or_fail(
        str(p),
        old=(
            '        """Hash join (v0.2.4). how ∈ {\'inner\', \'left\'}.\n'
            '\n'
            '        - how=\'inner\': только совпадающие ключи.\n'
            '        - how=\'left\':  все строки self сохраняются; right-колонки -> NULL\n'
            '          при отсутствии совпадения.'
        ),
        new=(
            '        """Hash join (v0.9.23j). how ∈ {\'inner\', \'left\',\n'
            '        \'outer\', \'anti\', \'semi\'}.\n'
            '\n'
            '        - how=\'inner\': совпадающие ключи (left ∪ right, on-coalesced).\n'
            '        - how=\'left\':  все left + right-колонки (null при нет совпадения).\n'
            '        - how=\'outer\': left ∪ right, обе стороны null-padded.\n'
            '        - how=\'anti\':  left без совпадения в right (SQL anti join).\n'
            '        - how=\'semi\':  left с совпадением в right (SQL semi join);\n'
            '                       output = left-колонки only.\n'
            '\n'
            '        NULL-семантика для anti/semi: null-ключ слева никогда\n'
            '        не совпадает (SQL); anti emit\'ит, semi — нет.'
        ),
        label="merge_docstring",
    )
    assert_compiles(p)
    print("  OK")


# ============ 5. Enum-stale-test grep BEFORE commit ============

def grep_stale_semi():
    print("\n=== grep: stale how='semi' in tests/ ===")
    r = subprocess.run(
        ["grep", "-rFn", 'how="semi"', "tests/"],
        capture_output=True, text=True,
    )
    lines = [ln for ln in r.stdout.strip().splitlines() if ln]
    if lines:
        print(f"  found {len(lines)} reference(s):")
        for ln in lines:
            print(f"    {ln}")
    else:
        print("  none")
    return lines


# ============ 6. Update stale tests (from grep) ============

def fix_stale_unsupported_how():
    print("\n=== update test_join.py::test_merge_unsupported_how ===")
    p = Path("tests/test_join.py")
    s = p.read_text()
    if '"semi"' in s and "NotImplementedError" not in s.split(
        "def test_merge_unsupported_how"
    )[1].split("def test_")[0] if "def test_merge_unsupported_how" in s else False:
        # heuristic — check the block
        pass

    # Read current block.
    import re
    m = re.search(
        r'def test_merge_unsupported_how\(tmp_path\):.*?(?=\ndef |\nclass |\Z)',
        s, re.DOTALL,
    )
    if not m:
        raise SystemExit("test_merge_unsupported_how not found")
    current = m.group(0)

    # v0.9.23j expected (v0.9.23i-fix5 form):
    old_block = (
        'def test_merge_unsupported_how(tmp_path):\n'
        '    """v0.9.23i-fix5: \'outer\' now supported; \'right\'/\'cross\' '
        'still raise.\n'
        '\n'
        '    Prior test (v0.2.x) expected only inner/left. Enum extended '
        'in\n'
        '    v0.9.23i (outer/anti/semi) and v0.9.23j (\'right\'). Update:\n'
        '    - outer -> supported (no longer in bad_how)\n'
        '    - right, cross -> raise with updated message\n'
        '    - semi -> raises NotImplementedError (v0.9.23j placeholder)\n'
        '    """\n'
        '    left = make_df(tmp_path, "L", {"k": [1], "a": [10]})\n'
        '    right = make_df(tmp_path, "R", {"k": [1], "b": [100]})\n'
        '    for bad_how in ("right", "cross"):\n'
        '        with pytest.raises(\n'
        '            NotImplementedError,\n'
        '            match="supported: inner, left, outer, anti, semi",\n'
        '        ):\n'
        '            left.merge(right, on="k", how=bad_how)\n'
        '    # semi: explicit placeholder message\n'
        '    with pytest.raises(NotImplementedError, match="semi"):\n'
        '        left.merge(right, on="k", how="semi")'
    )

    new_block = (
        'def test_merge_unsupported_how(tmp_path):\n'
        '    """v0.9.23j: how enum closed. Only \'right\'/\'cross\' raise.\n'
        '\n'
        '    Supported: inner, left, outer, anti, semi.\n'
        '    - right -> still NotImplementedError (deferred; SQL right\n'
        '      join = left join with sides swapped — not a v0.9.23 scope)\n'
        '    - cross -> still NotImplementedError (O(m*n) — out of scope)\n'
        '    """\n'
        '    left = make_df(tmp_path, "L", {"k": [1], "a": [10]})\n'
        '    right = make_df(tmp_path, "R", {"k": [1], "b": [100]})\n'
        '    for bad_how in ("right", "cross"):\n'
        '        with pytest.raises(\n'
        '            NotImplementedError,\n'
        '            match="supported: inner, left, outer, anti, semi",\n'
        '        ):\n'
        '            left.merge(right, on="k", how=bad_how)\n'
        '    # semi now works (v0.9.23j) — regression: must not raise.\n'
        '    result = left.merge(right, on="k", how="semi")\n'
        '    assert result.count() == 1'
    )

    if new_block in s:
        print("  already updated — skip")
        return

    if old_block in s:
        s = s.replace(old_block, new_block, 1)
        p.write_text(s)
        assert_compiles(p)
        print("  applied")
        return

    # Fallback: block matches current; use regex replacement.
    print("  WARNING: exact old_block not matched; inspecting current:")
    print("  current block (first 400 chars):")
    print("  " + current[:400].replace(chr(10), chr(10) + "  "))
    raise SystemExit(
        "test_merge_unsupported_how body differs from v0.9.23i-fix5. "
        "Update OLD_BLOCK in this patch script or fix manually."
    )


def fix_stale_semi_raises_test():
    print("\n=== update test_anti_outer_join.py::test_merge_how_semi_raises ===")
    p = Path("tests/test_anti_outer_join.py")
    s = p.read_text()
    old = (
        'def test_merge_how_semi_raises(tmp_path):\n'
        '    left = make_df(tmp_path, {"k": [1, 2]}, name="L")\n'
        '    right = make_df(tmp_path, {"k": [1]}, name="R")\n'
        '    with pytest.raises(NotImplementedError, match="semi"):\n'
        '        left.merge(right, on="k", how="semi")'
    )
    new = (
        'def test_merge_how_semi_works(tmp_path):\n'
        '    """v0.9.23j: semi join now works (was NotImplementedError).\n'
        '\n'
        '    Rename from test_merge_how_semi_raises; test semantics\n'
        '    flipped: emit left rows with match in right.\n'
        '    """\n'
        '    left = make_df(tmp_path, {"k": [1, 2]}, name="L")\n'
        '    right = make_df(tmp_path, {"k": [1]}, name="R")\n'
        '    result = left.merge(right, on="k", how="semi")\n'
        '    rows = _rows(result)\n'
        '    assert len(rows) == 1\n'
        '    assert rows[0]["k"] == 1'
    )
    if new in s:
        print("  already updated — skip")
        return
    if old not in s:
        print("  WARNING: test_merge_how_semi_raises not found — skip")
        return
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  applied")


# ============ 7. New test file: tests/test_semi_join.py ============

SEMI_TESTS = '''"""v0.9.23j: semi join (7.A.10 split 3/3 final).

Semi join = left rows with match in right. Output = left-schema only.
SQL null semantics: null left key -> NOT emitted (null never matches).
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L", mem_mb=50):
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


# ============ basic semantics ============

def test_semi_basic(tmp_path):
    left = make_df(tmp_path, {
        "k": [1, 2, 3, 4], "a": ["w", "x", "y", "z"],
    }, name="L")
    right = make_df(tmp_path, {"k": [1, 3], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="semi")
    rows = sorted(_rows(result), key=lambda r: r["k"])
    # Emit matched: keys 1, 3.
    assert [r["k"] for r in rows] == [1, 3]
    # Output = left-schema only (no right column b).
    assert "b" not in rows[0]


def test_semi_all_match(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2, 3]}, name="L")
    right = make_df(tmp_path, {"k": [1, 2, 3]}, name="R")
    result = left.merge(right, on="k", how="semi")
    assert result.count() == 3


def test_semi_none_match(tmp_path):
    left = make_df(tmp_path, {"k": [10, 20]}, name="L")
    right = make_df(tmp_path, {"k": [1, 2, 3]}, name="R")
    result = left.merge(right, on="k", how="semi")
    assert result.count() == 0


def test_semi_empty_right(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2]}, name="L")
    right_data = {"k": pa.array([], type=pa.int64())}
    right = make_df(tmp_path, right_data, name="R")
    result = left.merge(right, on="k", how="semi")
    assert result.count() == 0


def test_semi_null_left_not_emitted(tmp_path):
    """SQL semantics: null left key never matches -> NOT emitted (semi)."""
    left = make_df(tmp_path, {
        "k": pa.array([1, None, 3], type=pa.int64()),
    }, name="L")
    right = make_df(tmp_path, {"k": [1, 2]}, name="R")
    result = left.merge(right, on="k", how="semi")
    rows = _rows(result)
    # Emit only k=1 (None never matches, 3 not in right).
    keys = {r["k"] for r in rows}
    assert keys == {1}
    assert None not in keys


# ============ multi-key ============

def test_semi_multi_key(tmp_path):
    left = make_df(tmp_path, {
        "k1": [1, 1, 2, 2], "k2": ["a", "b", "a", "b"],
        "v": [10, 20, 30, 40],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1, 2], "k2": ["a", "a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="semi")
    rows = sorted(_rows(result), key=lambda r: (r["k1"], r["k2"]))
    # Emit matched: (1,a) and (2,a); (1,b),(2,b) unmatched.
    assert len(rows) == 2
    assert [(r["k1"], r["k2"]) for r in rows] == [(1, "a"), (2, "a")]


def test_semi_multi_key_crossed(tmp_path):
    """Canonical multi-key detector (v0.9.23i-fix3 lesson).

    left=(1,"a"), right={(1,"b"),(2,"a")}: cross-product match would
    falsely emit. Composite key: NOT emitted (correct).
    """
    left = make_df(tmp_path, {
        "k1": [1], "k2": ["a"], "v": [100],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1, 2], "k2": ["b", "a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="semi")
    rows = _rows(result)
    # (1,"a") crossed: (1,*) and (*,"a") in right, but no (1,"a") row.
    # Semi must NOT emit.
    assert len(rows) == 0, (
        f"multi-key semi cross-product bug: expected 0 rows, got {rows}"
    )


def test_semi_multi_key_actual_match(tmp_path):
    """left-tuple IS a right row -> emitted."""
    left = make_df(tmp_path, {
        "k1": [1, 1], "k2": ["a", "b"], "v": [10, 20],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1], "k2": ["a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="semi")
    rows = _rows(result)
    assert len(rows) == 1
    assert rows[0]["k1"] == 1 and rows[0]["k2"] == "a"


def test_semi_multi_key_null_not_emitted(tmp_path):
    """Null in one on-col -> not matched -> not emitted."""
    left = make_df(tmp_path, {
        "k1": [1, None], "k2": ["a", "b"], "v": [10, 20],
    }, name="L")
    right = make_df(tmp_path, {
        "k1": [1], "k2": ["a"],
    }, name="R")
    result = left.merge(right, on=["k1", "k2"], how="semi")
    rows = _rows(result)
    # (1, "a") matched -> emit. (None, "b") unmatched -> no emit.
    assert len(rows) == 1
    assert rows[0]["v"] == 10


# ============ AST guards ============

def test_semi_uses_hash_shuffle():
    import inspect
    from cadw.ops import join as j
    src = inspect.getsource(j.semi_join_streaming)
    assert "hash_shuffle" in src


def test_semi_uses_is_in():
    import inspect
    from cadw.ops import join as j
    src = inspect.getsource(j._process_partition_pair_semi)
    assert "is_in" in src


def test_semi_fill_null_false():
    """Semi: null left key must NOT be silently matched (fill_null False)."""
    import inspect
    from cadw.ops import join as j
    src = inspect.getsource(j._process_partition_pair_semi)
    assert "fill_null" in src
    assert "False" in src


def test_semi_uses_composite_key():
    """AST: multi-column via _combine_join_keys (fix3 lesson)."""
    import inspect
    from cadw.ops import join as j
    src = inspect.getsource(j._process_partition_pair_semi)
    assert "_combine_join_keys" in src, (
        "multi-column semi must use composite key, not per-column is_in"
    )


def test_semi_exported_from_cadw_ops():
    from cadw.ops import semi_join_streaming
    assert callable(semi_join_streaming)
'''


def write_test_file():
    print("\n=== write tests/test_semi_join.py ===")
    p = Path("tests/test_semi_join.py")
    if p.exists():
        print("  already exists — skip")
        return
    p.write_text(SEMI_TESTS)
    assert_compiles(p)
    from cadw.patch_utils import check_no_invalid_escape
    check_no_invalid_escape(p)
    print("  OK")


# ============ 8. Dogfood ============

def dogfood():
    print("\n=== dogfood: semi invariants (invariants only, no distrib) ===")

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

        # (a) Semi basic: emitted ⊆ matched keys.
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
                memory_guard=_MG(max_memory_mb=100),
                spill_dir=td / "spillL",
            )
            right = _CDF.from_parquet(
                src_r, td / "dsR", chunk_size=1,
                memory_guard=_MG(max_memory_mb=100),
                spill_dir=td / "spillR",
            )
            semi = left.merge(right, on="k", how="semi")
            rows = _rows(semi)
            right_keys = set(right_t.column("k").to_pylist())
            left_keys = set(left_t.column("k").to_pylist())
            # Invariant 1: every emitted k IS in right.
            for r in rows:
                assert r["k"] in right_keys, (
                    f"semi emitted unmatched key {r['k']}"
                )
            # Invariant 2: emitted keys = left_keys ∩ right_keys.
            assert {r["k"] for r in rows} == (left_keys & right_keys)

        # (b) Multi-key crossed: NOT emitted (canonical bug detector).
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
            semi = left.merge(right, on=["k1", "k2"], how="semi")
            rows = _rows(semi)
            assert len(rows) == 0, (
                f"crossed-tuple semi bug: expected 0 rows, got {rows}"
            )

        # (c) Null-left not emitted (SQL semantics).
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
            semi = left.merge(right, on="k", how="semi")
            rows = _rows(semi)
            keys = {r["k"] for r in rows}
            assert None not in keys
            assert keys == {1}

        # (d) Empty right -> 0 rows.
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            left_t = pa_.table({"k": [1, 2]})
            right_t = pa_.table({"k": pa_.array([], type=pa_.int64())})
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
            semi = left.merge(right, on="k", how="semi")
            assert semi.count() == 0

        # (e) Export from cadw.ops.
        from cadw.ops import semi_join_streaming as _sj
        assert callable(_sj)

        print(
            "  semi basic (not-in invariant) + multi-key crossed "
            "(no emit) + null-left not emitted + empty-right + "
            "cadw.ops export ok"
        )

    run_verify_in_subprocess(_verify)


# ============ 9. Tests ============

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
         "tests/test_semi_join.py",
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


# ============ 10. main ============

def main():
    print("=== v0.9.23j: semi join (7.A.10 split 3/3 final) ===")
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
    add_semi_fn()
    export_semi()
    fix_semi_branch()
    update_merge_docstring()
    grep_stale_semi()
    fix_stale_unsupported_how()
    fix_stale_semi_raises_test()
    write_test_file()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23j: semi join (AUDIT_FULL_v0922 item 7.A.10, split "
        "3/3 final, Top-10 gap #4 CLOSED). "
        "semi_join_streaming(left, right, on): left rows whose key "
        "IS matched in right. Output = left-schema only (SQL LEFT "
        "SEMI JOIN). Mirror of anti_join_streaming with "
        "filter(matched) — one-line diff from v0.9.23i's anti. "
        "Copy-not-share (independent function, no refactor of tested "
        "anti; documented choice). "
        "NULL semantics: null left key never matches -> NOT emitted "
        "(opposite of anti; SQL-correct). Null right key -> doesn't "
        "make any left row matched. fill_null(False) on is_in mask "
        "(null left -> treated as 'not in' -> not emitted). "
        "Multi-column: composite key via _combine_join_keys "
        "(v0.9.23i-fix3 lesson: never AND-of-per-column is_in — "
        "cross-product false positive). "
        "merge(how=...) enum CLOSED: inner | left | outer | anti | "
        "semi all supported. 'right'/'cross' still NotImplementedError "
        "(documented: right = left with sides swapped — out of v0.9.x "
        "scope; cross = O(m*n) — out). "
        "Export semi_join_streaming from cadw.ops (not from cadw "
        "top-level — same pre-existing anchor issue as v0.9.23i for "
        "cadw/__init__.py). "
        "merge() docstring updated (how list + null semantics for "
        "anti/semi). "
        "v0.9.23i lessons applied upfront (all 10): composite key, "
        "explicit _pc import, no dead code, module-level SEMI_TESTS "
        "constant, verify_state at start, enum-stale-test grep "
        "BEFORE commit, crossed-tuple test, pre-flight facade-as_py, "
        "dogfood invariants only, copy-not-share. "
        "Stale tests updated: "
        "  - test_join.py::test_merge_unsupported_how (removed semi "
        "from bad_how; added semi works regression); "
        "  - test_anti_outer_join.py::test_merge_how_semi_raises -> "
        "test_merge_how_semi_works (semantics flipped). "
        "+tests/test_semi_join.py (13 tests: basic, all-match, none-"
        "match, empty-right, null-left not-emitted, multi-key, "
        "multi-key-crossed (canonical detector), multi-key-actual-"
        "match, multi-key-null-not-emitted; 5 AST — hash_shuffle, "
        "is_in, fill_null(False), composite_key, cadw.ops export). "
        "Dogfood invariants only (v0.9.23h lesson): emitted ⊆ matched "
        "keys; emitted = left∩right; multi-key crossed NOT emitted; "
        "null-left not emitted; empty-right -> 0; cadw.ops export. "
        "Fast: 2152 -> ~2166 passed.",
        self_path="scripts/patch_v0923j_semi_join.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23j applied. 7.A.10 COMPLETE.")


if __name__ == "__main__":
    main()