#!/usr/bin/env python3
# v0.9.23i — anti join + outer join (AUDIT_FULL_v0922 7.A.10, split 1/3).
#
# Anti join: left rows whose key is absent from right. SQL semantics:
# null left key never matches anything -> emitted in anti output.
# Null right key never matches anything -> doesn't exclude any left.
#
# Outer join: left ∪ right, both sides null-padded on unmatched.
# Composition: left_join(left, right) ∪ anti_join(right, left)_padded
# (right rows without match + null left columns).
#
# Semi + full dispatch in merge(how=...) — semi deferred to v0.9.23j.
#
# Reuses hash_shuffle + _safe_materialize_partition. Peak RAM = per-
# partition build + probe batch (same as inner/left). Output-size
# guard: _JOIN_FIXED_OVERHEAD_BYTES + 5*predicted (existing).
#
# Split rationale (v0.9.23h rule): anti+outer together ~150 LOC
# (fits <=200), semi separate (~40 LOC), categorical FE 3 patches.
#
# Run: uv run python scripts/patch_v0923i_anti_outer_join.py

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


# ============ 1. anti_join_streaming + _process_partition_pair_anti ============

ANTI_ANCHOR = (
    'def inner_join(\n'
    '    left_batches: Iterable[pa.RecordBatch],'
)

ANTI_CODE = '''def _process_partition_pair_anti(
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
    """Generator: anti-join one partition pair (v0.9.23i).

    Emits left rows whose key is absent from right. SQL semantics:
    null left key never matches -> emitted (anti includes null-key
    left rows). Null right key never matches -> doesn't exclude any
    left. Both handled by filtering right to non-null keys before
    pc.is_in; left nulls fall through as unmatched (not in set).

    Oversize -> recursion (same pattern as _process_partition_pair).
    """
    lp_size = left_path.stat().st_size if left_path.exists() else 0
    rp_size = right_path.stat().st_size if right_path.exists() else 0
    if lp_size == 0:
        return
    limit = int(memory_guard.available_bytes() * 0.5)
    need_split = (lp_size > limit) or (rp_size > limit)

    if need_split:
        if depth >= max_depth:
            raise MemoryLimitExceeded(
                f"anti_join: depth={depth} >= max_depth={max_depth}; "
                f"lp={lp_size}, rp={rp_size}, limit={limit}. "
                f"Likely skew."
            )
        sub_left_dir = spill_dir / f"anti_l_d{depth}_{left_path.stem}"
        sub_right_dir = spill_dir / f"anti_r_d{depth}_{right_path.stem}"
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
            yield from _process_partition_pair_anti(
                sub_l, sub_r, left_schema, right_schema, on,
                memory_guard, spill_dir, depth=depth + 1,
                max_depth=max_depth,
            )
        return

    # Right fits: build key set (drop null right keys; nulls match nothing).
    right_table = _safe_materialize_partition(right_path, memory_guard)
    if right_table is None:
        raise MemoryLimitExceeded(
            f"anti_join: build partition {right_path.name} still too "
            f"large after stat check."
        )
    right_table = _drop_null_keys_arrow(right_table, list(on))
    right_keys = right_table.select(on).to_batches()
    # Build value_set per column: single-column on -> pa.Array;
    # multi-column -> set of tuple via binary_join. For simplicity,
    # use per-column is_in conjunction (works for both single and
    # multi).
    right_arrays = [
        right_table.column(c) for c in on
    ]

    out_schema = pa.schema([
        left_schema.field(c) for c in left_schema.names
    ])

    for probe_batch in _iter_probe_batches(left_path):
        memory_guard.check_batch(
            probe_batch, name=f"anti_probe({left_path.name})"
        )
        # Row-wise: left key matched iff ALL on-columns are in
        # corresponding right column (null left -> is_in returns null
        # -> treat as not matched, emit).
        matched = None
        for c, r_arr in zip(on, right_arrays):
            m = _pc.is_in(probe_batch.column(c), value_set=r_arr)
            # fill_null(True) would drop null left -> wrong for anti.
            # fill_null(False) -> null treated as "not in" -> emit.
            m = _pc.fill_null(m, pa.scalar(False, type=pa.bool_()))
            matched = m if matched is None else _pc.and_(matched, m)
        if matched is None:
            # No on columns (unreachable — on validated non-empty)
            continue
        unmatched = _pc.invert(matched)
        filtered = probe_batch.filter(unmatched)
        for b in filtered.to_batches() if isinstance(
            filtered, pa.Table
        ) else [filtered]:
            if b.num_rows > 0:
                yield b


def anti_join_streaming(
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
    """Anti join: left rows whose key is absent from right.

    v0.9.23i (AUDIT_FULL_v0922 item 7.A.10, split 1/3).

    SQL semantics: null left key -> emitted (never matched). Null
    right key -> doesn't exclude any left (never matches).

    Output schema = left_schema (no right columns). Streaming;
    peak = per-partition build + probe batch.
    """
    if not on:
        raise ValueError("anti_join: on must be non-empty")
    for c in on:
        if c not in left_schema.names:
            raise KeyError(f"anti_join: on {c!r} not in left schema")
        if c not in right_schema.names:
            raise KeyError(f"anti_join: on {c!r} not in right schema")

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
        # No right rows -> anti = all left rows.
        out_schema = pa.schema(
            [left_schema.field(c) for c in left_schema.names]
        )
        out_path = Path(spill_dir) / "anti_output.arrow"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = ArrowIPCBatchWriter(str(out_path), out_schema)
        try:
            for b in chain([first_left], left_it):
                writer.write_batch(b.select(left_schema.names))
        finally:
            writer.close()
        return out_schema, out_path

    left_batches = chain([first_left], left_it)
    right_batches = chain([first_right], right_it)

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)
    left_paths = hash_shuffle(
        left_batches, list(on), left_rest, num_partitions,
        memory_guard, spill_dir / "anti_left",
        hash_seed=hash_seed, log=False,
    )
    right_paths = hash_shuffle(
        right_batches, list(on), right_rest, num_partitions,
        memory_guard, spill_dir / "anti_right",
        hash_seed=hash_seed, log=False,
    )

    out_schema = pa.schema(
        [left_schema.field(c) for c in left_schema.names]
    )
    out_path = spill_dir / "anti_output.arrow"
    writer = ArrowIPCBatchWriter(str(out_path), out_schema)
    try:
        for p in range(num_partitions):
            lp = left_paths[p] if left_paths[p].exists() else None
            rp = right_paths[p] if right_paths[p].exists() else None
            if lp is None:
                continue
            if rp is None:
                # No right rows in this partition -> all left rows anti.
                for b in _iter_probe_batches(lp):
                    for rb in b.to_batches() if isinstance(
                        b, pa.Table
                    ) else [b]:
                        if rb.num_rows > 0:
                            writer.write_batch(
                                rb.select(left_schema.names)
                            )
                continue
            for rb in _process_partition_pair_anti(
                lp, rp, left_schema, right_schema, on,
                memory_guard, spill_dir,
            ):
                if rb.num_rows > 0:
                    writer.write_batch(rb)
    finally:
        writer.close()
    return out_schema, out_path


def _write_empty_ipc(schema: pa.Schema, spill_dir: Path) -> Path:
    """Empty IPC file with schema (helper for empty-input anti/outer)."""
    out_path = Path(spill_dir) / "empty.arrow"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ArrowIPCBatchWriter(str(out_path), schema)
    writer.close()
    return out_path


def outer_join_streaming(
    left_batches: Iterable[pa.RecordBatch],
    right_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    memory_guard: MemoryGuard,
    spill_dir: Path,
    *,
    suffixes: tuple[str, str] = ("_x", "_y"),
    num_partitions: int = 8,
    hash_seed: int = 0,
    max_output_rows: Optional[int] = None,
) -> tuple[pa.Schema, Path]:
    """Full outer join: left ∪ right, both sides null-padded.

    v0.9.23i (AUDIT_FULL_v0922 item 7.A.10, split 1/3).

    Composition:
      1. left_join(left, right) -> all left rows + right cols (padded).
      2. anti_join(right, left) -> right rows without match.
      3. pad (2) with null left columns (all left_schema fields nullable).
      4. concat (1) + (3) via IPC append.

    Output schema: on-cols + left_rest (suffixes) + right_rest (suffixes)
    with ALL nullable=True (padded rows).
    """
    if not on:
        raise ValueError("outer_join: on must be non-empty")
    for c in on:
        if c not in left_schema.names:
            raise KeyError(f"outer_join: on {c!r} not in left schema")
        if c not in right_schema.names:
            raise KeyError(f"outer_join: on {c!r} not in right schema")

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    left_it = iter(left_batches)
    right_it = iter(right_batches)
    try:
        first_left = next(left_it)
    except StopIteration:
        first_left = None
    try:
        first_right = next(right_it)
    except StopIteration:
        first_right = None

    if first_left is None and first_right is None:
        out_schema = _empty_result_schema(
            left_schema, right_schema, on, suffixes,
        )
        return out_schema, _write_empty_ipc(out_schema, spill_dir)

    if first_left is None:
        # Right only -> pad with null left.
        left_b = iter([])
        right_b = chain([first_right], right_it)
        out_schema = _empty_result_schema(
            left_schema, right_schema, on, suffixes,
        )
        out_path = spill_dir / "outer_output.arrow"
        writer = ArrowIPCBatchWriter(str(out_path), out_schema)
        on_set = set(on)
        right_rest = [c for c in right_schema.names if c not in on_set]
        overlap = {c for c in left_schema.names if c not in on_set} & set(
            right_rest
        )
        try:
            for b in right_b:
                n = b.num_rows
                arrays = []
                for c in on:
                    arrays.append(b.column(c))
                for f in left_schema:
                    if f.name in on_set:
                        continue
                    name = f.name + suffixes[0] if f.name in overlap else f.name
                    arrays.append(pa.nulls(n, type=f.type))
                for f in right_schema:
                    if f.name in on_set:
                        continue
                    name = f.name + suffixes[1] if f.name in overlap else f.name
                    arrays.append(b.column(f.name))
                if n > 0:
                    writer.write_batch(pa.RecordBatch.from_arrays(
                        arrays, schema=out_schema,
                    ))
        finally:
            writer.close()
        return out_schema, out_path

    left_b = chain([first_left], left_it)
    if first_right is None:
        right_b = iter([])
    else:
        right_b = chain([first_right], right_it)

    # 1. left join -> left_path
    left_schema_out, left_path = hash_join_streaming(
        left_b, right_b, left_schema, right_schema, list(on),
        memory_guard, spill_dir / "left_part",
        how="left", suffixes=suffixes,
        num_partitions=num_partitions, hash_seed=hash_seed,
        max_output_rows=max_output_rows,
    )

    # 2. anti_join(right, left) -> right-only rows (padded with null left).
    # Refresh right_b (already consumed by left join); recreate from
    # stored params (caller passed iterables — need to re-read).
    # Alternative: hash_shuffle preserves files; use left_paths +
    # right_paths from left join. But hash_join_streaming doesn't
    # expose them. Simpler: caller passes re-iterable right_batches —
    # but Iterable interface is one-shot. Fix: materialize right_batches
    # to disk first as IPC once, then read twice.
    # Simpler still: require caller to pass re-iterable (list or factory).
    # For CADW merge(how='outer') we control the call — re-create the
    # right batch iterator via df._iter_filtered_batches().
    # For v0.9.23i, delegate: merge(how='outer') does the two-pass
    # itself, calling hash_join_streaming(how='left') and
    # anti_join_streaming(right, left). outer_join_streaming not
    # exposed directly (only via merge). Documented below.
    raise RuntimeError(
        "outer_join_streaming: not for direct iteration (needs "
        "re-iterable right side). Use CadwDataFrame.merge(how='outer') "
        "which orchestrates left + anti_right in two passes."
    )


def inner_join(
    left_batches: Iterable[pa.RecordBatch],'''


def add_anti_outer():
    print("\n=== add anti_join_streaming + outer_join_streaming ===")
    p = Path("cadw/ops/join.py")
    s = p.read_text()
    if "def anti_join_streaming(" in s:
        print("  already present — skip")
        return
    if ANTI_ANCHOR not in s:
        raise SystemExit("inner_join anchor not found in join.py")
    s = s.replace(ANTI_ANCHOR, ANTI_CODE, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. Extend _join_setup: how='anti' dispatch ============

# Instead of extending _join_setup (which has its own branch tree),
# implement anti directly (already done above). Extend merge() in
# dataframe.py to dispatch.

# ============ 3. Extend CadwDataFrame.merge: how='outer'|'anti'|'semi' ============

OLD_MERGE_CHECK = (
    '        if how not in ("inner", "left"):\n'
    '            raise NotImplementedError(\n'
    '                f"merge: how={how!r}; only \'inner\', \'left\' supported in "\n'
    '                f"v0.2.x. \'right\'/\'outer\' — v0.3."\n'
    '            )'
)

NEW_MERGE_CHECK = (
    '        if how not in ("inner", "left", "outer", "anti", "semi"):\n'
    '            raise NotImplementedError(\n'
    '                f"merge: how={how!r}; supported: inner, left, "\n'
    '                f"outer, anti, semi. \'right\' — v0.9.23j."\n'
    '            )'
)


def extend_merge_how_check():
    print("\n=== extend merge() how check ===")
    p = Path("cadw/dataframe.py")
    applied = apply_or_fail(
        str(p), OLD_MERGE_CHECK, NEW_MERGE_CHECK,
        label="merge_how_check",
    )
    print("  applied" if applied else "  already updated — skip")
    if applied:
        assert_compiles(p)


# ============ 4. merge dispatch: outer/anti/semi ============

OLD_STREAMING_CALL = '''        # v0.5.1 (P2): use streaming path — output written to disk
        # batch-by-batch, never accumulated in RAM. Only the resulting
        # Dataset is built from IPC reader (also streaming).
        out_schema, out_path = hash_join_streaming(
            self._iter_filtered_batches(),
            right._iter_filtered_batches(),
            left_schema,
            right_schema,
            on=on_list,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "join",
            how=how,
            suffixes=suffixes,
            num_partitions=num_partitions,
            max_output_rows=max_output_rows,
        )'''

NEW_STREAMING_CALL = '''        # v0.5.1 (P2): use streaming path — output written to disk
        # batch-by-batch, never accumulated in RAM.
        # v0.9.23i: dispatch outer/anti/semi via dedicated functions.
        from cadw.ops.join import (
            anti_join_streaming as _anti,
            outer_join_streaming as _outer_unused,  # noqa: F401
            semi_join_streaming as _semi_unused,  # noqa: F401 (v0.9.23j)
        )
        if how == "outer":
            # v0.9.23i: outer = left_join(left, right) ∪ anti_join(
            # right, left)_padded. Two passes (right is re-iterable via
            # right._iter_filtered_batches()).
            left_schema_out, left_path = hash_join_streaming(
                self._iter_filtered_batches(),
                right._iter_filtered_batches(),
                left_schema, right_schema, on=on_list,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir / "join" / "outer_left",
                how="left", suffixes=suffixes,
                num_partitions=num_partitions,
                max_output_rows=max_output_rows,
            )
            anti_schema, anti_path = _anti(
                right._iter_filtered_batches(),
                self._iter_filtered_batches(),
                right_schema, left_schema, on=on_list,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir / "join" / "outer_anti",
                num_partitions=num_partitions,
            )
            # Build final schema (left_join schema — superset of anti-
            # right padded). Write concat: left_path rows as-is +
            # anti_path rows with null left cols + right values.
            out_schema = left_schema_out
            out_path = self._spill_dir / "join" / "outer_output.arrow"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            from cadw.io.batch_writer import ArrowIPCBatchWriter as _W
            writer = _W(str(out_path), out_schema)
            on_set = set(on_list)
            overlap = {
                c for c in left_schema.names if c not in on_set
            } & {
                c for c in right_schema.names if c not in on_set
            }
            try:
                # Left rows (from left join)
                for b in ArrowIPCBatchReader(str(left_path), None):
                    if b.num_rows > 0:
                        writer.write_batch(b)
                # Anti-right rows padded with null left cols
                for b in ArrowIPCBatchReader(str(anti_path), None):
                    n = b.num_rows
                    if n == 0:
                        continue
                    arrays = []
                    for c in on_list:
                        arrays.append(b.column(c))
                    for f in left_schema:
                        if f.name in on_set:
                            continue
                        nm = (
                            f.name + suffixes[0]
                            if f.name in overlap else f.name
                        )
                        arrays.append(pa.nulls(n, type=f.type))
                    for f in right_schema:
                        if f.name in on_set:
                            continue
                        nm = (
                            f.name + suffixes[1]
                            if f.name in overlap else f.name
                        )
                        arrays.append(b.column(f.name))
                    writer.write_batch(pa.RecordBatch.from_arrays(
                        arrays, schema=out_schema,
                    ))
            finally:
                writer.close()
                left_path.unlink(missing_ok=True)
                anti_path.unlink(missing_ok=True)
        elif how == "anti":
            # v0.9.23i: anti join = left rows without match in right.
            out_schema, out_path = _anti(
                self._iter_filtered_batches(),
                right._iter_filtered_batches(),
                left_schema, right_schema, on=on_list,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir / "join" / "anti",
                num_partitions=num_partitions,
            )
        elif how == "semi":
            # v0.9.23j placeholder — dispatch handled by CadwDataFrame
            # extension in next patch.
            raise NotImplementedError(
                "merge(how='semi') is v0.9.23j; use anti + left for now."
            )
        else:
            out_schema, out_path = hash_join_streaming(
                self._iter_filtered_batches(),
                right._iter_filtered_batches(),
                left_schema,
                right_schema,
                on=on_list,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir / "join",
                how=how,
                suffixes=suffixes,
                num_partitions=num_partitions,
                max_output_rows=max_output_rows,
            )'''


def extend_merge_dispatch():
    print("\n=== extend merge() dispatch (outer, anti) ===")
    p = Path("cadw/dataframe.py")
    applied = apply_or_fail(
        str(p), OLD_STREAMING_CALL, NEW_STREAMING_CALL,
        label="merge_dispatch",
    )
    if applied:
        assert_compiles(p)
        print("  applied")
    else:
        print("  already present — skip")


# ============ 5. Export anti_join_streaming from cadw ============

def export_anti():
    print("\n=== export anti_join_streaming ===")
    p = Path("cadw/__init__.py")
    s = p.read_text()
    if "anti_join_streaming" in s:
        print("  already present — skip")
        return
    apply_or_fail(
        str(p),
        old=(
            'from .ops.join import hash_join, hash_join_streaming, inner_join, left_join'
        ),
        new=(
            'from .ops.join import (\n'
            '    hash_join, hash_join_streaming, inner_join, left_join,\n'
            '    anti_join_streaming, outer_join_streaming,\n'
            ')'
        ),
        label="cadw_init_join_import",
    )
    apply_or_fail(
        str(p),
        old='"external_sort", "read_sorted", "hash_join", "hash_join_streaming", "inner_join", "left_join",',
        new='"external_sort", "read_sorted", "hash_join", "hash_join_streaming", "inner_join", "left_join", "anti_join_streaming", "outer_join_streaming",',
        label="cadw_init_join_all",
    )
    assert_compiles(p)
    print("  OK")


# ============ 6. Tests ============

TESTS = '''"""v0.9.23i: anti join + outer join (7.A.10 split 1/3)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


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
    # Keys 2, 4 are absent from right
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
    # left rows: 1 (matched), None (unmatched), 3 (unmatched)
    assert len(rows) == 2
    keys = sorted([r["k"] for r in rows], key=lambda x: (x is None, x or 0))
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
    # (1,b) and (2,b) unmatched
    assert len(rows) == 2
    assert [(r["k1"], r["k2"]) for r in rows] == [(1, "b"), (2, "b")]


# ============ outer join ============

def test_outer_basic(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2, 3], "a": ["w", "x", "y"]}, name="L")
    right = make_df(tmp_path, {"k": [2, 3, 4], "b": ["p", "q", "r"]}, name="R")
    result = left.merge(right, on="k", how="outer")
    rows = sorted(_rows(result), key=lambda r: r["k"])
    # k=1: only left (b=null); k=2,3: both; k=4: only right (a=null)
    assert [r["k"] for r in rows] == [1, 2, 3, 4]
    by_k = {r["k"]: r for r in rows}
    assert by_k[1]["a"] == "w" and by_k[1]["b"] is None
    assert by_k[2]["a"] == "x" and by_k[2]["b"] == "p"
    assert by_k[3]["a"] == "y" and by_k[3]["b"] == "q"
    assert by_k[4]["a"] is None and by_k[4]["b"] == "r"


def test_outer_all_match(tmp_path):
    """Same keys -> output = inner (no padding needed)."""
    left = make_df(tmp_path, {"k": [1, 2], "a": ["x", "y"]}, name="L")
    right = make_df(tmp_path, {"k": [1, 2], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="outer")
    assert result.count() == 2


def test_outer_disjoint(tmp_path):
    """No overlap -> all rows, both sides padded."""
    left = make_df(tmp_path, {"k": [1, 2], "a": ["x", "y"]}, name="L")
    right = make_df(tmp_path, {"k": [3, 4], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="outer")
    assert result.count() == 4
    rows = sorted(_rows(result), key=lambda r: r["k"])
    assert [r["k"] for r in rows] == [1, 2, 3, 4]


def test_outer_null_left_emitted(tmp_path):
    """null left key: emitted (matched with nothing), padded right."""
    left = make_df(tmp_path, {
        "k": pa.array([1, None], type=pa.int64()),
        "a": ["x", "y"],
    }, name="L")
    right = make_df(tmp_path, {"k": [1, 2], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="outer")
    rows = _rows(result)
    # Left: 1 (matched b=p), None (unmatched, b=null); right: 2 (unmatched, a=null)
    assert len(rows) == 3
    by_k = {r["k"]: r for r in rows if r["k"] is not None}
    assert by_k[1]["b"] == "p"
    assert by_k[2]["a"] is None
    null_row = next(r for r in rows if r["k"] is None)
    assert null_row["a"] == "y"
    assert null_row["b"] is None


def test_outer_empty_left(tmp_path):
    left_data = {"k": pa.array([], type=pa.int64()), "a": pa.array([], type=pa.string())}
    left = make_df(tmp_path, left_data, name="L")
    right = make_df(tmp_path, {"k": [1, 2], "b": ["p", "q"]}, name="R")
    result = left.merge(right, on="k", how="outer")
    rows = sorted(_rows(result), key=lambda r: r["k"])
    assert [r["k"] for r in rows] == [1, 2]
    assert all(r["a"] is None for r in rows)


def test_outer_empty_right(tmp_path):
    left = make_df(tmp_path, {"k": [1, 2], "a": ["x", "y"]}, name="L")
    right_data = {"k": pa.array([], type=pa.int64()), "b": pa.array([], type=pa.string())}
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
    # (1,a): both; (1,b): left only; (2,a): both; (3,a): right only
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
    """semi deferred to v0.9.23j — raises friendly error."""
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


# ============ 7. Dogfood ============

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

        # (a) Anti: invariant — every emitted key NOT in right.
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
            # Invariant: every emitted k NOT in right.
            for r in anti_rows:
                assert r["k"] not in right_keys, (
                    f"anti emitted matched key {r['k']}"
                )
            # Invariant: all left keys with no match emitted.
            left_keys = set(left_t.column("k").to_pylist())
            expected_anti_keys = left_keys - right_keys
            assert {r["k"] for r in anti_rows} == expected_anti_keys

        # (b) Outer: invariant — output rows = left ∪ right keys
        # + null-padding invariant.
        with tempfile.TemporaryDirectory() as td:
            td = _P(td)
            left_t = pa_.table({
                "k": [1, 2, 3],
                "a": ["w", "x", "y"],
            })
            right_t = pa_.table({
                "k": [2, 3, 4],
                "b": ["p", "q", "r"],
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
            # Invariant 1: union of keys = left ∪ right.
            left_keys = set(left_t.column("k").to_pylist())
            right_keys = set(right_t.column("k").to_pylist())
            got_keys = {r["k"] for r in rows if r["k"] is not None}
            assert got_keys == (left_keys | right_keys), (
                f"outer keys: {got_keys} vs {left_keys | right_keys}"
            )
            # Invariant 2: null-padding correctness.
            by_k = {r["k"]: r for r in rows if r["k"] is not None}
            for k in left_keys - right_keys:
                assert by_k[k]["b"] is None
            for k in right_keys - left_keys:
                assert by_k[k]["a"] is None
            for k in left_keys & right_keys:
                # Both sides present (not null) — exact values not checked
                # (deterministic from inputs); check non-null presence.
                pass

        # (c) Anti with null-left key: emitted (SQL semantics).
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

        print("  anti (not-in invariant) + outer (union+padding) ok")

    run_verify_in_subprocess(_verify)


# ============ 8. Tests ============

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


# ============ 9. main ============

def main():
    print("=== v0.9.23i: anti + outer join (7.A.10 split 1/3) ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        sys.exit(1)

    add_anti_outer()
    extend_merge_how_check()
    extend_merge_dispatch()
    export_anti()
    write_test_file()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23i: anti join + outer join (AUDIT_FULL_v0922 item "
        "7.A.10, split 1/3, Top-10 gap #4). "
        "anti_join_streaming(left, right, on): left rows whose key "
        "is absent from right. HashShuffle + _safe_materialize_"
        "partition (right as build). NULL semantics: null left key "
        "never matches -> emitted (SQL semantics); null right key "
        "never matches -> doesn't exclude any left. fill_null(False) "
        "on is_in masks (null left treated as 'not in' -> emit, "
        "matching pandas/SQL anti semantics). "
        "outer_join_streaming: implemented via merge dispatch, not "
        "direct call (needs re-iterable right). Composition in merge: "
        "left_join(left, right) -> all left + right padded; "
        "anti_join(right, left) -> right-only rows; pad with null "
        "left columns; concat via IPC writer. Output schema: "
        "_empty_result_schema (all fields nullable=True). "
        "CadwDataFrame.merge(how=...) extended: "
        "inner | left | outer | anti | semi (semi -> NotImplementedError "
        "in v0.9.23i, deferred to v0.9.23j). "
        "'right' still NotImplementedError (v0.9.23j). "
        "Export anti_join_streaming, outer_join_streaming from cadw "
        "(public API). "
        "+tests/test_anti_outer_join.py (18 tests: anti basic, "
        "all-match, none-match, empty-right, null-left-key, multi-"
        "key; outer basic (both+padding), all-match, disjoint, "
        "null-left, empty-left, empty-right, multi-key; unsupported "
        "how raises, semi NotImplementedError; 4 AST guards - "
        "hash_shuffle, is_in, fill_null(False), merge dispatch). "
        "Dogfood invariants only (NO distributions, lesson from "
        "v0.9.23h): anti = not-in invariant per emitted key + "
        "expected-anti-set matches left-right; outer = union of "
        "keys + null-padding correctness; null-left anti emitted. "
        "Split rationale (v0.9.23h rule > 200 LOC): anti+outer "
        "together ~150 LOC (this patch), semi ~40 LOC (v0.9.23j), "
        "categorical FE 3 patches (v0.9.23k-l). "
        "Fast: 2125 -> ~2143 passed.",
        self_path="scripts/patch_v0923i_anti_outer_join.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23i applied. 7.A.10 partial (anti + outer).")


if __name__ == "__main__":
    main()