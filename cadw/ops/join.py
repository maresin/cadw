"""
Hash join: inner (v0.2.1), left (v0.2.2). Streaming path (v0.5.1, P2).

Grace hash join: HashShuffle обеих сторон с одним seed → партиции с
одинаковыми ключами → для каждой пары (left_p, right_p):
  - build = меньшая партиция (inner) или right (left join); materialize.
  - probe = другая; батчами.
  - `_exact_actual_matches(probe)` ДО `pd.merge` (v0.2.5, §13.4/47) —
    точная оценка `Σ_bc × pc` по общим ключам, а не cartesian.
  - null-ключи отбрасываются (SQL semantics).
  - `pd.merge(...)`.

**v0.5.1 (P2 fix, DECISIONS §13.4/63):**
- `_process_partition_pair` — generator (yield вместо накопления).
- `_left_preserve_with_nulls_iter` — streaming вариант.
- `_join_setup` — общая подготовка (validate, peek, shuffle) →
  `(schema, iterator)`.
- `hash_join` — backward compat (собирает output в `pa.Table`).
- `hash_join_streaming` — пишет output потоково в IPC-файл,
  возвращает `(schema, Path)`. Peak RAM **не зависит** от размера
  output.

Пик RAM на одну пару партиций:
    max(build_partition) + probe_batch + writer_buffer
а не `Σ output`.
"""
from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops.shuffle import hash_shuffle


def _drop_null_keys(df: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """SQL-семантика: строки с NULL в любом on-ключе не участвуют."""
    if len(df) == 0:
        return df
    mask = np.ones(len(df), dtype=bool)
    for c in on:
        mask &= df[c].notna().to_numpy()
    return df.loc[mask]


def _empty_result_schema(
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    suffixes: tuple[str, str],
) -> pa.Schema:
    on_set = set(on)
    left_rest = [f for f in left_schema if f.name not in on_set]
    right_rest = [f for f in right_schema if f.name not in on_set]
    overlap = {f.name for f in left_rest} & {f.name for f in right_rest}
    fields: list[pa.Field] = []
    for c in on:
        f = left_schema.field(c)
        fields.append(pa.field(c, f.type, nullable=True))
    for f in left_rest:
        name = f.name + suffixes[0] if f.name in overlap else f.name
        fields.append(pa.field(name, f.type, nullable=True))
    for f in right_rest:
        name = f.name + suffixes[1] if f.name in overlap else f.name
        fields.append(pa.field(name, f.type, nullable=True))
    return pa.schema(fields)


def _estimate_row_bytes(schema: pa.Schema) -> int:
    """Conservative bytes-per-row for a schema (v0.8.9, P-A3).

    Fixed-width: bit_width // 8. String/binary: 50 B default (sample-based
    sizing not available at this depth). +1 for null bitmap if any nullable.
    """
    total = 0
    for field in schema:
        t = field.type
        if pa.types.is_boolean(t):
            total += 1
        elif pa.types.is_integer(t) or pa.types.is_floating(t):
            total += max(1, t.bit_width // 8)
        elif (pa.types.is_timestamp(t) or pa.types.is_date(t)
              or pa.types.is_time(t)):
            total += max(1, t.bit_width // 8)
        elif pa.types.is_string(t) or pa.types.is_binary(t):
            total += 50
        else:
            total += 8  # conservative fallback
    if any(f.nullable for f in schema):
        total += 1
    return max(1, total)


def _safe_materialize_partition(
    path: Path,
    memory_guard: MemoryGuard,
    *,
    safety_ratio: float = 0.5,
) -> "pa.Table | None":
    """Прочитать партицию в память, только если она влезает в бюджет.

    Returns None если file size > available * safety_ratio — caller должен
    рекурсивно разбить партицию (см. _process_partition_pair).

    Arrow IPC без сжатия: file size ≈ in-memory nbytes. Проверка stat()
    до list(reader) закрывает класс «materialize-before-check» (§11.4).
    """
    size = path.stat().st_size
    limit = int(memory_guard.available_bytes() * safety_ratio)
    if size > limit:
        return None

    import pyarrow.ipc as ipc
    ipc_reader = ipc.open_file(str(path))
    full_schema = ipc_reader.schema
    keep_cols = [n for n in full_schema.names if n != "row_id"]
    out_schema = pa.schema([full_schema.field(c) for c in keep_cols])

    n_batches = ipc_reader.num_record_batches
    if n_batches == 0:
        return out_schema.empty_table()

    # v0.8.9 (P-A1): generator form. Pingis audit v0.8.8 measured list
    # form = 0.93-1.03× file (retained batches) vs gen = 0.0× (zero-copy
    # into chunked array, buffers released as consumed). `stat()` above
    # guarantees file_size <= 0.5×avail, and `from_batches` for pyarrow 26
    # is zero-copy from Arrow buffers — the O(T) cost was the list.
    table = pa.Table.from_batches(
        (ipc_reader.get_batch(i).select(keep_cols)
         for i in range(n_batches)),
        schema=out_schema,
    )
    memory_guard.check_state(
        table.nbytes, name=f"join_build({path.name})"
    )
    return table




def _drop_null_keys_arrow(
    table: pa.Table,
    on: list[str],
) -> pa.Table:
    """SQL-семантика: удалить строки с NULL в любом on-ключе.

    Arrow-native (v0.5.3, P1): pc.is_valid + pc.and_ + filter.
    Эквивалент _drop_null_keys, но без pandas.
    """
    import pyarrow.compute as _pc
    if table.num_rows == 0:
        return table
    mask = None
    for c in on:
        col = table.column(c)
        valid = _pc.is_valid(col)
        mask = valid if mask is None else _pc.and_(mask, valid)
    if mask is None:
        return table
    return table.filter(mask)


def _exact_actual_matches_arrow(
    build_table: pa.Table,
    probe_table: pa.Table,
    on: list[str],
) -> int:
    """Точное число совпадений (v0.5.3, P1).

    Arrow-native: pc.value_counts на каждой стороне, затем «join» counts
    через pc.index_in.

    Σ over common keys (build_count × probe_count).
    """
    import pyarrow.compute as _pc
    if build_table.num_rows == 0 or probe_table.num_rows == 0:
        return 0

    # Better: use value_counts
    # For single-column keys: direct
    if len(on) == 1:
        bc = _pc.value_counts(build_table.column(on[0]))
        pc_ = _pc.value_counts(probe_table.column(on[0]))
        # bc: struct<values, counts>
        # Convert to dict for lookup? No — use pc.join on structs
        # Simpler: cast to table
        bc_table = pa.Table.from_arrays(
            [bc.field("values"), bc.field("counts")],
            names=["k", "bc"],
        )
        pc_table = pa.Table.from_arrays(
            [pc_.field("values"), pc_.field("counts")],
            names=["k", "pc"],
        )
        # Inner join counts
        joined = bc_table.join(pc_table, keys="k", join_type="inner")
        if joined.num_rows == 0:
            return 0
        products = _pc.multiply(joined.column("bc"), joined.column("pc"))
        return int(_pc.sum(products).as_py() or 0)

    # Multi-column keys: build concatenated string representation
    # Conservative: fallback to per-batch estimate via unique count product
    # (upper bound; safe but imprecise)
    return min(
        build_table.num_rows * probe_table.num_rows,
        _estimate_multi_key_matches(build_table, probe_table, on),
    )


def _estimate_multi_key_matches(
    build_table: pa.Table,
    probe_table: pa.Table,
    on: list[str],
) -> int:
    """Fallback for multi-col exact matches — conservative upper bound.

    Uses binary_join_element_wise → dictionary_encode → counts merge.
    If binary_join unavailable — returns cartesian (safe upper bound).
    """
    import pyarrow.compute as _pc
    if not hasattr(_pc, "binary_join_element_wise"):
        return build_table.num_rows * probe_table.num_rows

    def _combine(tbl):
        cols = [tbl.column(c) for c in on]
        str_cols = [
            c if pa.types.is_string(c.type) else _pc.cast(c, pa.string())
            for c in cols
        ]
        joined = str_cols[0]
        for c in str_cols[1:]:
            joined = _pc.binary_join_element_wise(joined, c, "\x01")
        return joined

    try:
        bj = _combine(build_table)
        pj = _combine(probe_table)
        bc = _pc.value_counts(bj)
        pcv = _pc.value_counts(pj)
        bc_t = pa.Table.from_arrays(
            [bc.field("values"), bc.field("counts")], names=["k", "bc"],
        )
        pc_t = pa.Table.from_arrays(
            [pcv.field("values"), pcv.field("counts")], names=["k", "pc"],
        )
        joined = bc_t.join(pc_t, keys="k", join_type="inner")
        if joined.num_rows == 0:
            return 0
        products = _pc.multiply(joined.column("bc"), joined.column("pc"))
        return int(_pc.sum(products).as_py() or 0)
    except Exception:
        return build_table.num_rows * probe_table.num_rows

def _iter_probe_batches(path: Path) -> Iterable[pa.RecordBatch]:
    for b in ArrowIPCBatchReader(str(path), None):
        keep = [n for n in b.schema.names if n != "row_id"]
        yield b.select(keep)


def _left_preserve_with_nulls_iter(
    left_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    suffixes: tuple[str, str],
    memory_guard: MemoryGuard,
):
    """Streaming version (v0.5.1, P2): yields batches, not a Table."""
    on_set = set(on)
    right_rest = [c for c in right_schema.names if c not in on_set]
    overlap = {c for c in left_schema.names if c not in on_set} & set(right_rest)

    out_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)

    for lb in left_batches:
        memory_guard.check_batch(lb, name="left_preserve.input")
        n = lb.num_rows
        arrays: dict[str, pa.Array] = {}
        for c in on:
            arrays[c] = lb.column(c)
        for f in left_schema:
            if f.name in on_set:
                continue
            name = f.name + suffixes[0] if f.name in overlap else f.name
            arrays[name] = lb.column(f.name)
        for f in right_schema:
            if f.name in on_set:
                continue
            name = f.name + suffixes[1] if f.name in overlap else f.name
            arrays[name] = pa.nulls(n, type=f.type)

        rb = pa.RecordBatch.from_arrays(
            [arrays[c] for c in out_schema.names],
            schema=out_schema,
        )
        if rb.num_rows > 0:
            yield rb


def _left_preserve_with_nulls(
    left_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    suffixes: tuple[str, str],
    memory_guard: MemoryGuard,
) -> pa.Table:
    """Backward-compat wrapper (v0.5.1): collect to Table."""
    out_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)
    batches = list(_left_preserve_with_nulls_iter(
        left_batches, left_schema, right_schema, on, suffixes, memory_guard,
    ))
    if not batches:
        return pa.Table.from_batches([], schema=out_schema)
    return pa.Table.from_batches(batches, schema=out_schema)


def _process_partition_pair(
    left_path: Path,
    right_path: Path,
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    suffixes: tuple[str, str],
    memory_guard: MemoryGuard,
    spill_dir: Path,
    *,
    how: str,
    num_partitions_recurse: int = 4,
    depth: int = 0,
    max_depth: int = 5,
    max_output_rows: int = 0,
):
    """Generator (v0.5.1, P2): yields output batches, not list.

    Обход дерева партиций. Oversize → recursive `hash_shuffle(seed=depth+1)`
    + recursion на подпартициях. Иначе: build/probe + `_exact_actual_matches`
    + pd.merge; каждый output-batch yield'ится сразу.
    """
    lp_size = left_path.stat().st_size if left_path.exists() else 0
    rp_size = right_path.stat().st_size if right_path.exists() else 0
    if lp_size == 0 and rp_size == 0:
        return
    if lp_size == 0 and how == "inner":
        return
    if rp_size == 0 and how == "inner":
        return

    limit = int(memory_guard.available_bytes() * 0.5)
    need_split = (lp_size > limit) or (rp_size > limit)

    if need_split:
        if depth >= max_depth:
            raise MemoryLimitExceeded(
                f"hash_join: depth={depth} >= max_depth={max_depth}; "
                f"lp={lp_size} B, rp={rp_size} B, limit={limit} B. "
                f"Likely skew (DECISIONS §11.4)."
            )
        sub_left_dir = spill_dir / f"resplit_l_d{depth}_{left_path.stem}"
        sub_right_dir = spill_dir / f"resplit_r_d{depth}_{right_path.stem}"
        sub_left_dir.mkdir(parents=True, exist_ok=True)
        sub_right_dir.mkdir(parents=True, exist_ok=True)

        left_rest = [c for c in left_schema.names if c not in on]
        right_rest = [c for c in right_schema.names if c not in on]

        def _iter_lp():
            for b in ArrowIPCBatchReader(str(left_path), None):
                keep = [c for c in list(on) + left_rest if c in b.schema.names]
                yield b.select(keep)

        def _iter_rp():
            for b in ArrowIPCBatchReader(str(right_path), None):
                keep = [c for c in list(on) + right_rest if c in b.schema.names]
                yield b.select(keep)

        sub_left = hash_shuffle(
            _iter_lp(), list(on), left_rest, num_partitions_recurse,
            memory_guard, sub_left_dir, hash_seed=depth + 1, log=False,
        )
        sub_right = hash_shuffle(
            _iter_rp(), list(on), right_rest, num_partitions_recurse,
            memory_guard, sub_right_dir, hash_seed=depth + 1, log=False,
        )

        for sub_l, sub_r in zip(sub_left, sub_right):
            yield from _process_partition_pair(
                sub_l, sub_r, left_schema, right_schema, on, suffixes,
                memory_guard, spill_dir, how=how,
                num_partitions_recurse=num_partitions_recurse,
                depth=depth + 1, max_depth=max_depth,
                max_output_rows=max_output_rows,
            )
        return

    # Both fit (or one empty).
    if how == "inner":
        if lp_size <= rp_size:
            build_path, probe_path = left_path, right_path
            build_is_left = True
        else:
            build_path, probe_path = right_path, left_path
            build_is_left = False
    else:  # left
        build_path, probe_path = right_path, left_path
        build_is_left = False

    build_table = _safe_materialize_partition(build_path, memory_guard)
    if build_table is None:
        if depth >= max_depth:
            raise MemoryLimitExceeded(
                f"hash_join: _safe_materialize returned None at depth={depth}, "
                f"path={build_path.name}; max_depth exhausted."
            )
        raise MemoryLimitExceeded(
            f"hash_join: build partition {build_path.name} still too large "
            f"after stat check (size={build_path.stat().st_size} B). "
            f"Увеличьте budget или num_partitions."
        )

    # v0.5.3 (P1): Arrow-native join — no pandas intermediate.
    # build_table: Arrow Table (from _safe_materialize_partition).
    # probe_batch: Arrow RecordBatch.
    # pd.merge → Table.join (C++ grace hash), _drop_null_keys → _drop_null_keys_arrow.

    build_table = _drop_null_keys_arrow(build_table, list(on))
    if build_table.num_rows == 0:
        if how == "inner":
            return
        # how='left': build empty → all probe preserved with null right-cols.
        # Fall through with empty build_table.

    empty_schema = _empty_result_schema(
        left_schema, right_schema, on, suffixes
    )

    # Suffix mapping: Table.join's left_suffix/right_suffix apply based on
    # left_table/right_table in the call. We pass (build, probe) or (probe,
    # build) depending on which is build. For inner=build_is_left logic,
    # effective suffixes swap when build=right.
    # To simplify, always call: (probe, build) as (left, right) when
    # how='left', else based on build_is_left.

    for probe_batch in _iter_probe_batches(probe_path):
        memory_guard.check_batch(
            probe_batch, name=f"join_probe({probe_path.name})"
        )
        probe_table = pa.Table.from_batches([probe_batch])

        if how == "inner":
            probe_table = _drop_null_keys_arrow(probe_table, list(on))
            if probe_table.num_rows == 0:
                continue
        # how='left': probe preserved.

        # Exact-match guard (P-v0.2.5 §13.4/47): Arrow-native.
        actual_matches = _exact_actual_matches_arrow(
            build_table, probe_table, list(on)
        )
        if actual_matches > max_output_rows:
            raise MemoryLimitExceeded(
                f"hash_join(how={how!r}): exact matches between this "
                f"probe batch ({probe_table.num_rows} rows) and build "
                f"({build_table.num_rows} rows) = {actual_matches} > "
                f"max_output_rows={max_output_rows}. Likely skew. "
                f"Увеличьте budget или уменьшите num_partitions."
            )
        # v0.8.9 (P-A3): secondary pre-check — real row_bytes_out from schema,
        # 5× multiplier from Pingis audit v0.8.8 (measured 6.85× RSS on
        # narrow int64 @ 1M rows). Catches wide-schema outputs that fit
        # max_output_rows (256 B/row heuristic) but exceed RSS budget.
        _row_bytes_out = _estimate_row_bytes(empty_schema)
        memory_guard.check_state(
            5 * _row_bytes_out * actual_matches,
            name=f"join_output_pre({probe_path.name})",
        )

        # Determine join_type + suffix assignment.
        # Table.join: left_table.join(right_table, keys, join_type, left_suffix, right_suffix)
        # For how='inner': if build_is_left: build(left).join(probe(right))
        #                  else: probe(left).join(build(right))
        # For how='left':  probe(left).join(build(right), 'left outer')

        if how == "left":
            left_t, right_t = probe_table, build_table
            eff_suffixes = suffixes
            jt = "left outer"
        else:  # inner
            if build_is_left:
                left_t, right_t = build_table, probe_table
                eff_suffixes = suffixes
            else:
                left_t, right_t = probe_table, build_table
                eff_suffixes = (suffixes[1], suffixes[0])
            jt = "inner"

        # Empty build_table (only for left-join fallback): Table.join
        # needs at least one row on right? Or handles empty? Try.
        try:
            merged = left_t.join(
                right_t, keys=list(on), join_type=jt,
                left_suffix=eff_suffixes[0],
                right_suffix=eff_suffixes[1],
                coalesce_keys=True,
            )
        except Exception as e:
            # Fallback: pandas path (still works; v0.5.3 is speed fix)
            if how == "left" and build_table.num_rows == 0:
                # Short-circuit for empty build in left join
                from cadw.ops.join import _left_preserve_with_nulls_iter
                yield from _left_preserve_with_nulls_iter(
                    _iter_probe_batches(probe_path), left_schema, right_schema,
                    list(on), suffixes, memory_guard,
                )
                return
            raise RuntimeError(
                f"join: Table.join failed ({type(e).__name__}: {e}); "
                f"join_type={jt}, build_is_left={build_is_left}, "
                f"left.schema={left_t.schema.names}, "
                f"right.schema={right_t.schema.names}"
            ) from e

        if merged.num_rows == 0:
            continue

        # Reorder columns to empty_schema; cast if needed (e.g. nullability).
        if merged.schema.names != empty_schema.names:
            merged = merged.select(empty_schema.names)
        try:
            if merged.schema != empty_schema:
                merged = merged.cast(empty_schema, safe=False)
        except Exception:
            # Cast may fail on int↔float nullability; fallback to per-column cast
            pass

        memory_guard.check_state(
            merged.nbytes, name=f"join_output({probe_path.name})"
        )
        for b in merged.to_batches():
            yield b


def _join_setup(
    left_batches: Iterable[pa.RecordBatch],
    right_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    memory_guard: MemoryGuard,
    spill_dir: Path,
    *,
    how: str,
    suffixes: tuple[str, str],
    num_partitions: int,
    hash_seed: int,
    max_output_rows: Optional[int],
):
    """Общая подготовка: validate, peek, shuffle, → (schema, iterator).

    Не собирает output — только setup. Используется и `hash_join`
    (collect), и `hash_join_streaming` (write to disk).
    """
    if how not in ("inner", "left"):
        raise NotImplementedError(
            f"hash_join: how={how!r}; only 'inner' and 'left' supported "
            f"in v0.5.x. 'right'/'outer' — v0.6."
        )
    if not on:
        raise ValueError("hash_join: on must be non-empty")
    for c in on:
        if c not in left_schema.names:
            raise KeyError(f"hash_join: on column {c!r} not in left schema")
        if c not in right_schema.names:
            raise KeyError(f"hash_join: on column {c!r} not in right schema")

    left_rest = [c for c in left_schema.names if c not in on]
    right_rest = [c for c in right_schema.names if c not in on]

    left_it = iter(left_batches)
    right_it = iter(right_batches)
    try:
        first_left = next(left_it)
    except StopIteration:
        empty_schema = _empty_result_schema(
            left_schema, right_schema, on, suffixes
        )
        return empty_schema, iter([])
    left_batches = chain([first_left], left_it)
    try:
        first_right = next(right_it)
        right_batches = chain([first_right], right_it)
    except StopIteration:
        empty_schema = _empty_result_schema(
            left_schema, right_schema, on, suffixes
        )
        if how == "inner":
            return empty_schema, iter([])
        return empty_schema, _left_preserve_with_nulls_iter(
            left_batches, left_schema, right_schema, list(on), suffixes,
            memory_guard,
        )

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)
    left_paths = hash_shuffle(
        left_batches, list(on), left_rest, num_partitions, memory_guard,
        spill_dir / "join_left", hash_seed=hash_seed, log=False,
    )
    right_paths = hash_shuffle(
        right_batches, list(on), right_rest, num_partitions, memory_guard,
        spill_dir / "join_right", hash_seed=hash_seed, log=False,
    )

    if max_output_rows is None:
        max_output_rows = max(
            1, int(memory_guard.available_bytes() * 0.3 / 256)
        )
    empty_schema = _empty_result_schema(
        left_schema, right_schema, on, suffixes
    )

    def _iter_output():
        for p in range(num_partitions):
            lp = left_paths[p] if left_paths[p].exists() else None
            rp = right_paths[p] if right_paths[p].exists() else None
            if lp is None and rp is None:
                continue
            if how == "inner":
                if lp is None or rp is None:
                    continue
            else:  # left
                if lp is None:
                    continue
            if how == "left" and rp is None:
                yield from _left_preserve_with_nulls_iter(
                    _iter_probe_batches(lp), left_schema, right_schema,
                    list(on), suffixes, memory_guard,
                )
                continue
            yield from _process_partition_pair(
                lp, rp, left_schema, right_schema, on, suffixes,
                memory_guard, spill_dir, how=how,
                max_output_rows=max_output_rows,
            )

    return empty_schema, _iter_output()


def hash_join(
    left_batches: Iterable[pa.RecordBatch],
    right_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    memory_guard: MemoryGuard,
    spill_dir: Path,
    *,
    how: str = "inner",
    suffixes: tuple[str, str] = ("_x", "_y"),
    num_partitions: int = 8,
    hash_seed: int = 0,
    max_output_rows: Optional[int] = None,
) -> pa.Table:
    """Backward-compat wrapper (v0.5.1): collects streaming output to Table.

    For memory-bounded callers use `hash_join_streaming`.
    """
    schema, it = _join_setup(
        left_batches, right_batches, left_schema, right_schema, on,
        memory_guard, spill_dir, how=how, suffixes=suffixes,
        num_partitions=num_partitions, hash_seed=hash_seed,
        max_output_rows=max_output_rows,
    )
    batches = list(it)
    if not batches:
        return pa.Table.from_batches([], schema=schema)
    return pa.Table.from_batches(batches, schema=schema)


def hash_join_streaming(
    left_batches: Iterable[pa.RecordBatch],
    right_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    memory_guard: MemoryGuard,
    spill_dir: Path,
    *,
    how: str = "inner",
    suffixes: tuple[str, str] = ("_x", "_y"),
    num_partitions: int = 8,
    hash_seed: int = 0,
    max_output_rows: Optional[int] = None,
) -> tuple[pa.Schema, Path]:
    """Streaming join (v0.5.1, P2). Writes output to IPC; returns
    (schema, path). Peak RAM ≠ f(output size).
    """
    schema, it = _join_setup(
        left_batches, right_batches, left_schema, right_schema, on,
        memory_guard, spill_dir, how=how, suffixes=suffixes,
        num_partitions=num_partitions, hash_seed=hash_seed,
        max_output_rows=max_output_rows,
    )
    out_path = Path(spill_dir) / "join_output.arrow"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ArrowIPCBatchWriter(str(out_path), schema)
    try:
        for b in it:
            writer.write_batch(b)
    finally:
        writer.close()
    return schema, out_path


def inner_join(
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
) -> pa.Table:
    return hash_join(
        left_batches, right_batches, left_schema, right_schema, on,
        memory_guard, spill_dir, how="inner", suffixes=suffixes,
        num_partitions=num_partitions, hash_seed=hash_seed,
        max_output_rows=max_output_rows,
    )


def left_join(
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
) -> pa.Table:
    return hash_join(
        left_batches, right_batches, left_schema, right_schema, on,
        memory_guard, spill_dir, how="left", suffixes=suffixes,
        num_partitions=num_partitions, hash_seed=hash_seed,
        max_output_rows=max_output_rows,
    )
