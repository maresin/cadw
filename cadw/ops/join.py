"""
Inner join (v0.2.1).

Grace hash join: HashShuffle обеих сторон с одним seed → партиции с
одинаковыми ключами → для каждой пары (left_p, right_p):
  - build = меньшая партиция (по file size), materialize как pd.DataFrame;
    check_state(build.nbytes);
  - probe = другая, батчами; check_batch(probe_batch);
  - проверка max_output_rows * probe_rows до pd.merge (защита от cartesian);
  - null-ключи отбрасываются (SQL-семантика; pandas.merge по умолчанию
    матчит NaN==NaN, что не то);
  - pd.merge(how="inner", suffixes=...).

Ограничения v0.2.1:
- how='inner' только.
- on — общие колонки в обеих сторонах (по имени).
- left_on/right_on, on=index — v0.2.2.
- row_id не сохраняется (join создаёт новую идентичность).
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
        fields.append(left_schema.field(c))
    for f in left_rest:
        name = f.name + suffixes[0] if f.name in overlap else f.name
        fields.append(pa.field(name, f.type))
    for f in right_rest:
        name = f.name + suffixes[1] if f.name in overlap else f.name
        fields.append(pa.field(name, f.type))
    return pa.schema(fields)


def _materialize_partition(path: Path, memory_guard: MemoryGuard) -> pa.Table:
    batches = []
    for b in ArrowIPCBatchReader(str(path), None):
        keep = [n for n in b.schema.names if n != "row_id"]
        batches.append(b.select(keep))
    if not batches:
        return pa.Table.from_batches([], schema=pa.schema([]))
    table = pa.Table.from_batches(batches)
    memory_guard.check_state(table.nbytes, name=f"join_build({path.name})")
    return table


def _iter_probe_batches(path: Path) -> Iterable[pa.RecordBatch]:
    for b in ArrowIPCBatchReader(str(path), None):
        keep = [n for n in b.schema.names if n != "row_id"]
        yield b.select(keep)


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
    """Grace hash inner join. Возвращает pa.Table."""
    if not on:
        raise ValueError("inner_join: on must be non-empty")
    for c in on:
        if c not in left_schema.names:
            raise KeyError(f"inner_join: on column {c!r} not in left schema")
        if c not in right_schema.names:
            raise KeyError(f"inner_join: on column {c!r} not in right schema")

    left_rest = [c for c in left_schema.names if c not in on]
    right_rest = [c for c in right_schema.names if c not in on]

    # Peek: пустой вход -> пустой результат
    left_it = iter(left_batches)
    right_it = iter(right_batches)
    try:
        first_left = next(left_it)
    except StopIteration:
        return pa.Table.from_batches(
            [], schema=_empty_result_schema(left_schema, right_schema, on, suffixes)
        )
    try:
        first_right = next(right_it)
    except StopIteration:
        return pa.Table.from_batches(
            [], schema=_empty_result_schema(left_schema, right_schema, on, suffixes)
        )
    left_batches = chain([first_left], left_it)
    right_batches = chain([first_right], right_it)

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
        # Консервативная оценка: 0.3 * available / 256 B на строку
        max_output_rows = max(
            1, int(memory_guard.available_bytes() * 0.3 / 256)
        )

    empty_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)
    all_batches: list[pa.RecordBatch] = []

    for p in range(num_partitions):
        lp = left_paths[p] if left_paths[p].exists() else None
        rp = right_paths[p] if right_paths[p].exists() else None
        if lp is None or rp is None:
            continue

        # build = меньшая партиция по file size (proxy для памяти)
        if lp.stat().st_size <= rp.stat().st_size:
            build_path, probe_path = lp, rp
        else:
            build_path, probe_path = rp, lp

        build_table = _materialize_partition(build_path, memory_guard)
        build_df = _drop_null_keys(build_table.to_pandas(), list(on))
        if len(build_df) == 0:
            continue

        for probe_batch in _iter_probe_batches(probe_path):
            memory_guard.check_batch(
                probe_batch, name=f"join_probe({probe_path.name})"
            )
            probe_df = _drop_null_keys(probe_batch.to_pandas(), list(on))
            if len(probe_df) == 0:
                continue
            # Верхняя оценка output до materialize
            if len(build_df) * len(probe_df) > max_output_rows:
                raise MemoryLimitExceeded(
                    f"inner_join: estimated output rows "
                    f"{len(build_df)} * {len(probe_df)} = "
                    f"{len(build_df) * len(probe_df)} > max_output_rows="
                    f"{max_output_rows}. Likely skew (DECISIONS §11.4)."
                )
            merged = pd.merge(
                build_df, probe_df, on=list(on), how="inner", suffixes=suffixes
            )
            if len(merged) == 0:
                continue
            out_table = pa.Table.from_pandas(merged, preserve_index=False)
            # Привести порядок колонок к empty_schema
            if out_table.schema.names != empty_schema.names:
                out_table = out_table.select(empty_schema.names)
            memory_guard.check_state(
                out_table.nbytes, name=f"join_output(p={p})"
            )
            for b in out_table.to_batches():
                all_batches.append(b)

    if not all_batches:
        return pa.Table.from_batches([], schema=empty_schema)
    return pa.Table.from_batches(all_batches)
