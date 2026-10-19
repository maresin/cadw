"""
Hash join: inner (v0.2.1), left (v0.2.2).

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
        f = left_schema.field(c)
        fields.append(pa.field(c, f.type, nullable=True))
    for f in left_rest:
        name = f.name + suffixes[0] if f.name in overlap else f.name
        fields.append(pa.field(name, f.type, nullable=True))
    for f in right_rest:
        name = f.name + suffixes[1] if f.name in overlap else f.name
        fields.append(pa.field(name, f.type, nullable=True))
    return pa.schema(fields)


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

    ВАЖНО: пустой файл (без record batches) НЕ 0 байт — Arrow IPC всегда
    пишет schema-заголовок (~200-500 B). Возвращаем schema-correct
    empty_table() с колонками, а не pa.schema([]) — иначе pd.merge
    получит DataFrame без on-колонок → KeyError.
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

    batches = [
        ipc_reader.get_batch(i).select(keep_cols)
        for i in range(n_batches)
    ]
    table = pa.Table.from_batches(batches)
    memory_guard.check_state(
        table.nbytes, name=f"join_build({path.name})"
    )
    return table


def _iter_probe_batches(path: Path) -> Iterable[pa.RecordBatch]:
    for b in ArrowIPCBatchReader(str(path), None):
        keep = [n for n in b.schema.names if n != "row_id"]
        yield b.select(keep)


def _left_preserve_with_nulls(
    left_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    suffixes: tuple[str, str],
    memory_guard: MemoryGuard,
) -> pa.Table:
    """how='left' при пустом right: left preserved, right-колонки -> null.

    Используется, когда right содержит 0 батчей: hash_shuffle не может
    принять пустой вход (см. §6.7), а left join не требует shuffle, потому
    что совпадений всё равно нет.
    """
    on_set = set(on)
    right_rest = [c for c in right_schema.names if c not in on_set]
    overlap = {c for c in left_schema.names if c not in on_set} & set(right_rest)

    out_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)

    all_batches: list[pa.RecordBatch] = []
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
            all_batches.append(rb)

    if not all_batches:
        return pa.Table.from_batches([], schema=out_schema)
    return pa.Table.from_batches(all_batches)


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
    max_cartesian_rows: int = 0,
) -> list[pa.RecordBatch]:
    """Обработать пару партиций (lp, rp) с учётом how.

    Если хотя бы одна из партиций не влезает в бюджет (stat > available*0.5),
    обе разбиваются hash_shuffle с seed=depth+1 на num_partitions_recurse
    подпартиций, и рекурсия продолжается для соответствующих подпар.

    Иначе: build = (right для left join; меньшая для inner), probe = другая;
    _safe_materialize_partition для build; probe батчами.
    """
    lp_size = left_path.stat().st_size if left_path.exists() else 0
    rp_size = right_path.stat().st_size if right_path.exists() else 0
    if lp_size == 0 and rp_size == 0:
        return []
    if lp_size == 0:
        if how == "inner":
            return []
        # how='left': left preserved, right null (right пуст)
        # Обрабатываем через _left_preserve_with_nulls? Но здесь есть
        # только одна партиция left. Проще: обработать как build_df пустой.
    if rp_size == 0 and how == "inner":
        return []

    limit = int(memory_guard.available_bytes() * 0.5)
    need_split = (lp_size > limit) or (rp_size > limit)

    if need_split:
        if depth >= max_depth:
            raise MemoryLimitExceeded(
                f"hash_join: depth={depth} >= max_depth={max_depth}; "
                f"lp={lp_size} B, rp={rp_size} B, limit={limit} B. "
                f"Likely skew (DECISIONS §11.4)."
            )
        # Разбиваем обе стороны согласованно.
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

        out: list[pa.RecordBatch] = []
        for sub_l, sub_r in zip(sub_left, sub_right):
            out.extend(
                _process_partition_pair(
                    sub_l, sub_r, left_schema, right_schema, on, suffixes,
                    memory_guard, spill_dir, how=how,
                    num_partitions_recurse=num_partitions_recurse,
                    depth=depth + 1, max_depth=max_depth,
                    max_output_rows=max_output_rows,
                    max_cartesian_rows=max_cartesian_rows,
                )
            )
        return out

    # Обе партиции влезают (или одна пуста).
    # Выбор build/probe.
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
        # Не должно произойти: мы проверили stat выше, но на случай
        # гонки/сжатия — уходим в рекурсию.
        if depth >= max_depth:
            raise MemoryLimitExceeded(
                f"hash_join: _safe_materialize returned None at depth={depth}, "
                f"path={build_path.name}; max_depth exhausted."
            )
        # Принудительный split через рекурсию (без повторного stat).
        # Упрощённо: поднимаем MemoryLimitExceeded, чтобы не дублировать логику.
        raise MemoryLimitExceeded(
            f"hash_join: build partition {build_path.name} still too large "
            f"after stat check (size={build_path.stat().st_size} B). "
            f"Увеличьте budget или num_partitions."
        )

    build_df = _drop_null_keys(build_table.to_pandas(), list(on))
    if len(build_df) == 0:
        if how == "inner":
            return []
        if build_df.shape[1] == 0:
            right_rest = [c for c in right_schema.names if c not in on]
            build_df = pd.DataFrame(columns=list(on) + right_rest)

    # Per-partition skew check: unique_build_keys + avg_matches_per_probe.
    # Ловит partition-level skew, которого глобальная модель не видит:
    # после hash_shuffle все строки с одним ключом идут в одну партицию,
    # и там build может дублироваться сильнее, чем в среднем.
    unique_build_keys = len(build_df)
    if len(build_df) > 0 and len(on) > 0:
        try:
            if len(on) == 1:
                unique_build_keys = int(build_df[on[0]].nunique(dropna=True))
            else:
                unique_build_keys = int(
                    build_df.drop_duplicates(subset=list(on)).shape[0]
                )
            unique_build_keys = max(1, unique_build_keys)
        except Exception:
            unique_build_keys = len(build_df)
    avg_matches_per_probe = (
        len(build_df) / max(1, unique_build_keys)
        if unique_build_keys > 0 else 1.0
    )

    out_batches: list[pa.RecordBatch] = []
    empty_schema = _empty_result_schema(
        left_schema, right_schema, on, suffixes
    )

    for probe_batch in _iter_probe_batches(probe_path):
        memory_guard.check_batch(
            probe_batch, name=f"join_probe({probe_path.name})"
        )
        probe_df = probe_batch.to_pandas()
        if how == "inner":
            probe_df = _drop_null_keys(probe_df, list(on))
            if len(probe_df) == 0:
                continue
        # how='left': probe preserved.

        # 1. Expected output (avg_matches × probe) против max_output_rows.
        #    Ловит per-partition skew: если в этой партиции build дублируется
        #    сильнее, чем в среднем по датасету.
        expected_probe_output = int(
            len(probe_df) * max(1.0, avg_matches_per_probe)
        )
        if expected_probe_output > max_output_rows:
            raise MemoryLimitExceeded(
                f"hash_join(how={how!r}): expected output from this "
                f"partition ≈ {expected_probe_output} rows "
                f"({len(probe_df)} probe × {avg_matches_per_probe:.2f} "
                f"avg_matches) > max_output_rows={max_output_rows}. "
                f"Likely partition-level skew."
            )
        # 2. Cartesian worst case (build × probe) против max_cartesian_rows.
        #    Ловит экстремальный skew, где даже probe-строка с уникальным
        #    ключом матчит много build-строк.
        cartesian = len(build_df) * len(probe_df)
        if cartesian > max_cartesian_rows:
            raise MemoryLimitExceeded(
                f"hash_join(how={how!r}): cartesian worst case "
                f"{len(build_df)} × {len(probe_df)} = {cartesian} > "
                f"max_cartesian_rows={max_cartesian_rows}. "
                f"Likely extreme skew."
            )

        if how == "left":
            merged = pd.merge(
                probe_df, build_df, on=list(on), how="left",
                suffixes=suffixes,
            )
        else:
            effective_suffixes = (
                suffixes if build_is_left else (suffixes[1], suffixes[0])
            )
            merged = pd.merge(
                build_df, probe_df, on=list(on), how="inner",
                suffixes=effective_suffixes,
            )
        if len(merged) == 0:
            continue
        merged = merged.reindex(columns=empty_schema.names)
        try:
            out_table = pa.Table.from_pandas(
                merged, schema=empty_schema, preserve_index=False
            )
        except Exception:
            out_table = pa.Table.from_pandas(merged, preserve_index=False)
            if out_table.schema.names != empty_schema.names:
                out_table = out_table.select(empty_schema.names)
            try:
                out_table = out_table.cast(empty_schema, safe=False)
            except Exception:
                pass
        memory_guard.check_state(
            out_table.nbytes, name=f"join_output({probe_path.name})"
        )
        out_batches.extend(out_table.to_batches())

    return out_batches


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
    max_cartesian_rows: Optional[int] = None,
) -> pa.Table:
    """Grace hash join. how ∈ {'inner', 'left'} (v0.2.2).

    - how='inner': build = меньшая партиция (по file size); NULL-ключи
      отбрасываются с обеих сторон (SQL).
    - how='left':  build = right всегда (left — preserved); NULL-ключи
      отбрасываются только из build (right), probe (left) сохраняется
      с NULL в right-колонках.
    """
    if how not in ("inner", "left"):
        raise NotImplementedError(
            f"hash_join: how={how!r}; only 'inner' and 'left' supported "
            f"in v0.2.x. 'right'/'outer' — v0.3."
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

    # Peek: пустой вход -> пустой результат
    left_it = iter(left_batches)
    right_it = iter(right_batches)
    try:
        first_left = next(left_it)
    except StopIteration:
        return pa.Table.from_batches(
            [], schema=_empty_result_schema(left_schema, right_schema, on, suffixes)
        )
    left_batches = chain([first_left], left_it)
    try:
        first_right = next(right_it)
        right_batches = chain([first_right], right_it)
    except StopIteration:
        if how == "inner":
            return pa.Table.from_batches(
                [], schema=_empty_result_schema(left_schema, right_schema, on, suffixes)
            )
        # how='left': right пуст -> left preserved с null right-cols
        # Без hash_shuffle (пустой вход не принимается, §6.7).
        return _left_preserve_with_nulls(
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
        # Консервативная оценка: 0.3 * available / 256 B на строку
        max_output_rows = max(
            1, int(memory_guard.available_bytes() * 0.3 / 256)
        )
    if max_cartesian_rows is None:
        # Default: 10× expected (c_cartesian из JoinMemoryModel)
        max_cartesian_rows = max_output_rows * 10

    empty_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)
    all_batches: list[pa.RecordBatch] = []

    for p in range(num_partitions):
        lp = left_paths[p] if left_paths[p].exists() else None
        rp = right_paths[p] if right_paths[p].exists() else None
        if lp is None and rp is None:
            continue
        # Для inner: если хотя бы одна сторона пуста — пропускаем пару.
        if how == "inner":
            if lp is None or rp is None:
                continue
        else:  # left
            if lp is None:
                continue
            # rp может быть None (пустая right) — _process_partition_pair
            # обработает как build_df пустой.

        # Если rp is None для left, создадим пустой временный файл? Нет,
        # _process_partition_pair принимает Path; для None используем
        # заглушку с пустым файлом. Проще: обработать отдельно.
        if how == "left" and rp is None:
            # left preserved, right null: используем _left_preserve_with_nulls
            for b in _iter_probe_batches(lp):
                # Собираем left-батчи и добавляем null right-колонки
                # (упрощённо: вызываем _left_preserve_with_nulls на одном батче)
                pass
            # Пока оставим как есть: _process_partition_pair с rp=lp?
            # Нет. Реализуем через _left_preserve_with_nulls на одном lp.
            # Для простоты: создадим пустой right-файл с нужной схемой.
            # Это редкий случай; в v0.2.2 оставим через _left_preserve_with_nulls
            # на всех left_paths сразу (см. ниже).
            all_batches.extend(
                _left_preserve_with_nulls(
                    _iter_probe_batches(lp), left_schema, right_schema,
                    list(on), suffixes, memory_guard,
                ).to_batches()
            )
            continue

        all_batches.extend(
            _process_partition_pair(
                lp, rp, left_schema, right_schema, on, suffixes,
                memory_guard, spill_dir, how=how,
                max_output_rows=max_output_rows,
                max_cartesian_rows=max_cartesian_rows,
            )
        )

    return pa.Table.from_batches(all_batches)


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
    max_cartesian_rows: Optional[int] = None,
) -> pa.Table:
    """Inner join. Обёртка над hash_join(how='inner')."""
    return hash_join(
        left_batches, right_batches, left_schema, right_schema, on,
        memory_guard, spill_dir,
        how="inner", suffixes=suffixes, num_partitions=num_partitions,
        hash_seed=hash_seed, max_output_rows=max_output_rows,
        max_cartesian_rows=max_cartesian_rows,
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
    max_cartesian_rows: Optional[int] = None,
) -> pa.Table:
    """Left join. Обёртка над hash_join(how='left')."""
    return hash_join(
        left_batches, right_batches, left_schema, right_schema, on,
        memory_guard, spill_dir,
        how="left", suffixes=suffixes, num_partitions=num_partitions,
        hash_seed=hash_seed, max_output_rows=max_output_rows,
        max_cartesian_rows=max_cartesian_rows,
    )
