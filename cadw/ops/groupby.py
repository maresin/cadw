# cadw/ops/groupby.py
"""
GroupBy: корректный при K > памяти.

Алгоритм (DECISIONS.md §6.1, Combine-Shuffle-Reduce):
  1. HashShuffle по ключу группы → P партиций (не пересекаются по ключам).
  2. Для каждой партиции — локальная агрегация в памяти
     (pa.Table.group_by(...).aggregate(...) — векторизованно, без построчных циклов).
  3. Конкатенация результатов партиций → финальный Dataset.
     Merge не нужен: партиции не пересекаются по ключам.

Пиковая RAM:
  - Фаза 1: batch + P буферов записи (см. hash_shuffle).
  - Фаза 2: один батч партиции + словарь агрегатов на партицию
    (ограничен числом групп в партиции, а не в целом датасете).
"""
from __future__ import annotations

import gc
import itertools

import numpy as np

from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops.shuffle import hash_shuffle, estimate_num_partitions, compute_hashes, _mod_p, _boundaries

logger = get_logger("ops.groupby")


_AGG_FUNCS = {"mean", "sum", "count", "min", "max"}
# Функции, для которых работает partial aggregation (combiner):
# - mean: sum + count partial
# - sum/count/min/max: прямой partial
# std/var требуют Welford state — v0.3.5.
_COMBINER_SUPPORTED = {"sum", "count", "min", "max", "mean"}
_COMB_PREFIX = "__comb_"



_MAX_DEPTH = 5


def _combine_batches(
    batch_iter,
    by_columns: Sequence[str],
    agg_columns: Sequence[str],
    agg_funcs: Sequence[str],
    memory_guard: "MemoryGuard | None" = None,
):
    """Partial aggregation батча: pandas.groupby(by).agg(partial).

    Yields compact batches: `by + __comb_{col}_{base_func}`.
      - mean(col) → __comb_{col}_sum + __comb_{col}_count.
      - sum/count/min/max(col) → __comb_{col}_{func}.

    State: O(batch_size) per batch, O(1) между батчами.
    Корректность: partial-агрегаты ассоциативны → merge_partials = global agg.
    """
    # Уникальные (col, base_func) пары для эмита.
    base_specs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for col, f in zip(agg_columns, agg_funcs):
        if f == "mean":
            for bf in ("sum", "count"):
                if (col, bf) not in seen:
                    seen.add((col, bf))
                    base_specs.append((col, bf))
        else:
            if (col, f) not in seen:
                seen.add((col, f))
                base_specs.append((col, f))

    select_cols = list(dict.fromkeys(list(by_columns) + list(agg_columns)))

    for batch in batch_iter:
        if memory_guard is not None:
            memory_guard.check_batch(batch, name="combiner.input")
        table = pa.Table.from_batches([batch.select(select_cols)])
        grouped = table.group_by(list(by_columns)).aggregate(base_specs)
        # v0.9.9 (N3): group_by output + table = peak (checked).
        if memory_guard is not None:
            memory_guard.check_state(
                grouped.nbytes + table.nbytes,
                name="combiner.group_by",
            )

        # Переименовываем {col}_{base_func} → __comb_{col}_{base_func}.
        # group_by().aggregate() формирует имена `{col}_{func}` — добавим префикс.
        # group_by().aggregate() returns pa.Table; .column() -> ChunkedArray.
        # pa.RecordBatch.from_arrays requires Array, not ChunkedArray.
        # Collect via pa.Table, then yield its (single, combined) batch.
        orig_names = list(by_columns) + [
            f"{col}_{bf}" for col, bf in base_specs
        ]
        new_names = list(by_columns) + [
            f"{_COMB_PREFIX}{col}_{bf}" for col, bf in base_specs
        ]
        out_table = grouped.select(orig_names).rename_columns(new_names)
        for b in out_table.to_batches():
            yield b


def _merge_partials(
    table: pa.Table,
    by_columns: Sequence[str],
    output_specs: Sequence[tuple[str, str]],  # [(col, func), ...] как в _local_groupby
    memory_guard: "MemoryGuard | None" = None,  # v0.9.9 (N3)
) -> pa.Table:
    """Merge partial aggregates → final aggregated table.

    `table` содержит `by + __comb_{col}_{base_func}` (из _combine_batches).
    `output_specs` — желаемый output: [(col, func)].

    Для mean — суммируем partial sums и counts, потом делим.
    Для остальных — merge по базовой функции (sum→sum, count→sum, min→min, max→max).
    """
    # Собираем merge-спеки: __comb_{col}_{base_func} с merge_func.
    merge_specs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for col, f in output_specs:
        if f == "mean":
            for bf in ("sum", "count"):
                src = f"{_COMB_PREFIX}{col}_{bf}"
                mf = "sum"
                if (src, mf) not in seen:
                    seen.add((src, mf))
                    merge_specs.append((src, mf))
        else:
            src = f"{_COMB_PREFIX}{col}_{f}"
            mf = "sum" if f == "count" else f
            if (src, mf) not in seen:
                seen.add((src, mf))
                merge_specs.append((src, mf))

    # Проверка: все нужные колонки есть в партиции.
    for src, _mf in merge_specs:
        if src not in table.schema.names:
            raise KeyError(
                f"_merge_partials: expected column {src!r} in partition "
                f"schema {table.schema.names}. Combiner output mismatch?"
            )

    merged = table.group_by(list(by_columns)).aggregate(merge_specs)
    # v0.9.9 (N3): group_by output + table = peak (checked).
    if memory_guard is not None:
        memory_guard.check_state(
            merged.nbytes + table.nbytes,
            name="combiner.merge_partials",
        )

    # Финальный output: by + {col}_{func} с правильным dtype.
    fields: list[pa.Field] = []
    arrays: list[pa.Array] = []
    by_set = set(by_columns)
    for c in by_columns:
        fld = merged.schema.field(c)
        fields.append(fld)
        arrays.append(merged.column(c))
    for col, f in output_specs:
        if f == "mean":
            sum_src = f"{_COMB_PREFIX}{col}_sum_sum"
            count_src = f"{_COMB_PREFIX}{col}_count_sum"
            if sum_src not in merged.schema.names or count_src not in merged.schema.names:
                raise KeyError(
                    f"_merge_partials: missing {sum_src!r}/{count_src!r} "
                    f"for mean; got {merged.schema.names}"
                )
            s_arr = merged.column(sum_src).cast(pa.float64())
            c_arr = merged.column(count_src).cast(pa.float64())
            # mean = sum / count; count=0 → NaN (не должно случаться для non-empty групп).
            import pyarrow.compute as pc
            div = pc.divide(s_arr, c_arr)
            arrays.append(div)
            fields.append(pa.field(f"{col}_{f}", pa.float64()))
        else:
            src = f"{_COMB_PREFIX}{col}_{f}_sum" if f == "count" else (
                f"{_COMB_PREFIX}{col}_{f}_{'sum' if f == 'sum' else f}"
            )
            # Проще: имя после merge_partials.aggregate = {src}_{mf}.
            mf = "sum" if f == "count" else f
            final_src = f"{_COMB_PREFIX}{col}_{f}_{mf}"
            if final_src not in merged.schema.names:
                raise KeyError(
                    f"_merge_partials: missing {final_src!r}; "
                    f"got {merged.schema.names}"
                )
            arr = merged.column(final_src)
            out_type = pa.int64() if f == "count" else pa.float64()
            arrays.append(arr.cast(out_type))
            fields.append(pa.field(f"{col}_{f}", out_type))

    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _empty_result_schema(by_columns, agg_columns, agg_funcs) -> pa.Schema:
    """Схема пустого результата агрегации (для пустых партиций)."""
    fields = [pa.field(c, pa.string()) for c in by_columns]
    for col, f in zip(agg_columns, agg_funcs):
        out_type = pa.int64() if f == "count" else pa.float64()
        fields.append(pa.field(f"{col}_{f}", out_type))
    return pa.schema(fields)


_DEFAULT_KEY_ENTRY_BYTES = 80  # CALIBRATION DEFAULT (см. 11-A-4b)


def _estimate_distinct_keys(
    table: pa.Table, by_columns: Sequence[str]
) -> int:
    """Upper bound на число уникальных ключей.

    Для multi-key: min(prod(nunique), num_rows). Без построения полного
    hash-таблицы (pc.count_distinct — single int per column).
    """
    import pyarrow.compute as pc

    product = 1
    for col in by_columns:
        try:
            n = pc.count_distinct(table.column(col)).as_py()
        except Exception:
            return table.num_rows
        if not isinstance(n, int) or n <= 0:
            return table.num_rows
        product *= n
        if product >= table.num_rows:
            return table.num_rows
    return min(product, table.num_rows)


def _read_partition_safe(
    partition_path: Path,
    memory_guard: "MemoryGuard | None",
) -> "pa.Table | None":
    """Прочитать всю партицию в память, только если она влезает в бюджет.

    Returns None если partition size > available*0.5 — caller должен
    рекурсивно разбить партицию (см. `_local_groupby`).

    ВАЖНО: pattern `list(ArrowIPCBatchReader(...))` разрешён ТОЛЬКО здесь.
    Любое materialize целой партиции в другом месте — must-fix
    (materialize-before-check, §11.4). Enforced AST guard'ом
    `test_materialize_guard.py::test_no_list_of_whole_reader_outside_helper`.

    Arrow IPC без сжатия: file size ≈ in-memory nbytes. Для сжатия
    (если появится) — увеличить safety или читать по-другому.
    """
    if memory_guard is not None:
        size = partition_path.stat().st_size
        limit = int(memory_guard.available_bytes() * 0.5)
        if size > limit:
            return None

    # v0.5.4-fix (P3/P10): single-pass materialize.
    # ArrowIPC file: schema доступна из ipc_reader без чтения batches.
    # Если партиция пустая (0 batches) — Table.from_batches требует schema.
    # Регулярный случай — generator (peak = Table only, не list + Table).
    import pyarrow.ipc as _ipc
    _reader = _ipc.open_file(str(partition_path))
    _schema = _reader.schema
    _n = _reader.num_record_batches
    if _n == 0:
        return pa.Table.from_batches([], schema=_schema)

    def _iter_batches():
        for i in range(_n):
            yield _reader.get_batch(i)

    table = pa.Table.from_batches(_iter_batches(), schema=_schema)
    if memory_guard is not None:
        memory_guard.check_state(
            table.nbytes,
            name=f"read_partition_safe({partition_path.name})",
        )
    return table


def _local_groupby(
    partition_path: Path,
    by_columns: Sequence[str],
    agg_columns: Sequence[str],
    agg_funcs: Sequence[str] | str,
    memory_guard: MemoryGuard | None = None,
    spill_dir: Path | None = None,
    depth: int = 0,
    max_depth: int = _MAX_DEPTH,
    *,
    size_of_key_entry_bytes: int | None = None,
    partial_merge: bool = False,
) -> pa.Table:
    """Локальная агрегация одной партиции. Векторизованно через pa.Table.group_by."""
    if isinstance(agg_funcs, str):
        agg_funcs = [agg_funcs] * len(agg_columns)
    if len(agg_funcs) != len(agg_columns):
        raise ValueError("agg_funcs и agg_columns должны быть одной длины")
    for f in agg_funcs:
        if f not in _AGG_FUNCS:
            raise ValueError(f"Неподдерживаемая агрегация: {f!r}. Доступно: {_AGG_FUNCS}")

    if size_of_key_entry_bytes is None:
        size_of_key_entry_bytes = _DEFAULT_KEY_ENTRY_BYTES

    # 1. Оценка размера файла ДО чтения.
    partition_bytes = (
        partition_path.stat().st_size if memory_guard is not None else 0
    )
    safety_limit = (
        int(memory_guard.available_bytes() * 0.5)
        if memory_guard is not None
        else partition_bytes
    )

    table: pa.Table | None = None
    estimated_dict_bytes: int = 0
    dict_fits: bool = True

    # 2. Файл влезает -> materialize -> dict check -> group_by
    if partition_bytes <= safety_limit:
        table = _read_partition_safe(partition_path, memory_guard)
        if table is None or table.num_rows == 0:
            return pa.Table.from_pylist(
                [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
            )

        # must-fix §11.4: dict check до group_by.
        # _read_partition_safe проверяет file_size, но group_by строит
        # hash-таблицу K_i * size_of_key_entry, которая для узких строк
        # больше файла -> OOM внутри pyarrow.
        if memory_guard is not None:
            estimated_k = _estimate_distinct_keys(table, by_columns)
            estimated_dict_bytes = estimated_k * size_of_key_entry_bytes
            dict_limit = int(memory_guard.available_bytes() * 0.5)
            dict_fits = estimated_dict_bytes <= dict_limit

        if dict_fits:
            # v0.5.4 (P10): gc before group_by — освободить reader/writer
            # buffers (Table уже materialized; old batches не нужны).
            gc.collect()
            if partial_merge:
                return _merge_partials(
                    table, by_columns,
                    [(col, f) for col, f in zip(agg_columns, agg_funcs)],
                    memory_guard=memory_guard,
                )
            agg_specs = [(col, f) for col, f in zip(agg_columns, agg_funcs)]
            return table.group_by(list(by_columns)).aggregate(agg_specs)
        # fall through: file fits, dict overflow -> split

    # 3. Файл или dict не влезает -> рекурсивное партиционирование
    if memory_guard is None or spill_dir is None:
        raise MemoryLimitExceeded(
            f"_local_groupby: file={partition_bytes} B, "
            f"dict~{estimated_dict_bytes} B, safety={safety_limit} B; "
            f"memory_guard/spill_dir not provided."
        )
    if depth >= max_depth:
        raise MemoryLimitExceeded(
            f"_local_groupby: depth={depth} >= max_depth={max_depth}; "
            f"file={partition_bytes} B, dict~{estimated_dict_bytes} B, "
            f"safety={safety_limit} B. Likely skew (DECISIONS §11.4)."
        )

    # v0.9.3e (C1, S2): prior (shadowed) `_read_for_resplit`
    # def removed. It was intended optimization: reuse in-memory
    # `table` when partition file fits but dict overflows,
    # avoiding disk re-read. Deferred to backlog (11.6) -- never
    # activated (second identical-name def shadowed it) and thus
    # never tested. See 13.4/177.

    def _read_for_resplit():
        for b in ArrowIPCBatchReader(str(partition_path), None):
            cols = [c for c in list(by_columns) + list(agg_columns) if c in b.schema.names]
            yield b.select(cols)

    sub_spill = spill_dir / f"resplit_d{depth}_{partition_path.stem}"
    sub_spill.mkdir(parents=True, exist_ok=True)

    sub_value_cols = [c for c in agg_columns if c not in by_columns]
    sub_paths = hash_shuffle(
        _read_for_resplit(),
        key_columns=list(by_columns),
        value_columns=sub_value_cols,
        num_partitions=4,
        memory_guard=memory_guard,
        spill_dir=sub_spill,
        hash_seed=depth + 1,
        log=False,
    )

    results: list[pa.Table] = []
    for sp in sub_paths:
        r = _local_groupby(
            sp, by_columns, agg_columns, agg_funcs,
            memory_guard=memory_guard, spill_dir=spill_dir,
            depth=depth + 1, max_depth=max_depth,
        )
        if r.num_rows > 0:
            results.append(r)
        try:
            sp.unlink()
        except OSError:
            pass

    if not results:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )
    return pa.concat_tables(results)



def _estimate_k_sample(
    sample_batches: list[pa.RecordBatch],
    by_columns: Sequence[str],
) -> int:
    """Unique-key count in sample. Arrow-native (v0.9.6c1).

    Composition of per-batch `dictionary_encode` (single-col) or
    binary_join + dictionary_encode (multi-col); union across batches
    via Python set on the *dictionary values* (small — bounded by
    sample_rows, typically K_sample).
    """
    from cadw.core._k_est import estimate_k_sample_arrow

    if not sample_batches:
        return 0
    cols = list(by_columns)
    if len(cols) == 1:
        # Single column: union of dictionaries across batches.
        seen: set = set()
        for b in sample_batches:
            arr = b.column(cols[0])
            enc = arr.dictionary_encode()
            if enc.dictionary is not None:
                for v in enc.dictionary.to_pylist():
                    if v is not None:
                        seen.add(v)
            else:
                for v in arr.to_pylist():
                    if v is not None:
                        seen.add(v)
        return max(1, len(seen))
    # Multi-column: union of binary-joined string keys.
    import pyarrow.compute as pc
    seen2: set = set()
    for b in sample_batches:
        str_cols = [
            b.column(c) if pa.types.is_string(b.column(c).type)
            else pc.cast(b.column(c), pa.string())
            for c in cols
        ]
        joined = str_cols[0]
        for c in str_cols[1:]:
            joined = pc.binary_join_element_wise(joined, c, "\x01")
        for v in joined.to_pylist():
            if v is not None:
                seen2.add(v)
    return max(1, len(seen2))



_HEAVY_RATIO = 0.05
# Skew isolation is worth the overhead only when the key is BOTH
# large in absolute terms and dominant in ratio. Without absolute
# min, tiny test data (5 rows, 3 keys, 40/40/20%) incorrectly
# triggers skew path — combiner never activates. §13.4/52b.
_HEAVY_ABSOLUTE_MIN = 100


def _detect_heavy_keys(sample_batches, by_columns, threshold_ratio: float = _HEAVY_RATIO) -> set:
    """Тяжёлые ключи по sample: доля в sample > threshold_ratio.

    v0.9.12 (F12-4, PF1): Arrow-native. Prior: per-row Python loop
    with .as_py() per column per row — 10k iterations per groupby
    call (~10-50 ms). Now: pa.concat_tables + group_by().aggregate(
    [(first_key, "count")]) — ~10x faster, single C++ pass.

    Threshold semantics preserved: heavy iff count >= 100 AND
    count > total * threshold_ratio. Integer arithmetic:
    int(total*ratio)+1 == strict `cnt > total*ratio` for all
    integer counts (verified: t*0.05 truncates, +1 restores).
    """
    if not sample_batches:
        return set()

    import pyarrow.compute as pc

    by_list = list(by_columns)
    tables = [
        pa.Table.from_batches([b.select(by_list)])
        for b in sample_batches
    ]
    combined = pa.concat_tables(tables, promote_options="default")
    if combined.num_rows == 0:
        return set()

    first_key = by_list[0]
    counts = combined.group_by(by_list).aggregate(
        [(first_key, "count")]
    )
    count_col = f"{first_key}_count"
    total = int(pc.sum(counts.column(count_col)).as_py() or 0)
    if total == 0:
        return set()

    threshold_abs = max(
        _HEAVY_ABSOLUTE_MIN,
        int(total * threshold_ratio) + 1,
    )
    mask = pc.greater_equal(
        counts.column(count_col),
        pa.scalar(threshold_abs, pa.int64()),
    )
    filtered = counts.filter(mask)
    return {
        tuple(row[c] for c in by_list)
        for row in filtered.select(by_list).to_pylist()
    }


def _compute_heavy_mask(batch: pa.RecordBatch, by_columns, heavy_keys: set) -> pa.Array:
    """Boolean mask: True where row's key is in heavy_keys.

    v0.9.12-fix2 (§13.4/218): fill_null(False). Prior code
    returned mask with null values on rows where any by_col
    was null: pc.equal(null, non_null) = null (Kleene logic),
    and_(True, null) = null, or_(False, null) = null. Then
    batch.filter(null) treats null as false → null-key rows
    dropped from heavy_batch AND from light_batch
    (invert(null) = null). Silent data loss: any groupby
    with nulls triggering skew path lost all null-key rows
    (v0.3.4 to v0.9.12-fix1, ~3 months, no test).

    Fix: fill_null(False) — null-key rows explicitly → light
    path (invert(False)=True), shuffled by _NULL_SENTINEL,
    grouped by pyarrow, preserved in output. Non-null data
    unaffected (mask has no nulls when no nulls in input).
    """
    n = batch.num_rows
    if not heavy_keys or n == 0:
        return pa.array([False] * n, type=pa.bool_())
    mask = pa.array([False] * n, type=pa.bool_())
    for key in heavy_keys:
        k_mask = pa.array([True] * n, type=pa.bool_())
        for col, v in zip(by_columns, key):
            k_mask = pc.and_(k_mask, pc.equal(batch.column(col), pa.scalar(v)))
        mask = pc.or_(mask, k_mask)
    # v0.9.12-fix2 (§13.4/218): Kleene null → False, not dropped.
    return pc.fill_null(mask, pa.scalar(False))


def _accumulate_heavy(accum: dict, batch: pa.RecordBatch, by_columns, agg_columns, agg_funcs) -> None:
    # Base-specs без дублей: (col, base_func).
    # mean -> (sum, count); остальное — как есть.
    needed: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for col, f in zip(agg_columns, agg_funcs):
        if f == "mean":
            for bf in ("sum", "count"):
                if (col, bf) not in seen:
                    seen.add((col, bf))
                    needed.append((col, bf))
        else:
            if (col, f) not in seen:
                seen.add((col, f))
                needed.append((col, f))

    select_cols = list(dict.fromkeys(list(by_columns) + list(agg_columns)))
    table = pa.Table.from_batches([batch.select(select_cols)])
    grouped = table.group_by(list(by_columns)).aggregate(needed)

    for row in grouped.to_pylist():
        key = tuple(row[c] for c in by_columns)
        if key not in accum:
            accum[key] = {}
        for col, f in zip(agg_columns, agg_funcs):
            if f == "mean":
                # Внутренние ключи — не пересекаются с явным sum/count.
                sk, ck = f"__mean_sum_{col}", f"__mean_count_{col}"
                accum[key][sk] = accum[key].get(sk, 0.0) + row[f"{col}_sum"]
                accum[key][ck] = accum[key].get(ck, 0) + row[f"{col}_count"]
            else:
                out = f"{col}_{f}"
                new = row[out]
                old = accum[key].get(out)
                if old is None:
                    accum[key][out] = new
                elif f in ("sum", "count"):
                    accum[key][out] = old + new
                elif f == "min":
                    accum[key][out] = min(old, new)
                elif f == "max":
                    accum[key][out] = max(old, new)


def _accum_to_table(accum: dict, by_columns, agg_columns, agg_funcs) -> pa.Table:
    rows = []
    for key, d in accum.items():
        row = dict(zip(by_columns, key))
        for col, f in zip(agg_columns, agg_funcs):
            if f == "mean":
                s = d.get(f"__mean_sum_{col}", 0.0)
                c = d.get(f"__mean_count_{col}", 0)
                row[f"{col}_mean"] = s / c if c > 0 else float("nan")
            else:
                row[f"{col}_{f}"] = d[f"{col}_{f}"]
        rows.append(row)
    if not rows:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )
    return pa.Table.from_pylist(
        rows, schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
    )


def _shuffle_write(writers, batch: pa.RecordBatch, key_columns, value_columns,
                   num_partitions: int, hash_seed: int, row_id_offset: int) -> None:
    hashes = compute_hashes(batch, key_columns, hash_seed)
    parts = _mod_p(hashes, num_partitions)
    order = pc.sort_indices(parts)
    # Dedup key ∩ value (§13.4/54): batch.select() на дубликатах имён
    # создаёт "cat, cat" → group_by("cat") падает с "Multiple matches".
    _seen: set[str] = set()
    _cols: list[str] = []
    for _c in list(key_columns) + list(value_columns):
        if _c not in _seen:
            _seen.add(_c)
            _cols.append(_c)
    sorted_batch = batch.select(_cols).take(order)
    sorted_parts = parts.take(order)
    parts_np = sorted_parts.to_numpy(zero_copy_only=False)
    order_np = order.to_numpy(zero_copy_only=False)
    if parts_np.size == 0:
        return
    bounds = _boundaries(parts_np)
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        p = int(parts_np[start])
        piece = sorted_batch.slice(start, end - start)
        row_ids = pa.array(
            (order_np[start:end].astype(np.int64) + row_id_offset).astype(np.uint64),
            type=pa.uint64(),
        )
        writers[p].write_batch(piece.append_column("row_id", row_ids))


def _groupby_with_skew(batch_iter, by_columns, agg_columns, agg_funcs, heavy_keys,
                       num_partitions, memory_guard, spill_dir, log) -> pa.Table:
    all_columns: list[str] = []
    seen: set[str] = set()
    for c in list(by_columns) + list(agg_columns):
        if c not in seen:
            seen.add(c)
            all_columns.append(c)

    it = iter(batch_iter)
    try:
        first_batch = next(it)
    except StopIteration:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )
    memory_guard.check_batch(first_batch, name="groupby_with_skew.first", extra_factor=3.0)

    fields = [first_batch.schema.field(c) for c in all_columns]
    fields.append(pa.field("row_id", pa.uint64()))
    schema = pa.schema(fields)
    shuf_dir = spill_dir / "shuffle"
    shuf_dir.mkdir(parents=True, exist_ok=True)
    paths = [shuf_dir / f"partition_{p:05d}.arrow" for p in range(num_partitions)]
    # v0.8.9 (P-G3): mirror hash_shuffle (shuffle.py:273) — check P writer
    # buffer cost BEFORE opening writers. Prior code opened P writers
    # without this check; the P × writer_buffer allocation was unbounded.
    _buf_bytes = memory_guard.recommended_partition_buffer_bytes(num_partitions)
    memory_guard.check_partition_buffers(
        num_partitions, min(_buf_bytes, 64 * 1024)
    )
    # v0.9.8a (R1): real per-writer check.
    memory_guard.check_partition_writers(num_partitions)
    writers = [ArrowIPCBatchWriter(str(p), schema) for p in paths]

    accum: dict = {}
    row_id_offset = 0
    try:
        for batch in itertools.chain([first_batch], it):
            memory_guard.check_batch(
                batch, name="groupby_with_skew.batch", extra_factor=3.0
            )
            heavy_mask = _compute_heavy_mask(batch, by_columns, heavy_keys)
            light_mask = pc.invert(heavy_mask)
            heavy_batch = batch.filter(heavy_mask)
            light_batch = batch.filter(light_mask)
            if heavy_batch.num_rows > 0:
                _accumulate_heavy(accum, heavy_batch, by_columns, agg_columns, agg_funcs)
            if light_batch.num_rows > 0:
                _shuffle_write(
                    writers, light_batch, by_columns, agg_columns,
                    num_partitions, hash_seed=0, row_id_offset=row_id_offset,
                )
            row_id_offset += batch.num_rows
    finally:
        for w in writers:
            w.close()

    light_results: list[pa.Table] = []
    for p in paths:
        if not p.exists() or p.stat().st_size == 0:
            continue
        t = _local_groupby(
            p, by_columns, agg_columns, agg_funcs,
            memory_guard=memory_guard, spill_dir=spill_dir,
        )
        if t.num_rows > 0:
            light_results.append(t)
        try:
            p.unlink()
        except OSError:
            pass

    heavy_result = _accum_to_table(accum, by_columns, agg_columns, agg_funcs)
    all_results = light_results + ([heavy_result] if heavy_result.num_rows > 0 else [])
    if not all_results:
        return pa.Table.from_pylist(
            [], schema=_empty_result_schema(by_columns, agg_columns, agg_funcs)
        )
    if log:
        logger.info(
            "groupby_with_skew: heavy=%d, light_partitions=%d",
            len(accum), len(light_results),
        )
    return pa.concat_tables(all_results)


def groupby_aggregate(
    batch_iter: Iterable[pa.RecordBatch],
    by_columns: Sequence[str] | str,
    agg_columns: Sequence[str] | str,
    agg_funcs: Sequence[str] | str,
    memory_guard: MemoryGuard,
    spill_dir: str | Path,
    *,
    num_partitions: int | None = None,
    hash_seed: int = 0,
    combiner: bool = False,
    log: bool = True,
) -> pa.Table:
    """
    Out-of-core groupby: HashShuffle + LocalGroupBy + Concat.

    Возвращает pa.Table с колонками:
        by_columns + [f"{col}_{func}" для каждой (col, func)].

    Корректен при K > памяти: число групп в каждой партиции ограничено
    только числом групп с данным хэшем, а не общим K.
    """
    if isinstance(by_columns, str):
        by_columns = [by_columns]
    if isinstance(agg_columns, str):
        agg_columns = [agg_columns]
    if isinstance(agg_funcs, str):
        agg_funcs = [agg_funcs] * len(agg_columns)

    by_columns = list(by_columns)
    agg_columns = list(agg_columns)
    agg_funcs = list(agg_funcs)

    if not by_columns:
        raise ValueError("by_columns must be non-empty")
    if len(agg_columns) != len(agg_funcs):
        raise ValueError("agg_columns и agg_funcs должны быть одной длины")
    for f in agg_funcs:
        if f not in _AGG_FUNCS:
            raise ValueError(f"Неподдерживаемая агрегация: {f!r}. Доступно: {_AGG_FUNCS}")

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    # --- Фаза 1: HashShuffle ---
    # Value columns для партиций: by + agg (после дедупликации)
    # HashShuffle пишет key_columns + value_columns. by_columns уже идут как key,
    # поэтому value_columns — только agg-колонки, которых нет в by (без дублей).
    seen: set[str] = set(by_columns)
    value_columns: list[str] = []
    for c in agg_columns:
        if c not in seen:
            seen.add(c)
            value_columns.append(c)
    if not value_columns:
        # Все agg-колонки совпадают с by — тогда в партицию пойдёт только by.
        # Такой groupby бессмысленен, но не должен падать.
        value_columns = list(by_columns)

    # --- Combiner (partial aggregation before shuffle) ---
    combiner_active = False
    if combiner:
        unsupported = [f for f in agg_funcs if f not in _COMBINER_SUPPORTED]
        if unsupported:
            import warnings
            warnings.warn(
                f"groupby_aggregate: combiner не поддерживает "
                f"agg_funcs={unsupported}; combiner отключён (fallback "
                f"на обычный путь). §13.4/52.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            combiner_active = True
            # После combiner value_columns заменяются на __comb_*.
            # Список будет пересчитан: unique (col, base_func) пары.
            comb_value_cols: list[str] = []
            seen_bc: set[str] = set()
            for col, f in zip(agg_columns, agg_funcs):
                bfs = ("sum", "count") if f == "mean" else (f,)
                for bf in bfs:
                    name = f"{_COMB_PREFIX}{col}_{bf}"
                    if name not in seen_bc:
                        seen_bc.add(name)
                        comb_value_cols.append(name)
            if log:
                logger.info(
                    "groupby_aggregate: combiner=on, %d value cols (was %d)",
                    len(comb_value_cols), len(value_columns),
                )


    # Sample строится ВСЕГДА: он нужен и для auto-P, и для skew detection.
    # Важно: break-and-chain, не tee. С tee chain(sample, rest) дублирует
    # sample (count/sum удваиваются, mean маскирует).
    # Если combiner_active — оборачиваем входной поток в partial-agg.
    # Это меняет схему: by + __comb_*.
    if combiner_active:
        # Sample (skew detection) идёт по СЫРЫМ батчам — но нам нужны
        # unique keys by_columns, которые не меняются от combiner.
        # Однако sample сохраняется как список сырых батчей; combiner
        # применится только к hash_shuffle. Порядок: sample → skew →
        # combine (если no skew) → shuffle.
        pass  # no-op; комбинирование применим после skew check

    sample_cap = 10_000
    # v0.8.9 (P-G5): cap by bytes too — wide schemas (>=10 KB/row) hit
    # 0.5×avail before 10k rows. Bound the sample to 0.3×avail.
    sample_cap_bytes = max(1, int(memory_guard.available_bytes() * 0.3))
    batch_iter = iter(batch_iter)
    sample_batches: list[pa.RecordBatch] = []
    n_sample = 0
    sample_bytes = 0
    for b in batch_iter:
        sample_batches.append(b)
        n_sample += b.num_rows
        sample_bytes += b.nbytes
        if n_sample >= sample_cap or sample_bytes >= sample_cap_bytes:
            break
    sample_limit = int(memory_guard.available_bytes() * 0.5)
    skip_skew = False
    if sample_bytes > sample_limit:
        if num_partitions is None:
            raise MemoryLimitExceeded(
                f"groupby_aggregate: sample ({sample_bytes} B) > "
                f"available*0.5 ({sample_limit} B) и num_partitions не задан. "
                f"Передайте num_partitions явно."
            )
        # P задан явно: sample не влезает, skew detection пропускаем,
        # но прочитанное не выбрасываем (batch_iter уже исчерпан sample-циклом).
        skip_skew = True
        if log:
            logger.warning(
                "groupby_aggregate: sample (%d B) > available*0.5 (%d B); "
                "skew detection skipped (num_partitions=%d задан явно)",
                sample_bytes, sample_limit, num_partitions,
            )
    batch_iter = itertools.chain(sample_batches, batch_iter)

    if num_partitions is None:
        k_est = _estimate_k_sample(sample_batches, by_columns)
        p_est = estimate_num_partitions(
            k_est,
            available_bytes=max(1, memory_guard.available_bytes() // 2),
        )
        num_partitions = max(8, p_est)
        if log:
            logger.info(
                "groupby_aggregate: auto num_partitions=%d (k_est=%d, sample=%d rows)",
                num_partitions, k_est, n_sample,
            )

    # --- Skew detection & isolation (§11.4 п.3) ---
    heavy_keys = set() if skip_skew else _detect_heavy_keys(sample_batches, by_columns)
    if heavy_keys:
        if combiner_active:
            import warnings
            warnings.warn(
                "groupby_aggregate: combiner отключён в skew-пути "
                "(heavy keys streaming + light без combiner). "
                "v0.3.5: combiner для light. §13.4/52.",
                RuntimeWarning,
                stacklevel=2,
            )
            combiner_active = False
            shuffled_cols = value_columns
            batches_for_shuffle = batch_iter
        if log:
            logger.info(
                "groupby_aggregate: skew detected, %d heavy key(s) -> streaming",
                len(heavy_keys),
            )
        return _groupby_with_skew(
            batch_iter, by_columns, agg_columns, agg_funcs, heavy_keys,
            num_partitions, memory_guard, spill_dir, log,
        )

    if combiner_active:
        # Wrap in partial-agg generator (state O(batch_size)).
        shuffled_cols = comb_value_cols
        batches_for_shuffle = _combine_batches(
            batch_iter, by_columns, agg_columns, agg_funcs,
            memory_guard=memory_guard,
        )
    else:
        shuffled_cols = value_columns
        batches_for_shuffle = batch_iter

    partitions = hash_shuffle(
        batches_for_shuffle,
        key_columns=by_columns,
        value_columns=shuffled_cols,
        num_partitions=num_partitions,
        memory_guard=memory_guard,
        spill_dir=spill_dir / "shuffle",
        hash_seed=hash_seed,
        log=log,
    )
    if log:
        logger.info("groupby_aggregate: %d partitions", len(partitions))

    # --- Фаза 2: LocalGroupBy по каждой партиции ---
    partial_tables: list[pa.Table] = []
    for i, p in enumerate(partitions):
        if combiner_active:
            # Частичные агрегаты → merge, а не group_by.
            t = _local_groupby(
                p, by_columns, agg_columns, agg_funcs,
                memory_guard=memory_guard,
                spill_dir=spill_dir,
                partial_merge=True,
            )
        else:
            t = _local_groupby(
                p, by_columns, agg_columns, agg_funcs,
                memory_guard=memory_guard,
                spill_dir=spill_dir,
            )
        # Пустая таблица без данных не мешает конкатенации
        if t.num_rows > 0:
            partial_tables.append(t)
        if log and t.num_rows > 0:
            memory_guard.log_step(
                "groupby_aggregate.partition",
                batch_id=i,
                n_rows=t.num_rows,
                batch_bytes=t.nbytes,
            )

    # --- Фаза 3: Конкатенация (merge не нужен — партиции не пересекаются по ключам) ---
    if not partial_tables:
        # Пустая схема результата
        fields = [pa.field(c, pa.string()) for c in by_columns]
        for col, f in zip(agg_columns, agg_funcs):
            out_type = pa.int64() if f == "count" else pa.float64()
            fields.append(pa.field(f"{col}_{f}", out_type))
        return pa.Table.from_pylist([], schema=pa.schema(fields))

    result = pa.concat_tables(partial_tables, promote_options="default")
    if log:
        logger.info("groupby_aggregate done: %d rows", result.num_rows)
    return result


def write_groupby_result(
    result_table: pa.Table,
    out_path: str | Path,
) -> Path:
    """Сохранить результат в Arrow IPC (для использования как Dataset)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ArrowIPCBatchWriter(str(out_path), result_table.schema)
    try:
        for batch in result_table.to_batches():
            writer.write_batch(batch)
    finally:
        writer.close()
    return out_path