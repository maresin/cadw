"""
Rank: post-processor of pre-sorted stream (v0.8.8 note).

`compute_ranks_to_disk(sorted_path: Path, ...)` accepts an ALREADY-sorted
IPC path; caller (`CadwDataFrame.rank` in dataframe.py) does
`external_sort`. Architectural pattern "post-processor": this module
does not import primitives. See §13.4/137.

Алгоритм (caller-side external_sort показан для контекста):
  1. external_sort по `by` (ascending/descending).
  2. Вычисление rank — ОДИН или ДВА прохода (зависит от method):
     - Streamable (`first`/`min`/`dense`): O(1) state, один проход.
     - Bufferable (`max`/`last`/`average`): требуют знать конец
       tie-группы. Двухпроходное решение O(1) RAM:
         Pass 1: scan sorted_by → emit `(first_rank, group_count)`
                 per tie-group на диск. State O(1).
         Pass 2: scan sorted_by + advance по group-файлу (sequential).
                 pos_in_group = global_pos - first_rank + 1.
                 last_rank = first_rank + group_count - 1.
                 max     → last_rank.
                 last    → first_rank + group_count - pos_in_group.
                 average → (first_rank + last_rank) / 2.
  3. external_sort ranks by row_id → original order.
  4. Zip с исходным DataFrame.

Nulls: участвуют как значения (null == null); pandas `skipna` не
поддерживается (v0.3.2).

Ranking formula:
  - `first`:   1-based позиция в sorted order (ties по row_id asc).
  - `min`:     минимум из рангов в tie-группе.
  - `dense`:   1, 2, 3, ... для уникальных значений.
  - `max`:     максимум из рангов в tie-группе.
  - `last`:    обратный min (max = +1, last = -1) по позициям в группе.
  - `average`: (min + max) / 2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter

logger = get_logger("ops.rank")

VALID_STREAMABLE = ("first", "min", "dense")
VALID_BUFFERABLE = ("max", "last", "average")
VALID_METHODS = VALID_STREAMABLE + VALID_BUFFERABLE


def _rank_dtype(method: str, pct: bool) -> pa.DataType:
    """int64 для целочисленных методов без pct; float64 для average или pct."""
    if pct or method == "average":
        return pa.float64()
    return pa.int64()


def _compute_streamable(
    sorted_batches: Iterable[pa.RecordBatch],
    *,
    by: str,
    output_col: str,
    method: str,
    pct: bool,
    total_n: int,
    rank_path: Path,
    memory_guard: MemoryGuard,
) -> Path:
    """One-pass: first/min/dense. O(1) state."""
    dtype = _rank_dtype(method, pct)
    schema = pa.schema([("row_id", pa.uint64()), (output_col, dtype)])
    writer = ArrowIPCBatchWriter(str(rank_path), schema)

    _UNSET = object()
    prev_val = _UNSET
    group_start = 0
    global_pos = 0
    dense_counter = 0

    import pyarrow.compute as _pc

    try:
        for batch in sorted_batches:
            # v0.9.24: extra_factor=7.0 — 3.5 × K pylist columns,
            # K=2 (by + row_id). Null fallback does to_pylist on both;
            # null-free numpy fast path needs no pylist. AUDIT_FULL §3.2.
            memory_guard.check_batch(
                batch, name="rank.streamable.input", extra_factor=7.0,
            )
            vals_arr = batch.column(by)
            row_ids_arr = batch.column("row_id")
            n = len(vals_arr)

            # v0.5.2b (P11): 'first' fast path — Arrow-native, no to_pylist.
            # Input sorted (external_sort); rank(i) = i+1 + global_pos —
            # ordinal position. np.arange — C-level, no per-row Python.
            if method == "first":
                # 'first' = ordinal position in sorted order. Input is
                # already sorted (external_sort), so rank(i) = i+1 within
                # batch, offset by global_pos. np.arange — Arrow-native,
                # no Python loop (P11).
                import numpy as _np
                local_rank = pa.array(
                    _np.arange(1, n + 1, dtype=_np.int64),
                    type=pa.int64(),
                )
                global_rank = _pc.add(
                    local_rank, pa.scalar(global_pos, type=pa.int64()),
                )
                global_pos += n
                if pct:
                    global_rank = _pc.divide(
                        _pc.cast(global_rank, pa.float64()),
                        pa.scalar(float(total_n), type=pa.float64()),
                    )
                global_rank = _pc.cast(global_rank, dtype)

                rb = pa.RecordBatch.from_arrays(
                    [row_ids_arr, global_rank], schema=schema,
                )
                writer.write_batch(rb)
                continue

            # v0.9.18 (AUDIT §1.2): numpy fast path for null-free batches.
            # Prior: to_pylist() + Python loop = O(n*28B) peak, 35.8ms/500k.
            # Now: numpy diff + cumsum/cummax, peak <8 MB, 1.4ms/500k (25.3x).
            # Null-containing batches fall back to Python loop (correct
            # null==null semantics preserved; §13.4/189).
            import numpy as _np

            if vals_arr.null_count == 0:
                vnp = vals_arr.to_numpy(zero_copy_only=False)
                n_b = len(vnp)
                if n_b == 0:
                    continue
                # is_new[k] = True iff new group starts at position k
                if prev_val is _UNSET:
                    is_new = _np.concatenate(
                        ([True], vnp[1:] != vnp[:-1])
                    )
                else:
                    is_new = _np.concatenate(
                        ([vnp[0] != prev_val], vnp[1:] != vnp[:-1])
                    )

                if method == "dense":
                    if dense_counter == 0:
                        # First batch, first group → counter becomes 1
                        dense_counter = 1
                    dense_ranks = dense_counter + _np.cumsum(
                        is_new.astype(_np.int64)
                    ) - 1
                    # If is_new[0]=True: dense_rank[0] = 1 (correct)
                    # If is_new[0]=False: dense_rank[0] = dense_counter (correct)
                    dense_counter = int(dense_ranks[-1])
                    ranks_np = dense_ranks
                else:  # min
                    positions_1based = _np.arange(
                        global_pos + 1, global_pos + n_b + 1,
                        dtype=_np.int64,
                    )
                    masked = _np.where(is_new, positions_1based, 0)
                    if not is_new[0]:
                        first_new = (
                            int(_np.argmax(is_new))
                            if is_new.any() else n_b
                        )
                        masked[:first_new] = group_start
                    min_ranks = _np.maximum.accumulate(masked)
                    ranks_np = min_ranks
                    group_start = int(min_ranks[-1])

                global_pos += n_b
                prev_val = vnp[-1].item() if hasattr(vnp[-1], "item") else vnp[-1]

                if pct:
                    ranks_arr = pa.array(
                        ranks_np.astype(_np.float64) / float(total_n),
                        type=dtype,
                    )
                else:
                    ranks_arr = pa.array(ranks_np, type=dtype)

                rb = pa.RecordBatch.from_arrays(
                    [row_ids_arr, ranks_arr], schema=schema,
                )
                writer.write_batch(rb)
                continue

            # Fallback: nulls present — Python loop (preserves null==null
            # as same group; §13.4/189 asymmetry intentional).
            vals = vals_arr.to_pylist()
            row_ids = row_ids_arr.to_pylist()

            ranks: list = []
            for v in vals:
                global_pos += 1
                if prev_val is _UNSET:
                    group_start = global_pos
                    dense_counter = 1
                elif v != prev_val:
                    group_start = global_pos
                    dense_counter += 1

                if method == "min":
                    r = group_start
                else:  # dense
                    r = dense_counter

                ranks.append((r / total_n) if pct else r)
                prev_val = v

            rb = pa.RecordBatch.from_arrays(
                [
                    pa.array(row_ids, type=pa.uint64()),
                    pa.array(ranks, type=dtype),
                ],
                schema=schema,
            )
            writer.write_batch(rb)
    finally:
        writer.close()
    return rank_path


def _pass1_groups_to_disk(
    sorted_batches: Iterable[pa.RecordBatch],
    *,
    by: str,
    groups_path: Path,
    memory_guard: MemoryGuard,
) -> Path:
    """Pass 1 of bufferable: scan sorted_by → (first_rank, group_count) file.

    State: prev_val, group_start, global_pos, plus per-batch accumulators
    `starts`/`counts` (bounded by batch size). O(batch_size) RAM.
    """
    schema = pa.schema([
        ("first_rank", pa.int64()),
        ("group_count", pa.int64()),
    ])
    writer = ArrowIPCBatchWriter(str(groups_path), schema)

    _UNSET = object()
    prev_val = _UNSET
    group_start = 0
    global_pos = 0
    starts: list[int] = []
    counts: list[int] = []

    def _flush() -> None:
        nonlocal starts, counts
        if not starts:
            return
        rb = pa.RecordBatch.from_arrays(
            [
                pa.array(starts, type=pa.int64()),
                pa.array(counts, type=pa.int64()),
            ],
            schema=schema,
        )
        writer.write_batch(rb)
        starts = []
        counts = []

    try:
        for batch in sorted_batches:
            # v0.9.24: extra_factor=3.5 — pass1 does to_pylist(by),
            # K=1 pylist column ⇒ 3.5 × batch.nbytes (AUDIT_FULL §3.2).
            memory_guard.check_batch(
                batch, name="rank.pass1.input", extra_factor=3.5,
            )
            vals = batch.column(by).to_pylist()
            for v in vals:
                global_pos += 1
                if prev_val is _UNSET:
                    group_start = global_pos
                elif v != prev_val:
                    starts.append(group_start)
                    counts.append(global_pos - group_start)
                    group_start = global_pos
                prev_val = v
            _flush()

        # Финальная группа (не эмитили её — она «продолжается» до EOF).
        if prev_val is not _UNSET:
            starts.append(group_start)
            counts.append(global_pos - group_start + 1)
            _flush()
    finally:
        writer.close()

    return groups_path


def _iter_groups(path: Path) -> Iterator[tuple[int, int]]:
    """Итерировать (first_rank, group_count) из group-файла."""
    for b in ArrowIPCBatchReader(str(path), ["first_rank", "group_count"]):
        firsts = b.column("first_rank").to_pylist()
        counts = b.column("group_count").to_pylist()
        for f, c in zip(firsts, counts):
            yield (int(f), int(c))


def _pass2_ranks_to_disk(
    sorted_batches: Iterable[pa.RecordBatch],
    *,
    by: str,
    groups_path: Path,
    output_col: str,
    method: str,
    pct: bool,
    total_n: int,
    rank_path: Path,
    memory_guard: MemoryGuard,
) -> Path:
    """Pass 2 of bufferable: scan sorted_by + advance по group file.

    State: prev_val, global_pos, cur_first, cur_count, cur_last. O(1) RAM.
    """
    dtype = _rank_dtype(method, pct)
    schema = pa.schema([("row_id", pa.uint64()), (output_col, dtype)])
    writer = ArrowIPCBatchWriter(str(rank_path), schema)

    groups_iter = _iter_groups(groups_path)
    _UNSET = object()
    prev_val = _UNSET
    global_pos = 0
    cur_first = 0
    cur_count = 0
    cur_last = 0

    def _advance() -> None:
        nonlocal cur_first, cur_count, cur_last
        try:
            f, c = next(groups_iter)
        except StopIteration as exc:
            raise RuntimeError(
                "rank.pass2: group iterator exhausted prematurely — "
                "inconsistent sorted_by / groups file"
            ) from exc
        cur_first = f
        cur_count = c
        cur_last = cur_first + cur_count - 1

    try:
        for batch in sorted_batches:
            # v0.9.24: extra_factor=7.0 — pass2 does to_pylist(by) +
            # to_pylist(row_id); K=2 pylist columns ⇒ 3.5 × 2.
            memory_guard.check_batch(
                batch, name="rank.pass2.input", extra_factor=7.0,
            )
            vals = batch.column(by).to_pylist()
            row_ids = batch.column("row_id").to_pylist()

            ranks: list = []
            for v, _rid in zip(vals, row_ids):
                global_pos += 1
                if prev_val is _UNSET or v != prev_val:
                    _advance()

                pos_in_group = global_pos - cur_first + 1
                if method == "max":
                    r = cur_last
                elif method == "last":
                    r = cur_first + cur_count - pos_in_group
                elif method == "average":
                    r = (cur_first + cur_last) / 2.0
                else:
                    raise NotImplementedError(f"rank.pass2: method={method!r}")

                ranks.append((r / total_n) if pct else r)
                prev_val = v

            rb = pa.RecordBatch.from_arrays(
                [
                    pa.array(row_ids, type=pa.uint64()),
                    pa.array(ranks, type=dtype),
                ],
                schema=schema,
            )
            writer.write_batch(rb)
    finally:
        writer.close()

    return rank_path


def compute_ranks_to_disk(
    sorted_path: Path,
    *,
    by: str,
    output_col: str,
    method: str,
    pct: bool,
    total_n: int,
    rank_path: Path,
    memory_guard: MemoryGuard,
    log: bool = False,
) -> Path:
    """Stream sorted_by (file) and write (row_id, rank) to rank_path.

    `sorted_path` — Arrow IPC file, columns [by, "row_id"], sorted by `by`.
    Передаём **Path** (не iterable), потому что bufferable-методы требуют
    двух проходов; файл открывается 1–2 раза.

    Streamable (`first`/`min`/`dense`): 1 pass.
    Bufferable (`max`/`last`/`average`): 2 passes (groups → ranks).
    """
    if method not in VALID_METHODS:
        raise NotImplementedError(
            f"rank: method={method!r} not supported. "
            f"Supported: {VALID_METHODS}."
        )
    if pct and total_n <= 0:
        raise ValueError("rank: pct=True requires total_n > 0")

    sorted_path = Path(sorted_path)

    def _batches() -> Iterator[pa.RecordBatch]:
        return ArrowIPCBatchReader(str(sorted_path), [by, "row_id"])

    if method in VALID_STREAMABLE:
        result = _compute_streamable(
            _batches(),
            by=by,
            output_col=output_col,
            method=method,
            pct=pct,
            total_n=total_n,
            rank_path=rank_path,
            memory_guard=memory_guard,
        )
    else:
        # Bufferable: two-pass.
        groups_path = rank_path.parent / (rank_path.stem + "_groups.arrow")
        _pass1_groups_to_disk(
            _batches(),
            by=by,
            groups_path=groups_path,
            memory_guard=memory_guard,
        )
        result = _pass2_ranks_to_disk(
            _batches(),
            by=by,
            groups_path=groups_path,
            output_col=output_col,
            method=method,
            pct=pct,
            total_n=total_n,
            rank_path=rank_path,
            memory_guard=memory_guard,
        )

    if log:
        logger.info(
            "compute_ranks_to_disk: method=%s pct=%s total_n=%d",
            method, pct, total_n,
        )
    return result
