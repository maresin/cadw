"""
Quantile (v0.3.3).

Алгоритм:
  1. external_sort(column) → Path (ascending).
  2. Streaming scan sorted: посчитать n_valid (non-null) и — при
     повторном проходе — собрать values на lo_i/hi_i.
  3. Интерполяция 'linear': v_lo + frac*(v_hi - v_lo).

Двух-проходный подход (sort + count/scan):
  - Sort: O(N log N) I/O; n_valid — параллельно с sort (counter).
  - Scan sorted: O(N) I/O; targets = {lo_i, hi_i} per q_i.

Один проход вместо двух: первый проход (после sort) может и считать
n_valid, и НЕ находить значения (targets зависят от n). Поэтому:
  1. Sort + count в одном проходе по исходным batches (count до sort).
  2. Sort → Path.
  3. Scan sorted → targets.

Nulls: skipna=True — nulls пропускаются; позиции 0..n_valid-1 по
не-null значениям.

Interpolation (v0.3.3): 'linear' только.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.ops.sort import external_sort

logger = get_logger("ops.quantile")

VALID_INTERPOLATIONS = ("linear",)


def _iter_counting(
    batch_iter: Iterable[pa.RecordBatch],
    column: str,
    memory_guard: MemoryGuard,
):
    """Iterate batches, yielding each; accumulate n_valid via shared state.

    Возвращает (generator, counter_getter). Counter заполняется по мере
    итерации.
    """
    state = {"n": 0}

    def _gen():
        for b in batch_iter:
            memory_guard.check_batch(b, name="quantile.count")
            arr = b.column(column)
            state["n"] += len(arr) - arr.null_count
            yield b

    return _gen(), state


def _stream_values_at_positions(
    sorted_path: Path,
    column: str,
    targets: Sequence[int],
    memory_guard: MemoryGuard,
) -> dict[int, object]:
    """Stream sorted file, collect values at 0-based non-null positions.

    v0.9.13 (F12-5, S3 perf): Arrow-native per batch. Prior: per-
    value Python loop (for v in vals) → O(max_target) Python
    iterations (~N for high quantiles). For N=10M / 5 q, ~10M
    Python iterations (~500ms-2s). Now: per-batch O(1) Arrow ops
    (filter + take on bounded target list). ~100-400x faster.

    Peak per batch: batch.nbytes (already check_batch'd) +
    selected array (len(targets) values, ≤ 2×len(q)). `filter`
    of null-only path allocates batch-size bool; bounded by
    check_batch. `to_pylist` on selected = O(K), K bounded by
    len(targets), not N.
    """
    import pyarrow.compute as _pc

    result: dict[int, object] = {}
    if not targets:
        return result
    targets_sorted = sorted(set(targets))
    max_target = targets_sorted[-1]
    pos = 0

    for batch in ArrowIPCBatchReader(str(sorted_path), [column]):
        memory_guard.check_batch(batch, name="quantile.sorted")
        arr = batch.column(column)
        n_valid_batch = len(arr) - arr.null_count
        covered = [
            t for t in targets_sorted
            if pos <= t < pos + n_valid_batch
        ]
        if covered:
            if arr.null_count == 0:
                non_null = arr
            else:
                non_null = arr.filter(_pc.is_valid(arr))
            local_idxs = [t - pos for t in covered]
            selected = _pc.take(
                non_null,
                pa.array(local_idxs, type=pa.int64()),
            )
            for t, v in zip(covered, selected.to_pylist()):
                result[t] = v
        pos += n_valid_batch
        if pos > max_target:
            return result
    return result


def quantile(
    batch_iter: Iterable[pa.RecordBatch],
    column: str,
    q: float | Sequence[float],
    memory_guard: MemoryGuard,
    spill_dir: str | Path,
    *,
    interpolation: str = "linear",
    log: bool = False,
) -> float | list[float]:
    """Compute quantile(s) of `column`.

    - q: scalar -> float; list -> list[float].
    - interpolation: 'linear' (default).
    - skipna=True (nulls не участвуют).

    Returns NaN (или list of NaN) для all-null / empty input.
    """
    if interpolation not in VALID_INTERPOLATIONS:
        raise NotImplementedError(
            f"quantile: interpolation={interpolation!r} not supported "
            f"in v0.3.3. Supported: {VALID_INTERPOLATIONS}. "
            f"'lower'/'higher'/'nearest'/'midpoint' — v0.3.4."
        )

    scalar_q = isinstance(q, (int, float))
    q_list = [float(q)] if scalar_q else [float(x) for x in q]
    if not q_list:
        raise ValueError("quantile: q must be non-empty")
    for qi in q_list:
        if not (0.0 <= qi <= 1.0):
            raise ValueError(f"quantile: q must be in [0, 1]; got {qi}")

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    it = iter(batch_iter)
    try:
        first = next(it)
    except StopIteration:
        nan_result = [float("nan")] * len(q_list)
        return nan_result[0] if scalar_q else nan_result

    if column not in first.schema.names:
        raise KeyError(f"quantile: column {column!r} not in schema")

    from itertools import chain
    counting_iter, count_state = _iter_counting(
        chain([first], it), column, memory_guard
    )

    # Pass 1 + sort: external_sort consumes counting_iter, счётчик
    # заполняется как побочный эффект.
    sorted_path = external_sort(
        counting_iter,
        key_columns=[column],
        value_columns=[column],
        memory_guard=memory_guard,
        spill_dir=spill_dir / "sort",
        ascending=True,
        log=False,
    )
    n_valid = count_state["n"]

    if n_valid == 0:
        nan_result = [float("nan")] * len(q_list)
        return nan_result[0] if scalar_q else nan_result

    # Compute target positions.
    positions: list[tuple[int, int, float]] = []
    all_targets: list[int] = []
    for qi in q_list:
        idx = qi * (n_valid - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        frac = idx - lo
        positions.append((lo, hi, frac))
        all_targets.extend([lo, hi])

    # Pass 2: scan sorted for values at positions.
    values = _stream_values_at_positions(
        sorted_path, column, all_targets, memory_guard
    )

    results: list[float] = []
    for qi, (lo, hi, frac) in zip(q_list, positions):
        v_lo = values.get(lo)
        v_hi = values.get(hi)
        if v_lo is None or v_hi is None:
            results.append(float("nan"))
            continue
        try:
            results.append(float(v_lo) + frac * (float(v_hi) - float(v_lo)))
        except (TypeError, ValueError):
            # non-numeric (strings) — только lower/higher, но в v0.3.3 только linear.
            raise TypeError(
                f"quantile: non-numeric column {column!r}; "
                f"interpolation='linear' requires numeric"
            )

    if log:
        logger.info(
            "quantile: column=%s q=%s n_valid=%d interpolation=%s",
            column, q_list, n_valid, interpolation,
        )
    return results[0] if scalar_q else results
