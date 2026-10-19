"""
Sampling helpers для planner integration.

Первые батчи читаются в `CadwDataFrame.merge` для построения `DataChars`
и вызова `MemoryGuard.plan(...)`. `as_py()` здесь допустим: O(1) на
сэмпл (одна строка на колонку для оценки `avg_key_size`), не O(N).
Наружу отдаётся `DataChars` — фасад (`dataframe.py`) не вызывает `as_py`
(см. AST guard `test_facade_has_no_as_py`).
"""
from __future__ import annotations

import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.core.models import DataChars, data_chars_from_sample


def peek_first_batch(gen) -> "pa.RecordBatch | None":
    """Первый батч генератора или None (для sampling в merge)."""
    for b in gen:
        return b
    return None


def data_chars_for_join(
    sample: pa.RecordBatch,
    n_total: int,
    on_cols: list[str],
    memory_guard: MemoryGuard,
) -> DataChars:
    """DataChars для join: k_est — из unique на on-колонках sample,
    экстраполяция на n_total.

    `as_py()` на одной строке (repr().encode() для avg_key_size) — O(1),
    не нарушает гарантию памяти. `size_of_key_entry` — консервативно
    `row_size + 80` (agg state).
    """
    if sample.num_rows == 0:
        raise ValueError("data_chars_for_join: empty sample")
    on_sample = sample.select([c for c in on_cols if c in sample.schema.names])
    k_est = memory_guard.estimate_unique_keys(on_sample, n_total)

    # O(1): одна строка на колонку, не цикл по batch.num_rows.
    first = {
        c: sample.column(c)[0].as_py()
        for c in on_cols
        if c in sample.schema.names
    }
    avg_key_size = len(repr(tuple(first.values())).encode())

    return data_chars_from_sample(
        sample, n_total,
        k_est=k_est,
        avg_key_size_bytes=avg_key_size,
        size_of_key_entry_bytes=avg_key_size + 80,
    )
