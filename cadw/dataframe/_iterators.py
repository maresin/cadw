# cadw/dataframe/_iterators.py
# v0.9.26i (B.1 step 3): _IteratorsMixin.
# Extracted verbatim from _core.py; no logic change.
# Helpers _convert_batch/_concat_batches imported from
# dataframe_helpers (v0.9.25f-a - already extracted).

from __future__ import annotations

from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from cadw.core.memory_guard import MemoryGuard
from cadw.dataset import (
    Dataset, MaskFilter, PredicateFilter, RowFilter,
    SliceFilter,
)
from cadw.dataframe_helpers import (
    _concat_batches, _convert_batch,
)
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import (
    ArrowIPCBatchWriter, ParquetBatchWriter,
)


class _IteratorsMixin:
    """Batch iteration + filter composition mixin (v0.9.26i)."""

    def _iter_arrow_batches(
        self, batch_size: int | None = None
    ) -> Iterator[pa.RecordBatch]:
        """Внутренний поток RecordBatch с учётом фильтров и re-chunk."""
        if not self._row_filters and batch_size is None:
            yield from self._dataset.iter_batches(columns=self._columns)
            return
        if not self._row_filters:
            yield from self._dataset.iter_batches(
                batch_size=batch_size, columns=self._columns
            )
            return
        if batch_size is None:
            yield from self._iter_filtered_batches()
            return
        accum: list[pa.RecordBatch] = []
        accum_n = 0
        for b in self._iter_filtered_batches():
            j = 0
            n = b.num_rows
            while j < n:
                take = min(batch_size - accum_n, n - j)
                accum.append(b.slice(j, take))
                accum_n += take
                j += take
                if accum_n == batch_size:
                    yield _concat_batches(accum)
                    accum = []
                    accum_n = 0
        if accum_n > 0:
            yield _concat_batches(accum)



    def _iter_filtered_batches(self) -> Iterator[pa.RecordBatch]:
        """Поток с учётом всех фильтров; на выходе — только self._columns."""
        if not self._row_filters:
            yield from self._dataset.iter_batches(columns=self._columns)
            return

        needed = list(self._columns)
        extra: set[str] = set()
        for f in self._row_filters:
            extra |= f.columns_required()
        for c in extra:
            if c not in needed:
                needed.append(c)

        it = self._dataset.iter_batches(columns=needed)
        for f in self._row_filters:
            it = f.iter_filtered(it, self._dataset.schema)
        for b in it:
            yield b.select(self._columns)



    def _iter_filtered_batches_with_positions(self):
        """Yield (filtered_batch, positions_in_raw_dataset).

        positions — numpy int64 array of 0-based позиций в **unfiltered**
        dataset (после последнего set_index/reset_index). Используется для
        восстановления parent index при filter (v0.4.0d).
        """
        from ._core import _apply_filter_with_positions
        if not self._row_filters:
            offset = 0
            for b in self._dataset.iter_batches(columns=self._columns):
                positions = np.arange(offset, offset + b.num_rows, dtype=np.int64)
                yield b, positions
                offset += b.num_rows
            return

        needed = list(self._columns)
        extra: set[str] = set()
        for f in self._row_filters:
            extra |= f.columns_required()
        for c in extra:
            if c not in needed:
                needed.append(c)

        raw_it = self._dataset.iter_batches(columns=needed)

        offset_box = [0]

        def _init():
            for b in raw_it:
                positions = np.arange(
                    offset_box[0],
                    offset_box[0] + b.num_rows,
                    dtype=np.int64,
                )
                yield b, positions
                offset_box[0] += b.num_rows

        it = _init()
        for f in self._row_filters:
            it = _apply_filter_with_positions(f, it)

        for b, positions in it:
            yield b.select(self._columns), positions



    def _with_filter(self, f: RowFilter) -> "CadwDataFrame":
        from ._core import CadwDataFrame
        new = CadwDataFrame.__new__(CadwDataFrame)
        new._dataset = self._dataset
        new._memory_guard = self._memory_guard
        new._spill_dir = self._spill_dir
        new._columns = list(self._columns)
        new._row_filters = list(self._row_filters) + [f]
        # Index fields copied; .index будет raise при наличии filters
        # (v0.4.0a). v0.4.0b (iloc) — синхронизация через row_id.
        new._index_kind = self._index_kind
        new._index_start = self._index_start
        new._index_stop = self._index_stop
        new._index_values = self._index_values
        new._index_levels = self._index_levels
        return new



    def _apply_column_transform(
        self,
        column: str,
        feature_name: str,
        transform_fn,
        out_type: pa.DataType,
        *,
        accessor: str,
    ) -> "CadwDataFrame":
        """Apply per-batch Arrow transform to one column; append {col}_{feature}."""
        from ._core import CadwDataFrame
        out_col = f"{column}_{feature_name}"
        if out_col in self._columns:
            raise KeyError(
                f"{accessor}.{feature_name}: output column {out_col!r} "
                f"already exists in {self._columns}"
            )
        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_fields.append(pa.field(out_col, out_type, nullable=True))
        out_schema = pa.schema(out_fields)

        def _iter():
            for b in self._iter_arrow_batches():
                self._memory_guard.check_batch(
                    b, name=f"{accessor}.{feature_name}",
                )
                src = b.column(column)
                new_arr = transform_fn(src)
                arrays = [b.column(c) for c in self._columns]
                arrays.append(new_arr)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


