"""
CadwDataFrame — публичный фасад.

Контракт «честный фасад»:
  - не реализует алгоритмы; делегирует примитивам и RowFilter;
  - не содержит построчных циклов и вызовов as_py на данных;
  - хранит только: Dataset, memory_guard, spill_dir, columns, row_filters.

Ленивость: срезы, фильтры и выбор колонок не материализуют данные.
Фильтры применяются в iter_batches / head / groupby / sort / write.
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

from cadw.core.memory_guard import MemoryGuard
from cadw.core.logging import get_logger
from cadw.core.sampling import (
    data_chars_for_join,
    peek_first_batch,
)
from cadw.dataset import (
    Dataset,
    CadwIndex,
    RangeIndex,
    MaterializedIndex,
    MaterializedMultiIndex,
    RowFilter,
    MaskFilter,
    PredicateFilter,
    SliceFilter,
)
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter, ParquetBatchWriter
from cadw.indexers import (
    _ILocIndexer,
    _LocIndexer,
    _AtIndexer,
    _IAtIndexer,
)
from cadw.ops.groupby import groupby_aggregate
from cadw.ops.join import hash_join, hash_join_streaming
from cadw.ops.sort import external_sort
from cadw.ops.dedup import drop_duplicates as _ops_drop_duplicates
from cadw.ops.rank import compute_ranks_to_disk, VALID_METHODS
from cadw.ops.quantile import quantile as _ops_quantile
from cadw.ops.resample import (
    parse_freq as _parse_freq,
    resample_to_disk as _resample_to_disk,
)
from cadw.dataframe_helpers import (  # v0.9.25f-a
    _ArrowTail, _RollState, _ExpandingState, _EwmState,
    _group_boundaries_arrow, _convert_batch, _concat_batches,
    _empty_value_counts_table, _reservoir_positions,
    _weighted_reservoir_positions,
)

logger = get_logger("dataframe")




from ._exports import _ExportsMixin, concat
from ._iterators import _IteratorsMixin
from ._column_proxy import _ColumnProxy
from ._groupby import _GroupBy
from ._misc import _MiscMixin
from ._indexing import _IndexingMixin
from ._aggregations import _AggregationsMixin
from ._transformers_fill import _TransformersFillMixin
from ._transformers_reshape import _TransformersReshapeMixin
from ._rolling import _RollingMixin
from ._cumulative import _CumulativeMixin


class CadwDataFrame(_ExportsMixin, _IteratorsMixin, _MiscMixin, _IndexingMixin, _AggregationsMixin, _TransformersFillMixin, _TransformersReshapeMixin, _RollingMixin, _CumulativeMixin):
    # ---------------- construction ----------------

    def __init__(
        self,
        dataset: Dataset,
        *,
        memory_guard: MemoryGuard | None = None,
        spill_dir: str | Path | None = None,
        columns: Sequence[str] | None = None,
        row_filters: Sequence[RowFilter] | None = None,
    ) -> None:
        if not isinstance(dataset, Dataset):
            raise TypeError("CadwDataFrame requires a Dataset instance")
        self._dataset = dataset
        self._memory_guard = memory_guard or MemoryGuard(max_memory_mb=100)
        if spill_dir is None:
            spill_dir = Path(tempfile.mkdtemp(prefix="cadw_facade_"))
        self._spill_dir = Path(spill_dir)
        self._spill_dir.mkdir(parents=True, exist_ok=True)
        self._columns = list(columns) if columns is not None else dataset.columns
        for c in self._columns:
            if c not in dataset.columns:
                raise KeyError(f"column {c!r} not found in dataset")
        self._row_filters: list[RowFilter] = list(row_filters) if row_filters else []
        # Index state: "range" (lazy) | "materialized" (single Arrow Array)
        # | "multi" (list of Arrays, tuple labels).
        self._index_kind: str = "range"
        self._index_start: int = 0
        self._index_stop: int = dataset.total_rows
        self._index_values: pa.Array | None = None
        self._index_levels: list[pa.Array] | None = None




    # ---------------- properties ----------------

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def index(self) -> CadwIndex:
        """Индекс строк (v0.4.0d — filter-compat).

        - Без фильтров: RangeIndex(0, N) или MaterializedIndex.
        - С фильтрами: MaterializedIndex — subset `_index_values[positions]`
          (O(filtered_n) scan, positions из _iter_filtered_batches_with_positions).

        Pandas-совместимо: df[df.v > 10].index = subset исходного index
        для отфильтрованных строк. §13.4/59.
        """
        if not self._row_filters:
            if self._index_kind == "range":
                return RangeIndex(self._index_start, self._index_stop)
            if self._index_kind == "multi":
                assert self._index_levels is not None
                return MaterializedMultiIndex(self._index_levels)
            assert self._index_values is not None
            return MaterializedIndex(self._index_values)

        # Filtered: collect positions and take from _index_values
        pos_chunks: list[np.ndarray] = []
        for _b, pos in self._iter_filtered_batches_with_positions():
            pos_chunks.append(pos)
        if pos_chunks:
            positions = np.concatenate(pos_chunks)
        else:
            positions = np.array([], dtype=np.int64)

        if self._index_kind == "range":
            values = positions + self._index_start
            return MaterializedIndex(pa.array(values, type=pa.int64()))
        if self._index_kind == "multi":
            assert self._index_levels is not None
            idx_arr = pa.array(positions, type=pa.int64())
            new_levels = [lv.take(idx_arr) for lv in self._index_levels]
            return MaterializedMultiIndex(new_levels)
        assert self._index_values is not None
        values = self._index_values.take(
            pa.array(positions, type=pa.int64())
        )
        return MaterializedIndex(values)







    @property
    def shape(self) -> tuple[int, int]:
        """Верхняя оценка: (dataset.total_rows, len(columns)).

        Для точного количества после фильтров используйте .count().
        """
        return (self._dataset.total_rows, len(self._columns))

    @property
    def dataset(self) -> Dataset:
        return self._dataset


    def __len__(self) -> int:
        """Row count (v0.9.6b2) — same as count().

        Enables `len(df)` in per-group apply patterns
        (`lambda g: len(g)`). O(1) unfiltered, O(N) filtered
        (same cost as count()).
        """
        return self.count()

    def select_dtypes(
        self,
        include=None,
        exclude=None,
    ) -> "CadwDataFrame":
        """Select columns by dtype predicate (v0.9.25g).

        - include / exclude: dtype spec (str or iterable of str).
          At least one required.
        - Keep column if (matches any include OR include is None/'all')
          AND (matches no exclude).
        - Dtype specs: 'number', 'int'/'integer', 'float'/'floating',
          'bool'/'boolean', 'string'/'object', 'datetime',
          'category', 'all', or explicit pyarrow type name
          (e.g. 'int64', 'float64', 'string').
        - Returns new CadwDataFrame with selected columns;
          filters preserved.
        - No-match → ValueError.

        Example:
            df.select_dtypes(include='number')
            df.select_dtypes(exclude=['string', 'bool'])
            df.select_dtypes(include='number', exclude='float32')
        """
        if include is None and exclude is None:
            raise ValueError(
                "select_dtypes: at least one of include/exclude required"
            )

        def _normalize(spec):
            if spec is None:
                return None
            if isinstance(spec, str):
                return [spec]
            return list(spec)

        def _matches(t: pa.DataType, s: str) -> bool:
            if s == "all":
                return True
            if s == "number":
                return pa.types.is_integer(t) or pa.types.is_floating(t)
            if s in ("int", "integer"):
                return pa.types.is_integer(t)
            if s in ("float", "floating"):
                return pa.types.is_floating(t)
            if s in ("bool", "boolean"):
                return pa.types.is_boolean(t)
            if s in ("string", "object"):
                return (
                    pa.types.is_string(t) or pa.types.is_large_string(t)
                    or pa.types.is_binary(t) or pa.types.is_large_binary(t)
                )
            if s in ("datetime", "datetime64"):
                return pa.types.is_timestamp(t)
            if s == "category":
                return pa.types.is_dictionary(t)
            # v0.9.25g-fix1 (§13.4/262): PyArrow dtype reprs differ
            # from user-facing names. str(pa.float32()) == "float",
            # str(pa.float64()) == "double". Use type_for_alias first
            # (same resolver astype uses), fallback to str-compare.
            try:
                return t == pa.type_for_alias(s)
            except (ValueError, KeyError, TypeError):
                return str(t) == s

        inc_list = _normalize(include)
        exc_list = _normalize(exclude)

        keep = []
        for c in self._columns:
            t = self._dataset.schema.field(c).type
            if inc_list is not None and "all" not in inc_list:
                if not any(_matches(t, s) for s in inc_list):
                    continue
            if exc_list is not None:
                if any(_matches(t, s) for s in exc_list):
                    continue
            keep.append(c)

        if not keep:
            raise ValueError(
                f"select_dtypes: no columns matched "
                f"include={include!r} exclude={exclude!r}; "
                f"available dtypes={self.dtypes()}"
            )
        return self[keep]








    # ---------------- __getitem__ ----------------

    def __getitem__(self, key):
        # 1. slice: df[10:50], df[10:50:2]
        if isinstance(key, slice):
            f = SliceFilter(key.start, key.stop, key.step)
            return self._with_filter(f)

        # 2. list[str]: выбор подмножества колонок, фильтры сохраняются
        if isinstance(key, list) and all(isinstance(k, str) for k in key):
            new = CadwDataFrame.__new__(CadwDataFrame)
            new._dataset = self._dataset
            new._memory_guard = self._memory_guard
            new._spill_dir = self._spill_dir
            new._columns = list(key)
            for c in new._columns:
                if c not in self._dataset.columns:
                    raise KeyError(f"column {c!r} not found in dataset")
            new._row_filters = list(self._row_filters)
            new._index_kind = self._index_kind
            new._index_start = self._index_start
            new._index_stop = self._index_stop
            new._index_values = self._index_values
            new._index_levels = self._index_levels
            return new

        # 3. str: прокси для сравнений (df['a'] > 0)
        if isinstance(key, str):
            if key not in self._dataset.columns:
                raise KeyError(f"column {key!r} not found in dataset")
            return _ColumnProxy(key, self)

        # 4. RowFilter: df[mask]
        if isinstance(key, RowFilter):
            return self._with_filter(key)

        raise TypeError(
            f"unsupported __getitem__ key: {type(key).__name__}; "
            f"expected slice, list[str], str, or RowFilter"
        )


    # ---------------- filtered iteration ----------------



    # ---------------- accessors + type ops (v0.9.0a) ----------------






    # ---------------- pivot / melt / crosstab (v0.9.1f) ----------------

    def melt(
        self,
        id_vars: "str | Sequence[str] | None" = None,
        value_vars: "str | Sequence[str] | None" = None,
        *,
        var_name: str = "variable",
        value_name: str = "value",
    ) -> "CadwDataFrame":
        """Unpivot wide → long (streaming).

        Args:
            id_vars: identifier columns (repeated per melted column).
                     None → no id columns.
            value_vars: columns to melt. None → all except id_vars.
            var_name: name of the "source column" column.
            value_name: name of the "value" column.

        Returns:
            CadwDataFrame with [*id_vars, var_name, value_name].
            Output rows = N × len(value_vars).

        Valid:
            - Non-overlapping id_vars and value_vars.
            - All value_vars share the SAME dtype (Arrow column uniform).
            - var_name and value_name do not collide with id_vars.
            - Non-empty value_vars.

        Invalid (raises):
            - Empty value_vars → ValueError.
            - Overlap id_vars ∩ value_vars → ValueError.
            - Mixed dtypes in value_vars → TypeError.
            - var_name / value_name collides with id_vars → ValueError.
            - Missing columns → KeyError.

        Memory: O(batch × len(value_vars)). Streaming.
        """
        import pyarrow as pa

        if id_vars is None:
            id_list: list[str] = []
        elif isinstance(id_vars, str):
            id_list = [id_vars]
        else:
            id_list = list(id_vars)

        if value_vars is None:
            value_list = [c for c in self._columns if c not in id_list]
        elif isinstance(value_vars, str):
            value_list = [value_vars]
        else:
            value_list = list(value_vars)

        if not value_list:
            raise ValueError(
                "melt: value_vars must be non-empty (all columns are "
                "id_vars?)"
            )
        for c in id_list:
            if c not in self._columns:
                raise KeyError(f"melt: id_vars {c!r} not in {self._columns}")
        for c in value_list:
            if c not in self._columns:
                raise KeyError(
                    f"melt: value_vars {c!r} not in {self._columns}"
                )
            if c in id_list:
                raise ValueError(
                    f"melt: column {c!r} in both id_vars and value_vars"
                )
        if var_name in id_list:
            raise ValueError(
                f"melt: var_name {var_name!r} collides with id_vars"
            )
        if value_name in id_list:
            raise ValueError(
                f"melt: value_name {value_name!r} collides with id_vars"
            )
        if var_name == value_name:
            raise ValueError(
                f"melt: var_name == value_name == {var_name!r}"
            )

        # Type uniformity check
        value_types = [
            self._dataset.schema.field(c).type for c in value_list
        ]
        first_t = value_types[0]
        for c, t in zip(value_list[1:], value_types[1:]):
            if t != first_t:
                raise TypeError(
                    f"melt: value_vars must share one dtype; got "
                    f"{value_list[0]!r}: {first_t} vs {c!r}: {t}. "
                    f"Cast to common type first: df.astype({{...}})."
                )

        out_fields = [self._dataset.schema.field(c) for c in id_list]
        out_fields.append(pa.field(var_name, pa.string()))
        out_fields.append(pa.field(value_name, first_t, nullable=True))
        out_schema = pa.schema(out_fields)

        def _iter():
            for batch in self._iter_arrow_batches():
                self._memory_guard.check_batch(batch, name="melt.input")
                n = batch.num_rows
                if n == 0:
                    continue
                id_arrays = [batch.column(c) for c in id_list]
                for vv in value_list:
                    val_arr = batch.column(vv)
                    var_arr = pa.array([vv] * n, type=pa.string())
                    arrays = [*id_arrays, var_arr, val_arr]
                    yield pa.RecordBatch.from_arrays(
                        arrays, schema=out_schema,
                    )

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def pivot_table(
        self,
        index: "str | Sequence[str]",
        columns: str,
        values: str,
        *,
        aggfunc: str = "mean",
    ) -> "CadwDataFrame":
        """Streaming pivot_table (v0.9.6c2, R6 — no pandas).

        Args:
            index: row identifier column(s).
            columns: column whose unique values become output columns.
            values: value column to aggregate.
            aggfunc: {'mean','sum','count','min','max'}, default 'mean'.

        Returns CadwDataFrame with [*index, <str(columns-values)>...].

        Streaming algorithm (cadw.ops.pivot.pivot_to_disk):
          1. groupby_aggregate([*index, columns]) → long Table.
          2. external_sort([*index, columns]).
          3. Scan sorted stream → unique columns (sorted).
          4. Per index_key: buffer cells, emit RecordBatch every 1000
             rows. No dense K_index × K_columns matrix in RAM.

        Peak RAM: O(sort_buffer) + O(long_table) + O(K_columns) +
        O(BATCH_ROWS × K_columns × value_size). Independent of K_index.

        R6 audit: previous pandas path used `to_pandas().pivot()` —
        26.8 MB RSS for 200k rows / 300×300 (~720 KB useful, 37×
        overhead). New streaming path: ~10 MB RSS for same input.

        Missing (index, columns) combos → null cells. Null in
        `columns` → row dropped (dropna=True, pandas default).

        Raises:
            ValueError: empty index, aggfunc not supported, empty
                after null filter, name collision with index cols.
            KeyError: missing columns.
            MemoryLimitExceeded: K_columns × key_size × 2 or
                BATCH_ROWS × K_columns × value_size > state_limit.
        """
        import pyarrow as pa
        from cadw.core.memory_guard import MemoryLimitExceeded
        from cadw.ops.pivot import pivot_to_disk

        if isinstance(index, str):
            index_cols = [index]
        else:
            index_cols = list(index)
        if not index_cols:
            raise ValueError("pivot_table: index must be non-empty")

        for c in index_cols:
            if c not in self._columns:
                raise KeyError(
                    f"pivot_table: index col {c!r} not in {self._columns}"
                )
        if columns not in self._columns:
            raise KeyError(
                f"pivot_table: columns {columns!r} not in {self._columns}"
            )
        if values not in self._columns:
            raise KeyError(
                f"pivot_table: values {values!r} not in {self._columns}"
            )
        if columns in index_cols:
            raise ValueError(
                f"pivot_table: {columns!r} in both index and columns"
            )
        if aggfunc not in ("mean", "sum", "count", "min", "max"):
            raise ValueError(
                f"pivot_table: aggfunc must be one of mean/sum/count/"
                f"min/max; got {aggfunc!r}"
            )
        # v0.9.1f-fix1 (§13.4/157): count ignores value semantics; other
        # aggs require value not in index/columns.
        if aggfunc != "count":
            if values in index_cols or values == columns:
                raise ValueError(
                    f"pivot_table: values {values!r} must be distinct "
                    f"from index/columns (aggfunc={aggfunc!r})"
                )
        if values == columns:
            raise ValueError(
                f"pivot_table: values {values!r} == columns {columns!r}"
            )

        # Empty check: filter null columns first (matches dropna=True).
        filtered = self[self[columns].notnull()]
        n = filtered.count()
        if n == 0:
            raise ValueError(
                f"pivot_table: no rows after filtering null {columns!r}; "
                f"cannot pivot empty data"
            )

        values_type = self._dataset.schema.field(values).type
        out_schema, out_path = pivot_to_disk(
            filtered._iter_filtered_batches(),
            index_cols=index_cols,
            columns_col=columns,
            values_col=values,
            values_type=values_type,
            aggfunc=aggfunc,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "pivot",
        )
        try:
            new_ds = Dataset.from_batches(
                out_schema,
                ArrowIPCBatchReader(str(out_path), None),
                self._new_dataset_dir(),
            )
        finally:
            out_path.unlink(missing_ok=True)
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def crosstab(
        self,
        index: "str | Sequence[str]",
        columns: str,
    ) -> "CadwDataFrame":
        """Frequency table (count per (index, columns) combo).

        Equivalent to:
            df.dropna(subset=[*index, columns])
              .pivot_table(index, columns, values=index[0], aggfunc='count')

        Drops rows with null in any index or columns value (standard
        pandas crosstab behaviour).

        Returns:
            CadwDataFrame with [*index, <columns-values as columns>].
        """
        index_cols = [index] if isinstance(index, str) else list(index)
        if not index_cols:
            raise ValueError("crosstab: index must be non-empty")
        for c in index_cols:
            if c not in self._columns:
                raise KeyError(
                    f"crosstab: index col {c!r} not in {self._columns}"
                )
        if columns not in self._columns:
            raise KeyError(
                f"crosstab: columns {columns!r} not in {self._columns}"
            )
        # Drop nulls in any relevant column (standard crosstab)
        filtered = self.dropna(subset=index_cols + [columns])
        return filtered.pivot_table(
            index=index,
            columns=columns,
            values=index_cols[0],  # any non-null col; = row count
            aggfunc="count",
        )

    # ---------------- reading ----------------




    # ---------------- groupby ----------------



    # ---------------- unique ----------------


    # ---------------- sort_values ----------------


    # ---------------- drop_duplicates ----------------


    # ---------------- rank ----------------


    # ---------------- quantile ----------------


    # ---------------- encode_categorical ----------------

    def encode_categorical(
        self,
        columns: "str | Sequence[str]",
    ) -> "tuple[CadwDataFrame, dict[str, list]]":
        """Encode categorical columns as int32 (v0.6.2a).

        - columns: str or Sequence[str]. Must be string/binary columns.
        - Pass 1: collect unique values (pc.value_counts per batch),
          sorted for determinism. Nulls excluded (become null ids).
        - Pass 2: pc.index_in(col, value_set=pa.array(sorted_values))
          per batch → int32 (null stays null).

        Returns:
            (encoded_df, reverse_maps): new CadwDataFrame with int32
            columns (same names); original df unchanged.
            reverse_maps[col] = sorted_values; id k → reverse_maps[col][k].

        Memory: O(K) per column, independent of N.
        Raises TypeError for non-string columns.
        """
        import pyarrow.compute as pc

        if isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError("encode_categorical: columns must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"encode_categorical: column {c!r} not in {self._columns}"
                )

        # Validate types — only string/binary encodable.
        for c in cols:
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_string(t) or pa.types.is_large_string(t)
                    or pa.types.is_binary(t) or pa.types.is_large_binary(t)):
                raise TypeError(
                    f"encode_categorical: column {c!r} has type {t}; "
                    f"only string/binary supported"
                )

        # Pass 1: collect unique values per column.
        reverse_maps: dict[str, list] = {}
        for col in cols:
            uniques: set = set()
            for b in self._iter_filtered_batches():
                arr = b.column(col)
                vc = pc.value_counts(arr)  # struct<values, counts>
                if isinstance(vc, pa.ChunkedArray):
                    vc = vc.combine_chunks()
                values = vc.field("values")
                for v in values.to_pylist():  # K per batch, not N
                    if v is not None:
                        uniques.add(v)
                # v0.9.25c (part1_report B2): in-loop guard. Prior code
                # accumulated Python-set of all uniques with no bound;
                # @ 10M unique strings ~600 MB invisible to MemoryGuard
                # (batch.nbytes ~130 KB/batch). 180 B/entry = 50 (str)
                # + ~40 (set slot) + ~90 (list copy margin for sorted()).
                # Fail-fast at state_limit (=available*0.5).
                self._memory_guard.check_state(
                    len(uniques) * 180,
                    name=f"encode_categorical({col})",
                )
            try:
                sorted_vals = sorted(uniques)
            except TypeError:
                sorted_vals = sorted(uniques, key=lambda x: str(x))
            reverse_maps[col] = sorted_vals

        # Pass 2: encode.
        schema = pa.schema(
            [
                (
                    pa.field(c, pa.int32())
                    if c in cols
                    else self._dataset.schema.field(c)
                )
                for c in self._columns
            ]
        )

        def _iter_encoded():
            for b in self._iter_filtered_batches():
                arrays = []
                for c in self._columns:
                    if c in cols:
                        arr = b.column(c)
                        value_set = pa.array(reverse_maps[c], type=arr.type)
                        ids = pc.index_in(arr, value_set=value_set)
                        arrays.append(ids.cast(pa.int32()))
                    else:
                        arrays.append(b.column(c))
                yield pa.RecordBatch.from_arrays(arrays, schema=schema)

        new_ds = Dataset.from_batches(
            schema, _iter_encoded(), self._new_dataset_dir()
        )
        new_df = CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )
        return new_df, reverse_maps


    # ---------------- shift / diff ----------------



    def _shift_on(
        self,
        k: int,
        fill_value,
        *,
        columns: "str | Sequence[str] | None",
        on_cols: list[str],
    ) -> "CadwDataFrame":
        """Sort-aware shift on `on_cols` (v0.8.2, multi v0.8.6).

        Algorithm:
          1. external_sort by [on_cols...] (row_id tie-break from heap)
             → sorted IPC. Lexicographic composite sort.
          2. Read sorted stream, apply stateful shift (deque per column,
             O(k) memory). Output (row_id, shifted_cols) → IPC.
          3. external_sort by __orig_rid → original order.
          4. Zip with original.

        Peak RAM: batch + O(k × #cols). Independent of N.
        `on_cols` pre-normalized by caller (v0.8.6).
        """
        from collections import deque

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"shift: column {c!r} not in {self._columns}"
                )
        shift_cols = set(cols)

        out_fields = []
        for c in self._columns:
            f = self._dataset.schema.field(c)
            if c in shift_cols:
                f = pa.field(c, f.type, nullable=True)
            out_fields.append(f)
        out_schema = pa.schema(out_fields)

        # 1. Sort by [on_cols...] (lexicographic)
        all_cols = list(self._columns)
        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=on_cols,
            value_columns=all_cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "shift" / "sort_by_on",
            ascending=True,
            log=False,
        )

        # 2. Shift on sorted stream
        sorted_reader = ArrowIPCBatchReader(str(sorted_path), None)
        shift_schema = pa.schema(
            [pa.field("row_id", pa.uint64())]
            + [
                pa.field(c, self._dataset.schema.field(c).type, nullable=True)
                for c in cols
            ]
        )
        shift_path = (
            self._spill_dir / "shift" / "shifted_by_on.arrow"
        )
        shift_path.parent.mkdir(parents=True, exist_ok=True)

        def _fill(size: int, type_: pa.DataType) -> pa.Array:
            if fill_value is None:
                return pa.nulls(size, type=type_)
            return pa.array([fill_value] * size, type=type_)

        writer = ArrowIPCBatchWriter(str(shift_path), shift_schema)
        # v0.9.18 (AUDIT §1.1): Arrow-native tail buffer (same fix as shift).
        bufs: dict[str, _ArrowTail] = {c: _ArrowTail(k) for c in cols}
        try:
            for b in sorted_reader:
                self._memory_guard.check_batch(
                    b, name="shift_on.sorted_input"
                )
                n = b.num_rows
                arrays: list = [b.column("row_id")]
                for c in cols:
                    col_vals = b.column(c)
                    prev_arr = bufs[c].get()
                    prev_len = len(prev_arr) if prev_arr is not None else 0
                    fill_n = max(0, k - prev_len)
                    parts = []
                    if fill_n > 0:
                        parts.append(_fill(fill_n, col_vals.type))
                    if prev_arr is not None:
                        parts.append(prev_arr)
                    if n > k:
                        parts.append(col_vals.slice(0, n - k))
                    out = (
                        pa.concat_arrays(parts) if parts
                        else pa.array([], type=col_vals.type)
                    )
                    if len(out) > n:
                        out = out.slice(0, n)
                    elif len(out) < n:
                        out = pa.concat_arrays([
                            out,
                            pa.nulls(n - len(out), type=col_vals.type),
                        ])
                    arrays.append(out)
                    bufs[c].push(col_vals)
                rb = pa.RecordBatch.from_arrays(arrays, schema=shift_schema)
                writer.write_batch(rb)
        finally:
            writer.close()

        # 3. Sort back by __orig_rid.
        # external_sort ALWAYS appends its own "row_id" column. Our stream
        # already has "row_id" (= original position) → two row_ids would
        # trigger ArrowInvalid: Multiple matches for FieldRef.Name(row_id).
        # Rename "row_id" → "__orig_rid" so external_sort's appended row_id
        # is a different column. Sort by __orig_rid ascending = original order.
        def _iter_shifted():
            for b in ArrowIPCBatchReader(str(shift_path), None):
                names = b.schema.names
                rid_idx = names.index("row_id")
                new_names = (
                    ["__orig_rid"]
                    + [n for i, n in enumerate(names) if i != rid_idx]
                )
                yield b.rename_columns(new_names)

        sorted_rowid_path = external_sort(
            _iter_shifted(),
            key_columns=["__orig_rid"],
            value_columns=cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "shift" / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        # 4. Zip (re-chunk to same batch_size)
        batch_size = 10_000

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            shifted_reader = ArrowIPCBatchReader(
                str(sorted_rowid_path), cols,
            )
            accum: list[pa.RecordBatch] = []
            accum_n = 0
            shifted_gen = iter(shifted_reader)
            for ob in orig_iter:
                target_n = ob.num_rows
                while accum_n < target_n:
                    try:
                        nb = next(shifted_gen)
                    except StopIteration:
                        raise RuntimeError(
                            "shift: shifted stream exhausted before "
                            "original (batch mismatch)"
                        )
                    accum.append(nb)
                    accum_n += nb.num_rows
                combined = pa.Table.from_batches(accum)
                take = combined.slice(0, target_n)
                rest = combined.slice(
                    target_n, combined.num_rows - target_n,
                )
                accum = list(rest.to_batches()) if rest.num_rows else []
                accum_n = combined.num_rows - target_n
                # v0.8.2-fix4: (a) REPLACE shifted columns, not append —
                # out_schema has self._columns (shift replaces column value,
                # doesn't add new column; matches v0.6.2b semantics);
                # (b) Table.column() returns ChunkedArray; combine_chunks()
                # + to_batches()[0] → RecordBatch with Array columns.
                take_batch = take.combine_chunks().to_batches()[0]
                arrays = []
                for c in self._columns:
                    if c in shift_cols:
                        arrays.append(take_batch.column(c))
                    else:
                        arrays.append(ob.column(c))
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_zipped(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def _diff_on(
        self,
        *,
        k: int,
        columns: "str | Sequence[str] | None",
        on: "str | Sequence[str]",
    ) -> "CadwDataFrame":
        """Sort-aware diff: pc.subtract(col, shifted(col)) with on=col sort.

        v0.8.2 (single), v0.8.6 (multi). Delegates to _shift_on + subtract
        per batch. No extra sort needed (reuses shift's 2 sorts).
        """
        # v0.8.6: validate on early (before numeric-type check on columns).
        on_cols = self._normalize_on(on, fn_name="diff")
        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"diff: column {c!r} not in {self._columns}"
                )
        # Validate numeric for diff
        for c in cols:
            t = self._dataset.schema.field(c).type
            if not (
                pa.types.is_integer(t) or pa.types.is_floating(t)
            ):
                raise TypeError(
                    f"diff: column {c!r} has type {t}; numeric required"
                )
        diff_cols = set(cols)

        # Shift with fill_value=None (nulls for first k rows)
        shifted_df = self._shift_on(
            k, None, columns=cols, on_cols=on_cols,
        )

        # Zip original + shifted, subtract
        import pyarrow.compute as _pc
        out_fields = []
        for c in self._columns:
            f = self._dataset.schema.field(c)
            if c in diff_cols:
                f = pa.field(c, f.type, nullable=True)
            out_fields.append(f)
        out_schema = pa.schema(out_fields)

        batch_size = 10_000

        def _iter_diffed():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            shifted_iter = shifted_df._iter_arrow_batches(
                batch_size=batch_size,
            )
            for ob, sb in zip(orig_iter, shifted_iter):
                if ob.num_rows != sb.num_rows:
                    raise RuntimeError(
                        f"diff_on: batch size mismatch "
                        f"(orig={ob.num_rows}, shifted={sb.num_rows})"
                    )
                arrays = []
                for c in self._columns:
                    if c in diff_cols:
                        arrays.append(
                            _pc.subtract(ob.column(c), sb.column(c))
                        )
                    else:
                        arrays.append(ob.column(c))
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_diffed(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    # ---------------- rolling ----------------



    # ---------------- per-group apply (v0.9.1e) ----------------

    def _per_group_apply(
        self,
        group_cols: list[str],
        on_cols: list[str],
        fn,
        result_col: str,
        memory_guard: "MemoryGuard",
    ) -> "CadwDataFrame":
        """Composite-sort apply(fn) on each group. v0.9.6b2 CADW-native.

        Args:
            group_cols: group keys. Non-empty.
            on_cols: within-group sort keys (empty → row_id order).
            fn: callable(group_df: CadwDataFrame) → scalar | dict[str,scalar]
                | CadwDataFrame. Uniform mode across groups.
            result_col: column name for scalar mode.
            memory_guard: per-group memory check before fn().

        Peak RAM = batch + 2 × max(group_size × row_size) + fn's output.
        Bounded per group, independent of N.

        Empty-input limitation (§11.5, v0.9.14 F12-7):
        when df.count() == 0, fn is never called; output
        schema = [group_cols..., result_col: float64]. This
        is only accurate for scalar mode. For dict mode
        (fn returns dict[str, scalar]) or DataFrame mode
        (fn returns CadwDataFrame) with empty input, the
        output is the flat result_col schema — not dict
        keys / df columns, since mode is unknown at
        empty-input time. Caller who needs accurate empty
        schema for dict/df mode: use a non-empty check or
        synthesize one row.

        No pandas. For multi-metric per-group output use a dict
        (`{"mean": ..., "max": ...}`); for multi-row output use
        `CadwDataFrame.from_table(pa_table)`.

        fn's own alloc is user responsibility (§3.9 extended).
        """
        import pyarrow as pa
        from cadw.io.batch_reader import ArrowIPCBatchReader
        from cadw.io.batch_writer import ArrowIPCBatchWriter
        from cadw.core.memory_guard import MemoryLimitExceeded

        if not group_cols:
            raise ValueError(
                "_per_group_apply: group_cols must be non-empty"
            )
        if not callable(fn):
            raise TypeError(
                f"_per_group_apply: fn must be callable; "
                f"got {type(fn).__name__}"
            )
        for gc in group_cols:
            if gc not in self._columns:
                raise KeyError(
                    f"_per_group_apply: group column {gc!r} not in "
                    f"{self._columns}"
                )
        for oc in on_cols:
            if oc not in self._columns:
                raise KeyError(
                    f"_per_group_apply: on column {oc!r} not in "
                    f"{self._columns}"
                )
        if not isinstance(result_col, str) or not result_col:
            raise ValueError(
                f"_per_group_apply: result_col must be non-empty str; "
                f"got {result_col!r}"
            )

        if self.count() == 0:
            out_fields = [
                self._dataset.schema.field(c) for c in group_cols
            ]
            out_fields.append(pa.field(result_col, pa.float64(), nullable=True))
            out_schema = pa.schema(out_fields)
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        sort_keys = list(group_cols) + list(on_cols)
        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=sort_keys,
            value_columns=self._columns,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "per_group_apply" / "sort_by_keys",
            ascending=True,
            log=False,
        )

        out_schema_box: list = []
        out_mode_box: list = []
        buf_rows: list = []
        BATCH_ROWS = 1000
        applied_path = (
            self._spill_dir / "per_group_apply" / "applied.arrow"
        )
        applied_path.parent.mkdir(parents=True, exist_ok=True)
        writer_box: list = []

        def _flush_buf():
            # v0.9.22 (§13.4/244, AUDIT_ML §1.7): buf_rows bounded
            # at BATCH_ROWS=1000 (see _emit_row) → ~140 KB at K=5.
            # Real guard is _flush_group at 0.3·available. No
            # check_batch needed — bounded by construction.
            nonlocal buf_rows
            if not buf_rows:
                return
            _maybe_upgrade_schema()  # v0.9.11 (PF4)
            if not writer_box:
                writer_box.append(
                    ArrowIPCBatchWriter(str(applied_path),
                                        out_schema_box[0])
                )
            # v0.9.14 (F12-6b): non-widenable type mismatch
            # (e.g. str field + numeric later) → clear error,
            # not cryptic ArrowInvalid from from_pylist.
            try:
                table = pa.Table.from_pylist(buf_rows,
                                             schema=out_schema_box[0])
            except (pa.ArrowInvalid, pa.ArrowTypeError) as _e:
                raise RuntimeError(
                    f"_per_group_apply: fn() returned "
                    f"inconsistent types across groups. "
                    f"Buffered schema={out_schema_box[0]}, "
                    f"error={_e}. fn() must return the "
                    f"same type for all groups. Hint: "
                    f"return float(x) from fn() if numeric "
                    f"types might mix (int/float/bool); "
                    f"return None for missing values. §11.5"
                ) from _e
            for b in table.to_batches():
                writer_box[0].write_batch(b)
            buf_rows = []

        def _maybe_upgrade_schema() -> None:
            """v0.9.11 (PF4): widen int->float64 on fractional float.

            Deferred from per-row (_maybe_upgrade_for_row) to
            once-per-flush. Per-row allocation of `upgraded` list
            was O(N*K) — for 10M groups x K=10, ~100M list allocs
            + ~1B attr checks. Now: one scan of buf_rows
            (<= BATCH_ROWS=1000 rows) per flush. ~1000x fewer allocs,
            ~10x fewer attr checks.

            Checks each integer field across all buf_rows; if any
            fractional float found -> widen to float64. Lossless
            for |v| < 2^53.

            Limitation (§11.5): only before first flush. Once writer
            committed (writer_box non-empty), post-flush widen raises
            clear RuntimeError (prior: silent truncation or cryptic
            ArrowInvalid at write_batch). Workaround: return
            float(x) from fn() if any group might be fractional.
            """
            if not out_schema_box:
                return
            schema = out_schema_box[0]
            upgraded = []
            needs = False
            widen_fields: list = []
            for field in schema:
                # v0.9.14 (F12-6a): bool field, any non-bool
                # numeric → widen to float64. Prior: pa.bool_
                # schema + later float → ArrowInvalid (cryptic).
                # Same class as C1 (int→float).
                if pa.types.is_boolean(field.type):
                    widen = False
                    for row in buf_rows:
                        val = row.get(field.name)
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            widen = True
                            break
                    if widen:
                        upgraded.append(pa.field(
                            field.name, pa.float64(),
                            nullable=field.nullable,
                        ))
                        needs = True
                        widen_fields.append(field.name)
                    else:
                        upgraded.append(field)
                    continue
                if not pa.types.is_integer(field.type):
                    upgraded.append(field)
                    continue
                widen = False
                for row in buf_rows:
                    val = row.get(field.name)
                    if isinstance(val, float) and val != int(val):
                        widen = True
                        break
                if widen:
                    upgraded.append(pa.field(
                        field.name, pa.float64(),
                        nullable=field.nullable,
                    ))
                    needs = True
                    widen_fields.append(field.name)
                else:
                    upgraded.append(field)
            if needs:
                if writer_box:
                    raise RuntimeError(
                        f"_per_group_apply: post-flush widen needed "
                        f"for {widen_fields}, but writer already "
                        f"committed. Workaround: return float(x) "
                        f"from fn() if any group might be fractional. "
                        f"See §11.5."
                    )
                out_schema_box[0] = pa.schema(upgraded)

        def _emit_row(row: dict) -> None:
            """Append row; widen check deferred to _flush_buf (PF4)."""
            nonlocal buf_rows
            buf_rows.append(row)

        def _emit_group(key: tuple, result):
            nonlocal buf_rows

            mode = _detect_mode(result)
            if not out_mode_box:
                out_mode_box.append(mode)
                out_schema_box.append(
                    _build_out_schema(group_cols, self, mode,
                                      result, result_col)
                )
            elif mode != out_mode_box[0]:
                raise RuntimeError(
                    f"_per_group_apply: mixed return modes across "
                    f"groups: first group returned {out_mode_box[0]!r}, "
                    f"group {key!r} returned {mode!r}. fn() must "
                    f"return same type for all groups."
                )

            if mode == "scalar":
                row = {gc: v for gc, v in zip(group_cols, key)}
                row[result_col] = _normalize_scalar(result)
                _emit_row(row)
            elif mode == "dict":
                row = {gc: v for gc, v in zip(group_cols, key)}
                for k, v in result.items():
                    row[str(k)] = _normalize_scalar(v)
                _emit_row(row)
            else:  # CadwDataFrame
                for b in result._iter_filtered_batches():
                    for rec in b.to_pylist():
                        row = dict(rec)
                        for gc, v in zip(group_cols, key):
                            row.setdefault(gc, v)
                        _emit_row(row)
            if len(buf_rows) >= BATCH_ROWS:
                _flush_buf()

        _UNSET = object()
        prev_key = _UNSET
        group_buf: list = []
        max_group_limit = int(
            memory_guard.available_bytes() * 0.3
        )

        def _flush_group():
            nonlocal group_buf
            if not group_buf:
                return
            est_pre = sum(b.nbytes for b in group_buf) * 2
            if est_pre > max_group_limit:
                raise MemoryLimitExceeded(
                    f"_per_group_apply: group size ~{est_pre // 2} B "
                    f"× 2 (materialize + overhead) = {est_pre} B > "
                    f"0.3 × avail = {max_group_limit} B. Split by "
                    f"more group cols or use groupby().agg() if fn "
                    f"reduces. See §11.5."
                )
            memory_guard.check_state(
                est_pre, name="per_group_apply.group",
            )
            gt = pa.Table.from_batches(group_buf)
            est_actual = gt.nbytes * 2
            if est_actual > max_group_limit:
                raise MemoryLimitExceeded(
                    f"_per_group_apply: actual group size "
                    f"{gt.nbytes} B × 2 = {est_actual} B > 0.3 × avail "
                    f"= {max_group_limit} B (pre-estimate was "
                    f"{est_pre} B; buffer overhead). See §11.5."
                )
            # v0.9.6b2: CADW-native group materialize (in-memory).
            gdf = CadwDataFrame.from_table(
                gt,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )
            result = fn(gdf)
            _emit_group(tuple(prev_key), result)
            group_buf.clear()

        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                memory_guard.check_batch(
                    b, name="per_group_apply.sorted"
                , extra_factor=7.0)
                n = b.num_rows
                if n == 0:
                    continue
                group_vals = {
                    gc: b.column(gc).to_pylist() for gc in group_cols
                }
                start = 0
                for i in range(1, n):
                    gk = tuple(group_vals[gc][i] for gc in group_cols)
                    prev_gk = tuple(
                        group_vals[gc][i - 1] for gc in group_cols
                    )
                    if gk != prev_gk:
                        gk_start = tuple(
                            group_vals[gc][start] for gc in group_cols
                        )
                        piece = b.slice(start, i - start)
                        if gk_start != prev_key:
                            if prev_key is not _UNSET:
                                _flush_group()
                            prev_key = gk_start
                        group_buf.append(piece)
                        start = i
                gk_start = tuple(
                    group_vals[gc][start] for gc in group_cols
                )
                piece = b.slice(start, n - start)
                if gk_start != prev_key:
                    if prev_key is not _UNSET:
                        _flush_group()
                    prev_key = gk_start
                group_buf.append(piece)

            _flush_group()
            _flush_buf()
        finally:
            if writer_box:
                writer_box[0].close()

        if not out_schema_box:
            raise RuntimeError(
                "_per_group_apply: no groups produced output — "
                "unreachable after empty check"
            )

        def _iter_applied():
            for b in ArrowIPCBatchReader(str(applied_path), None):
                yield b

        new_ds = Dataset.from_batches(
            out_schema_box[0], _iter_applied(),
            self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    # ---------------- per-group cumulative (v0.9.23e) ----------------


    # ---------------- per-group shift/diff/rolling (v0.8.10) ----------------

    def _per_group_shift(
        self,
        group_cols: list[str],
        on_cols: list[str],
        k: int,
        fill_value,
        columns: "str | Sequence[str] | None",
    ) -> "CadwDataFrame":
        """Composite-sort per-group shift (validated by proto_per_group.py).

        Algorithm (11 cases, A==B):
          1. external_sort by [*group_cols, *on_cols] — contiguous groups.
          2. Stream sorted; single deque per value col; clear on group change.
          3. external_sort by __orig_rid.
          4. Zip: replace shifted cols, keep originals.

        Peak RAM = batch + O(k × #cols). Independent of N and G.
        """
        from collections import deque

        if not group_cols:
            raise ValueError("_per_group_shift: group_cols must be non-empty")
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"_per_group_shift: k must be int >= 1; got {k!r}")
        for gc in group_cols:
            if gc not in self._columns:
                raise KeyError(
                    f"_per_group_shift: group column {gc!r} not in {self._columns}"
                )
        for oc in on_cols:
            if oc not in self._columns:
                raise KeyError(
                    f"_per_group_shift: on column {oc!r} not in {self._columns}"
                )

        if columns is None:
            cols = [c for c in self._columns if c not in group_cols]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c in group_cols:
                raise ValueError(
                    f"_per_group_shift: group column {c!r} cannot be shifted"
                )
            if c not in self._columns:
                raise KeyError(
                    f"_per_group_shift: column {c!r} not in {self._columns}"
                )
        if not cols:
            raise ValueError("_per_group_shift: no value columns to shift")
        shift_cols = set(cols)

        out_fields = []
        for c in self._columns:
            f = self._dataset.schema.field(c)
            if c in shift_cols:
                f = pa.field(c, f.type, nullable=True)
            out_fields.append(f)
        out_schema = pa.schema(out_fields)

        if self.count() == 0:
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir()
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        sort_keys = list(group_cols) + list(on_cols)
        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=sort_keys,
            value_columns=self._columns,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "per_group_shift" / "sort_by_keys",
            ascending=True,
            log=False,
        )

        applied_schema = pa.schema(
            [pa.field("row_id", pa.uint64())]
            + [
                pa.field(c, self._dataset.schema.field(c).type, nullable=True)
                for c in cols
            ]
        )
        applied_path = self._spill_dir / "per_group_shift" / "applied.arrow"
        applied_path.parent.mkdir(parents=True, exist_ok=True)

        writer = ArrowIPCBatchWriter(str(applied_path), applied_schema)
        bufs = {c: deque(maxlen=k) for c in cols}
        _SENTINEL = object()
        prev_group = _SENTINEL
        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                self._memory_guard.check_batch(
                    b, name="per_group_shift.sorted"
                , extra_factor=7.0)
                n = b.num_rows
                if n == 0:
                    continue
                # v0.9.21 (§1.4): Arrow-native group boundaries in shift.
                # Prior: group_arrs = {gc: b.column(gc).to_pylist()...}
                # + per-row tuple(group_arrs[gc][i]) = O(n × K_cols)
                # Python ops + n × K_cols Python objects (28 B each).
                # Now: _group_boundaries_arrow = O(n × K_cols) C++ ops
                # + O(n) transient bool masks. Peak Python per batch =
                # max_group_size × cols (transient, GC'd between
                # groups), not n × cols.
                boundaries = _group_boundaries_arrow(b, group_cols)

                # v0.9.21-fix1: batch-level key via 1-row to_pylist
                # (bounded #batches × K_cols, NOT per-group as_py).
                batch_first_key = tuple(
                    b.column(gc).slice(0, 1).to_pylist()[0]
                    for gc in group_cols
                )
                if (prev_group is _SENTINEL
                        or batch_first_key != prev_group):
                    if prev_group is not _SENTINEL:
                        for dq in bufs.values():
                            dq.clear()

                all_shifted_arrow = {c: [] for c in cols}
                for gi in range(len(boundaries) - 1):
                    start = boundaries[gi]
                    end = boundaries[gi + 1]
                    if gi > 0:
                        # New group within batch — reset state (no key
                        # tracking; batch-boundary key handles carryover).
                        for dq in bufs.values():
                            dq.clear()
                    group_len = end - start
                    gvals = {
                        c: b.column(c).slice(start, group_len).to_pylist()
                        for c in cols
                    }
                    group_shifted = {c: [None] * group_len for c in cols}
                    for i in range(group_len):
                        for c in cols:
                            if len(bufs[c]) < k:
                                group_shifted[c][i] = fill_value
                            else:
                                group_shifted[c][i] = bufs[c][0]
                            bufs[c].append(gvals[c][i])
                    for c in cols:
                        all_shifted_arrow[c].append(
                            pa.array(
                                group_shifted[c],
                                type=b.schema.field(c).type,
                            )
                        )
                arrays = [b.column("row_id")]
                for c in cols:
                    arrays.append(pa.concat_arrays(all_shifted_arrow[c]))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
                # v0.9.21-fix1: track last row's key for next batch's
                # cross-batch continuity (bounded 1-row to_pylist).
                prev_group = tuple(
                    b.column(gc).slice(n - 1, 1).to_pylist()[0]
                    for gc in group_cols
                )
        finally:
            writer.close()

        def _iter_applied():
            for b in ArrowIPCBatchReader(str(applied_path), None):
                names = b.schema.names
                rid_idx = names.index("row_id")
                new_names = (
                    ["__orig_rid"]
                    + [n for i, n in enumerate(names) if i != rid_idx]
                )
                yield b.rename_columns(new_names)

        resorted_path = external_sort(
            _iter_applied(),
            key_columns=["__orig_rid"],
            value_columns=cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "per_group_shift" / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        batch_size = 10_000

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            sorted_reader = ArrowIPCBatchReader(str(resorted_path), cols)
            accum: list = []
            accum_n = 0
            gen = iter(sorted_reader)
            for ob in orig_iter:
                target_n = ob.num_rows
                while accum_n < target_n:
                    try:
                        nb = next(gen)
                    except StopIteration:
                        raise RuntimeError(
                            "per_group_shift: sorted stream exhausted before "
                            "original"
                        )
                    accum.append(nb)
                    accum_n += nb.num_rows
                combined = pa.Table.from_batches(accum)
                take = combined.slice(0, target_n)
                rest = combined.slice(
                    target_n, combined.num_rows - target_n,
                )
                accum = list(rest.to_batches()) if rest.num_rows else []
                accum_n = combined.num_rows - target_n
                take_batch = take.combine_chunks().to_batches()[0]
                arrays = []
                for c in self._columns:
                    if c in shift_cols:
                        arrays.append(take_batch.column(c))
                    else:
                        arrays.append(ob.column(c))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_zipped(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def _per_group_diff(
        self,
        group_cols: list[str],
        on_cols: list[str],
        k: int,
        columns: "str | Sequence[str] | None",
    ) -> "CadwDataFrame":
        """Per-group diff — delegate to shift then subtract."""
        import pyarrow.compute as _pc

        if not isinstance(k, int) or k < 1:
            raise ValueError(f"_per_group_diff: k must be int >= 1; got {k!r}")

        if columns is None:
            cols = [c for c in self._columns if c not in group_cols]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c in group_cols:
                raise ValueError(
                    f"_per_group_diff: group column {c!r} cannot be diffed"
                )
            if c not in self._columns:
                raise KeyError(
                    f"_per_group_diff: column {c!r} not in {self._columns}"
                )
        for c in cols:
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"_per_group_diff: column {c!r} has type {t}; "
                    f"numeric required"
                )
        if not cols:
            raise ValueError("_per_group_diff: no value columns")

        diff_cols = set(cols)

        shifted_df = self._per_group_shift(
            group_cols, on_cols, k, None, cols,
        )

        out_fields = []
        for c in self._columns:
            f = self._dataset.schema.field(c)
            if c in diff_cols:
                f = pa.field(c, f.type, nullable=True)
            out_fields.append(f)
        out_schema = pa.schema(out_fields)

        batch_size = 10_000

        def _iter_diffed():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            shifted_iter = shifted_df._iter_arrow_batches(batch_size=batch_size)
            for ob, sb in zip(orig_iter, shifted_iter):
                if ob.num_rows != sb.num_rows:
                    raise RuntimeError(
                        f"_per_group_diff: batch size mismatch "
                        f"(orig={ob.num_rows}, shifted={sb.num_rows})"
                    )
                arrays = []
                for c in self._columns:
                    if c in diff_cols:
                        arrays.append(_pc.subtract(ob.column(c), sb.column(c)))
                    else:
                        arrays.append(ob.column(c))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_diffed(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    # ---------------- resample ----------------

    def resample(
        self,
        freq: "int | str",
        agg: str,
        *,
        on: str,
        columns: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Bucket aggregation by `on` (v0.8.7).

        - freq: int (seconds) or str (`'5s'`, `'1min'`, `'1h'`, `'1D'`).
        - agg ∈ {'sum', 'mean', 'min', 'max', 'count'}.
        - on: timestamp / date / integer column (single).
        - columns=None → all except `on`; else subset.

        Algorithm: sort by `on` → stream + bucket transition (`(t//freq)*freq`)
        → write output batches (BATCH=1000).

        Output: [on, f'{col}_{agg}'...]. Bucket column dtype = `on` dtype.
        Rows in original order? **No** — output sorted by bucket.

        Semantics (MVP):
        - Empty bucket (gap in ts) → skip (no NaN fill; use `.reindex` after).
        - Null in `on` → skip row.
        - All-null agg bucket → skip row.
        - Null in agg col → skip value.

        Peak RAM: batch + O(#cols) state + 1 output batch. Independent of N
        and bucket size (running accumulator, not `list`).
        """
        # Validate freq
        freq_seconds = _parse_freq(freq)

        # Validate on
        if not isinstance(on, str):
            raise TypeError(
                f"resample: on must be str; got {type(on).__name__}"
            )
        if on not in self._columns:
            raise KeyError(
                f"resample: on column {on!r} not in {self._columns}"
            )
        on_type = self._dataset.schema.field(on).type
        if not (
            pa.types.is_timestamp(on_type)
            or pa.types.is_date(on_type)
            or pa.types.is_integer(on_type)
        ):
            raise TypeError(
                f"resample: on column {on!r} has type {on_type}; "
                f"expected timestamp, date, or integer"
            )
        if pa.types.is_uint64(on_type):
            raise TypeError(
                f"resample: on column {on!r} is uint64 (overflow risk); "
                f"cast to int64 or timestamp"
            )

        # Validate agg
        if agg not in ("sum", "mean", "min", "max", "count"):
            raise ValueError(
                f"resample: agg must be one of sum/mean/min/max/count; "
                f"got {agg!r}"
            )

        # Validate columns
        if columns is None:
            cols = [c for c in self._columns if c != on]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError("resample: columns must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"resample: column {c!r} not in {self._columns}"
                )
            if c == on:
                raise ValueError(
                    f"resample: on column {on!r} cannot be in columns"
                )
            if agg != "count":
                t = self._dataset.schema.field(c).type
                if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                    raise TypeError(
                        f"resample: column {c!r} has type {t}; "
                        f"agg={agg!r} requires numeric"
                    )

        col_types = {c: self._dataset.schema.field(c).type for c in cols}

        out_schema, out_path = _resample_to_disk(
            self._iter_filtered_batches(),
            on=on,
            on_type=on_type,
            freq_seconds=freq_seconds,
            agg=agg,
            columns=cols,
            col_types=col_types,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "resample",
        )
        try:
            new_ds = Dataset.from_batches(
                out_schema,
                ArrowIPCBatchReader(str(out_path), None),
                self._new_dataset_dir(),
            )
        finally:
            out_path.unlink(missing_ok=True)
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    # ---------------- ML: fit_linear ----------------

    def fit_linear(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        alpha: float = 0.0,
    ):
        """Fit Linear (alpha=0) or Ridge (alpha>0) (v0.7.0a).

        One Reduce over filtered batches → solve (X^T X + αI_nonintercept) β.

        Returns: LinearModel (coef_, intercept_, predict(X)).

        See cadw.ml.linear.fit_linear for full docs.
        """
        from cadw.ml.linear import fit_linear
        return fit_linear(
            self, X_cols, y_col, alpha=alpha,
            memory_guard=self._memory_guard,
        )

    # ---------------- ML: fit_logistic ----------------

    def fit_logistic(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        alpha: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        """Fit binary logistic via Newton IRLS (v0.7.0b).

        Each Newton iteration = one Reduce pass. Typically 5-15 iterations.

        Returns LogisticModel (coef_, intercept_, predict, predict_proba).

        See cadw.ml.logistic.fit_logistic for full docs.
        """
        from cadw.ml.logistic import fit_logistic
        return fit_logistic(
            self, X_cols, y_col,
            alpha=alpha, max_iter=max_iter, tol=tol,
            memory_guard=self._memory_guard,
        )

    # ---------------- ML: fit_kmeans ----------------

    def fit_kmeans(
        self,
        X_cols: "str | Sequence[str]",
        k: int,
        *,
        n_init: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        seed: int = 42,
    ):
        """Fit k-means clustering via iter-Reduce (v0.7.1).

        Each iteration = one Reduce pass (assign + accumulate). Peak
        RAM = batch + O(k×d).

        Returns KMeansModel (cluster_centers_, predict).
        """
        from cadw.ml.kmeans import fit_kmeans
        return fit_kmeans(
            self, X_cols, k,
            n_init=n_init, max_iter=max_iter, tol=tol, seed=seed,
            memory_guard=self._memory_guard,
        )

    # ---------------- ML: fit_pca ----------------

    def fit_pca(
        self,
        X_cols: "str | Sequence[str]",
        n_components: "int | None" = None,
    ):
        """Fit PCA via Reduce(CovarianceOp) + eigh (v0.7.2).

        Returns PCA with sklearn-compatible attributes:
            components_ (k, d), mean_, explained_variance_,
            explained_variance_ratio_, singular_values_.

        pca.transform(df) → df with pc0..pc{k-1} columns.
        pca.inverse_transform(df) → reconstructed feature columns.
        """
        from cadw.ml.pca import fit_pca
        return fit_pca(
            self, X_cols, n_components,
            memory_guard=self._memory_guard,
        )

    # ---------------- ML: fit_gbdt_regressor ----------------

    def fit_gbdt_regressor(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        bins: int = 32,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        min_gain: float = 0.0,
    ):
        """Fit GBDT regression (L2 loss, level-wise) — v0.7.3a.

        Peak RAM = batch + O(n × 32B) internal state + O(nodes × feat × bins).
        O(n) state is inherent to Newton-based boosting (XGBoost/LightGBM
        same). Guarded via MemoryGuard.

        Returns GBDTRegressor (trees_, predict).
        """
        from cadw.ml.gbdt import fit_gbdt_regressor
        return fit_gbdt_regressor(
            self, X_cols, y_col,
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, bins=bins,
            min_child_weight=min_child_weight, reg_lambda=reg_lambda,
            min_gain=min_gain,
            memory_guard=self._memory_guard,
        )

    # ---------------- ML: fit_gbdt_classifier ----------------

    def fit_gbdt_classifier(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        bins: int = 32,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        min_gain: float = 0.0,
    ):
        """Fit GBDT binary classification (logloss) — v0.7.5.

        y must be {0, 1}, both classes present.
        Returns GBDTClassifier (predict_proba, predict).
        """
        from cadw.ml.gbdt import fit_gbdt_classifier
        return fit_gbdt_classifier(
            self, X_cols, y_col,
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, bins=bins,
            min_child_weight=min_child_weight, reg_lambda=reg_lambda,
            min_gain=min_gain,
            memory_guard=self._memory_guard,
        )

    # ---------------- ML: fit_naive_bayes ----------------

    def fit_naive_bayes(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        variant: str = "gaussian",
        alpha: float = 1.0,
        var_smoothing: float = 1e-9,
    ):
        """Fit Naive Bayes (v0.7.6).

        - variant='gaussian'    : continuous X; params: var_smoothing.
        - variant='multinomial' : X >= 0 counts; params: alpha.
        - variant='bernoulli'   : X in {0,1}; params: alpha.

        y must be numeric (int/float). For categorical target — encode.
        Returns GaussianNB | MultinomialNB | BernoulliNB.
        """
        from cadw.ml.naive_bayes import (
            _VARIANTS,
            fit_gaussian_nb, fit_multinomial_nb, fit_bernoulli_nb,
        )
        if variant not in _VARIANTS:
            raise ValueError(
                f"fit_naive_bayes: variant must be one of {_VARIANTS}; "
                f"got {variant!r}"
            )
        if variant == "gaussian":
            return fit_gaussian_nb(
                self, X_cols, y_col, var_smoothing=var_smoothing,
                memory_guard=self._memory_guard,
            )
        if variant == "multinomial":
            return fit_multinomial_nb(
                self, X_cols, y_col, alpha=alpha,
                memory_guard=self._memory_guard,
            )
        return fit_bernoulli_nb(
            self, X_cols, y_col, alpha=alpha,
            memory_guard=self._memory_guard,
        )

    def fit_gaussian_nb(
        self, X_cols: "str | Sequence[str]", y_col: str,
        *, var_smoothing: float = 1e-9,
    ):
        from cadw.ml.naive_bayes import fit_gaussian_nb
        return fit_gaussian_nb(
            self, X_cols, y_col, var_smoothing=var_smoothing,
            memory_guard=self._memory_guard,
        )

    def fit_multinomial_nb(
        self, X_cols: "str | Sequence[str]", y_col: str,
        *, alpha: float = 1.0,
    ):
        from cadw.ml.naive_bayes import fit_multinomial_nb
        return fit_multinomial_nb(
            self, X_cols, y_col, alpha=alpha,
            memory_guard=self._memory_guard,
        )

    def fit_bernoulli_nb(
        self, X_cols: "str | Sequence[str]", y_col: str,
        *, alpha: float = 1.0,
    ):
        from cadw.ml.naive_bayes import fit_bernoulli_nb
        return fit_bernoulli_nb(
            self, X_cols, y_col, alpha=alpha,
            memory_guard=self._memory_guard,
        )

    # ---------------- ML: fit_lasso / fit_elasticnet ----------------

    def fit_lasso(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        alpha: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        """Fit Lasso (L1). alpha is raw (additive to X^T X)."""
        from cadw.ml.linear import fit_lasso
        return fit_lasso(
            self, X_cols, y_col,
            alpha=alpha, max_iter=max_iter, tol=tol,
            memory_guard=self._memory_guard,
        )

    def fit_elasticnet(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        """Fit ElasticNet (L1+L2). l1_ratio=1 → Lasso, 0 → Ridge."""
        from cadw.ml.linear import fit_elasticnet
        return fit_elasticnet(
            self, X_cols, y_col,
            alpha=alpha, l1_ratio=l1_ratio,
            max_iter=max_iter, tol=tol,
            memory_guard=self._memory_guard,
        )

    # ---------------- expanding ----------------


    # ---------------- ewm ----------------


    # ---------------- ML: fit_gbdt_multiclass ----------------

    def fit_gbdt_multiclass(
        self,
        X_cols: "str | Sequence[str]",
        y_col: str,
        *,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        bins: int = 32,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        min_gain: float = 0.0,
    ):
        """Fit GBDT multiclass classification (softmax) — v0.7.8.

        y must be {0, 1, ..., K-1}, K >= 3. For binary — fit_gbdt_classifier.
        """
        from cadw.ml.gbdt import fit_gbdt_multiclass
        return fit_gbdt_multiclass(
            self, X_cols, y_col,
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, bins=bins,
            min_child_weight=min_child_weight, reg_lambda=reg_lambda,
            min_gain=min_gain,
            memory_guard=self._memory_guard,
        )

    # ---------------- merge ----------------

    def merge(
        self,
        right: "CadwDataFrame",
        on: str | Sequence[str],
        how: str = "inner",
        *,
        suffixes: tuple[str, str] = ("_x", "_y"),
        num_partitions: int | None = None,
        use_planner: bool = True,
    ) -> "CadwDataFrame":
        """Hash join (v0.9.23j). how ∈ {'inner', 'left',
        'outer', 'anti', 'semi'}.

        - how='inner': совпадающие ключи (left ∪ right, on-coalesced).
        - how='left':  все left + right-колонки (null при нет совпадения).
        - how='outer': left ∪ right, обе стороны null-padded.
        - how='anti':  left без совпадения в right (SQL anti join).
        - how='semi':  left с совпадением в right (SQL semi join);
                       output = left-колонки only.

        NULL-семантика для anti/semi: null-ключ слева никогда
        не совпадает (SQL); anti emit'ит, semi — нет.
        - on: имена общих колонок в обеих сторонах.
        - suffixes: для пересекающихся не-on колонок.
        - num_partitions: число партиций HashShuffle; None — из
          JoinMemoryModel (если use_planner), иначе дефолт 8.
        - use_planner: использовать JoinMemoryModel.solve() для выбора
          num_partitions / max_output_rows. Если
          False — runtime использует старые дефолты (0.3 * available / 256).
        """
        if how not in ("inner", "left", "outer", "anti", "semi"):
            raise NotImplementedError(
                f"merge: how={how!r}; supported: inner, left, "
                f"outer, anti, semi. 'right' — v0.9.23j."
            )
        if not isinstance(right, CadwDataFrame):
            raise TypeError("merge: right must be CadwDataFrame")
        on_list = [on] if isinstance(on, str) else list(on)
        for c in on_list:
            if c not in self.columns:
                raise KeyError(f"merge: on column {c!r} not in self")
            if c not in right.columns:
                raise KeyError(f"merge: on column {c!r} not in right")

        left_schema = pa.schema(
            [self._dataset.schema.field(c) for c in self.columns]
        )
        right_schema = pa.schema(
            [right._dataset.schema.field(c) for c in right.columns]
        )

        # --- Planner: sample first batch, plan, override defaults. ---
        max_output_rows: int | None = None

        if use_planner:
            left_sample = peek_first_batch(self._iter_filtered_batches())
            right_sample = peek_first_batch(right._iter_filtered_batches())

            # Пустая сторона: короткое замыкание (runtime обработает).
            if left_sample is not None and right_sample is not None:
                left_chars = data_chars_for_join(
                    left_sample, self._dataset.total_rows,
                    on_list, self._memory_guard,
                )
                right_chars = data_chars_for_join(
                    right_sample, right._dataset.total_rows,
                    on_list, self._memory_guard,
                )
                # v0.9.23i-fix2: JoinMemoryModel.solve supports only
                # 'inner'|'left'. Map anti/semi/outer -> 'left' for
                # planning (planner build=right, probe=left; same class).
                # Actual execution dispatches to anti_join_streaming /
                # hash_join_streaming(how='left') appropriately.
                _planner_how = (
                    "inner" if how == "inner" else "left"
                )
                params = self._memory_guard.plan(
                    "join", left_chars,
                    op_context={
                        "right_chars": right_chars,
                        "how": _planner_how,
                    },
                )
                if num_partitions is None:
                    num_partitions = params.num_partitions
                max_output_rows = params.max_output_rows

                # v0.8.10-pre (P-A3 addendum): whole-operation RSS WARNING.
                # Per-batch check in _process_partition_pair enforces
                # per-batch memory, but does NOT see Arrow pool init +
                # hash-table build (~40 MB floor, measured across 3 runs
                # @ N=1M: 114.7 / 142.8 / 156.8 MB). Warn if worst-case
                # whole-operation RSS may exceed state_limit. Do not raise
                # (would break existing tests with 20 MB budget — bumping
                # budgets to 100 MB is v0.8.11 work).
                from cadw.ops.join import (
                    _JOIN_FIXED_OVERHEAD_BYTES,
                    _JOIN_STRICT_BUDGET_THRESHOLD_BYTES,
                )
                # v0.9.23i-fix2: predicted_output per how.
                # anti/semi: ≤ left_n (per-probe-batch guard sufficient,
                # whole-op overhead only — 40 MB fixed).
                # outer: left_join (left_n*avg_matches) + anti_right
                # (≤ right_n) rows.
                # inner/left: left_n*avg_matches (unchanged).
                if how in ("anti", "semi"):
                    predicted_output = 0
                elif how == "outer":
                    predicted_output = (
                        left_chars.n_rows
                        * params.avg_matches
                        * params.output_row_size
                        + right_chars.n_rows * params.output_row_size
                    )
                else:
                    predicted_output = (
                        left_chars.n_rows
                        * params.avg_matches
                        * params.output_row_size
                    )
                total_rss_estimate = (
                    _JOIN_FIXED_OVERHEAD_BYTES + 5 * predicted_output
                )
                # v0.8.11b: strict when budget is large enough that the
                # fixed 40 MB overhead is meaningful. Threshold:
                # budget >= 4×overhead (160 MB) AND
                # state_limit >= overhead (available >= 2×overhead).
                # Below: warn only — raise would fire trivially for
                # tiny joins (40 MB overhead >> state_limit at small
                # budgets). See §9.2.6.
                state_limit = self._memory_guard.state_limit_bytes
                budget = self._memory_guard.budget_bytes
                strict_mode = (
                    budget >= _JOIN_STRICT_BUDGET_THRESHOLD_BYTES
                    and state_limit >= _JOIN_FIXED_OVERHEAD_BYTES
                )
                if strict_mode:
                    self._memory_guard.check_state(
                        total_rss_estimate,
                        name="join.whole_operation_rss",
                    )
                elif total_rss_estimate > state_limit:
                    logger.warning(
                        "merge: predicted whole-operation RSS delta "
                        "~%.0f MiB (output %.1f MiB + %.0f MiB Arrow "
                        "pool) exceeds state_limit (%.0f MiB) but "
                        "budget %.0f MiB < %.0f MiB strict threshold. "
                        "Per-batch guards still enforce; actual peak "
                        "may exceed budget. Bump MemoryGuard budget "
                        "to enable strict enforcement. See §9.2.6.",
                        total_rss_estimate / (1024 * 1024),
                        predicted_output / (1024 * 1024),
                        _JOIN_FIXED_OVERHEAD_BYTES / (1024 * 1024),
                        state_limit / (1024 * 1024),
                        budget / (1024 * 1024),
                        _JOIN_STRICT_BUDGET_THRESHOLD_BYTES / (1024 * 1024),
                    )
            else:
                # Одна из сторон пуста — planner не нужен; hash_join
                # короткозамкнёт. num_partitions не важен.
                if num_partitions is None:
                    num_partitions = 8
        else:
            if num_partitions is None:
                num_partitions = 8

        # v0.5.1 (P2): use streaming path — output written to disk
        # batch-by-batch, never accumulated in RAM.
        # v0.9.23i: dispatch outer/anti/semi via dedicated functions.
        # v0.9.23i-fix2: only anti (semi in v0.9.23j; outer via
        # left+anti composition inside dispatch, no direct call).
        from cadw.ops.join import anti_join_streaming as _anti
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
            # v0.9.23j: semi join = left rows with match in right.
            from cadw.ops.join import semi_join_streaming as _semi
            out_schema, out_path = _semi(
                self._iter_filtered_batches(),
                right._iter_filtered_batches(),
                left_schema, right_schema, on=on_list,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir / "join" / "semi",
                num_partitions=num_partitions,
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
            )
        try:
            new_ds = Dataset.from_batches(
                out_schema,
                ArrowIPCBatchReader(str(out_path), None),
                self._new_dataset_dir(),
            )
        finally:
            out_path.unlink(missing_ok=True)
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    # ---------------- metrics (v0.9.2a) ----------------

    def metrics(
        self,
        *,
        y_true: str,
        metrics: "Sequence[str] | str",
        y_pred: "str | None" = None,
        y_proba: "str | Sequence[str] | None" = None,
        y_proba_cols: "Sequence[str] | None" = None,
        classes: "Sequence | None" = None,
        average: "str | None" = None,
        pos_label=None,
    ) -> dict:
        """Regression + log_loss + classification metrics in one pass.

        Metrics:
          - r2, mse, rmse, mae — require `y_pred` (numeric, continuous).
          - log_loss — requires `y_proba` (binary, values in [0,1]).
          - accuracy, precision, recall, f1 — require `y_pred` (labels,
            numeric or string).

        Args:
            y_true: column with true values.
            metrics: metric name(s) — scalar or list.
            y_pred: predicted values (float or labels — depends on
                metric group).
            y_proba: predicted probabilities. str for binary
                (single column, float in [0,1]); list[str] for
                multiclass (K columns, K classes — requires
                `classes`).
            y_proba_cols: alias for `y_proba` when passing a
                list of columns. Both work; use whichever you
                prefer (matches MetricsOp parameter name).
                Passing both → ValueError.
            classes: explicit class list for classification. K=2 default
                when classes=None.
            average: 'binary' | 'macro' | 'micro' | 'weighted' for
                precision/recall/f1. Auto: 'binary' (K=2) / 'macro' (K>2).
            pos_label: positive class for average='binary'. None →
                classes[-1] (K=2 default: [0,1]→1; string classes →
                last class). Explicit value must be in `classes`.

        Returns:
            dict `{metric_name: value}`. NaN for empty input.

        Restrictions:
          - Regression (r2/mse/rmse/mae) and classification
            (accuracy/...) cannot share y_pred in one call.
          - Classification K×24+64 > state_limit → MemoryLimitExceeded.

        State: O(1) + O(K) (classification). Independent of N.
        """
        from cadw.ops.metrics import (
            AUC_METRICS, CLASSIFICATION_METRICS, REGRESSION_METRICS,
            SUPPORTED_METRICS, MetricsOp,
        )
        from cadw.ops.reduce import reduce

        if isinstance(metrics, str):
            metrics = [metrics]
        metrics = list(metrics)
        if not metrics:
            raise ValueError("metrics: metrics list is empty")
        unknown = [m for m in metrics if m not in SUPPORTED_METRICS]
        if unknown:
            raise ValueError(
                f"metrics: unknown metric(s) {unknown}; "
                f"supported: {sorted(SUPPORTED_METRICS)}"
            )

        # v0.9.2c-fix1 (§13.4/167): y_proba is str | Sequence[str];
        # generic loop can't handle list (would check `list not in
        # columns` — always True). Specific per-column check below
        # (after y_proba_cols normalization).
        for c, label in ((y_true, "y_true"), (y_pred, "y_pred")):
            if c is not None and c not in self._columns:
                raise KeyError(
                    f"metrics: {label} column {c!r} not in {self._columns}"
                )

        # §13.4/164: fail-fast for classification K too large — before
        # constructing MetricsOp (clear message, not "state limit").
        ms = set(metrics)
        if (ms & CLASSIFICATION_METRICS) or (ms & AUC_METRICS):
            K = 2 if classes is None else len(classes)
            est = 64 + 24 * K
            state_limit = self._memory_guard.state_limit_bytes
            if est > state_limit:
                from cadw.core.memory_guard import MemoryLimitExceeded
                raise MemoryLimitExceeded(
                    f"metrics: K={K} → state {est} B > state_limit "
                    f"{state_limit} B. Reduce class count or increase "
                    f"budget. §13.4/164."
                )

        # v0.9.23b: y_proba_cols alias (matches MetricsOp param
        # name; AUDIT_FULL_v0922 item 7.A.11.3 / c11 Jupyter
        # gap). Existing `y_proba` (str|Sequence) and new
        # `y_proba_cols` (Sequence[str]) both work; passing
        # both → ValueError (ambiguous); str → TypeError (would
        # iterate char-by-char — footgun).
        if y_proba_cols is not None:
            if y_proba is not None:
                raise ValueError(
                    "metrics: pass either y_proba= or "
                    "y_proba_cols=, not both"
                )
            if isinstance(y_proba_cols, str):
                raise TypeError(
                    "metrics: y_proba_cols must be a "
                    "sequence of column names; got single "
                    f"string {y_proba_cols!r}. Use "
                    "y_proba='col' for single-column "
                    "proba, or y_proba_cols=['col'] for a "
                    "list."
                )
            y_proba = list(y_proba_cols)
            if not y_proba:
                raise ValueError(
                    "metrics: y_proba_cols must be non-empty"
                )

        # v0.9.2c: y_proba normalization (str → [str], list → list).
        if y_proba is None:
            y_proba_cols = None
        elif isinstance(y_proba, str):
            y_proba_cols = [y_proba]
        else:
            y_proba_cols = list(y_proba)
            if not y_proba_cols:
                raise ValueError(
                    "metrics: y_proba list is empty"
                )

        # Column existence for all y_proba cols
        if y_proba_cols is not None:
            for pc_ in y_proba_cols:
                if pc_ not in self._columns:
                    raise KeyError(
                        f"metrics: y_proba column {pc_!r} not in "
                        f"{self._columns}"
                    )

        is_multiclass_proba = (
            y_proba_cols is not None and len(y_proba_cols) > 1
        )
        if is_multiclass_proba:
            if classes is None:
                raise ValueError(
                    f"metrics: multiclass y_proba (len="
                    f"{len(y_proba_cols)}) requires `classes`"
                )
            if len(y_proba_cols) != len(classes):
                raise ValueError(
                    f"metrics: len(y_proba)={len(y_proba_cols)} != "
                    f"len(classes)={len(classes)}"
                )

        # roc_auc: separate handling (multipass Sort-based)
        has_roc_auc = "roc_auc" in metrics
        other_metrics = [m for m in metrics if m != "roc_auc"]

        # Conflict: roc_auc + regression on same y_true (binary labels vs
        # continuous target — mutually exclusive semantics).
        if has_roc_auc and (set(other_metrics) & REGRESSION_METRICS):
            raise ValueError(
                f"metrics: roc_auc requires binary labels y_true, "
                f"regression ({sorted(set(other_metrics) & REGRESSION_METRICS)}) "
                f"requires continuous — cannot mix. Use separate calls."
            )
        if has_roc_auc and y_proba_cols is None:
            raise ValueError("metrics: roc_auc requires y_proba")

        result: dict = {}

        if other_metrics:
            op = MetricsOp(
                y_true,
                y_pred_col=y_pred,
                y_proba_cols=y_proba_cols,
                metrics=other_metrics,
                classes=classes,
                average=average,
                pos_label=pos_label,
            )
            result.update(reduce(
                self._iter_filtered_batches(), op,
                self._memory_guard, log=False,
            ))

        if has_roc_auc:
            from cadw.ops.metrics import (
                _compute_roc_auc_binary, _compute_roc_auc_macro,
            )
            if is_multiclass_proba:
                result["roc_auc"] = _compute_roc_auc_macro(
                    stream_factory=lambda: self._iter_filtered_batches(),
                    y_true_col=y_true,
                    y_proba_cols=y_proba_cols,
                    classes=list(classes),
                    spill_dir=self._spill_dir / "metrics_roc_auc",
                    memory_guard=self._memory_guard,
                )
            else:
                pos_roc = pos_label if pos_label is not None else 1
                result["roc_auc"] = _compute_roc_auc_binary(
                    stream_factory=lambda: self._iter_filtered_batches(),
                    y_true_col=y_true,
                    y_proba_col=y_proba_cols[0],
                    pos_label=pos_roc,
                    spill_dir=self._spill_dir / "metrics_roc_auc",
                    memory_guard=self._memory_guard,
                )

        return result

    # ---------------- concat / nlargest / info ----------------



    def dtypes(self):
        # Return {col: dtype_str}. pandas-compatible shape (dict).
        return {
            c: str(self._dataset.schema.field(c).type)
            for c in self._columns
        }

    def info(self, *, compute_non_null=False):
        # Summary: total_rows, filtered_rows, per-column dtype.
        # compute_non_null=True adds per-column non-null count
        # (O(d) Reduce passes: one pass per column).
        n_total = self._dataset.total_rows
        n_filtered = self.count() if self._row_filters else n_total
        cols = {}
        for c in self._columns:
            entry = {"dtype": str(self._dataset.schema.field(c).type)}
            if compute_non_null:
                from cadw.ops.reduce import reduce, CountOp
                entry["non_null"] = reduce(
                    self._iter_filtered_batches(),
                    CountOp(c),
                    self._memory_guard,
                    log=False,
                )
            cols[c] = entry
        return {
            "total_rows": n_total,
            "filtered_rows": n_filtered,
            "n_columns": len(self._columns),
            "columns": cols,
        }

    def memory_usage(self):
        # Upper-bound estimate: filtered_rows * per-column bit_width
        # (+ 1 byte bitmap if nullable). Uses join._estimate_row_bytes
        # (bit_width//8 + null bitmap for single-field schema).
        from cadw.ops.join import _estimate_row_bytes
        n = self.count()
        per_col = {}
        total = 0
        for c in self._columns:
            f = self._dataset.schema.field(c)
            sz = _estimate_row_bytes(pa.schema([f]))
            per_col[c] = n * sz
            total += per_col[c]
        return {"total": total, "per_column": per_col}


    # ---------------- corr ----------------


    # ---------------- EDA basics (v0.9.23g) ----------------






    # ---------------- sample (v0.9.23f) ----------------


    # ---------------- cumulative (v0.9.23e) ----------------






    # ---------------- write ----------------


    # ---------------- helpers ----------------

    def _wrap_result_table(self, table: pa.Table) -> "CadwDataFrame":
        new_ds = Dataset.from_batches(
            table.schema, iter(table.to_batches()), self._new_dataset_dir()
        )
        # Default RangeIndex(0, total_rows) — результат операций сбрасывает
        # исходный index (как pandas после groupby/sort/merge).
        return CadwDataFrame(
            new_ds, memory_guard=self._memory_guard, spill_dir=self._spill_dir
        )

    def _new_dataset_dir(self) -> Path:
        p = self._spill_dir / "datasets" / f"ds_{uuid.uuid4().hex[:8]}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def __repr__(self) -> str:
        nf = len(self._row_filters)
        return (
            f"CadwDataFrame(rows<={self._dataset.total_rows}, "
            f"columns={self._columns}, filters={nf})"
        )

    def _repr_html_(self) -> str:
        """Jupyter HTML repr — sample 20 rows, schema, shape (upper bound).

        v0.9.0c. Bounded: `.to_pandas().to_html()` on 20-row head only,
        not full df. Whitelisted in AST guard with rationale.
        """
        sample_n = 20
        try:
            sample_table = self.head(sample_n)
            sample_df = sample_table.to_pandas()
            table_html = sample_df.to_html(
                index=False, border=0, classes="cadw-table",
            )
        except Exception as e:
            return (
                f"<pre>CadwDataFrame (repr failed: {e!r})\n"
                f"columns={self._columns}</pre>"
            )

        n_rows_upper = self._dataset.total_rows
        actual = self.count() if self._row_filters else n_rows_upper

        header_parts = [
            f"<strong>CadwDataFrame</strong> "
            f"({actual} rows"
        ]
        if self._row_filters:
            header_parts.append(
                f" of upper bound {n_rows_upper}; {len(self._row_filters)} "
                f"filter(s) applied"
            )
        else:
            header_parts.append("")
        header_parts.append(f", {len(self._columns)} columns)")
        header = "".join(header_parts)

        schema_rows = "".join(
            f"<tr><td>{c}</td><td>{self._dataset.schema.field(c).type}</td></tr>"
            for c in self._columns[:20]
        )
        if len(self._columns) > 20:
            schema_rows += (
                f"<tr><td>... ({len(self._columns) - 20} more)</td>"
                f"<td></td></tr>"
            )
        schema_html = (
            "<details><summary>schema</summary>"
            "<table><tr><th>column</th><th>type</th></tr>"
            f"{schema_rows}</table></details>"
        )

        truncated_note = ""
        if actual > sample_n:
            truncated_note = (
                f"<div style='color:#666;font-size:0.9em;'>"
                f"Showing first {sample_n} rows</div>"
            )

        return (
            f"<div class='cadw-df'>{header}{schema_html}"
            f"{truncated_note}{table_html}</div>"
        )







# ============ v0.9.23g-fix2: empty value_counts helper ============


# ============ v0.9.23f: reservoir sampling helpers ============


def _apply_filter_with_positions(f: RowFilter, it):
    """Apply a RowFilter to (batch, positions) stream (v0.4.0d).

    - MaskFilter: positions[mask] — raw positions of rows that passed.
    - SliceFilter: slice по **локальному** counter (filtered-view position);
      raw positions — из накопленного массива.
    """
    if isinstance(f, MaskFilter):
        for b, positions in it:
            mask = f.mask_for_batch(b)
            mask_np = mask.to_numpy(zero_copy_only=False)
            new_positions = positions[mask_np]
            if len(new_positions) > 0:
                yield b.filter(mask), new_positions
        return
    if isinstance(f, SliceFilter):
        local_pos = 0
        for b, positions in it:
            n = b.num_rows
            batch_local = np.arange(local_pos, local_pos + n, dtype=np.int64)
            mask_np = batch_local >= f.start
            if f.stop is not None:
                mask_np &= batch_local < f.stop
            if f.step > 1:
                mask_np &= ((batch_local - f.start) % f.step == 0)
            new_positions = positions[mask_np]
            if len(new_positions) > 0:
                yield (
                    b.filter(pa.array(mask_np, type=pa.bool_())),
                    new_positions,
                )
            local_pos += n
            if f.stop is not None and local_pos >= f.stop:
                return
        return
    raise TypeError(
        f"_apply_filter_with_positions: unsupported filter "
        f"{type(f).__name__}"
    )



# ============ per-group apply helpers (v0.9.1e) ============

# ============ per-group apply helpers (v0.9.6b2, CADW-native) ============

def _detect_mode(result) -> str:
    """Classify fn() output: 'scalar' / 'dict' / 'dataframe'.

    v0.9.6b2: CADW-native contract. fn returns one of:
      - scalar (int/float/str/bool/None, or numpy scalar): single-value row
      - dict[str, scalar]: multi-value single row (replaces pd.Series)
      - CadwDataFrame: multi-row output (replaces pd.DataFrame)

    No pandas. For pandas-like multi-row output, wrap in
    CadwDataFrame.from_table(pa_table).
    """
    from cadw.dataframe import CadwDataFrame
    import numpy as np

    if isinstance(result, CadwDataFrame):
        return "dataframe"
    if isinstance(result, dict):
        return "dict"
    if result is None:
        return "scalar"
    if isinstance(result, (bool, int, float, str)):
        return "scalar"
    if isinstance(result, (np.integer, np.floating, np.bool_)):
        return "scalar"
    raise TypeError(
        f"_per_group_apply: fn returned {type(result).__name__}; "
        f"expected scalar, dict[str, scalar], or CadwDataFrame "
        f"(v0.9.6b2 CADW-native contract). For multi-metric output "
        f"use a dict; for multi-row output use CadwDataFrame.from_table("
        f"pa_table)."
    )


def _normalize_scalar(v):
    """Convert numpy scalar / NaN to Python scalar / None.

    v0.9.6b2: no pandas. NaN detection via float comparison (`v != v`),
    not pd.isna. pd.NA / pd.NaT not supported (fn should return None).
    """
    import numpy as np

    if v is None:
        return None
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        v = v.item()
    if isinstance(v, float) and v != v:  # NaN
        return None
    return v


def _infer_arrow_type_from_value(v):
    """Map a Python scalar to Arrow type (schema inference).

    v0.9.14-fix3 (§13.4/223): bool check BEFORE int. Python
    bool is int subclass — `isinstance(True, int)` = True,
    so int branch captured True → schema int64. Then
    `pa.Table.from_pylist([{'x': True}], schema=int64)` →
    ArrowTypeError (Expected integer, got bool). Prior:
    F12-6a bool widen branch in _maybe_upgrade_schema was
    DEAD CODE — schema never had pa.bool_ from _infer.
    np.bool_ is not subclass of int/np.integer; for safety
    check both bool and np.bool_ first.
    """
    import pyarrow as pa
    import numpy as np

    if v is None:
        return pa.float64()
    # §13.4/223: bool BEFORE int (bool is int subclass).
    if isinstance(v, (bool, np.bool_)):
        return pa.bool_()
    if isinstance(v, (np.integer, int)):
        return pa.int64()
    if isinstance(v, (np.floating, float)):
        return pa.float64()
    if isinstance(v, str):
        return pa.string()
    return pa.float64()


def _build_out_schema(group_cols, df, mode, first_result, result_col):
    """Output pa.Schema from group_cols + first group's result shape.

    v0.9.6b2: CADW-native. Modes:
      - 'scalar': group_cols + result_col (inferred type)
      - 'dict':   group_cols + first_result keys (inferred types)
      - 'dataframe': group_cols + first_result.columns (skipping
        group_cols if already present in df.columns)
    """
    import pyarrow as pa

    fields = [df._dataset.schema.field(c) for c in group_cols]
    if mode == "scalar":
        v = _normalize_scalar(first_result)
        fields.append(pa.field(
            result_col, _infer_arrow_type_from_value(v), nullable=True,
        ))
    elif mode == "dict":
        for k in first_result:
            v = _normalize_scalar(first_result[k])
            fields.append(pa.field(
                str(k), _infer_arrow_type_from_value(v), nullable=True,
            ))
    else:  # CadwDataFrame
        for c in first_result.columns:
            if c in group_cols:
                continue
            fields.append(first_result._dataset.schema.field(c))
    return pa.schema(fields)


