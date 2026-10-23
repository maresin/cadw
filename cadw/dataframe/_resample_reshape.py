# cadw/dataframe/_resample_reshape.py
# v0.9.26t (B.1 step 14): _ResampleReshapeMixin.
# Symbols: resample, pivot_table, melt, crosstab.
# Extracted verbatim from _core.py; no logic change.

from __future__ import annotations

import pyarrow as pa
from cadw.dataset import Dataset
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.ops.resample import parse_freq as _parse_freq
from cadw.ops.resample import resample_to_disk as _resample_to_disk


class _ResampleReshapeMixin:
    """Resample + reshape mixin (v0.9.26t)."""
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
        from ._core import CadwDataFrame
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
        from ._core import CadwDataFrame
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
        from ._core import CadwDataFrame
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

