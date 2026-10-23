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


class CadwDataFrame(_ExportsMixin, _IteratorsMixin, _MiscMixin, _IndexingMixin, _AggregationsMixin):
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

    def pct_change(self, periods: int = 1, columns=None) -> "CadwDataFrame":
        """Percent change between rows (v0.9.25h).

        Formula: `(x / shift(x, periods)) - 1` per row. Composition
        of `shift(periods)` + `pc.cast` to float64 + `pc.divide` +
        `pc.subtract`.

        - periods: int >= 1. Default 1 (adjacent rows).
        - columns: None → all numeric (int/float). Else str or
          Sequence[str]; non-numeric → TypeError.
        - Output: original + `{col}_pct_change` (float64, nullable).
        - First `periods` rows → null (no previous value).
        - Null in current or shifted → null row (skipna).
        - **Div-by-zero matches pandas**: `pc.divide` uses IEEE 754
          float semantics → `x / 0` (x != 0) = `inf`; `0 / 0` = `nan`.
          Then `inf - 1 = inf`, `nan - 1 = nan`. `pa.array` and Parquet
          preserve inf/nan (IEEE 754 round-trip).
          **CadwDataFrame.pct_change does NOT null-out inf/nan** — user
          who wants null semantics should filter: `df[df.pct.is_finite()]`
          or post-process. Rationale: pandas parity + honesty; changing
          this to null-out-by-default would be a policy decision for
          ALL float ops, not just pct_change (§11.5).
        - Empty filtered view → empty CadwDataFrame with full schema.
        - Original `{col}` columns preserved; new `{col}_pct_change`
          appended.

        Peak RAM = batch × (1 + #cols) × 8 B + _ArrowTail(k × 8 B) per
        col (from shift) — O(batch + k × #cols). Independent of N.
        """
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset

        if not isinstance(periods, int) or periods < 1:
            raise ValueError(
                f"pct_change: periods must be int >= 1; got {periods!r}"
            )
        if columns is None:
            cols = [
                c for c in self._columns
                if pa.types.is_integer(
                    self._dataset.schema.field(c).type
                ) or pa.types.is_floating(
                    self._dataset.schema.field(c).type
                )
            ]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError(
                "pct_change: no numeric columns selected "
                "(pass columns=None for auto int/float, or explicit)"
            )
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"pct_change: column {c!r} not in {self._columns}"
                )
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"pct_change: column {c!r} has type {t}; numeric "
                    f"required"
                )

        shifted = self.shift(periods, columns=cols)

        out_fields = [
            self._dataset.schema.field(c) for c in self._columns
        ]
        for c in cols:
            out_fields.append(
                pa.field(f"{c}_pct_change", pa.float64(), nullable=True)
            )
        out_schema = pa.schema(out_fields)

        batch_size = 10_000

        def _iter():
            orig_it = self._iter_arrow_batches(batch_size=batch_size)
            shifted_it = shifted._iter_arrow_batches(batch_size=batch_size)
            for ob, sb in zip(orig_it, shifted_it):
                if ob.num_rows != sb.num_rows:
                    raise RuntimeError(
                        f"pct_change: batch size mismatch "
                        f"(orig={ob.num_rows}, shifted={sb.num_rows})"
                    )
                arrays = [ob.column(c) for c in self._columns]
                for c in cols:
                    curr_f = _pc.cast(ob.column(c), pa.float64())
                    prev_f = _pc.cast(sb.column(c), pa.float64())
                    # pc.divide uses IEEE 754 (inf for x/0, nan for 0/0)
                    # — matches pandas. Nulls preserved (null/null = null).
                    ratio = _pc.divide(curr_f, prev_f)
                    pct = _pc.subtract(
                        ratio, pa.scalar(1.0, type=pa.float64())
                    )
                    arrays.append(pct)
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

    def ffill(self, columns=None) -> "CadwDataFrame":
        """Forward-fill nulls per column (v0.9.25i).

        Streaming per-batch via pc.fill_null_forward + carry for
        leading-null cross-batch fix. O(#cols) state, O(N) I/O.

        - columns: None → all columns; str → one; Sequence[str] →
          subset. Non-fillable columns pass through unchanged.
        - Leading nulls stay null (no prior value).
        - Null-safe across batches: last non-null carries forward.
        - Original column types preserved (nullable=True forced).
        - Int cols stay int (fill doesn't widen types).

        Peak RAM = batch + #cols × 8 B (carry buffers).
        """
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError("ffill: columns must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"ffill: column {c!r} not in {self._columns}"
                )

        fill_set = set(cols)
        out_fields = []
        for c in self._columns:
            orig = self._dataset.schema.field(c)
            if c in fill_set:
                out_fields.append(
                    pa.field(c, orig.type, nullable=True)
                )
            else:
                out_fields.append(orig)
        out_schema = pa.schema(out_fields)

        carry: dict = {c: None for c in cols}

        def _iter():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="ffill.input")
                n = b.num_rows
                if n == 0:
                    yield b.select(self._columns)
                    continue
                arrays = []
                for c in self._columns:
                    arr = b.column(c)
                    if c not in fill_set:
                        arrays.append(arr)
                        continue
                    car = carry[c]
                    if car is not None:
                        extended = pa.concat_arrays([car, arr])
                        filled = _pc.fill_null_forward(extended).slice(1)
                    else:
                        filled = _pc.fill_null_forward(arr)
                    if len(filled) > 0:
                        last_slice = filled.slice(len(filled) - 1, 1)
                        # v0.9.25i-fix2 (§13.4/269): null_count is
                        # O(1) Arrow metadata — replaces
                        # `is_valid(last_slice).to_pylist()[0]`
                        # (1-element to_pylist still flagged by
                        # diagnostic as low_factor since check_batch
                        # factor<3.0). Semantically identical for
                        # 1-element slice.
                        if last_slice.null_count == 0:
                            carry[c] = last_slice
                    arrays.append(filled)
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

    def bfill(self, columns=None) -> "CadwDataFrame":
        """Backward-fill nulls per column (v0.9.25i).

        Implementation: reverse-sort (__orig_rid descending) +
        ffill-internal + re-sort (__orig_rid ascending). 2×
        external_sort — O(N log N) I/O, O(batch) RAM. **Heavier than
        ffill** (O(N) I/O). Documented.

        - columns: None → all; str → one; Sequence[str] → subset.
        - Trailing nulls stay null (no later value).
        - Original order preserved (2× sort round-trip).
        """
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset
        from cadw.io.batch_reader import ArrowIPCBatchReader
        from cadw.io.batch_writer import ArrowIPCBatchWriter
        from cadw.ops.sort import external_sort

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError("bfill: columns must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"bfill: column {c!r} not in {self._columns}"
                )

        fill_set = set(cols)
        out_fields = []
        for c in self._columns:
            orig = self._dataset.schema.field(c)
            if c in fill_set:
                out_fields.append(
                    pa.field(c, orig.type, nullable=True)
                )
            else:
                out_fields.append(orig)
        out_schema = pa.schema(out_fields)

        if self.count() == 0:
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        # Step 1: add __orig_rid, external_sort descending
        def _iter_with_rid():
            offset = 0
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="bfill.step1")
                n = b.num_rows
                rid = pa.array(
                    range(offset, offset + n), type=pa.uint64(),
                )
                offset += n
                yield b.append_column("__orig_rid", rid)

        sorted_desc = external_sort(
            _iter_with_rid(),
            key_columns=["__orig_rid"],
            value_columns=list(self._columns),
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "bfill" / "sort_desc",
            ascending=False,
            log=False,
        )

        # Step 2: ffill pass
        filled_path = self._spill_dir / "bfill" / "filled.arrow"
        filled_path.parent.mkdir(parents=True, exist_ok=True)
        filled_schema = pa.schema(
            [pa.field("__orig_rid", pa.uint64())]
            + [
                pa.field(c, self._dataset.schema.field(c).type,
                         nullable=True)
                for c in self._columns
            ]
        )
        writer = ArrowIPCBatchWriter(str(filled_path), filled_schema)
        carry: dict = {c: None for c in self._columns}
        try:
            for b in ArrowIPCBatchReader(str(sorted_desc), None):
                self._memory_guard.check_batch(
                    b, name="bfill.step2.read",
                )
                n = b.num_rows
                if n == 0:
                    continue
                arrays = [b.column("__orig_rid")]
                for c in self._columns:
                    arr = b.column(c)
                    # v0.9.25i-fix1 (§13.4/267): only fill columns in
                    # fill_set. Prior code filled ALL columns
                    # regardless of columns= — bug caught by
                    # test_bfill_columns_subset. Mirror-symmetry with
                    # ffill (which correctly has this guard).
                    if c not in fill_set:
                        arrays.append(arr)
                        continue
                    car = carry[c]
                    if car is not None:
                        extended = pa.concat_arrays([car, arr])
                        filled = _pc.fill_null_forward(extended).slice(1)
                    else:
                        filled = _pc.fill_null_forward(arr)
                    if len(filled) > 0:
                        last_slice = filled.slice(len(filled) - 1, 1)
                        # v0.9.25i-fix2 (§13.4/269): null_count O(1).
                        if last_slice.null_count == 0:
                            carry[c] = last_slice
                    arrays.append(filled)
                writer.write_batch(
                    pa.RecordBatch.from_arrays(
                        arrays, schema=filled_schema,
                    )
                )
        finally:
            writer.close()

        # Step 3: external_sort ascending by __orig_rid
        def _iter_filled():
            for b in ArrowIPCBatchReader(str(filled_path), None):
                yield b

        sorted_asc = external_sort(
            _iter_filled(),
            key_columns=["__orig_rid"],
            value_columns=list(self._columns),
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "bfill" / "sort_asc",
            ascending=True,
            log=False,
        )

        def _iter_out():
            for b in ArrowIPCBatchReader(str(sorted_asc), None):
                yield b.select(list(self._columns))

        new_ds = Dataset.from_batches(
            out_schema, _iter_out(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )


    def cut(
        self, column, bins, *, right: bool = True,
        include_lowest: bool = True,
    ):
        """Bin numeric column into discrete intervals (v0.9.25j).

        Args:
            column: numeric column to bin.
            bins: int → k equal-width bins spanning [min, max].
                list[float] → explicit edge values (ascending,
                len >= 2).
            right: True (default) → intervals are (a, b], matching
                pandas `cut(right=True)` default.
            include_lowest: True (default, CADW choice — DIFFERS from
                pandas) → lowest edge is included. Value == min goes
                to bin 0 instead of null. Implemented via
                np.nextafter(min, -inf) as first edge.

        Returns:
            (CadwDataFrame, edges: list[float]): augmented df with
            `{column}_bin` (int32 nullable) appended; edges list.

        Labels: 0..k-1. Out-of-range (below min_edge or above
        max_edge) → null. Null input → null label.

        Peak RAM = batch + O(k). Independent of N.
        """
        import numpy as np
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset

        if not isinstance(right, bool):
            raise TypeError(
                f"cut: right must be bool; got {type(right).__name__}"
            )
        if not isinstance(include_lowest, bool):
            raise TypeError(
                f"cut: include_lowest must be bool; got "
                f"{type(include_lowest).__name__}"
            )
        if column not in self._columns:
            raise KeyError(
                f"cut: column {column!r} not in {self._columns}"
            )
        col_type = self._dataset.schema.field(column).type
        if not (
            pa.types.is_integer(col_type) or pa.types.is_floating(col_type)
        ):
            raise TypeError(
                f"cut: column {column!r} has type {col_type}; "
                f"numeric required"
            )

        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_fields.append(pa.field(f"{column}_bin", pa.int32(), nullable=True))
        out_schema = pa.schema(out_fields)

        if self.count() == 0:
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            ), []

        # Compute edges
        if isinstance(bins, int):
            if bins < 1:
                raise ValueError(
                    f"cut: bins (int) must be >= 1; got {bins}"
                )
            from cadw.ops.reduce import MinMaxOp, reduce
            min_val = reduce(
                self._iter_filtered_batches(),
                MinMaxOp(column, "min"),
                self._memory_guard, log=False,
            )
            max_val = reduce(
                self._iter_filtered_batches(),
                MinMaxOp(column, "max"),
                self._memory_guard, log=False,
            )
            if min_val is None or max_val is None:
                raise ValueError(
                    f"cut: column {column!r} has no non-null values"
                )
            min_f = float(min_val)
            max_f = float(max_val)
            if min_f == max_f:
                edges = [min_f, max_f]
            else:
                width = (max_f - min_f) / bins
                edges = [min_f + i * width for i in range(bins + 1)]
        else:
            edges = [float(b) for b in bins]
            if len(edges) < 2:
                raise ValueError(
                    f"cut: bins list must have >= 2 edges; got "
                    f"{len(edges)}"
                )
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    raise ValueError(
                        f"cut: bins must be strictly ascending; got "
                        f"{edges} at index {i}"
                    )

        edges_np = np.asarray(edges, dtype=np.float64)
        k_bins = len(edges) - 1

        # include_lowest: extend first edge to just below (via
        # np.nextafter) so that value == min goes to bucket 1 (bin 0)
        if include_lowest and len(edges_np) >= 2:
            edges_used = edges_np.copy()
            edges_used[0] = np.nextafter(edges_used[0], -np.inf)
        else:
            edges_used = edges_np

        def _iter():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(
                    b, name="cut.input", extra_factor=3.0,
                )
                arr = b.column(column)
                n = b.num_rows
                if n == 0:
                    yield b.select(self._columns)
                    continue
                null_np = (
                    arr.is_null().to_numpy(zero_copy_only=False)
                    if arr.null_count > 0 else None
                )
                arr_f64 = _pc.cast(arr, pa.float64())
                if arr.null_count > 0:
                    safe = _pc.fill_null(arr_f64, float("-inf"))
                else:
                    safe = arr_f64
                arr_np = safe.to_numpy(zero_copy_only=False)
                labels_np = (
                    np.digitize(arr_np, edges_used, right=right) - 1
                ).astype(np.int32)
                oob = (labels_np < 0) | (labels_np >= k_bins)
                if null_np is not None:
                    oob = oob | null_np
                if oob.any():
                    label_arr = pa.array(
                        labels_np, type=pa.int32(), mask=oob,
                    )
                else:
                    label_arr = pa.array(labels_np, type=pa.int32())
                arrays = [b.column(c) for c in self._columns]
                arrays.append(label_arr)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        ), edges

    def qcut(
        self, column, q, *, right: bool = True,
        include_lowest: bool = True,
    ):
        """Quantile-based binning of a numeric column (v0.9.25j).

        Args:
            column: numeric column to bin.
            q: int k → k equal-frequency bins (quantiles at
                [0, 1/k, ..., 1]). list[float] in (0, 1) → interior
                boundaries; edges = [min] + quantile(q) + [max].
            right: True (default) → intervals are (a, b].
            include_lowest: True (default) → lowest edge included.

        Returns: (CadwDataFrame, edges: list[float]).

        If quantile computation produces tied edges (insufficient
        distinct values in data), edges are deduped; fewer bins than
        requested. Raises ValueError if all values identical.

        Peak RAM = batch + O(N log N) I/O for external_sort
        (quantile). Independent of N for RAM.
        """
        import numpy as np
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset

        if not isinstance(right, bool):
            raise TypeError(
                f"qcut: right must be bool; got {type(right).__name__}"
            )
        if not isinstance(include_lowest, bool):
            raise TypeError(
                f"qcut: include_lowest must be bool; got "
                f"{type(include_lowest).__name__}"
            )
        if column not in self._columns:
            raise KeyError(
                f"qcut: column {column!r} not in {self._columns}"
            )
        col_type = self._dataset.schema.field(column).type
        if not (
            pa.types.is_integer(col_type) or pa.types.is_floating(col_type)
        ):
            raise TypeError(
                f"qcut: column {column!r} has type {col_type}; "
                f"numeric required"
            )

        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_fields.append(pa.field(f"{column}_bin", pa.int32(), nullable=True))
        out_schema = pa.schema(out_fields)

        if self.count() == 0:
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            ), []

        # Build q positions (full range with 0.0 and 1.0)
        if isinstance(q, int):
            if q < 1:
                raise ValueError(
                    f"qcut: q (int) must be >= 1; got {q}"
                )
            q_positions = [i / q for i in range(q + 1)]
        elif isinstance(q, (list, tuple)):
            q_list = [float(v) for v in q]
            if not q_list:
                raise ValueError("qcut: q list must be non-empty")
            for qi in q_list:
                if not (0.0 <= qi <= 1.0):
                    raise ValueError(
                        f"qcut: q positions must be in [0, 1]; "
                        f"got {qi}"
                    )
            q_positions = [0.0] + q_list + [1.0]
        else:
            raise TypeError(
                f"qcut: q must be int or list[float]; got "
                f"{type(q).__name__}"
            )

        # Compute edges via quantile (uses external_sort for N>batch)
        edges = list(self.quantile(column, q_positions))
        # Handle NaN (all-null column)
        for e in edges:
            if e != e:  # NaN check
                raise ValueError(
                    f"qcut: column {column!r} has no non-null values"
                )

        # Dedupe tied edges (§13.4: quantile can tie on low-card data)
        unique_edges = sorted(set(edges))
        if len(unique_edges) < 2:
            _first_or_empty = (
                unique_edges[0] if unique_edges else "empty"
            )
            raise ValueError(
                f"qcut: all values identical ({_first_or_empty}); "
                f"cannot create bins"
            )
        if len(unique_edges) < len(edges):
            # Some quantile positions tied; use unique edges
            import logging
            logging.getLogger("cadw.dataframe").warning(
                "qcut: quantile edges contained ties (insufficient "
                "distinct values); %d unique edges instead of %d",
                len(unique_edges), len(edges),
            )
            edges = unique_edges

        edges_np = np.asarray(edges, dtype=np.float64)
        k_bins = len(edges) - 1

        if include_lowest and len(edges_np) >= 2:
            edges_used = edges_np.copy()
            edges_used[0] = np.nextafter(edges_used[0], -np.inf)
        else:
            edges_used = edges_np

        def _iter():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(
                    b, name="qcut.input", extra_factor=3.0,
                )
                arr = b.column(column)
                n = b.num_rows
                if n == 0:
                    yield b.select(self._columns)
                    continue
                null_np = (
                    arr.is_null().to_numpy(zero_copy_only=False)
                    if arr.null_count > 0 else None
                )
                arr_f64 = _pc.cast(arr, pa.float64())
                if arr.null_count > 0:
                    safe = _pc.fill_null(arr_f64, float("-inf"))
                else:
                    safe = arr_f64
                arr_np = safe.to_numpy(zero_copy_only=False)
                labels_np = (
                    np.digitize(arr_np, edges_used, right=right) - 1
                ).astype(np.int32)
                oob = (labels_np < 0) | (labels_np >= k_bins)
                if null_np is not None:
                    oob = oob | null_np
                if oob.any():
                    label_arr = pa.array(
                        labels_np, type=pa.int32(), mask=oob,
                    )
                else:
                    label_arr = pa.array(labels_np, type=pa.int32())
                arrays = [b.column(c) for c in self._columns]
                arrays.append(label_arr)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        ), edges

    def duplicated(self, subset=None, keep="first") -> "CadwDataFrame":
        """Mark duplicate rows (v0.9.25j).

        Returns augmented CadwDataFrame: original columns + `duplicated`
        bool column (nullable=True). Row order = current filtered
        view order.

        - subset: None → all columns. str → one. Sequence[str] → subset.
        - keep: 'first' (default) → first occurrence per key is
          False (not duplicate); rest True.
          'last' → last occurrence False; earlier True.
          False → all rows in duplicate groups (count>1) True;
          unique-key rows False.

        NULL in subset: null == null for grouping (own group).

        Implementation: (1) HashShuffle + dedup on (subset, __rid)
        to find kept row_ids; (2) per-batch mask via
        pc.is_in(rid, kept_set); mask = NOT is_in.

        Peak RAM = partition_size (bounded by existing shuffle guard)
        + O(K_kept × 8 × 2) hash set for kept_rids. K_kept ≤ N_unique
        keys; guard fails fast if exceeds state_limit.

        Use to filter df: `df2 = df.duplicated(subset=["k"]);
        filtered = df2[df2["duplicated"] == True]`.
        Or extract mask: `df.duplicated()["duplicated"].to_list()`.
        """
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset
        from cadw.ops.dedup import drop_duplicates as _ops_dedup

        if keep not in ("first", "last", False):
            raise ValueError(
                f"duplicated: keep must be 'first', 'last', or False; "
                f"got {keep!r}"
            )
        if subset is None:
            subset_cols = list(self._columns)
        elif isinstance(subset, str):
            subset_cols = [subset]
        else:
            subset_cols = list(subset)
        if not subset_cols:
            raise ValueError("duplicated: subset must be non-empty")
        for c in subset_cols:
            if c not in self._columns:
                raise KeyError(
                    f"duplicated: subset column {c!r} not in "
                    f"{self._columns}"
                )

        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_fields.append(pa.field("duplicated", pa.bool_(), nullable=True))
        out_schema = pa.schema(out_fields)

        if self.count() == 0:
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        # Pass 1: HashShuffle + dedup on (subset, __rid) → kept __rids
        def _subset_with_rid():
            for b, positions in self._iter_filtered_batches_with_positions():
                self._memory_guard.check_batch(
                    b, name="duplicated.shuffle", extra_factor=2.0,
                )
                sub_b = b.select(subset_cols)
                rid = pa.array(positions, type=pa.uint64())
                yield sub_b.append_column("__rid", rid)

        kept_table = _ops_dedup(
            _subset_with_rid(),
            subset=subset_cols,
            keep=keep,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "duplicated",
            total_rows=self._dataset.total_rows,
            log=False,
        )
        kept_rids = kept_table.column("__rid").combine_chunks()

        # Guard: kept_rids hash set must fit budget (~2x raw bytes)
        kept_est = max(1, kept_rids.nbytes) * 2
        self._memory_guard.check_state(
            kept_est,
            name=f"duplicated.kept_set(K_kept={kept_table.num_rows})",
        )

        # Pass 2: mask = NOT is_in(rid, kept_rids) per batch
        def _iter_augmented():
            for b, positions in self._iter_filtered_batches_with_positions():
                self._memory_guard.check_batch(
                    b, name="duplicated.mask", extra_factor=2.5,
                )
                rid = pa.array(positions, type=pa.uint64())
                is_kept = _pc.is_in(rid, value_set=kept_rids)
                dup = _pc.invert(is_kept)
                arrays = [b.column(c) for c in self._columns]
                arrays.append(dup)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_augmented(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

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


    def astype(
        self,
        dtype_map: dict,
        *,
        safe: bool = True,
    ) -> "CadwDataFrame":
        """Cast columns to new dtypes. Streaming, no materialization.

        - dtype_map: {col: pa.DataType | str-alias}
        - safe=True: raise on overflow (default). safe=False: wrap/truncate.
        """
        import pyarrow.compute as _pc

        resolved: dict[str, pa.DataType] = {}
        for c, dt in dtype_map.items():
            if c not in self._columns:
                raise KeyError(
                    f"astype: column {c!r} not in {self._columns}"
                )
            if isinstance(dt, pa.DataType):
                resolved[c] = dt
            elif isinstance(dt, str):
                try:
                    resolved[c] = pa.type_for_alias(dt)
                except (ValueError, KeyError) as e:
                    raise ValueError(
                        f"astype: unknown dtype alias {dt!r}: {e}"
                    ) from e
            else:
                raise TypeError(
                    f"astype: dtype for {c!r} must be str or pa.DataType; "
                    f"got {type(dt).__name__}"
                )

        out_fields = []
        for c in self._columns:
            if c in resolved:
                out_fields.append(pa.field(c, resolved[c], nullable=True))
            else:
                out_fields.append(self._dataset.schema.field(c))
        out_schema = pa.schema(out_fields)

        def _iter():
            for b in self._iter_arrow_batches():
                self._memory_guard.check_batch(b, name="astype")
                arrays = []
                for c in self._columns:
                    if c in resolved:
                        arrays.append(_pc.cast(
                            b.column(c), resolved[c], safe=safe,
                        ))
                    else:
                        arrays.append(b.column(c))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def dropna(
        self,
        how: str = "any",
        subset: "Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Drop rows with nulls (lazy — composes notnull() filters).

        - how='any': drop if ANY subset column is null.
        - how='all': drop only if ALL subset columns are null.
        - subset=None: all columns.
        """
        if how not in ("any", "all"):
            raise ValueError(
                f"dropna: how must be 'any' or 'all'; got {how!r}"
            )
        cols = list(subset) if subset is not None else list(self._columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"dropna: subset column {c!r} not in {self._columns}"
                )
        if not cols:
            return self

        combined = self[cols[0]].notnull()
        for c in cols[1:]:
            nf = self[c].notnull()
            if how == "any":
                combined = combined & nf
            else:
                combined = combined | nf
        return self[combined]

    def drop(self, cols) -> "CadwDataFrame":
        """Drop columns (selection minus). Lazy."""
        if isinstance(cols, str):
            cols = [cols]
        else:
            cols = list(cols)
        for c in cols:
            if c not in self._columns:
                raise KeyError(f"drop: column {c!r} not in {self._columns}")
        keep = [c for c in self._columns if c not in cols]
        return self[keep]

    def rename(self, mapping: dict[str, str]) -> "CadwDataFrame":
        """Rename columns. Materializes new Dataset (schema change).

        - mapping: {old: new}. All `old` must exist; `new` must not collide
          with an existing column (unless it's the column being renamed).
        """
        if not isinstance(mapping, dict):
            raise TypeError(
                f"rename: mapping must be dict; got {type(mapping).__name__}"
            )
        for old, new in mapping.items():
            if old not in self._columns:
                raise KeyError(
                    f"rename: column {old!r} not in {self._columns}"
                )
            if new != old and new in self._columns:
                raise ValueError(
                    f"rename: target {new!r} collides with existing column"
                )
        new_names = [mapping.get(c, c) for c in self._columns]
        new_fields = []
        for c in self._columns:
            orig = self._dataset.schema.field(c)
            new_fields.append(
                pa.field(mapping.get(c, c), orig.type, nullable=orig.nullable)
            )
        new_schema = pa.schema(new_fields)

        def _iter():
            for b in self._iter_arrow_batches():
                yield b.rename_columns(new_names)

        new_ds = Dataset.from_batches(
            new_schema, _iter(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

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

    def drop_duplicates(
        self,
        subset: str | Sequence[str] | None = None,
        keep: str | bool = "first",
        *,
        num_partitions: int | None = None,
    ) -> "CadwDataFrame":
        """Удалить дубликаты строк (v0.3.0).

        - subset=None: все колонки — как pandas.
        - subset=str | Sequence[str]: ключ(и) для определения дубликата.
        - keep: 'first' | 'last' | False (drop all duplicates).
        - num_partitions: None — auto из sample (по k_est на subset).

        **Порядок результата**: partition-concatenation, **не** pandas-order.
        Если нужен стабильный порядок — `.sort_values(...)` после. §11.5.

        **Skew**: при oversized partition — `MemoryLimitExceeded`;
        recursion не поддерживается в v0.3.0. Увеличьте `num_partitions`.
        """
        if keep not in ("first", "last", False):
            raise ValueError(
                f"drop_duplicates: keep must be 'first', 'last', or False; "
                f"got {keep!r}"
            )
        if subset is None:
            subset_cols = list(self._columns)
        elif isinstance(subset, str):
            subset_cols = [subset]
        else:
            subset_cols = list(subset)
        if not subset_cols:
            raise ValueError("drop_duplicates: subset must be non-empty")
        for c in subset_cols:
            if c not in self._columns:
                raise KeyError(
                    f"drop_duplicates: subset column {c!r} not in "
                    f"columns {self._columns}"
                )
        # subset=None означает dedup по всем; subset явный — dedup по списку,
        # но output сохраняет все колонки. Проверим совпадение с columns
        # для случая subset=None.
        if subset is None:
            subset_cols = list(self._columns)

        result = _ops_drop_duplicates(
            self._iter_filtered_batches(),
            subset=subset_cols,
            keep=keep,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "dedup",
            total_rows=self._dataset.total_rows,
            num_partitions=num_partitions,
            log=False,
        )
        return self._wrap_result_table(result)

    # ---------------- rank ----------------

    def rank(
        self,
        by: str,
        method: str = "first",
        ascending: bool = True,
        pct: bool = False,
        *,
        output_col: str | None = None,
        batch_size: int = 10_000,
    ) -> "CadwDataFrame":
        """Ранг по колонке `by` (v0.3.1).

        - method ∈ {'first', 'min', 'dense', 'max', 'last', 'average'}:
            - 'first':   1-based позиция в sorted order (ties по row_id asc).
            - 'min':     минимум из рангов в tie-группе.
            - 'dense':   1, 2, 3, ... для уникальных значений.
            - 'max':     максимум из рангов в tie-группе.
            - 'last':    reverse of 'first' (CADW extension, НЕ в pandas).
            - 'average': (min + max) / 2 (float).
        - ascending: True — меньшие значения получают меньший ранг.
        - pct: делить rank на total_n.
        - output_col: имя новой колонки (default `{by}_rank`).

        Возвращает новый CadwDataFrame со всеми исходными колонками +
        rank в **исходном порядке** строк.

        Null values участвуют как значения (null == null). pandas skipna
        не поддерживается в v0.3.1.

        Методы 'max'/'last'/'average' реализованы двухпроходным
        streaming (O(1) RAM): pass 1 → group file, pass 2 → join.
        """
        if method not in VALID_METHODS:
            raise NotImplementedError(
                f"rank: method={method!r} not supported. "
                f"Supported: {VALID_METHODS}."
            )
        if by not in self._columns:
            raise KeyError(f"rank: by={by!r} not in columns {self._columns}")
        if output_col is None:
            output_col = f"{by}_rank"
        if output_col in self._columns:
            raise KeyError(
                f"rank: output_col={output_col!r} already in columns"
            )

        # Count filtered rows (needed for pct, and sanity).
        total_n = self.count()
        if total_n == 0:
            # Empty: append an empty rank column of proper dtype.
            rank_dtype = pa.float64() if (pct or method == "average") else pa.int64()
            out_fields = [
                self._dataset.schema.field(c) for c in self._columns
            ]
            out_fields.append(pa.field(output_col, rank_dtype))
            out_schema = pa.schema(out_fields)

            def _iter_empty():
                for b in self._iter_arrow_batches(batch_size=batch_size):
                    n = b.num_rows
                    if pct:
                        col = pa.array([], type=pa.float64())
                    else:
                        col = pa.array([], type=pa.int64())
                    yield b.append_column(output_col, col)

            new_ds = Dataset.from_batches(
                out_schema, _iter_empty(), self._new_dataset_dir()
            )
            return CadwDataFrame(
                new_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        spill = self._spill_dir / "rank" / f"r_{uuid.uuid4().hex[:8]}"
        spill.mkdir(parents=True, exist_ok=True)

        # Step 1: sort by `by`.
        sorted_by_val = external_sort(
            self._iter_filtered_batches(),
            key_columns=[by],
            value_columns=[by],
            memory_guard=self._memory_guard,
            spill_dir=spill / "sort_by_val",
            ascending=ascending,
            log=False,
        )

        # Step 2: rank per row → (row_id, rank) on disk.
        # compute_ranks_to_disk принимает Path (не iterable) — bufferable
        # методы требуют двух проходов по sorted_by.
        from cadw.io.batch_reader import ArrowIPCBatchReader as _IPCReader
        rank_pairs_path = compute_ranks_to_disk(
            sorted_by_val,
            by=by,
            output_col=output_col,
            method=method,
            pct=pct,
            total_n=total_n,
            rank_path=spill / "ranks.arrow",
            memory_guard=self._memory_guard,
            log=False,
        )

        # Step 3: sort rank pairs by original row_id.
        def _iter_rank_pairs():
            for b in _IPCReader(str(rank_pairs_path), None):
                # columns: row_id, output_col → rename row_id to __orig_rid
                yield b.rename_columns(["__orig_rid", output_col])

        sorted_by_rowid = external_sort(
            _iter_rank_pairs(),
            key_columns=["__orig_rid"],
            value_columns=[output_col],
            memory_guard=self._memory_guard,
            spill_dir=spill / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        # Step 4: build rank Dataset (in original order via __orig_rid).
        rank_dtype = pa.float64() if (pct or method == "average") else pa.int64()
        rank_schema = pa.schema([
            ("__orig_rid", pa.uint64()),
            (output_col, rank_dtype),
        ])

        def _iter_rank_after_sort():
            for b in _IPCReader(str(sorted_by_rowid), ["__orig_rid", output_col]):
                yield b

        rank_ds = Dataset.from_batches(
            rank_schema, _iter_rank_after_sort(), spill / "rank_ds"
        )

        # Step 5: zip with original (both in original order; same batch_size).
        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_fields.append(pa.field(output_col, rank_dtype))
        out_schema = pa.schema(out_fields)

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            rank_iter = rank_ds.iter_batches(
                batch_size=batch_size, columns=[output_col]
            )
            for ob, rb in zip(orig_iter, rank_iter):
                if ob.num_rows != rb.num_rows:
                    raise RuntimeError(
                        f"rank: batch-size mismatch: orig={ob.num_rows} "
                        f"rank={rb.num_rows}"
                    )
                yield ob.append_column(output_col, rb.column(output_col))

        new_ds = Dataset.from_batches(
            out_schema, _iter_zipped(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

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

    def shift(
        self,
        k: int = 1,
        fill_value=None,
        *,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Shift values by `k` positions (v0.6.2b, lag; v0.8.6 multi-on).

        - k >= 0: `out[i] = x[i-k]` for i>=k, else `fill_value`.
        - k == 0: returns self (no-op).
        - k < 0 (lead): NotImplementedError (v0.6.3).
        - columns=None: all columns; else subset (others pass-through).
        - fill_value=None: null for first k rows. Non-None: that value.
        - on: positional shift if None; sort-aware if str/Sequence.
          `on=["ts", "sensor"]` — lexicographic composite sort
          (NOT per-group; for per-group see v0.8.5a).

        Streamable: O(k) memory (deque per column). Position-based,
        iterates filtered batches in row order.

        Shifted columns become **nullable** (fill = null).
        """
        from collections import deque

        if not isinstance(k, int):
            raise TypeError(f"shift: k must be int; got {type(k).__name__}")
        if k < 0:
            raise NotImplementedError(
                "shift: negative k (lead) not supported in v0.6.2b; "
                "use reverse + shift + reverse, or v0.6.3"
            )
        if k == 0:
            return self

        # v0.8.2: sort-aware shift; v0.8.6: multi-column on.
        if on is not None:
            on_cols = self._normalize_on(on, fn_name="shift")
            return self._shift_on(
                k, fill_value, columns=columns, on_cols=on_cols,
            )

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

        def _fill(size: int, type_) -> pa.Array:
            if fill_value is None:
                return pa.nulls(size, type=type_)
            return pa.array([fill_value] * size, type=type_)

        def _iter_shifted():
            # v0.9.18 (AUDIT §1.1): Arrow-native tail buffer.
            # Prior: `deque(maxlen=k)` of Python scalars +
            # `col_vals.to_pylist()` per batch (O(batch_size * 28B) peak).
            # Now: `_ArrowTail` (deque of Arrow slices, O(k) peak).
            # Peak 2.05 MB -> <400 KB for 300k rows / k=2 (proven).
            bufs: dict[str, _ArrowTail] = {
                c: _ArrowTail(k) for c in shift_cols
            }
            for b in self._iter_filtered_batches():
                n = b.num_rows
                arrays = []
                for c in self._columns:
                    col_vals = b.column(c)
                    if c not in shift_cols:
                        arrays.append(col_vals)
                        continue
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
                            out, pa.nulls(n - len(out), type=col_vals.type),
                        ])
                    arrays.append(out)
                    bufs[c].push(col_vals)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_shifted(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def diff(
        self,
        k: int = 1,
        *,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """First difference: `out[i] = x[i] - x[i-k]` (v0.6.2b; multi-on v0.8.6).

        - k >= 1. First k rows -> null.
        - columns=None: all; else subset (others pass-through).
        - on: positional if None; sort-aware if str/Sequence
          (lexicographic composite sort).
        - Output columns nullable (first k = null).

        Streamable: O(k) memory.
        """
        from collections import deque

        if not isinstance(k, int):
            raise TypeError(f"diff: k must be int; got {type(k).__name__}")
        if k < 1:
            raise ValueError(f"diff: k must be >= 1; got {k}")

        # v0.8.2: sort-aware diff = shift(on) - original.
        if on is not None:
            return self._diff_on(k=k, columns=columns, on=on)

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
        diff_cols = set(cols)

        out_fields = []
        for c in self._columns:
            f = self._dataset.schema.field(c)
            if c in diff_cols:
                f = pa.field(c, f.type, nullable=True)
            out_fields.append(f)
        out_schema = pa.schema(out_fields)

        import pyarrow.compute as _pc

        def _iter_diffed():
            bufs: dict[str, deque] = {
                c: deque(maxlen=k) for c in diff_cols
            }
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="diff.input", extra_factor=3.5)
                n = b.num_rows
                arrays = []
                for c in self._columns:
                    col_vals = b.column(c)
                    if c not in diff_cols:
                        arrays.append(col_vals)
                        continue
                    prev = list(bufs[c])
                    fill_n = max(0, k - len(prev))
                    parts = []
                    if fill_n > 0:
                        parts.append(pa.nulls(fill_n, type=col_vals.type))
                    if prev:
                        parts.append(pa.array(prev, type=col_vals.type))
                    if n > k:
                        parts.append(col_vals.slice(0, n - k))
                    shifted = (
                        pa.concat_arrays(parts) if parts
                        else pa.array([], type=col_vals.type)
                    )
                    if len(shifted) > n:
                        shifted = shifted.slice(0, n)
                    elif len(shifted) < n:
                        shifted = pa.concat_arrays([
                            shifted,
                            pa.nulls(n - len(shifted), type=col_vals.type),
                        ])
                    diff_arr = _pc.subtract(col_vals, shifted)
                    arrays.append(diff_arr)
                    for v in col_vals.to_pylist():
                        bufs[c].append(v)
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_diffed(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

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

    def rolling(
        self,
        window: int,
        agg: str,
        *,
        min_periods: "int | None" = None,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Rolling window aggregation (v0.6.2c; sort-aware v0.8.4; multi v0.8.6).

        - window: int >= 1.
        - agg: {'sum', 'mean', 'min', 'max', 'count'}.
        - min_periods=None -> window.
        - columns=None -> all numeric columns.
        - on: positional if None; sort-aware if str/Sequence
          (`on=["ts", "sensor"]` — lexicographic composite sort, NOT
          per-group; for per-group see v0.8.5a).

        Output: original + `{col}_rolling_{agg}` (APPEND).
        Nulls skip min/max/sum/mean but **consume window slot** (positional,
        matches pandas skipna=True). Peak O(window × #cols).
        """
        from collections import deque

        if not isinstance(window, int):
            raise TypeError(
                f"rolling: window must be int; got {type(window).__name__}"
            )
        if window < 1:
            raise ValueError(f"rolling: window must be >= 1; got {window}")
        if agg not in {"sum", "mean", "min", "max", "count"}:
            raise ValueError(
                f"rolling: agg must be one of sum/mean/min/max/count; "
                f"got {agg!r}"
            )
        if min_periods is None:
            mp = window
        elif not isinstance(min_periods, int):
            raise TypeError(
                f"rolling: min_periods must be int; "
                f"got {type(min_periods).__name__}"
            )
        elif not (1 <= min_periods <= window):
            raise ValueError(
                f"rolling: min_periods must be in [1, window={window}]; "
                f"got {min_periods}"
            )
        else:
            mp = min_periods

        # v0.8.4: sort-aware; v0.8.6: multi-column on.
        if on is not None:
            on_cols = self._normalize_on(on, fn_name="rolling")
            return self._rolling_on(
                window=window, agg=agg, min_periods=mp,
                columns=columns, on_cols=on_cols,
            )

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"rolling: column {c!r} not in {self._columns}"
                )

        # Validate numeric for sum/mean/min/max (count works for any type).
        for c in cols:
            t = self._dataset.schema.field(c).type
            if agg != "count" and not (
                pa.types.is_integer(t) or pa.types.is_floating(t)
            ):
                raise TypeError(
                    f"rolling: column {c!r} has type {t}; "
                    f"agg={agg!r} requires numeric"
                )

        # Output schema: original cols + rolling output per (col, agg).
        out_fields = []
        for c in self._columns:
            out_fields.append(self._dataset.schema.field(c))
        out_type_by_col: dict[str, pa.DataType] = {}
        for c in cols:
            in_type = self._dataset.schema.field(c).type
            if agg in ("sum", "mean"):
                ot = pa.float64()
            elif agg == "count":
                ot = pa.int64()
            else:  # min/max
                ot = in_type
            out_name = f"{c}_rolling_{agg}"
            out_type_by_col[c] = ot
            out_fields.append(pa.field(out_name, ot, nullable=True))
        out_schema = pa.schema(out_fields)

        def _iter_rolling():
            states = {c: _RollState(window, mp, agg) for c in cols}
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="rolling.input", extra_factor=3.5)
                n = b.num_rows
                out_arrays = []
                for c in self._columns:
                    out_arrays.append(b.column(c))
                for c in cols:
                    state = states[c]
                    src_vals = b.column(c).to_pylist()
                    rolling_vals = []
                    for v in src_vals:
                        rv = state.push(v)
                        rolling_vals.append(rv)
                    out_arrays.append(
                        pa.array(rolling_vals, type=out_type_by_col[c])
                    )
                yield pa.RecordBatch.from_arrays(
                    out_arrays, schema=out_schema
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_rolling(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def _rolling_on(
        self,
        *,
        window: int,
        agg: str,
        min_periods: int,
        columns: "str | Sequence[str] | None",
        on_cols: list[str],
    ) -> "CadwDataFrame":
        """Sort-aware rolling on `on_cols` (v0.8.4; multi v0.8.6).

        Validated by prototype (scripts/proto_rolling_on.py):
          A (sort_values+rolling+sort_values) ==
          B (external_sort + _RollState + external_sort) ==
          C (pure Python deque).

        Algorithm (identical to v0.8.4, multi-key `on_cols`):
          1. external_sort by [on_cols...] (row_id tie-break from heap).
          2. _RollState.push per row (per column) → IPC
             (row_id, {col}_rolling_{agg}).
          3. external_sort by __orig_rid (§13.4/122).
          4. Zip original + rolling cols (APPEND).

        Peak RAM: batch + O(window × #cols). Independent of N.
        `on_cols` pre-normalized by caller (v0.8.6).
        """
        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"rolling: column {c!r} not in {self._columns}"
                )

        # Validate numeric (agg != count requires numeric; same v0.6.2c)
        for c in cols:
            t = self._dataset.schema.field(c).type
            if agg != "count" and not (
                pa.types.is_integer(t) or pa.types.is_floating(t)
            ):
                raise TypeError(
                    f"rolling: column {c!r} has type {t}; "
                    f"agg={agg!r} requires numeric"
                )

        # Output schema: original + {col}_rolling_{agg} (APPEND semantics).
        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_type_by_col: dict[str, tuple[str, pa.DataType]] = {}
        for c in cols:
            in_type = self._dataset.schema.field(c).type
            if agg in ("sum", "mean"):
                ot = pa.float64()
            elif agg == "count":
                ot = pa.int64()
            else:  # min/max
                ot = in_type
            out_name = f"{c}_rolling_{agg}"
            out_type_by_col[c] = (out_name, ot)
            out_fields.append(pa.field(out_name, ot, nullable=True))
        out_schema = pa.schema(out_fields)
        rolling_out_cols = [out_type_by_col[c][0] for c in cols]

        # 1. Sort by `on` (row_id tie-break from external_sort's heap).
        sorted_path = external_sort(
            self._iter_filtered_batches(),
            key_columns=on_cols,
            value_columns=list(self._columns),
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "rolling" / "sort_by_on",
            ascending=True,
            log=False,
        )

        # 2. Roll on sorted stream, write (row_id, {col}_rolling_{agg}).
        rolling_schema_fields: list[pa.Field] = [
            pa.field("row_id", pa.uint64()),
        ]
        for c in cols:
            out_name, ot = out_type_by_col[c]
            rolling_schema_fields.append(
                pa.field(out_name, ot, nullable=True)
            )
        rolling_schema = pa.schema(rolling_schema_fields)

        rolling_path = self._spill_dir / "rolling" / "rolling_by_on.arrow"
        rolling_path.parent.mkdir(parents=True, exist_ok=True)

        writer = ArrowIPCBatchWriter(str(rolling_path), rolling_schema)
        states = {c: _RollState(window, min_periods, agg) for c in cols}
        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                self._memory_guard.check_batch(
                    b, name="rolling_on.sorted_input"
                , extra_factor=7.0)
                rid_list = b.column("row_id").to_pylist()
                arrays: list[pa.Array] = [
                    pa.array(rid_list, type=pa.uint64()),
                ]
                for c in cols:
                    state = states[c]
                    src_vals = b.column(c).to_pylist()
                    rolled = [state.push(v) for v in src_vals]
                    out_name, ot = out_type_by_col[c]
                    arrays.append(pa.array(rolled, type=ot))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(
                        arrays, schema=rolling_schema,
                    )
                )
        finally:
            writer.close()

        # 3. Sort back by __orig_rid. external_sort appends its own
        # "row_id"; input already has "row_id" (original pos) → duplicate
        # → ArrowInvalid. Rename to "__orig_rid" (§13.4/122).
        def _iter_rolled():
            for b in ArrowIPCBatchReader(str(rolling_path), None):
                names = b.schema.names
                rid_idx = names.index("row_id")
                new_names = (
                    ["__orig_rid"]
                    + [n for i, n in enumerate(names) if i != rid_idx]
                )
                yield b.rename_columns(new_names)

        sorted_rowid_path = external_sort(
            _iter_rolled(),
            key_columns=["__orig_rid"],
            value_columns=rolling_out_cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "rolling" / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        # 4. Zip original + rolling cols (APPEND; rolling doesn't replace).
        batch_size = 10_000

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            rolled_reader = ArrowIPCBatchReader(
                str(sorted_rowid_path), rolling_out_cols,
            )
            accum: list[pa.RecordBatch] = []
            accum_n = 0
            rolled_gen = iter(rolled_reader)
            for ob in orig_iter:
                target_n = ob.num_rows
                while accum_n < target_n:
                    try:
                        nb = next(rolled_gen)
                    except StopIteration:
                        raise RuntimeError(
                            "rolling: rolled stream exhausted before "
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
                take_batch = take.combine_chunks().to_batches()[0]
                arrays = [ob.column(c) for c in self._columns]
                for out_name in rolling_out_cols:
                    arrays.append(take_batch.column(out_name))
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

    def _per_group_cumulative(
        self,
        group_cols: list[str],
        on_cols: list[str],
        op: str,
        columns,
    ) -> "CadwDataFrame":
        """Composite-sort per-group cumulative (sum/prod/max/min).

        Algorithm (mirrors _per_group_shift, v0.9.23e):
          1. external_sort by [*group_cols, *on_cols] — contiguous groups
             (on_cols empty -> [*group_cols]).
          2. Stream sorted; single _ExpandingState per col; reset on
             group change.
          3. external_sort by __orig_rid.
          4. Zip: APPEND {col}_{op} (not replace, unlike shift).

        Peak RAM = batch + O(K_groups * n_cols * 8). Independent of N.
        """
        import pyarrow.compute as _pc
        from cadw.io.batch_reader import ArrowIPCBatchReader as _IPCReader
        from cadw.io.batch_writer import ArrowIPCBatchWriter as _W
        from cadw.ops.sort import external_sort as _ext_sort

        _SUFFIX = {"sum": "cumsum", "prod": "cumprod",
                   "max": "cummax", "min": "cummin"}
        suffix = _SUFFIX[op]

        if not group_cols:
            raise ValueError(
                f"_per_group_cumulative: group_cols must be non-empty"
            )
        for gc in group_cols:
            if gc not in self._columns:
                raise KeyError(
                    f"_per_group_cumulative: group column {gc!r} not in "
                    f"{self._columns}"
                )
        for oc in on_cols:
            if oc not in self._columns:
                raise KeyError(
                    f"_per_group_cumulative: on column {oc!r} not in "
                    f"{self._columns}"
                )
        if columns is None:
            cols = [c for c in self._columns if c not in group_cols
                    and (pa.types.is_integer(
                        self._dataset.schema.field(c).type)
                        or pa.types.is_floating(
                            self._dataset.schema.field(c).type))]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError(
                f"_per_group_cumulative: no numeric columns selected"
            )
        for c in cols:
            if c in group_cols:
                raise ValueError(
                    f"_per_group_cumulative: group column {c!r} cannot "
                    f"be accumulated"
                )
            if c not in self._columns:
                raise KeyError(
                    f"_per_group_cumulative: column {c!r} not in "
                    f"{self._columns}"
                )
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"_per_group_cumulative: column {c!r} has type {t}; "
                    f"numeric required"
                )

        out_cols = {c: f"{c}_{suffix}" for c in cols}
        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        for c in cols:
            in_t = self._dataset.schema.field(c).type
            out_fields.append(
                pa.field(out_cols[c], in_t, nullable=True)
            )
        out_schema = pa.schema(out_fields)

        if self.count() == 0:
            empty_ds = Dataset.from_batches(
                out_schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        sort_keys = list(group_cols) + list(on_cols)
        sorted_path = _ext_sort(
            self._iter_filtered_batches(),
            key_columns=sort_keys,
            value_columns=list(self._columns),
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / suffix / "sort_by_keys",
            ascending=True,
            log=False,
        )

        applied_schema = pa.schema(
            [pa.field("row_id", pa.uint64())]
            + [out_schema.field(out_cols[c]) for c in cols]
        )
        applied_path = self._spill_dir / suffix / "cumulative.arrow"
        applied_path.parent.mkdir(parents=True, exist_ok=True)

        writer = _W(str(applied_path), applied_schema)
        states = {c: _ExpandingState(op, 1) for c in cols}
        _SENTINEL = object()
        prev_group = _SENTINEL
        try:
            for b in _IPCReader(str(sorted_path), None):
                self._memory_guard.check_batch(
                    b, name=f"per_group_{suffix}.sorted",
                    extra_factor=3.5,
                )
                n = b.num_rows
                if n == 0:
                    continue
                boundaries = _group_boundaries_arrow(b, group_cols)
                batch_first_key = tuple(
                    b.column(gc).slice(0, 1).to_pylist()[0]
                    for gc in group_cols
                )
                if (prev_group is _SENTINEL
                        or batch_first_key != prev_group):
                    if prev_group is not _SENTINEL:
                        for st in states.values():
                            st.reset()
                all_cum_arrow = {c: [] for c in cols}
                for gi in range(len(boundaries) - 1):
                    start = boundaries[gi]
                    end = boundaries[gi + 1]
                    if gi > 0:
                        for st in states.values():
                            st.reset()
                    group_len = end - start
                    gvals = {
                        c: b.column(c).slice(start, group_len).to_pylist()
                        for c in cols
                    }
                    group_cum = {c: [None] * group_len for c in cols}
                    for i in range(group_len):
                        for c in cols:
                            group_cum[c][i] = states[c].push(gvals[c][i])
                    for c in cols:
                        all_cum_arrow[c].append(pa.array(
                            group_cum[c],
                            type=out_schema.field(out_cols[c]).type,
                        ))
                arrays = [b.column("row_id")]
                for c in cols:
                    arrays.append(pa.concat_arrays(all_cum_arrow[c]))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
                prev_group = tuple(
                    b.column(gc).slice(n - 1, 1).to_pylist()[0]
                    for gc in group_cols
                )
        finally:
            writer.close()

        def _iter_applied():
            for b in _IPCReader(str(applied_path), None):
                names = b.schema.names
                rid_idx = names.index("row_id")
                new_names = (
                    ["__orig_rid"]
                    + [n for i, n in enumerate(names) if i != rid_idx]
                )
                yield b.rename_columns(new_names)

        out_col_list = list(out_cols.values())
        resorted_path = _ext_sort(
            _iter_applied(),
            key_columns=["__orig_rid"],
            value_columns=out_col_list,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / suffix / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        batch_size = 10_000

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            sorted_reader = _IPCReader(str(resorted_path), out_col_list)
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
                            f"per_group_{suffix}: sorted stream "
                            f"exhausted before original"
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
                arrays = [ob.column(c) for c in self._columns]
                for oc in out_col_list:
                    arrays.append(take_batch.column(oc))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_zipped(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

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

    def _per_group_rolling(
        self,
        group_cols: list[str],
        on_cols: list[str],
        window: int,
        agg: str,
        min_periods: "int | None",
        columns: "str | Sequence[str] | None",
    ) -> "CadwDataFrame":
        """Composite-sort per-group rolling — APPEND semantics."""
        if not group_cols:
            raise ValueError("_per_group_rolling: group_cols must be non-empty")
        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"_per_group_rolling: window must be int >= 1; got {window!r}"
            )
        if agg not in {"sum", "mean", "min", "max", "count"}:
            raise ValueError(
                f"_per_group_rolling: agg must be one of sum/mean/min/max/"
                f"count; got {agg!r}"
            )
        if min_periods is None:
            mp = window
        elif not isinstance(min_periods, int):
            raise TypeError(
                f"_per_group_rolling: min_periods must be int; "
                f"got {type(min_periods).__name__}"
            )
        elif not (1 <= min_periods <= window):
            raise ValueError(
                f"_per_group_rolling: min_periods must be in [1, window="
                f"{window}]; got {min_periods}"
            )
        else:
            mp = min_periods

        for gc in group_cols:
            if gc not in self._columns:
                raise KeyError(
                    f"_per_group_rolling: group column {gc!r} not in "
                    f"{self._columns}"
                )
        for oc in on_cols:
            if oc not in self._columns:
                raise KeyError(
                    f"_per_group_rolling: on column {oc!r} not in "
                    f"{self._columns}"
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
                    f"_per_group_rolling: group column {c!r} cannot be rolled"
                )
            if c not in self._columns:
                raise KeyError(
                    f"_per_group_rolling: column {c!r} not in {self._columns}"
                )
        if not cols:
            raise ValueError("_per_group_rolling: no value columns")
        for c in cols:
            t = self._dataset.schema.field(c).type
            if agg != "count" and not (
                pa.types.is_integer(t) or pa.types.is_floating(t)
            ):
                raise TypeError(
                    f"_per_group_rolling: column {c!r} has type {t}; "
                    f"agg={agg!r} requires numeric"
                )

        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        out_type_by_col: dict[str, tuple[str, pa.DataType]] = {}
        for c in cols:
            in_type = self._dataset.schema.field(c).type
            if agg in ("sum", "mean"):
                ot = pa.float64()
            elif agg == "count":
                ot = pa.int64()
            else:
                ot = in_type
            out_name = f"{c}_rolling_{agg}"
            out_type_by_col[c] = (out_name, ot)
            out_fields.append(pa.field(out_name, ot, nullable=True))
        out_schema = pa.schema(out_fields)
        rolling_out_cols = [out_type_by_col[c][0] for c in cols]

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
            spill_dir=self._spill_dir / "per_group_rolling" / "sort_by_keys",
            ascending=True,
            log=False,
        )

        applied_schema = pa.schema(
            [pa.field("row_id", pa.uint64())]
            + [
                pa.field(
                    out_type_by_col[c][0],
                    out_type_by_col[c][1],
                    nullable=True,
                )
                for c in cols
            ]
        )
        applied_path = self._spill_dir / "per_group_rolling" / "applied.arrow"
        applied_path.parent.mkdir(parents=True, exist_ok=True)

        writer = ArrowIPCBatchWriter(str(applied_path), applied_schema)
        states = {c: _RollState(window, mp, agg) for c in cols}
        _SENTINEL = object()
        prev_group = _SENTINEL
        try:
            for b in ArrowIPCBatchReader(str(sorted_path), None):
                self._memory_guard.check_batch(
                    b, name="per_group_rolling.sorted"
                , extra_factor=7.0)
                n = b.num_rows
                if n == 0:
                    continue
                # v0.9.21 (§1.4): Arrow-native group boundaries in rolling
                # (same rationale as _per_group_shift).
                boundaries = _group_boundaries_arrow(b, group_cols)

                # v0.9.21-fix1: batch-level key via 1-row to_pylist.
                batch_first_key = tuple(
                    b.column(gc).slice(0, 1).to_pylist()[0]
                    for gc in group_cols
                )
                if (prev_group is _SENTINEL
                        or batch_first_key != prev_group):
                    if prev_group is not _SENTINEL:
                        for st in states.values():
                            st.reset()

                all_rolled_arrow = {c: [] for c in cols}
                for gi in range(len(boundaries) - 1):
                    start = boundaries[gi]
                    end = boundaries[gi + 1]
                    if gi > 0:
                        # New group within batch — reset state.
                        for st in states.values():
                            st.reset()
                    group_len = end - start
                    gvals = {
                        c: b.column(c).slice(start, group_len).to_pylist()
                        for c in cols
                    }
                    group_rolled = {c: [None] * group_len for c in cols}
                    for i in range(group_len):
                        for c in cols:
                            group_rolled[c][i] = states[c].push(gvals[c][i])
                    for c in cols:
                        out_name, ot = out_type_by_col[c]
                        all_rolled_arrow[c].append(
                            pa.array(group_rolled[c], type=ot)
                        )
                arrays = [b.column("row_id")]
                for c in cols:
                    arrays.append(pa.concat_arrays(all_rolled_arrow[c]))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
                )
                # v0.9.21-fix1: track last row's key for next batch.
                prev_group = tuple(
                    b.column(gc).slice(n - 1, 1).to_pylist()[0]
                    for gc in group_cols
                )
        finally:
            writer.close()

        def _iter_rolled():
            for b in ArrowIPCBatchReader(str(applied_path), None):
                names = b.schema.names
                rid_idx = names.index("row_id")
                new_names = (
                    ["__orig_rid"]
                    + [n for i, n in enumerate(names) if i != rid_idx]
                )
                yield b.rename_columns(new_names)

        resorted_path = external_sort(
            _iter_rolled(),
            key_columns=["__orig_rid"],
            value_columns=rolling_out_cols,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "per_group_rolling" / "sort_by_rowid",
            ascending=True,
            log=False,
        )

        batch_size = 10_000

        def _iter_zipped():
            orig_iter = self._iter_arrow_batches(batch_size=batch_size)
            sorted_reader = ArrowIPCBatchReader(
                str(resorted_path), rolling_out_cols,
            )
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
                            "per_group_rolling: sorted stream exhausted "
                            "before original"
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
                arrays = [ob.column(c) for c in self._columns]
                for out_name in rolling_out_cols:
                    arrays.append(take_batch.column(out_name))
                yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

        new_ds = Dataset.from_batches(
            out_schema, _iter_zipped(), self._new_dataset_dir()
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

    def expanding(
        self,
        agg: str,
        *,
        min_periods: int = 1,
        columns: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Cumulative (growing) window aggregation (v0.8.0).

        Equivalent to `rolling(window=n_so_far, agg)` but O(1) state
        without deque. For min/max, cumulative min/max is monotone —
        no monotonic deque needed (unlike rolling).

        - agg ∈ {'sum', 'mean', 'min', 'max', 'count'}.
        - min_periods=1: first non-null row → non-null output.
        - columns=None: all numeric columns (count works on any type).
        - Nulls: skip; null row → null output.

        Output: original columns + `{col}_expanding_{agg}` per column.

        Not supported: `center`, `on=col` (sort-first via `.sort_values`).
        """
        if agg not in {"sum", "mean", "min", "max", "count"}:
            raise ValueError(
                f"expanding: agg must be one of sum/mean/min/max/count; "
                f"got {agg!r}"
            )
        if min_periods < 1:
            raise ValueError(
                f"expanding: min_periods must be >= 1; got {min_periods}"
            )
        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"expanding: column {c!r} not in {self._columns}"
                )
        for c in cols:
            t = self._dataset.schema.field(c).type
            if agg != "count" and not (
                pa.types.is_integer(t) or pa.types.is_floating(t)
            ):
                raise TypeError(
                    f"expanding: column {c!r} has type {t}; "
                    f"agg={agg!r} requires numeric"
                )

        out_fields = [
            self._dataset.schema.field(c) for c in self._columns
        ]
        out_type_by_col: dict[str, pa.DataType] = {}
        for c in cols:
            in_type = self._dataset.schema.field(c).type
            if agg in ("sum", "mean"):
                ot = pa.float64()
            elif agg == "count":
                ot = pa.int64()
            else:  # min/max
                ot = in_type
            out_name = f"{c}_expanding_{agg}"
            out_type_by_col[c] = ot
            out_fields.append(pa.field(out_name, ot, nullable=True))
        out_schema = pa.schema(out_fields)

        def _iter_expanding():
            states = {c: _ExpandingState(agg, min_periods) for c in cols}
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="expanding.input", extra_factor=3.5)
                out_arrays = [b.column(c) for c in self._columns]
                for c in cols:
                    state = states[c]
                    src_vals = b.column(c).to_pylist()
                    vals = [state.push(v) for v in src_vals]
                    out_arrays.append(
                        pa.array(vals, type=out_type_by_col[c])
                    )
                yield pa.RecordBatch.from_arrays(
                    out_arrays, schema=out_schema
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_expanding(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    # ---------------- ewm ----------------

    def ewm(
        self,
        *,
        alpha: "float | None" = None,
        span: "float | None" = None,
        halflife: "float | None" = None,
        com: "float | None" = None,
        adjust: bool = True,
        min_periods: int = 0,
        columns: "str | Sequence[str] | None" = None,
    ) -> "CadwDataFrame":
        """Exponentially weighted mean (v0.8.0).

        Exactly one of `alpha`, `span`, `halflife`, `com` must be given.

        - alpha ∈ (0, 1]: direct smoothing factor.
        - span >= 1: alpha = 2 / (span + 1).
        - halflife > 0: alpha = 1 - exp(-ln2 / halflife).
        - com >= 0: alpha = 1 / (1 + com).

        - adjust=True (pandas default): y_t = s_num/s_den where
            s_num = (1-α)·s_num + x_t,  s_den = (1-α)·s_den + 1.
        - adjust=False: y_t = (1-α)·y_{t-1} + α·x_t,  y_0 = x_0.
        - min_periods: minimum non-null rows before non-null output.
        - columns=None: all numeric columns.

        Null semantics: null row → null output; state preserved (skip).
        This matches `ignore_na=True` in pandas. `ignore_na=False` is NOT
        implemented.

        Output: original columns + `{col}_ewm` per column.

        **Not supported**: `bias=True` correction for `adjust=False`
        (rarely used); `ignore_na=False`.
        """
        # Exactly one parameter
        given = sum(p is not None for p in (alpha, span, halflife, com))
        if given != 1:
            raise ValueError(
                f"ewm: exactly one of alpha/span/halflife/com required; "
                f"got {given}"
            )
        if alpha is not None:
            if not (0.0 < alpha <= 1.0):
                raise ValueError(
                    f"ewm: alpha must be in (0, 1]; got {alpha}"
                )
            a = float(alpha)
        elif span is not None:
            if span < 1:
                raise ValueError(f"ewm: span must be >= 1; got {span}")
            a = 2.0 / (float(span) + 1.0)
        elif halflife is not None:
            if halflife <= 0:
                raise ValueError(
                    f"ewm: halflife must be > 0; got {halflife}"
                )
            import math
            a = 1.0 - math.exp(-math.log(2.0) / float(halflife))
        else:  # com
            if com < 0:
                raise ValueError(f"ewm: com must be >= 0; got {com}")
            a = 1.0 / (1.0 + float(com))

        if min_periods < 0:
            raise ValueError(
                f"ewm: min_periods must be >= 0; got {min_periods}"
            )
        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"ewm: column {c!r} not in {self._columns}"
                )
        for c in cols:
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"ewm: column {c!r} has type {t}; numeric required"
                )

        out_fields = [
            self._dataset.schema.field(c) for c in self._columns
        ]
        for c in cols:
            out_fields.append(
                pa.field(f"{c}_ewm", pa.float64(), nullable=True)
            )
        out_schema = pa.schema(out_fields)

        def _iter_ewm():
            states = {c: _EwmState(a, adjust, min_periods) for c in cols}
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="ewm.input", extra_factor=3.5)
                out_arrays = [b.column(c) for c in self._columns]
                for c in cols:
                    state = states[c]
                    src_vals = b.column(c).to_pylist()
                    vals = [state.push(v) for v in src_vals]
                    out_arrays.append(
                        pa.array(vals, type=pa.float64())
                    )
                yield pa.RecordBatch.from_arrays(
                    out_arrays, schema=out_schema
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_ewm(), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

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



    def fillna(self, value, columns=None) -> "CadwDataFrame":
        """Fill nulls with a scalar `value`. Streaming.

        v0.9.23g (AUDIT_FULL_v0922 item 7.A.5). Per-batch `pc.fill_null`
        — no materialize. `value` must be a scalar compatible with all
        target columns' dtypes (Arrow will raise on incompatible fill
        — no silent type coercion).

        Args:
            value: scalar fill value.
            columns: None → all columns; str → one; Sequence[str] →
                subset. Non-numeric columns fillable with matching-type
                scalar (e.g. str fill on string column).

        Returns: CadwDataFrame with nulls filled in target columns.
        """
        import pyarrow.compute as _pc
        from cadw.dataset import Dataset as _DS

        if columns is None:
            cols = list(self._columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError("fillna: columns must be non-empty")
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"fillna: column {c!r} not in {self._columns}"
                )
        fill_set = set(cols)

        # All fields nullable=True except unchanged.
        out_fields = []
        for c in self._columns:
            orig = self._dataset.schema.field(c)
            if c in fill_set:
                out_fields.append(pa.field(c, orig.type, nullable=True))
            else:
                out_fields.append(orig)
        out_schema = pa.schema(out_fields)

        # Pre-construct the Arrow fill scalar per column, so any type
        # mismatch fails fast at function entry, not mid-iteration.
        fill_scalars = {}
        for c in cols:
            t = self._dataset.schema.field(c).type
            try:
                fill_scalars[c] = pa.scalar(value, type=t)
            except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
                raise TypeError(
                    f"fillna: fill value {value!r} incompatible with "
                    f"column {c!r} type {t}: {e}"
                ) from e

        def _iter_filled():
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(b, name="fillna.input")
                arrays = []
                for c in self._columns:
                    if c in fill_set:
                        arrays.append(
                            _pc.fill_null(b.column(c), fill_scalars[c])
                        )
                    else:
                        arrays.append(b.column(c))
                yield pa.RecordBatch.from_arrays(
                    arrays, schema=out_schema,
                )

        new_ds = _DS.from_batches(
            out_schema, _iter_filled(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )



    # ---------------- sample (v0.9.23f) ----------------

    def sample(
        self,
        n=None,
        frac=None,
        replace: bool = False,
        weights: "str | None" = None,
        random_state=None,
    ) -> "CadwDataFrame":
        """Random sample of rows. Streaming, bounded memory.

        v0.9.23f (AUDIT_FULL_v0922 item 7.A.2, Top-10 gap #1).
        Prototype validated (stratified_sample, RAM 1.01 MB @ 32 MB).

        Args:
            n: number of rows (int >= 0). Exactly one of n / frac.
            frac: fraction of total rows [0, 1]. n = max(1,
                round(N*frac)) for frac > 0; frac=0 -> empty.
            replace: sample with replacement. Default False. n can
                exceed N only if replace=True.
            weights: name of a numeric weight column. Sampling is
                proportional to weight (Efraimidis-Spirakis A-Res).
                Not supported with replace=True (NotImplementedError).
                Non-positive / null weights are skipped.
            random_state: int seed; None -> 42 (deterministic).

        Returns:
            CadwDataFrame, rows in sample order (random).

        Raises:
            ValueError: exactly one of n/frac not given; n > N with
                replace=False; frac out of [0, 1]; n < 0; both n and
                frac; weights all-zero/null.
            NotImplementedError: weights + replace=True.
            KeyError: weights column missing.
            TypeError: weights column non-numeric.
            MemoryLimitExceeded: 4*n*row_bytes > state_limit.
        """
        import numpy as np
        from cadw.core.memory_guard import MemoryLimitExceeded as _MLE
        from cadw.dataset import Dataset as _DS

        # ---- validation ----
        if (n is None) == (frac is None):
            raise ValueError(
                "sample: exactly one of n or frac must be provided"
            )
        if frac is not None:
            # v0.9.23f-fix2: [0, 1] (pandas-parity; frac=0 -> empty).
            if not (0.0 <= float(frac) <= 1.0):
                raise ValueError(
                    f"sample: frac must be in [0, 1]; got {frac!r}"
                )
        if not isinstance(replace, bool):
            raise TypeError(
                f"sample: replace must be bool; got {type(replace).__name__}"
            )

        N = self.count()
        if N == 0:
            # Empty view
            schema = pa.schema(
                [self._dataset.schema.field(c) for c in self._columns]
            )
            empty_ds = _DS.from_batches(
                schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        if frac is not None:
            if float(frac) == 0.0:
                n = 0
            else:
                n = max(1, int(round(N * float(frac))))

        if not isinstance(n, int):
            raise TypeError(
                f"sample: n must be int; got {type(n).__name__}"
            )
        if n < 0:
            raise ValueError(f"sample: n must be >= 0; got {n}")

        if weights is not None and replace:
            raise NotImplementedError(
                "sample: weights + replace=True is not supported "
                "(would need O(N) full-weight materialization). "
                "Use replace=False with weights, or use unweighted "
                "sampling with replacement."
            )

        if weights is not None:
            if weights not in self._columns:
                raise KeyError(
                    f"sample: weights column {weights!r} not in "
                    f"{self._columns}"
                )
            t = self._dataset.schema.field(weights).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"sample: weights column {weights!r} has type "
                    f"{t}; numeric required"
                )

        if n == 0:
            schema = pa.schema(
                [self._dataset.schema.field(c) for c in self._columns]
            )
            empty_ds = _DS.from_batches(
                schema, iter([]), self._new_dataset_dir(),
            )
            return CadwDataFrame(
                empty_ds,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )

        if n > N and not replace:
            raise ValueError(
                f"sample: n={n} > N={N} with replace=False. "
                f"Set replace=True or reduce n."
            )

        seed = 42 if random_state is None else int(random_state)
        rng = np.random.default_rng(seed)

        # ---- memory guard: 4*n*row_bytes for iloc materialize ----
        from cadw.core.sampling import peek_first_batch
        first_batch = peek_first_batch(self._iter_filtered_batches())
        if first_batch is not None:
            row_bytes_est = self._memory_guard.reestimate_from_batch(
                first_batch,
            )
        else:
            row_bytes_est = max(
                1,
                self._memory_guard.estimate_row_size_from_schema(
                    pa.schema(
                        [self._dataset.schema.field(c)
                         for c in self._columns]
                    )
                ),
            )
        self._memory_guard.check_state(
            4 * n * row_bytes_est,
            name=f"sample(n={n})",
        )

        # ---- replace=True: vectorized positions ----
        if replace:
            positions = rng.integers(0, N, size=n).tolist()
            return self.iloc[positions]

        # ---- n == N without replacement: shuffle ----
        if n == N:
            perm = rng.permutation(N).tolist()
            return self.iloc[perm]

        # ---- n < N without replacement ----
        if weights is None:
            positions = _reservoir_positions(
                self._iter_filtered_batches(), n, N, rng,
                self._memory_guard,
            )
        else:
            positions = _weighted_reservoir_positions(
                self._iter_filtered_batches(), n, weights, rng,
                self._memory_guard,
            )

        return self.iloc[positions]

    # ---------------- cumulative (v0.9.23e) ----------------

    def _cumulative_impl(
        self, op: str, columns, on,
    ) -> "CadwDataFrame":
        """Internal: cumulative sum/prod/max/min (global, not grouped).

        op ∈ {'sum', 'prod', 'max', 'min'}. Streaming per batch; on=None
        (row_id order) or on=col (sort-aware via external_sort + sort
        back). Output: APPEND `{col}_{op}` names to match expanding/ewm:
          sum -> `{col}_cumsum`, prod -> `{col}_cumprod`,
          max -> `{col}_cummax`, min -> `{col}_cummin`.
        """
        import pyarrow.compute as _pc
        from cadw.io.batch_reader import ArrowIPCBatchReader as _IPCReader
        from cadw.io.batch_writer import ArrowIPCBatchWriter as _W
        from cadw.ops.sort import external_sort as _ext_sort

        _SUFFIX = {"sum": "cumsum", "prod": "cumprod",
                   "max": "cummax", "min": "cummin"}
        suffix = _SUFFIX[op]

        if columns is None:
            cols = [c for c in self._columns
                    if pa.types.is_integer(
                        self._dataset.schema.field(c).type)
                    or pa.types.is_floating(
                        self._dataset.schema.field(c).type)]
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)
        if not cols:
            raise ValueError(
                f"{suffix}: no numeric columns selected"
            )
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"{suffix}: column {c!r} not in {self._columns}"
                )
            t = self._dataset.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise TypeError(
                    f"{suffix}: column {c!r} has type {t}; numeric required"
                )

        out_cols = {c: f"{c}_{suffix}" for c in cols}
        out_fields = [self._dataset.schema.field(c) for c in self._columns]
        for c in cols:
            # Same dtype as input (int -> int, float -> float) to
            # preserve identity; mean is the only op that promotes to
            # float64. cum* are all identity-preserving.
            in_t = self._dataset.schema.field(c).type
            out_fields.append(pa.field(out_cols[c], in_t, nullable=True))
        out_schema = pa.schema(out_fields)

        # Sort-aware path
        if on is not None:
            on_cols = self._normalize_on(on, fn_name=suffix)
            sorted_path = _ext_sort(
                self._iter_filtered_batches(),
                key_columns=on_cols,
                value_columns=list(self._columns),
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir / suffix / "sort_by_on",
                ascending=True,
                log=False,
            )
            applied_path = (
                self._spill_dir / suffix / "cumulative.arrow"
            )
            applied_path.parent.mkdir(parents=True, exist_ok=True)
            applied_schema = pa.schema(
                [pa.field("row_id", pa.uint64())]
                + [out_schema.field(out_cols[c]) for c in cols]
            )
            writer = _W(str(applied_path), applied_schema)
            states = {c: _ExpandingState(op, 1) for c in cols}
            try:
                for b in _IPCReader(str(sorted_path), None):
                    self._memory_guard.check_batch(
                        b, name=f"{suffix}.sorted_input",
                        extra_factor=3.5,  # 1 pylist per column
                    )
                    rid_arr = b.column("row_id")
                    out_arrays: list = [rid_arr]
                    for c in cols:
                        vals = b.column(c).to_pylist()
                        state = states[c]
                        rolled = [state.push(v) for v in vals]
                        out_arrays.append(
                            pa.array(rolled, type=out_schema.field(
                                out_cols[c]).type)
                        )
                    writer.write_batch(pa.RecordBatch.from_arrays(
                        out_arrays, schema=applied_schema,
                    ))
            finally:
                writer.close()

            # Sort back by __orig_rid
            def _iter_applied():
                for b in _IPCReader(str(applied_path), None):
                    names = b.schema.names
                    rid_idx = names.index("row_id")
                    new_names = (
                        ["__orig_rid"]
                        + [n for i, n in enumerate(names) if i != rid_idx]
                    )
                    yield b.rename_columns(new_names)

            resorted_path = _ext_sort(
                _iter_applied(),
                key_columns=["__orig_rid"],
                value_columns=list(out_cols.values()),
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir / suffix / "sort_by_rowid",
                ascending=True,
                log=False,
            )

            out_col_list = list(out_cols.values())
            batch_size = 10_000

            def _iter_zipped():
                orig_iter = self._iter_arrow_batches(
                    batch_size=batch_size)
                sorted_reader = _IPCReader(str(resorted_path), out_col_list)
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
                                f"{suffix}: sorted stream exhausted "
                                f"before original"
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
                    arrays = [ob.column(c) for c in self._columns]
                    for oc in out_col_list:
                        arrays.append(take_batch.column(oc))
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

        # Streaming path (no on): global cumulative, row_id order
        def _iter_cumulative():
            states = {c: _ExpandingState(op, 1) for c in cols}
            for b in self._iter_filtered_batches():
                self._memory_guard.check_batch(
                    b, name=f"{suffix}.input", extra_factor=3.5,
                )
                out_arrays = [b.column(c) for c in self._columns]
                for c in cols:
                    state = states[c]
                    vals = b.column(c).to_pylist()
                    cum_vals = [state.push(v) for v in vals]
                    out_arrays.append(
                        pa.array(cum_vals, type=out_schema.field(
                            out_cols[c]).type)
                    )
                yield pa.RecordBatch.from_arrays(
                    out_arrays, schema=out_schema,
                )

        new_ds = Dataset.from_batches(
            out_schema, _iter_cumulative(), self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def cumsum(self, columns=None, *, on=None) -> "CadwDataFrame":
        """Cumulative sum. Global (row_id order) or sort-aware if on=col.

        v0.9.23e (AUDIT_FULL_v0922 item 7.A.6). APPEND `{col}_cumsum`.
        Nulls preserved (null row -> null output; state frozen, matches
        expanding semantics).
        """
        return self._cumulative_impl("sum", columns, on)

    def cumprod(self, columns=None, *, on=None) -> "CadwDataFrame":
        """Cumulative product. See `cumsum`. APPEND `{col}_cumprod`."""
        return self._cumulative_impl("prod", columns, on)

    def cummax(self, columns=None, *, on=None) -> "CadwDataFrame":
        """Cumulative max. See `cumsum`. APPEND `{col}_cummax`."""
        return self._cumulative_impl("max", columns, on)

    def cummin(self, columns=None, *, on=None) -> "CadwDataFrame":
        """Cumulative min. See `cumsum`. APPEND `{col}_cummin`."""
        return self._cumulative_impl("min", columns, on)

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


