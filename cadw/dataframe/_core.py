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

from cadw.core.memory_guard import MemoryGuard
from cadw.core.logging import get_logger
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
from ._per_group import _PerGroupMixin
from ._resample_reshape import _ResampleReshapeMixin
from ._join import _JoinMixin
from ._ml_bindings import _MLBindingsMixin
from ._columns import _ColumnsMixin
from ._helpers import (
    _apply_filter_with_positions,
    _detect_mode,
    _normalize_scalar,
    _infer_arrow_type_from_value,
    _build_out_schema,
)



class CadwDataFrame(_ExportsMixin, _IteratorsMixin, _MiscMixin, _IndexingMixin, _AggregationsMixin, _TransformersFillMixin, _TransformersReshapeMixin, _RollingMixin, _CumulativeMixin, _PerGroupMixin, _ResampleReshapeMixin, _JoinMixin, _MLBindingsMixin, _ColumnsMixin):
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




    # ---------------- reading ----------------




    # ---------------- groupby ----------------



    # ---------------- unique ----------------


    # ---------------- sort_values ----------------


    # ---------------- drop_duplicates ----------------


    # ---------------- rank ----------------


    # ---------------- quantile ----------------


    # ---------------- encode_categorical ----------------



    # ---------------- shift / diff ----------------





    # ---------------- rolling ----------------



    # ---------------- per-group apply (v0.9.1e) ----------------


    # ---------------- per-group cumulative (v0.9.23e) ----------------


    # ---------------- per-group shift/diff/rolling (v0.8.10) ----------------




    # ---------------- resample ----------------


    # ---------------- ML: fit_linear ----------------


    # ---------------- ML: fit_logistic ----------------


    # ---------------- ML: fit_kmeans ----------------


    # ---------------- ML: fit_pca ----------------


    # ---------------- ML: fit_gbdt_regressor ----------------


    # ---------------- ML: fit_gbdt_classifier ----------------


    # ---------------- ML: fit_naive_bayes ----------------





    # ---------------- ML: fit_lasso / fit_elasticnet ----------------



    # ---------------- expanding ----------------


    # ---------------- ewm ----------------


    # ---------------- ML: fit_gbdt_multiclass ----------------


    # ---------------- merge ----------------


    # ---------------- metrics (v0.9.2a) ----------------


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

    # ─────────────────────────────────────────────────────────────
    # v0.9.94-unified-narrow: adapters from Auto*Result to CadwDataFrame.
    # Non-breaking — producers keep returning AutoResult by default.
    # Use these to wrap any producer's output into a CadwDataFrame for
    # further chaining (groupby, sort, Pipeline.transform, ...).
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def from_producer_result(
        cls,
        result,
        *,
        col_names=None,
        memory_guard=None,
        spill_dir=None,
    ) -> "CadwDataFrame":
        """Universal wrap: AutoResult | AutoArrayResult → CadwDataFrame.

        v0.9.95: single entry point for all 13 producers that flipped
        default output="auto" → CadwDataFrame.

        - InMemoryResult → from_table (no disk).
        - OnDiskResult → from_auto_result (disk-backed Dataset, no
          extra copy — re-uses original ArrowIPC file dir).
        - InMemoryArrayResult → _from_numpy (no disk).
        - OnDiskArrayResult → from_autoarray_result.

        Prefer this over type-specific from_auto* when the caller
        doesn't know which branch the producer returned (small data
        → in-memory; large → spilled).
        """
        from cadw.core.spill import AutoResult, AutoArrayResult
        # v0.9.95: also accept pa.Table / np.ndarray (legacy output=
        # "memory" branches). Unified return type = CadwDataFrame.
        if isinstance(result, pa.Table):
            return cls.from_table(result, memory_guard=memory_guard)
        if isinstance(result, np.ndarray):
            return cls._from_numpy(
                result, col_names=col_names, memory_guard=memory_guard)
        if isinstance(result, AutoResult):
            if not result.is_spilled:
                return cls.from_table(result.to_table())
            return cls.from_auto_result(
                result, memory_guard=memory_guard, spill_dir=spill_dir)
        if isinstance(result, AutoArrayResult):
            if not result.is_spilled:
                return cls._from_numpy(
                    result.to_numpy(), col_names=col_names,
                    memory_guard=memory_guard)
            return cls.from_autoarray_result(
                result, col_names=col_names,
                memory_guard=memory_guard, spill_dir=spill_dir)
        raise TypeError(
            f"from_producer_result: expected pa.Table | np.ndarray | "
            f"AutoResult | AutoArrayResult; got {type(result).__name__}")

    @classmethod
    def _from_numpy(cls, arr, *, col_names=None, memory_guard=None):
        """Wrap (n,) or (n, K) numpy as in-memory CadwDataFrame (no disk).

        Read-only (_InMemoryDataset). For persistent results, call
        .to_parquet() or similar downstream.
        """
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(
                f"_from_numpy: expected 1D or 2D array; got ndim={arr.ndim}")
        n, k = arr.shape
        if col_names is None:
            names = [f"c{i}" for i in range(k)]
        else:
            names = list(col_names)
            if len(names) != k:
                raise ValueError(
                    f"_from_numpy: col_names length {len(names)} "
                    f"!= width {k}")
        arrays = [pa.array(arr[:, i]) for i in range(k)]
        table = pa.Table.from_arrays(arrays, names=names)
        return cls.from_table(table, memory_guard=memory_guard)

    @classmethod
    def from_auto_result(
        cls,
        result,
        *,
        memory_guard=None,
        spill_dir=None,
    ) -> "CadwDataFrame":
        """Wrap AutoResult (InMemoryResult | OnDiskResult) as
        CadwDataFrame.

        The iterable is consumed eagerly by Dataset.from_batches
        (v0.9.92 — verified: from_batches fully iterates before
        returning). OnDiskResult's owned_tmpdir cannot be GC'd
        mid-consumption — safe.

        Parameters
        ----------
        result : AutoResult
            Output of `groupby_aggregate(..., output="auto")`,
            `drop_duplicates(...)`, `topk_to_disk(...)`,
            `gmm.score_samples(...)`, `chains.predict_proba(...)`,
            etc.
        memory_guard : MemoryGuard, optional
            Guard for the resulting CadwDataFrame's downstream
            operations. None → default guard.
        spill_dir : str | Path, optional
            Root directory for the new Dataset. None → tempfile
            mkdtemp under /tmp/cadw_unified_*.

        Raises
        ------
        TypeError : result is not AutoResult.
        """
        from cadw.core.spill import AutoResult
        if not isinstance(result, AutoResult):
            raise TypeError(
                f"from_auto_result: expected AutoResult (e.g. "
                f"InMemoryResult | OnDiskResult); got "
                f"{type(result).__name__}. If you have pa.Table, use "
                f"the existing CadwDataFrame(Dataset.from_batches(...)) "
                f"pattern.")
        if spill_dir is None:
            spill_dir = Path(tempfile.mkdtemp(prefix="cadw_unified_"))
        else:
            spill_dir = Path(spill_dir)
        p = spill_dir / "datasets" / f"ds_{uuid.uuid4().hex[:8]}"
        p.mkdir(parents=True, exist_ok=True)
        new_ds = Dataset.from_batches(result.schema, iter(result), p)
        return cls(new_ds, memory_guard=memory_guard, spill_dir=spill_dir)

    @classmethod
    def from_autoarray_result(
        cls,
        result,
        *,
        col_names=None,
        memory_guard=None,
        spill_dir=None,
        width=None,
        dtype=None,
    ) -> "CadwDataFrame":
        """Wrap AutoArrayResult (InMemoryArrayResult |
        OnDiskArrayResult) as CadwDataFrame.

        AutoArrayResult from ML producers (iforest.score_samples,
        tfidf.transform_dense, randsvd.transform, ordinal.predict,
        coxph.predict_risk, ...).

        Peek first batch to determine (width, dtype). Re-entrant
        iteration — InMemoryArrayResult and OnDiskArrayResult both
        open fresh readers on `iter()`. Safe.

        Parameters
        ----------
        result : AutoArrayResult
            (n,) or (n, K) numpy-like; spill-on-demand already
            handled by the producer's SpillArray.
        col_names : Sequence[str], optional
            Column names. Length must equal width. Default: c0..cK-1
            (or c0 for width=1).
        width, dtype :
            Required for empty result (n=0) — cannot peek shape.

        Raises
        ------
        TypeError : result is not AutoArrayResult.
        ValueError : empty result without width; col_names length
            mismatch.
        """
        from cadw.core.spill import (
            AutoArrayResult, array_schema, array_to_record_batch,
        )
        if not isinstance(result, AutoArrayResult):
            raise TypeError(
                f"from_autoarray_result: expected AutoArrayResult "
                f"(e.g. InMemoryArrayResult | OnDiskArrayResult); got "
                f"{type(result).__name__}")
        # Peek first batch — re-entrant iteration; no state consumed.
        first = None
        for arr in result:
            arr = np.asarray(arr)
            if arr.size > 0:
                first = arr
                break
        if first is None:
            if width is None:
                raise ValueError(
                    "from_autoarray_result: result is empty; "
                    "pass width= (and optionally dtype=) to construct "
                    "schema without peeking")
            actual_width = int(width)
            actual_dtype = np.dtype(dtype) if dtype is not None else (
                np.float64)
        else:
            if first.ndim == 1:
                actual_width = 1
            else:
                actual_width = int(first.shape[1])
            actual_dtype = first.dtype

        if col_names is None:
            names = [f"c{i}" for i in range(actual_width)]
        else:
            names = list(col_names)
            if len(names) != actual_width:
                raise ValueError(
                    f"from_autoarray_result: col_names length "
                    f"{len(names)} != width {actual_width}")
        # Build schema from array_schema (handles dtype mapping); then
        # replace names.
        base = array_schema(actual_width, actual_dtype)
        schema = pa.schema(
            [(names[i], base.field(i).type) for i in range(actual_width)])

        if spill_dir is None:
            spill_dir = Path(tempfile.mkdtemp(prefix="cadw_unified_"))
        else:
            spill_dir = Path(spill_dir)
        p = spill_dir / "datasets" / f"ds_{uuid.uuid4().hex[:8]}"
        p.mkdir(parents=True, exist_ok=True)

        def _iter():
            for arr in result:
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                rb = array_to_record_batch(arr)
                # Rename columns if custom names given.
                if col_names is not None:
                    rb = pa.RecordBatch.from_arrays(
                        [rb.column(i) for i in range(rb.num_columns)],
                        names=names)
                yield rb

        new_ds = Dataset.from_batches(schema, _iter(), p)
        return cls(new_ds, memory_guard=memory_guard, spill_dir=spill_dir)

    def _wrap_result_table(self, table) -> "CadwDataFrame":
        # v0.9.92: accept AutoResult (default output="auto") in addition
        # to pa.Table (explicit output="memory"). Facade API always
        # returns CadwDataFrame; internally Dataset.from_batches is
        # eager (fully consumes iterable before returning), so
        # OnDiskResult's tmpdir cannot be GC'd mid-consumption.
        from cadw.core.spill import AutoResult
        if isinstance(table, AutoResult):
            schema = table.schema
            batches = iter(table)
        elif isinstance(table, pa.Table):
            schema = table.schema
            batches = iter(table.to_batches())
        else:
            raise TypeError(
                f"_wrap_result_table: expected pa.Table or AutoResult; "
                f"got {type(table).__name__}")
        new_ds = Dataset.from_batches(
            schema, batches, self._new_dataset_dir())
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





# ============ per-group apply helpers (v0.9.1e) ============

# ============ per-group apply helpers (v0.9.6b2, CADW-native) ============









