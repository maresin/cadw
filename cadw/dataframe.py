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
from cadw.core.sampling import (
    data_chars_for_join,
    peek_first_batch,
)
from cadw.dataset import (
    Dataset,
    RowFilter,
    PredicateFilter,
    SliceFilter,
)
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ParquetBatchWriter
from cadw.ops.groupby import groupby_aggregate
from cadw.ops.join import hash_join
from cadw.ops.sort import external_sort
from cadw.ops.dedup import drop_duplicates as _ops_drop_duplicates
from cadw.ops.rank import compute_ranks_to_disk, VALID_METHODS
from cadw.ops.quantile import quantile as _ops_quantile


class _ColumnProxy:
    """df['col']: сравнения возвращают ленивый PredicateFilter."""
    __slots__ = ("_column",)

    def __init__(self, column: str) -> None:
        self._column = column

    def _pred(self, op: str, value) -> PredicateFilter:
        return PredicateFilter(self._column, op, value)

    def __gt__(self, v): return self._pred("gt", v)
    def __ge__(self, v): return self._pred("ge", v)
    def __lt__(self, v): return self._pred("lt", v)
    def __le__(self, v): return self._pred("le", v)
    def __eq__(self, v): return self._pred("eq", v)  # type: ignore[override]
    def __ne__(self, v): return self._pred("ne", v)  # type: ignore[override]
    def __hash__(self): return id(self)

    def __repr__(self) -> str:
        return f"ColumnProxy({self._column!r})"


class CadwDataFrame:
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

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str | Path,
        base_dir: str | Path,
        *,
        chunk_size: int = 10_000,
        memory_guard: MemoryGuard | None = None,
        spill_dir: str | Path | None = None,
    ) -> "CadwDataFrame":
        ds = Dataset.from_parquet(parquet_path, base_dir, chunk_size=chunk_size)
        return cls(ds, memory_guard=memory_guard, spill_dir=spill_dir)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        *,
        memory_guard: MemoryGuard | None = None,
        spill_dir: str | Path | None = None,
    ) -> "CadwDataFrame":
        return cls(dataset, memory_guard=memory_guard, spill_dir=spill_dir)

    # ---------------- properties ----------------

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def shape(self) -> tuple[int, int]:
        """Верхняя оценка: (dataset.total_rows, len(columns)).

        Для точного количества после фильтров используйте .count().
        """
        return (self._dataset.total_rows, len(self._columns))

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def count(self) -> int:
        """Точное число строк с учётом фильтров (сканирует)."""
        if not self._row_filters:
            return self._dataset.total_rows
        n = 0
        for b in self._iter_filtered_batches():
            n += b.num_rows
        return n

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
            return new

        # 3. str: прокси для сравнений (df['a'] > 0)
        if isinstance(key, str):
            if key not in self._dataset.columns:
                raise KeyError(f"column {key!r} not found in dataset")
            return _ColumnProxy(key)

        # 4. RowFilter: df[mask]
        if isinstance(key, RowFilter):
            return self._with_filter(key)

        raise TypeError(
            f"unsupported __getitem__ key: {type(key).__name__}; "
            f"expected slice, list[str], str, or RowFilter"
        )

    def _with_filter(self, f: RowFilter) -> "CadwDataFrame":
        new = CadwDataFrame.__new__(CadwDataFrame)
        new._dataset = self._dataset
        new._memory_guard = self._memory_guard
        new._spill_dir = self._spill_dir
        new._columns = list(self._columns)
        new._row_filters = list(self._row_filters) + [f]
        return new

    # ---------------- filtered iteration ----------------

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

    # ---------------- reading ----------------

    def to_batches(
        self,
        batch_size: int | None = None,
        format: str = "arrow",
    ):
        """Поток батчей с учётом фильтров.

        format:
          - "arrow": pa.RecordBatch (default, обратная совместимость);
          - "pandas": pd.DataFrame per batch;
          - "numpy": dict[str, np.ndarray] per batch;
          - "torch": dict[str, torch.Tensor] per batch (ленивый импорт torch).

        batch_size=None — как в Dataset (stored boundaries); N — re-chunk.
        """
        if format not in ("arrow", "pandas", "numpy", "torch"):
            raise ValueError(
                f"unsupported format {format!r}; "
                f"expected 'arrow' | 'pandas' | 'numpy' | 'torch'"
            )

        for b in self._iter_arrow_batches(batch_size=batch_size):
            yield _convert_batch(b, format)

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

    def head(self, n: int = 5) -> pa.Table:
        if n <= 0:
            schema = pa.schema([self._dataset.schema.field(c) for c in self._columns])
            return pa.Table.from_batches([], schema=schema)
        acc: list[pa.RecordBatch] = []
        remaining = n
        for b in self._iter_filtered_batches():
            take = min(b.num_rows, remaining)
            acc.append(b.slice(0, take))
            remaining -= take
            if remaining <= 0:
                break
        if not acc:
            schema = pa.schema([self._dataset.schema.field(c) for c in self._columns])
            return pa.Table.from_batches([], schema=schema)
        return pa.Table.from_batches(acc).slice(0, n)

    # ---------------- groupby ----------------

    def groupby(self, by: str | Sequence[str]) -> "_GroupBy":
        by_cols = [by] if isinstance(by, str) else list(by)
        return _GroupBy(self, by_cols)

    def _groupby_impl(
        self, by_cols, agg_columns, agg_funcs, *, combiner: bool = False
    ) -> "CadwDataFrame":
        for c in list(by_cols) + list(agg_columns):
            if c not in self._columns:
                raise KeyError(
                    f"groupby: column {c!r} not in current selection {self._columns}"
                )
        result = groupby_aggregate(
            self._iter_filtered_batches(),
            by_cols,
            agg_columns,
            agg_funcs,
            self._memory_guard,
            self._spill_dir / "groupby",
            combiner=combiner,
            log=False,
        )
        return self._wrap_result_table(result)

    # ---------------- unique ----------------

    def unique(self, column: str) -> "CadwDataFrame":
        if column not in self._columns:
            raise KeyError(f"unique: column {column!r} not in selection")
        cols = list(self._columns)
        result = groupby_aggregate(
            self._iter_filtered_batches(),
            [column],
            [column],
            ["count"],
            self._memory_guard,
            self._spill_dir / "unique",
            log=False,
        )
        # результат: by=column, val_count; оставляем только сам столбец
        values_only = result.select([column])
        # восстанавливаем остальные колонки? нет — unique возвращает только колонку
        return self._wrap_result_table(values_only)

    # ---------------- sort_values ----------------

    def sort_values(self, by, *, ascending=True) -> "CadwDataFrame":
        by_cols = [by] if isinstance(by, str) else list(by)
        for c in by_cols:
            if c not in self._columns:
                raise KeyError(f"sort_values: column {c!r} not in selection")
        path = external_sort(
            self._iter_filtered_batches(),
            key_columns=by_cols,
            value_columns=self._columns,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "sort",
            ascending=ascending,
            log=False,
        )
        schema = pa.schema([self._dataset.schema.field(c) for c in self._columns])

        def _iter():
            reader = ArrowIPCBatchReader(str(path), self._columns)
            for b in reader:
                yield b

        new_ds = Dataset.from_batches(schema, _iter(), self._new_dataset_dir())
        return CadwDataFrame(
            new_ds, memory_guard=self._memory_guard, spill_dir=self._spill_dir
        )

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

    def quantile(
        self,
        column: str,
        q: float | Sequence[float],
        *,
        interpolation: str = "linear",
    ) -> float | list[float]:
        """Quantile(s) of a single column (v0.3.3).

        - q: scalar (returns float) или list (returns list[float]) ∈ [0, 1].
        - interpolation='linear' (pandas default; v0.3.3 only).
        - skipna=True (nulls не участвуют в n_valid и позициях).

        Sort: всегда ascending (квантиль от порядка не зависит).

        Для all-null или empty — NaN (или list of NaN).

        Другие интерполяции ('lower'/'higher'/'nearest'/'midpoint') —
        v0.3.4.
        """
        if column not in self._columns:
            raise KeyError(
                f"quantile: column {column!r} not in columns {self._columns}"
            )
        return _ops_quantile(
            self._iter_filtered_batches(),
            column=column,
            q=q,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "quantile",
            interpolation=interpolation,
            log=False,
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
        """Hash join (v0.2.4). how ∈ {'inner', 'left'}.

        - how='inner': только совпадающие ключи.
        - how='left':  все строки self сохраняются; right-колонки -> NULL
          при отсутствии совпадения.
        - on: имена общих колонок в обеих сторонах.
        - suffixes: для пересекающихся не-on колонок.
        - num_partitions: число партиций HashShuffle; None — из
          JoinMemoryModel (если use_planner), иначе дефолт 8.
        - use_planner: использовать JoinMemoryModel.solve() для выбора
          num_partitions / max_output_rows. Если
          False — runtime использует старые дефолты (0.3 * available / 256).
        """
        if how not in ("inner", "left"):
            raise NotImplementedError(
                f"merge: how={how!r}; only 'inner', 'left' supported in "
                f"v0.2.x. 'right'/'outer' — v0.3."
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
                params = self._memory_guard.plan(
                    "join", left_chars,
                    op_context={"right_chars": right_chars, "how": how},
                )
                if num_partitions is None:
                    num_partitions = params.num_partitions
                max_output_rows = params.max_output_rows
            else:
                # Одна из сторон пуста — planner не нужен; hash_join
                # короткозамкнёт. num_partitions не важен.
                if num_partitions is None:
                    num_partitions = 8
        else:
            if num_partitions is None:
                num_partitions = 8

        result_table = hash_join(
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
        return self._wrap_result_table(result_table)

    # ---------------- write ----------------

    def write_parquet(self, path: str | Path, *, compression: str = "snappy") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([self._dataset.schema.field(c) for c in self._columns])
        writer = ParquetBatchWriter(str(path), schema, compression=compression)
        try:
            for b in self._iter_filtered_batches():
                writer.write_batch(b)
        finally:
            writer.close()
        return path

    # ---------------- helpers ----------------

    def _wrap_result_table(self, table: pa.Table) -> "CadwDataFrame":
        new_ds = Dataset.from_batches(
            table.schema, iter(table.to_batches()), self._new_dataset_dir()
        )
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


class _GroupBy:
    def __init__(self, parent: CadwDataFrame, by_cols: Sequence[str]) -> None:
        self._parent = parent
        self._by = list(by_cols)

    def agg(self, agg_columns, agg_funcs, *, combiner: bool = False) -> CadwDataFrame:
        """Aggregate группами.

        - combiner=True: partial aggregation до shuffle для
          sum/count/min/max/mean (экономия I/O в N/K раз для низкой
          кардинальности). std → warning + fallback. §13.4/52.
        """
        agg_cols = [agg_columns] if isinstance(agg_columns, str) else list(agg_columns)
        funcs = [agg_funcs] if isinstance(agg_funcs, str) else list(agg_funcs)
        return self._parent._groupby_impl(
            self._by, agg_cols, funcs, combiner=combiner
        )


def _concat_batches(batches: list[pa.RecordBatch]) -> pa.RecordBatch:
    if len(batches) == 1:
        return batches[0]
    return pa.Table.from_batches(batches).combine_chunks().to_batches()[0]


def _convert_batch(batch: pa.RecordBatch, format: str):
    """Конвертация одного батча. Векторизованно, без .as_py()."""
    if format == "arrow":
        return batch
    if format == "pandas":
        return batch.to_pandas()
    if format == "numpy":
        return {
            name: batch.column(i).to_numpy(zero_copy_only=False)
            for i, name in enumerate(batch.schema.names)
        }
    if format == "torch":
        import torch  # ленивый импорт
        return {
            name: torch.from_numpy(
                batch.column(i).to_numpy(zero_copy_only=False)
            )
            for i, name in enumerate(batch.schema.names)
        }
    raise ValueError(f"unsupported format {format!r}")
