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
from cadw.io.batch_writer import ParquetBatchWriter
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
        # Index state: "range" (lazy) | "materialized" (single Arrow Array)
        # | "multi" (list of Arrays, tuple labels).
        self._index_kind: str = "range"
        self._index_start: int = 0
        self._index_stop: int = dataset.total_rows
        self._index_values: pa.Array | None = None
        self._index_levels: list[pa.Array] | None = None

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
    def iloc(self) -> "_ILocIndexer":
        """Pandas-style positional indexer (v0.4.0b).

        - `df.iloc[5]` → dict {col: value} — 6-я строка текущего view.
        - `df.iloc[1:3]` → CadwDataFrame — позиции 1..2 (lazy, SliceFilter).
        - `df.iloc[[0, 5, 7]]` → CadwDataFrame — три строки (materialize K).

        Negative indices работают: для slice — стандартная Python semantics;
        для int/list — resolve через count() (O(N) с фильтрами, O(1) без).

        `iloc[bool_array]` — raise (use df[mask]).
        """
        return _ILocIndexer(self)

    @property
    def loc(self) -> "_LocIndexer":
        """Pandas-style label-based indexer (v0.4.0c).

        - `df.loc[label]` → dict — строка с индексом `label` (KeyError если нет).
        - `df.loc[lo:hi]` → CadwDataFrame — inclusive labels (как pandas).
        - `df.loc[[l1, l2, ...]]` → CadwDataFrame — несколько строк.

        Требует `MaterializedIndex` (после `set_index`). `RangeIndex` →
        `TypeError`; используйте `iloc`.

        `loc[bool_array]` → `NotImplementedError` (use `df[mask]`).
        Дубликаты в индексе → первое вхождение (v0.4.0c); v0.4.1 — все.
        """
        return _LocIndexer(self)

    @property
    def at(self) -> "_AtIndexer":
        """`df.at[label, col]` — scalar (v0.4.0c). Read-only."""
        return _AtIndexer(self)

    @property
    def iat(self) -> "_IAtIndexer":
        """`df.iat[pos, col]` — scalar по позиции (v0.4.0c). Read-only."""
        return _IAtIndexer(self)

    def set_index(
        self,
        col: "str | Sequence[str]",
        drop: bool = True,
        inplace: bool = False,
    ) -> "CadwDataFrame":
        """Set DataFrame index from column `col` (v0.4.0a).

        - col: name of existing column.
        - drop: True — remove col from columns; False — keep.
        - inplace: True — modify self; False — return new CadwDataFrame.

        Материализует колонку в память (Arrow Array). Для больших N — O(N).
        Требует отсутствия активных фильтров (v0.4.0a).
        """
        # v0.6.1a: str or list[str] (MultiIndex if len >= 2).
        if isinstance(col, str):
            cols = [col]
        elif isinstance(col, (list, tuple)):
            cols = list(col)
            if not cols:
                raise ValueError("set_index: empty column list")
            for c in cols:
                if not isinstance(c, str):
                    raise TypeError(
                        f"set_index: col[{c!r}] must be str"
                    )
        else:
            raise TypeError(
                f"set_index: col must be str | Sequence[str]; "
                f"got {type(col).__name__}"
            )
        for c in cols:
            if c not in self._columns:
                raise KeyError(
                    f"set_index: column {c!r} not in {self._columns}"
                )
        if not isinstance(drop, bool):
            raise TypeError("set_index: drop must be bool")

        chunks_by_col = {c: [] for c in cols}
        for b in self._dataset.iter_batches(columns=cols):
            for c in cols:
                chunks_by_col[c].append(b.column(c))

        def _combine(chunks, c):
            if not chunks:
                return pa.array(
                    [], type=self._dataset.schema.field(c).type
                )
            if len(chunks) == 1:
                return chunks[0]
            return pa.chunked_array(chunks).combine_chunks()

        level_arrays = [_combine(chunks_by_col[c], c) for c in cols]

        new_columns = list(self._columns)
        if drop:
            for c in cols:
                new_columns.remove(c)

        def _apply(df):
            if len(cols) == 1:
                df._index_kind = "materialized"
                df._index_values = level_arrays[0]
                df._index_levels = None
                df._index_start = 0
                df._index_stop = len(level_arrays[0])
            else:
                df._index_kind = "multi"
                df._index_values = None
                df._index_levels = level_arrays
                df._index_start = 0
                df._index_stop = len(level_arrays[0])

        if inplace:
            self._columns = new_columns
            _apply(self)
            return self

        new = CadwDataFrame.__new__(CadwDataFrame)
        new._dataset = self._dataset
        new._memory_guard = self._memory_guard
        new._spill_dir = self._spill_dir
        new._columns = new_columns
        new._row_filters = list(self._row_filters)
        _apply(new)
        return new

    def reset_index(
        self,
        drop: bool = False,
        inplace: bool = False,
    ) -> "CadwDataFrame":
        """Reset index to RangeIndex(0, N) (v0.4.0a).

        - drop=True: index просто отбрасывается (columns без изменений).
        - drop=False: index становится колонкой `index` (или `level_0`
          при коллизии).
        - inplace=True: modify self; False — return new CadwDataFrame.

        Если index уже RangeIndex: drop=True — no-op; drop=False — raise
        ValueError (pandas: "cannot insert index, already have default").
        """
        if not isinstance(drop, bool):
            raise TypeError("reset_index: drop must be bool")

        if self._index_kind == "range":
            if drop:
                if inplace:
                    return self
                # new copy с тем же RangeIndex
                new = CadwDataFrame.__new__(CadwDataFrame)
                new._dataset = self._dataset
                new._memory_guard = self._memory_guard
                new._spill_dir = self._spill_dir
                new._columns = list(self._columns)
                new._row_filters = list(self._row_filters)
                new._index_kind = "range"
                new._index_start = self._index_start
                new._index_stop = self._index_stop
                new._index_values = None
                return new
            raise ValueError(
                "reset_index(drop=False): already have a default "
                "RangeIndex — cannot insert. "
                "Use reset_index(drop=True) or set_index(col) first."
            )

        if self._index_kind == "multi":
            assert self._index_levels is not None
            levels = self._index_levels
            nlevels = len(levels)
            if drop:
                if inplace:
                    self._index_kind = "range"
                    self._index_start = 0
                    self._index_stop = len(levels[0])
                    self._index_values = None
                    self._index_levels = None
                    return self
                new = CadwDataFrame.__new__(CadwDataFrame)
                new._dataset = self._dataset
                new._memory_guard = self._memory_guard
                new._spill_dir = self._spill_dir
                new._columns = list(self._columns)
                new._row_filters = list(self._row_filters)
                new._index_kind = "range"
                new._index_start = 0
                new._index_stop = len(levels[0])
                new._index_values = None
                new._index_levels = None
                return new

            new_names = []
            for i in range(nlevels):
                base = f"level_{i}"
                name = base
                j = 0
                while name in self._columns or name in new_names:
                    name = f"level_{i}_{j}"
                    j += 1
                new_names.append(name)

            new_columns = list(self._columns) + new_names
            new_schema = pa.schema(
                [self._dataset.schema.field(c) for c in self._columns]
                + [
                    pa.field(new_names[i], levels[i].type)
                    for i in range(nlevels)
                ]
            )

            if self._row_filters:
                def _iter_with_multi():
                    for b, positions in self._iter_filtered_batches_with_positions():
                        idx = pa.array(positions, type=pa.int64())
                        out = b
                        for i, lv in enumerate(levels):
                            out = out.append_column(
                                new_names[i], lv.take(idx)
                            )
                        yield out
                _iter_impl = _iter_with_multi
            else:
                offset = [0]

                def _iter_with_multi_unfiltered():
                    for b in self._iter_filtered_batches():
                        n = b.num_rows
                        out = b
                        for i, lv in enumerate(levels):
                            out = out.append_column(
                                new_names[i],
                                lv.slice(offset[0], n),
                            )
                        offset[0] += n
                        yield out
                _iter_impl = _iter_with_multi_unfiltered

            new_ds = Dataset.from_batches(
                new_schema, _iter_impl(), self._new_dataset_dir()
            )

            if inplace:
                self._dataset = new_ds
                self._columns = new_columns
                self._row_filters = []
                self._index_kind = "range"
                self._index_start = 0
                self._index_stop = new_ds.total_rows
                self._index_values = None
                self._index_levels = None
                return self

            new = CadwDataFrame.__new__(CadwDataFrame)
            new._dataset = new_ds
            new._memory_guard = self._memory_guard
            new._spill_dir = self._spill_dir
            new._columns = new_columns
            new._row_filters = []
            new._index_kind = "range"
            new._index_start = 0
            new._index_stop = new_ds.total_rows
            new._index_values = None
            new._index_levels = None
            return new

        # materialized (single level)
        assert self._index_values is not None
        values = self._index_values

        if drop:
            if self._row_filters:
                # v0.4.0d: filtered + drop=True → materialize filtered view
                # as new Dataset with RangeIndex(0, filtered_n).
                filtered_n = self.count()
                if inplace:
                    new_ds = Dataset.from_batches(
                        pa.schema(
                            [self._dataset.schema.field(c) for c in self._columns]
                        ),
                        self._iter_filtered_batches(),
                        self._new_dataset_dir(),
                    )
                    self._dataset = new_ds
                    self._row_filters = []
                    self._index_kind = "range"
                    self._index_start = 0
                    self._index_stop = filtered_n
                    self._index_values = None
                    return self
                new_ds = Dataset.from_batches(
                    pa.schema(
                        [self._dataset.schema.field(c) for c in self._columns]
                    ),
                    self._iter_filtered_batches(),
                    self._new_dataset_dir(),
                )
                new = CadwDataFrame.__new__(CadwDataFrame)
                new._dataset = new_ds
                new._memory_guard = self._memory_guard
                new._spill_dir = self._spill_dir
                new._columns = list(self._columns)
                new._row_filters = []
                new._index_kind = "range"
                new._index_start = 0
                new._index_stop = filtered_n
                new._index_values = None
                return new
            if inplace:
                self._index_kind = "range"
                self._index_start = 0
                self._index_stop = len(values)
                self._index_values = None
                return self
            new = CadwDataFrame.__new__(CadwDataFrame)
            new._dataset = self._dataset
            new._memory_guard = self._memory_guard
            new._spill_dir = self._spill_dir
            new._columns = list(self._columns)
            new._row_filters = list(self._row_filters)
            new._index_kind = "range"
            new._index_start = 0
            new._index_stop = len(values)
            new._index_values = None
            return new

        # drop=False: add "index" or "level_N" column
        name = "index"
        if name in self._columns:
            i = 0
            name = f"level_{i}"
            while name in self._columns:
                i += 1
                name = f"level_{i}"

        new_columns = list(self._columns) + [name]
        new_schema = pa.schema(
            [self._dataset.schema.field(c) for c in self._columns]
            + [pa.field(name, values.type)]
        )

        if self._row_filters:
            # v0.4.0d: filtered — index column = _index_values[positions],
            # позиции из _iter_filtered_batches_with_positions.
            def _iter_with_index_filtered():
                for b, positions in self._iter_filtered_batches_with_positions():
                    idx_slice = values.take(
                        pa.array(positions, type=pa.int64())
                    )
                    yield b.append_column(name, idx_slice)
            _iter_with_index_impl = _iter_with_index_filtered
        else:
            offset = 0

            def _iter_with_index_unfiltered():
                nonlocal offset
                for b in self._iter_filtered_batches():
                    n = b.num_rows
                    idx_slice = values.slice(offset, n)
                    yield b.append_column(name, idx_slice)
                    offset += n
            _iter_with_index_impl = _iter_with_index_unfiltered

        new_ds = Dataset.from_batches(
            new_schema, _iter_with_index_impl(), self._new_dataset_dir()
        )

        if inplace:
            self._dataset = new_ds
            self._columns = new_columns
            self._index_kind = "range"
            self._index_start = 0
            self._index_stop = new_ds.total_rows
            self._index_values = None
            return self

        new = CadwDataFrame.__new__(CadwDataFrame)
        new._dataset = new_ds
        new._memory_guard = self._memory_guard
        new._spill_dir = self._spill_dir
        new._columns = new_columns
        new._row_filters = []
        new._index_kind = "range"
        new._index_start = 0
        new._index_stop = new_ds.total_rows
        new._index_values = None
        return new

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
        # Index fields copied; .index будет raise при наличии filters
        # (v0.4.0a). v0.4.0b (iloc) — синхронизация через row_id.
        new._index_kind = self._index_kind
        new._index_start = self._index_start
        new._index_stop = self._index_stop
        new._index_values = self._index_values
        new._index_levels = self._index_levels
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

    def _iter_filtered_batches_with_positions(self):
        """Yield (filtered_batch, positions_in_raw_dataset).

        positions — numpy int64 array of 0-based позиций в **unfiltered**
        dataset (после последнего set_index/reset_index). Используется для
        восстановления parent index при filter (v0.4.0d).
        """
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

        # v0.5.1 (P2): use streaming path — output written to disk
        # batch-by-batch, never accumulated in RAM. Only the resulting
        # Dataset is built from IPC reader (also streaming).
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

