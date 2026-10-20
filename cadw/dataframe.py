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

logger = get_logger("dataframe")


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

    def _normalize_on(
        self, on: "str | Sequence[str]", *, fn_name: str,
    ) -> list[str]:
        """Normalize `on` to list[str], validate each column exists.

        v0.8.6: multi-column `on` (`on=["ts", "sensor"]`) — lexicographic
        composite sort. Accepts str (backward compat) or Sequence[str].
        """
        if isinstance(on, str):
            on_cols = [on]
        else:
            on_cols = list(on)
        if not on_cols:
            raise ValueError(f"{fn_name}: on must be non-empty")
        for c in on_cols:
            if c not in self._columns:
                raise KeyError(
                    f"{fn_name}: on column {c!r} not in {self._columns}"
                )
        return on_cols

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
            bufs: dict[str, deque] = {
                c: deque(maxlen=k) for c in shift_cols
            }
            for b in self._iter_filtered_batches():
                n = b.num_rows
                arrays = []
                for c in self._columns:
                    col_vals = b.column(c)
                    if c not in shift_cols:
                        arrays.append(col_vals)
                        continue
                    prev = list(bufs[c])
                    fill_n = max(0, k - len(prev))
                    parts = []
                    if fill_n > 0:
                        parts.append(_fill(fill_n, col_vals.type))
                    if prev:
                        parts.append(pa.array(prev, type=col_vals.type))
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
                    for v in col_vals.to_pylist():
                        bufs[c].append(v)
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
        bufs: dict[str, deque] = {c: deque(maxlen=k) for c in cols}
        try:
            for b in sorted_reader:
                self._memory_guard.check_batch(
                    b, name="shift_on.sorted_input"
                )
                n = b.num_rows
                arrays: list = [b.column("row_id")]
                for c in cols:
                    col_vals = b.column(c)
                    prev = list(bufs[c])
                    fill_n = max(0, k - len(prev))
                    parts = []
                    if fill_n > 0:
                        parts.append(_fill(fill_n, col_vals.type))
                    if prev:
                        parts.append(pa.array(prev, type=col_vals.type))
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
                    for v in col_vals.to_pylist():
                        bufs[c].append(v)
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
                )
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
                self._memory_guard.check_batch(b, name="per_group_shift.sorted")
                n = b.num_rows
                if n == 0:
                    continue
                group_arrs = {gc: b.column(gc).to_pylist() for gc in group_cols}
                col_vals = {c: b.column(c).to_pylist() for c in cols}
                shifted_vals = {c: [None] * n for c in cols}
                for i in range(n):
                    gk = tuple(group_arrs[gc][i] for gc in group_cols)
                    if gk != prev_group:
                        for dq in bufs.values():
                            dq.clear()
                        prev_group = gk
                    for c in cols:
                        if len(bufs[c]) < k:
                            shifted_vals[c][i] = fill_value
                        else:
                            shifted_vals[c][i] = bufs[c][0]
                        bufs[c].append(col_vals[c][i])
                arrays = [b.column("row_id")]
                for c in cols:
                    arrays.append(
                        pa.array(
                            shifted_vals[c],
                            type=b.schema.field(c).type,
                        )
                    )
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
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
                )
                n = b.num_rows
                if n == 0:
                    continue
                group_arrs = {gc: b.column(gc).to_pylist() for gc in group_cols}
                col_vals = {c: b.column(c).to_pylist() for c in cols}
                rolled_vals = {c: [None] * n for c in cols}
                for i in range(n):
                    gk = tuple(group_arrs[gc][i] for gc in group_cols)
                    if gk != prev_group:
                        for st in states.values():
                            st.reset()
                        prev_group = gk
                    for c in cols:
                        rolled_vals[c][i] = states[c].push(col_vals[c][i])
                arrays = [b.column("row_id")]
                for c in cols:
                    out_name, ot = out_type_by_col[c]
                    arrays.append(pa.array(rolled_vals[c], type=ot))
                writer.write_batch(
                    pa.RecordBatch.from_arrays(arrays, schema=applied_schema)
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


class _RollState:
    """Incremental rolling aggregation state (v0.6.2c).

    - sum/mean/count: running accumulators (O(1) push).
    - min/max: monotonic deque (O(1) amortized push).

    Nulls are skipped (pandas default). Count = non-null in window.
    """

    __slots__ = (
        "_window", "_min_periods", "_agg", "_buf", "_idx",
        "_sum", "_count", "_dq",
    )

    def __init__(self, window: int, min_periods: int, agg: str) -> None:
        from collections import deque as _dq_mod
        self._window = window
        self._min_periods = min_periods
        self._agg = agg
        self._buf = _dq_mod()  # (idx, is_null, value)
        self._idx = 0
        self._sum = 0.0
        self._count = 0  # non-null in current window
        self._dq = _dq_mod()  # for min/max: (idx, value)

    def push(self, v):
        window = self._window
        i = self._idx
        # 1. Add current
        is_null = v is None
        if not is_null:
            self._count += 1
            if self._agg in ("sum", "mean"):
                self._sum += float(v)
        self._buf.append((i, is_null, v))
        # v0.8.4-fix (S1): null + non-empty _dq → `_dq[-1][1] >= None`
        # raises TypeError. Nulls skip monotone-deque maintenance entirely
        # (they don't contribute to min/max; window slot still consumed).
        if self._agg == "min":
            if not is_null:
                while self._dq and self._dq[-1][1] >= v:
                    self._dq.pop()
                self._dq.append((i, v))
        elif self._agg == "max":
            if not is_null:
                while self._dq and self._dq[-1][1] <= v:
                    self._dq.pop()
                self._dq.append((i, v))

        # 2. Evict if len > window
        while len(self._buf) > window:
            old_i, old_null, old_v = self._buf.popleft()
            if not old_null:
                self._count -= 1
                if self._agg in ("sum", "mean"):
                    self._sum -= float(old_v)
            if self._agg in ("min", "max") and self._dq and self._dq[0][0] == old_i:
                self._dq.popleft()

        self._idx += 1

        # 3. Compute result
        if self._count < self._min_periods:
            return None
        if self._agg == "count":
            return self._count
        if self._agg == "sum":
            return self._sum
        if self._agg == "mean":
            return self._sum / self._count
        if self._agg in ("min", "max"):
            return self._dq[0][1] if self._dq else None
        raise RuntimeError(f"_RollState: unknown agg {self._agg!r}")

    def reset(self) -> None:
        """Reset state for per-group rolling (v0.8.10).

        Called on group change. Clears deque/history/accumulators; next
        push starts fresh (min_periods applies from group start).
        """
        self._buf.clear()
        self._idx = 0
        self._sum = 0.0
        self._count = 0
        self._dq.clear()


class _ExpandingState:
    """Cumulative (growing) window aggregation state (v0.8.0).

    O(1) per row. Unlike rolling, no deque needed:
    - sum/mean/count: scalar accumulators.
    - min/max: current min/max (monotone by construction).

    Nulls: skip (count/sum/min/max not updated). Output for a null row is
    null — the input row stays null-visible (unlike rolling, where a null
    row with min_periods=1 may inherit prior output).
    """

    __slots__ = ("_agg", "_min_periods", "_count", "_sum",
                 "_min", "_max")

    def __init__(self, agg: str, min_periods: int) -> None:
        self._agg = agg
        self._min_periods = min_periods
        self._count = 0
        self._sum = 0.0
        self._min = None
        self._max = None

    def push(self, v):
        if v is None:
            return None
        self._count += 1
        x = float(v)
        if self._agg == "count":
            result = self._count
        elif self._agg == "sum":
            self._sum += x
            result = self._sum
        elif self._agg == "mean":
            self._sum += x
            result = self._sum / self._count
        elif self._agg == "min":
            self._min = x if self._min is None else min(self._min, x)
            result = self._min
        elif self._agg == "max":
            self._max = x if self._max is None else max(self._max, x)
            result = self._max
        else:
            raise RuntimeError(f"_ExpandingState: unknown agg {self._agg!r}")
        if self._count < self._min_periods:
            return None
        return result


class _EwmState:
    """Exponentially weighted mean state (v0.8.0). O(1) per row.

    adjust=True:  s_num = (1-α)·s_num + x_t,  s_den = (1-α)·s_den + 1
                  y_t = s_num / s_den. Init s_num=x_0, s_den=1.
    adjust=False: y_t = (1-α)·y_{t-1} + α·x_t. Init y_0 = x_0.

    Nulls: skip (state frozen). Output for null row is null.
    min_periods counts non-null rows.
    """

    __slots__ = ("_alpha", "_adjust", "_min_periods", "_count",
                 "_s_num", "_s_den", "_ewm")

    def __init__(self, alpha: float, adjust: bool, min_periods: int) -> None:
        self._alpha = alpha
        self._adjust = adjust
        self._min_periods = min_periods
        self._count = 0
        self._s_num = 0.0
        self._s_den = 0.0
        self._ewm = None

    def push(self, v):
        if v is None:
            return None
        self._count += 1
        x = float(v)
        if self._adjust:
            if self._count == 1:
                self._s_num = x
                self._s_den = 1.0
            else:
                self._s_num = (1.0 - self._alpha) * self._s_num + x
                self._s_den = (1.0 - self._alpha) * self._s_den + 1.0
            result = self._s_num / self._s_den
        else:
            if self._count == 1:
                self._ewm = x
            else:
                self._ewm = (
                    (1.0 - self._alpha) * self._ewm
                    + self._alpha * x
                )
            result = self._ewm
        if self._count < self._min_periods:
            return None
        return result


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

    def _normalize_on(self, on) -> list:
        if on is None:
            return []
        if isinstance(on, str):
            return [on]
        return list(on)

    def shift(
        self,
        k: int = 1,
        fill_value=None,
        *,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group shift (v0.8.10).

        Shift by `k` within each group, optionally order-aware (`on=col`).
        `on=None` → within-group order = original row_id order.

        - k: >= 1. First k rows of EACH GROUP → `fill_value`.
        - columns: value columns to shift (default: all except group keys).
        - on: sort key within group (timestamp/integer), or None.

        Group columns are NOT shifted by default (they're the key).
        See `_per_group_shift` docstring for full semantics.
        """
        on_cols = self._normalize_on(on)
        return self._parent._per_group_shift(
            self._by, on_cols, k, fill_value, columns
        )

    def diff(
        self,
        k: int = 1,
        *,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group first difference (v0.8.10).

        `out[i] = x[i] - x[i-k]` within each group. First k rows of each
        group → null (per-group, not global).
        """
        on_cols = self._normalize_on(on)
        return self._parent._per_group_diff(
            self._by, on_cols, k, columns
        )

    def rolling(
        self,
        window: int,
        agg: str,
        *,
        min_periods: "int | None" = None,
        columns: "str | Sequence[str] | None" = None,
        on: "str | Sequence[str] | None" = None,
    ) -> CadwDataFrame:
        """Per-group rolling (v0.8.10). APPENDS `{col}_rolling_{agg}`.

        Rolling computed within each group independently (min_periods
        applies from group start). Output preserves original row order;
        rolling columns are appended.
        """
        on_cols = self._normalize_on(on)
        return self._parent._per_group_rolling(
            self._by, on_cols, window, agg, min_periods, columns
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

