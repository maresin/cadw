# cadw/dataframe/_indexing.py
# v0.9.26m (B.1 step 7): _IndexingMixin.
# Symbols: iloc, loc, at, iat (properties), head, tail,
# set_index, reset_index, reindex, sort_values.
# Extracted verbatim from _core.py; no logic change.

from __future__ import annotations

import pyarrow as pa
from cadw.dataset import Dataset
from cadw.indexers import (
    _AtIndexer,
    _IAtIndexer,
    _ILocIndexer,
    _LocIndexer,
)
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.ops.sort import external_sort


class _IndexingMixin:
    """Row/column indexing, slicing, sort, reindex (v0.9.26m)."""
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
        from ._core import CadwDataFrame
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
        from ._core import CadwDataFrame
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


    def reindex(self, labels) -> "CadwDataFrame":
        """Row alignment to target labels (v0.9.25i).

        For each label in `labels`:
          - found in source index → source row
          - not found → null-filled row

        - labels: iterable of scalars.
        - Returns CadwDataFrame with rows in `labels` order.
        - Duplicate source labels: first occurrence wins.
        - MultiIndex source: NotImplementedError.
        - RangeIndex source: labels interpreted as ORIGINAL positional
          ints (0-based). Filtered RangeIndex: labels are ORIGINAL
          positions, not filtered-view enumeration.
        - Filtered source: labels matched against filtered view's index
          (via self.index property — MaterializedIndex with original
          labels).

        Memory: O(K × row_size) where K = len(labels). Requires
        K × row_size ≤ state_limit.

        Peak RAM = K × row_size + batch.
        """
        from cadw.dataset import Dataset
        from cadw.dataset.index import MaterializedIndex
        from cadw.core.memory_guard import MemoryLimitExceeded

        labels = list(labels)
        if not labels:
            raise ValueError("reindex: labels must be non-empty")

        if self._index_kind == "multi":
            raise NotImplementedError(
                "reindex: MultiIndex source not supported; use "
                ".reset_index() first"
            )

        # Resolve each label → position in FILTERED view (which is
        # what iloc[] expects).
        positions: list = []
        if self._index_kind == "range" and not self._row_filters:
            # Unfiltered RangeIndex: labels are positional ints.
            start = self._index_start
            stop = self._index_stop
            for lab in labels:
                try:
                    lab_int = int(lab)
                except (TypeError, ValueError):
                    positions.append(None)
                    continue
                if start <= lab_int < stop:
                    positions.append(lab_int - start)
                else:
                    positions.append(None)
        else:
            # Filtered (any kind) OR materialized:
            # self.index → MaterializedIndex with correct labels;
            # get_positions_all returns filtered-view positions.
            # Build label→pos map once (O(N + K) not O(K*N)).
            idx = self.index
            label_to_pos: dict = {}
            for i, x in enumerate(idx):
                try:
                    if x not in label_to_pos:
                        label_to_pos[x] = i
                except TypeError:
                    continue
            for lab in labels:
                try:
                    positions.append(label_to_pos.get(lab, None))
                except TypeError:
                    positions.append(None)

        valid_positions = [p for p in positions if p is not None]
        k_valid = len(valid_positions)

        # All-missing shortcut: null-filled table (avoid take-on-empty)
        if k_valid == 0:
            n = len(labels)
            arrays = [
                pa.nulls(n, type=self._dataset.schema.field(c).type)
                for c in self._columns
            ]
            result_table = pa.Table.from_arrays(
                arrays, names=self._columns,
            )
            return self._wrap_result_table(result_table)

        # Memory check
        first_batch = None
        for b in self._iter_filtered_batches():
            first_batch = b
            break
        if first_batch is None:
            row_bytes = max(
                1, self._memory_guard.estimate_row_size_from_schema(
                    pa.schema([self._dataset.schema.field(c)
                               for c in self._columns]),
                )
            )
        else:
            row_bytes = self._memory_guard.reestimate_from_batch(
                first_batch,
            )
        est = k_valid * row_bytes
        state_limit = self._memory_guard.state_limit_bytes
        if est > state_limit:
            raise MemoryLimitExceeded(
                f"reindex: K={k_valid} × {row_bytes} B/row = "
                f"{est} B > state_limit {state_limit} B. "
                f"Reduce labels count or increase budget."
            )
        self._memory_guard.check_state(
            est, name=f"reindex(K={k_valid})",
        )

        # Gather valid rows via iloc[list]
        valid_df = self.iloc[valid_positions]
        valid_table = valid_df.head(k_valid)

        # Map: output[i] = valid_table[valid_idx_counter] if
        # positions[i] is not None, else null.
        valid_idx_arr: list = []
        valid_idx_counter = 0
        for pos in positions:
            if pos is None:
                valid_idx_arr.append(None)
            else:
                valid_idx_arr.append(valid_idx_counter)
                valid_idx_counter += 1

        try:
            result_table = valid_table.take(
                pa.array(valid_idx_arr, type=pa.int64())
            )
        except (pa.ArrowInvalid, pa.ArrowIndexError) as e:
            raise RuntimeError(
                f"reindex: Table.take with null indices failed: {e}. "
                f"This PyArrow build may not support null-safe take."
            ) from e

        return self._wrap_result_table(result_table)


    def sort_values(self, by, *, ascending=True) -> "CadwDataFrame":
        from ._core import CadwDataFrame
        by_cols = [by] if isinstance(by, str) else list(by)
        for c in by_cols:
            if c not in self._columns:
                raise KeyError(f"sort_values: column {c!r} not in selection")
        # v0.9.24: per-column ascending. A list/tuple ascending is
        # interpreted positionally against by_cols (pandas semantics).
        # Prior code did `"ascending" if ascending else "descending"`
        # which treated any non-empty list as truthy -> all columns
        # ascending. That silently broke mixed-direction sorts.
        if isinstance(ascending, (list, tuple)):
            if len(ascending) != len(by_cols):
                raise ValueError(
                    f"sort_values: ascending has {len(ascending)} "
                    f"entries but by has {len(by_cols)} columns"
                )
            asc_seq = [bool(a) for a in ascending]
        else:
            asc_seq = [bool(ascending)] * len(by_cols)
        # in-memory fast path (v0.9.6b1): pa.Table.sort_by directly.
        # Avoids 1-5 ms external_sort overhead per small group —
        # critical for per-group apply (100k groups × 5 ms = 500 s).
        from cadw.dataset._in_memory import _InMemoryDataset
        if isinstance(self._dataset, _InMemoryDataset):
            sorted_table = self._dataset._table.select(self._columns).sort_by(
                [(c, "ascending" if a else "descending")
                 for c, a in zip(by_cols, asc_seq)],
            )
            return CadwDataFrame.from_table(
                sorted_table,
                memory_guard=self._memory_guard,
                spill_dir=self._spill_dir,
            )
        path = external_sort(
            self._iter_filtered_batches(),
            key_columns=by_cols,
            value_columns=self._columns,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "sort",
            ascending=asc_seq,
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

