"""
Indexers: .iloc / .loc (pandas-compatible), v0.4.0b MVP.

`_ILocIndexer` — positional indexer. Позиция = позиция в текущем view
(filtered stream или raw). Возвращает:
  - `iloc[int]` → dict[str, Any] (row values by column).
  - `iloc[slice]` → CadwDataFrame (lazy, via SliceFilter).
  - `iloc[list[int]]` → CadwDataFrame (materialize K=len(list) rows).

Ограничения v0.4.0b:
- `iloc[bool_array]` — raise (use df[mask]).
- `iloc[list]` на filtered view → default RangeIndex (не сохраняет
  parent index; v0.4.0c — filter-compat).
- `iloc[list]` на unfiltered → preserves parent index (RangeIndex →
  MaterializedIndex; MaterializedIndex → subset values).
- `iloc[slice]` полностью lazy (не читает данные).

Почему indexers в отдельном модуле от dataframe.py: AST guard
`test_facade_has_no_as_py` проверяет весь модуль dataframe.py. `iloc[int]`
использует `.as_py()` (O(K), K=1 или len(list)) — это не O(N), но facade
guard не различает. Отдельный модуль — обход, и семантически правильнее
(facade без data-примитивов; indexers — новый layer).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from cadw.dataframe import CadwDataFrame


class _ILocIndexer:
    def __init__(self, parent: "CadwDataFrame") -> None:
        self._parent = parent

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._slice(key)
        if isinstance(key, int) and not isinstance(key, bool):
            return self._one(key)
        if isinstance(key, list) and all(
            isinstance(x, int) and not isinstance(x, bool) for x in key
        ):
            return self._many(key)
        if isinstance(key, pa.Array) and pa.types.is_boolean(key.type):
            raise NotImplementedError(
                "iloc[bool_array]: not supported in v0.4.0b. Use df[mask]."
            )
        raise TypeError(
            f"iloc: unsupported key type {type(key).__name__}; "
            f"expected int, slice, or list[int]"
        )

    def _view_size(self) -> int:
        p = self._parent
        if p._row_filters:
            return p.count()
        return p._dataset.total_rows

    def _resolve_int(self, i: int) -> int:
        if i < 0:
            n = self._view_size()
            i = n + i
            if i < 0:
                raise IndexError(f"iloc: index out of range (n={n})")
        return i

    def _one(self, i: int) -> dict[str, Any]:
        i = self._resolve_int(i)
        running = 0
        for b in self._parent._iter_filtered_batches():
            if running + b.num_rows > i:
                local = i - running
                return {
                    name: b.column(name)[local].as_py()
                    for name in b.schema.names
                }
            running += b.num_rows
        raise IndexError(f"iloc: position {i} out of range (n={running})")

    def _slice(self, key: slice):
        n = self._view_size()
        start, stop, step = key.indices(n)
        # Delegates to __getitem__(slice) = SliceFilter — lazy + already
        # composes with existing filters.
        return self._parent[slice(start, stop, step)]

    def _many(self, positions: list[int]):
        resolved = [self._resolve_int(p) for p in positions]
        if not resolved:
            return self._empty_view()

        targets = sorted(set(resolved))
        target_set = set(targets)

        # v0.4.0d: use _iter_filtered_batches_with_positions to also
        # capture raw-dataset positions (for index preservation).
        collected: dict[int, dict[str, Any]] = {}
        raw_pos_collected: dict[int, int] = {}
        running = 0
        for b, raw_positions in self._parent._iter_filtered_batches_with_positions():
            n = b.num_rows
            batch_start = running
            batch_end = running + n
            for p in targets:
                if batch_start <= p < batch_end:
                    local = p - batch_start
                    collected[p] = {
                        name: b.column(name)[local].as_py()
                        for name in b.schema.names
                    }
                    raw_pos_collected[p] = int(raw_positions[local])
            running = batch_end
            if len(collected) == len(targets):
                break

        missing = target_set - set(collected.keys())
        if missing:
            raise IndexError(
                f"iloc: positions out of range: {sorted(missing)[:5]} "
                f"(n={running})"
            )

        rows = [collected[p] for p in resolved]  # keep order + duplicates
        raw_pos = [raw_pos_collected[p] for p in resolved]
        return self._rows_to_cadwdf(rows, raw_pos)

    def _empty_view(self):
        parent = self._parent
        schema = pa.schema(
            [parent._dataset.schema.field(c) for c in parent._columns]
        )
        table = pa.Table.from_batches([], schema=schema)
        return _table_to_cadwdf(table, parent, None)

    def _rows_to_cadwdf(self, rows, resolved_positions):
        parent = self._parent
        schema = pa.schema(
            [parent._dataset.schema.field(c) for c in parent._columns]
        )
        table = pa.Table.from_pylist(rows, schema=schema)
        new_index = _rebuild_index(parent, resolved_positions)
        return _table_to_cadwdf(table, parent, new_index)


def _rebuild_index(parent: "CadwDataFrame", raw_positions: list[int]):
    """Rebuild index for iloc[list] result (v0.4.0d — filter-compat).

    `raw_positions` — позиции в **unfiltered** dataset (из
    _iter_filtered_batches_with_positions).

    - parent RangeIndex → MaterializedIndex([start + p, ...]).
    - parent MaterializedIndex → subset `_index_values[p]`.
    """
    from cadw.dataset.index import MaterializedIndex

    if parent._index_kind == "range":
        vals = [parent._index_start + p for p in raw_positions]
        return MaterializedIndex(pa.array(vals, type=pa.int64()))

    src = parent._index_values
    vals = [src[p] for p in raw_positions]
    return MaterializedIndex(pa.array(vals, type=src.type))


def _table_to_cadwdf(table: pa.Table, parent: "CadwDataFrame", new_index):
    """Build a CadwDataFrame from a materialized table (+ optional index)."""
    from cadw.dataframe import CadwDataFrame
    from cadw.dataset import Dataset

    new_ds = Dataset.from_batches(
        table.schema, iter(table.to_batches()), parent._new_dataset_dir()
    )
    new = CadwDataFrame(
        new_ds,
        memory_guard=parent._memory_guard,
        spill_dir=parent._spill_dir,
    )
    if new_index is not None:
        new._index_kind = "materialized"
        new._index_values = new_index.to_arrow()
        new._index_start = 0
        new._index_stop = len(new_index)
    return new

class _LocIndexer:
    """Label-based indexer (pandas-style). Требует MaterializedIndex.

    v0.4.1: open endpoints (`loc[:hi]`, `loc[lo:]`, `loc[:]`) на monotonic
    index (asc или desc). Duplicate labels: `loc[label]` → CadwDataFrame
    (несколько строк) вместо dict. `loc[slice]` требует уникального
    индекса (pandas: cannot slice non-unique) и монотонности.
    """

    def __init__(self, parent: "CadwDataFrame") -> None:
        self._parent = parent

    def _materialized(self):
        from cadw.dataset.index import MaterializedIndex
        p = self._parent
        if p._index_kind == "range":
            raise TypeError(
                "loc: requires MaterializedIndex (after set_index). "
                "RangeIndex — use iloc."
            )
        if p._row_filters:
            idx = p.index
            if not isinstance(idx, MaterializedIndex):
                raise TypeError(
                    f"loc: filtered index is {type(idx).__name__}, "
                    f"expected MaterializedIndex"
                )
            return idx
        assert p._index_values is not None
        return MaterializedIndex(p._index_values)

    def __getitem__(self, key):
        idx = self._materialized()
        if isinstance(key, slice):
            return self._slice(key, idx)
        if isinstance(key, list):
            return self._many(key, idx)
        if isinstance(key, pa.Array) and pa.types.is_boolean(key.type):
            raise NotImplementedError(
                "loc[bool_array]: not supported. Use df[mask]."
            )
        # scalar label — duplicates → CadwDataFrame; unique → dict.
        positions = idx.get_positions_all(key)
        if not positions:
            raise KeyError(f"loc: label {key!r} not in index")
        if len(positions) == 1:
            return self._parent.iloc[positions[0]]
        return self._parent.iloc[positions]

    def _slice(self, key, idx):
        if key.step is not None and key.step != 1:
            raise NotImplementedError(
                "loc[slice]: step not supported in v0.4.1."
            )
        if idx.has_duplicates():
            raise NotImplementedError(
                "loc[slice]: requires unique index in v0.4.1. "
                "Duplicate labels — use loc[list_of_labels]."
            )

        n = len(idx)
        if key.start is None and key.stop is None:
            # loc[:] — все строки (lazy view).
            return self._parent[slice(0, n)]

        pl = None
        ph = None
        if key.start is not None:
            pl = idx.get_indexer(key.start)[0]
            if pl < 0:
                raise KeyError(f"loc: label {key.start!r} not in index")
        if key.stop is not None:
            ph = idx.get_indexer(key.stop)[0]
            if ph < 0:
                raise KeyError(f"loc: label {key.stop!r} not in index")

        if pl is not None and ph is not None:
            # explicit [lo:hi]
            if idx.is_monotonic_increasing():
                if pl <= ph:
                    return self._parent.iloc[list(range(pl, ph + 1))]
                return self._parent.iloc[[]]
            if idx.is_monotonic_decreasing():
                if ph <= pl:
                    return self._parent.iloc[list(range(ph, pl + 1))]
                return self._parent.iloc[[]]
            raise NotImplementedError(
                "loc[slice]: requires monotonic index (incr or decr)."
            )

        if pl is None:
            # [ : hi] — from start up to hi (inclusive), any monotonic dir.
            if not idx.is_monotonic():
                raise NotImplementedError(
                    "loc[slice]: requires monotonic index (incr or decr)."
                )
            return self._parent.iloc[list(range(0, ph + 1))]

        # [lo : ] — from lo to end.
        if not idx.is_monotonic():
            raise NotImplementedError(
                "loc[slice]: requires monotonic index (incr or decr)."
            )
        return self._parent.iloc[list(range(pl, n))]

    def _many(self, labels: list, idx):
        all_positions: list[int] = []
        missing: list = []
        for lab in labels:
            pos = idx.get_positions_all(lab)
            if not pos:
                missing.append(lab)
            else:
                all_positions.extend(pos)
        if missing:
            raise KeyError(f"loc: labels not in index: {missing[:5]}")
        return self._parent.iloc[all_positions]


class _AtIndexer:
    """`.at[label, col]` — scalar access by label + column (v0.4.1)."""

    def __init__(self, parent: "CadwDataFrame") -> None:
        self._parent = parent

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError(
                f".at: key must be (label, col); got {type(key).__name__}"
            )
        label, col = key
        row = self._parent.loc[label]
        if not isinstance(row, dict):
            raise KeyError(
                f".at: label {label!r} is not unique in index; "
                f"use .loc[{label!r}] for multiple rows"
            )
        if col not in row:
            raise KeyError(f".at: column {col!r} not in df.columns")
        return row[col]

    def __setitem__(self, key, value):
        raise NotImplementedError(
            ".at.__setitem__: CadwDataFrame immutable in v0.4.0c. "
            "Use df.write_parquet + reload to mutate."
        )


class _IAtIndexer:
    """`.iat[pos, col]` — scalar access by position + column."""

    def __init__(self, parent: "CadwDataFrame") -> None:
        self._parent = parent

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError(
                f".iat: key must be (pos, col); got {type(key).__name__}"
            )
        pos, col = key
        row = self._parent.iloc[pos]
        if col not in row:
            raise KeyError(f".iat: column {col!r} not in df.columns")
        return row[col]

    def __setitem__(self, key, value):
        raise NotImplementedError(
            ".iat.__setitem__: CadwDataFrame immutable in v0.4.0c."
        )

