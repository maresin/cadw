#!/usr/bin/env python3
"""v0.6.1a: MaterializedMultiIndex + set_index(list) + reset_index explode.

Scope:
- MaterializedMultiIndex: tuple labels from k equal-length pa.Array.
- df.set_index(["a","b"]) -> MaterializedMultiIndex.
- df.reset_index(drop=True/False) -> RangeIndex or k exploded columns.
- df.index returns MaterializedMultiIndex when multi.
- iloc[list] preserves multi index via raw positions.
- loc on multi -> NotImplementedError (v0.6.1b adds loc[tuple]).

Run: uv run python scripts/patch_v061a.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


# ============ helpers ============

def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str),
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:8]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-800:]}")
        sys.exit(r.returncode)
    return r.returncode


def git_current_branch():
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return r.stdout.strip()


def ensure_branch(name):
    cur = git_current_branch()
    if cur == name:
        print(f"  branch: already on {name}")
        return
    r = subprocess.run(
        ["git", "rev-parse", "--verify", name],
        capture_output=True,
    )
    if r.returncode == 0:
        print(f"  branch: checkout existing {name}")
        sh(["git", "checkout", name])
    else:
        print(f"  branch: create {name}")
        sh(["git", "checkout", "-b", name])


def require_clean():
    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)


def ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-1000:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def replace_exact(s, old, new, *, tag):
    if old not in s:
        raise SystemExit(f"[{tag}] pattern not found")
    n = s.count(old)
    if n > 1:
        raise SystemExit(f"[{tag}] pattern found {n} times (ambiguous)")
    return s.replace(old, new, 1)


# ============ MaterializedMultiIndex class ============

MULTI_INDEX_SRC = '''


class MaterializedMultiIndex(CadwIndex):
    """Multi-level index: tuple labels from k equal-length Arrow Arrays.

    pandas-style MultiIndex. Not subclass of MaterializedIndex — tuple
    semantics differ. Call sites check `_index_kind` discriminator:
      "range" | "materialized" | "multi".
    """

    __slots__ = ("_levels", "_n")

    def __init__(self, levels):
        if len(levels) < 2:
            raise ValueError(
                f"MaterializedMultiIndex requires at least 2 levels; "
                f"got {len(levels)}"
            )
        combined = []
        n = None
        for i, lv in enumerate(levels):
            if isinstance(lv, pa.ChunkedArray):
                if lv.num_chunks > 1:
                    lv = lv.combine_chunks()
                if lv.num_chunks == 1:
                    lv = lv.chunk(0)
            if n is None:
                n = len(lv)
            elif len(lv) != n:
                raise ValueError(
                    f"level {i} has {len(lv)} values, expected {n}"
                )
            combined.append(lv)
        self._levels = combined
        self._n = n if n is not None else 0

    @property
    def nlevels(self):
        return len(self._levels)

    @property
    def level_types(self):
        return [lv.type for lv in self._levels]

    def __len__(self):
        return self._n

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key += self._n
            if not (0 <= key < self._n):
                raise IndexError(f"index out of range: {key}")
            return tuple(lv[key].as_py() for lv in self._levels)
        if isinstance(key, slice):
            start, stop, _step = key.indices(self._n)
            return MaterializedMultiIndex([
                lv.slice(start, stop - start) for lv in self._levels
            ])
        raise TypeError(
            f"MaterializedMultiIndex: unsupported key {type(key).__name__}"
        )

    def __iter__(self):
        for i in range(self._n):
            yield tuple(lv[i].as_py() for lv in self._levels)

    def to_arrow(self):
        return pa.StructArray.from_arrays(
            self._levels,
            names=[f"level_{i}" for i in range(self.nlevels)],
        )

    def equals(self, other):
        if not isinstance(other, MaterializedMultiIndex):
            return False
        if self.nlevels != other.nlevels or self._n != other._n:
            return False
        return all(a.equals(b) for a, b in zip(self._levels, other._levels))

    def has_duplicates(self):
        seen = set()
        for t in self:
            if t in seen:
                return True
            seen.add(t)
        return False

    def get_indexer(self, labels):
        single = isinstance(labels, tuple)
        targets = [labels] if single else list(labels)
        m = {}
        for i, t in enumerate(self):
            if t not in m:
                m[t] = i
        return [m.get(t, -1) for t in targets]

    def get_positions_all(self, label):
        result = []
        for i, t in enumerate(self):
            if t == label:
                result.append(i)
        return result

    def is_monotonic_increasing(self):
        prev = None
        first = True
        for t in self:
            if first:
                prev = t
                first = False
                continue
            try:
                if t < prev:
                    return False
            except TypeError:
                return False
            prev = t
        return True

    def is_monotonic_decreasing(self):
        prev = None
        first = True
        for t in self:
            if first:
                prev = t
                first = False
                continue
            try:
                if t > prev:
                    return False
            except TypeError:
                return False
            prev = t
        return True

    def is_monotonic(self):
        if self._n <= 1:
            return True
        return self.is_monotonic_increasing() or self.is_monotonic_decreasing()

    def __repr__(self):
        preview = list(self)[:5]
        suffix = "..." if self._n > 5 else ""
        return (
            f"MaterializedMultiIndex({preview}{suffix}, "
            f"nlevels={self.nlevels}, n={self._n})"
        )
'''


def patch_index_py():
    print("\n=== cadw/dataset/index.py: MaterializedMultiIndex ===")
    p = Path("cadw/dataset/index.py")
    s = p.read_text()
    if "class MaterializedMultiIndex" in s:
        print("  already present — skip")
        return
    s = s.rstrip() + "\n" + MULTI_INDEX_SRC + "\n"
    p.write_text(s)
    compile_check(p)
    print("  OK: MaterializedMultiIndex appended")


# ============ cadw/dataframe.py ============

def patch_dataframe_py():
    print("\n=== cadw/dataframe.py ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    # 1. Import MaterializedMultiIndex
    old = '''from cadw.dataset import (
    Dataset,
    CadwIndex,
    RangeIndex,
    MaterializedIndex,
    RowFilter,
    MaskFilter,
    PredicateFilter,
    SliceFilter,
)'''
    new = '''from cadw.dataset import (
    Dataset,
    CadwIndex,
    RangeIndex,
    MaterializedIndex,
    MaterializedMultiIndex,
    RowFilter,
    MaskFilter,
    PredicateFilter,
    SliceFilter,
)'''
    s = replace_exact(s, old, new, tag="import-block")

    # 2. __init__: add _index_levels
    old = '''        # Index state: "range" (lazy) or "materialized" (Arrow Array).
        self._index_kind: str = "range"
        self._index_start: int = 0
        self._index_stop: int = dataset.total_rows
        self._index_values: pa.Array | None = None'''
    new = '''        # Index state: "range" (lazy) | "materialized" (single Arrow Array)
        # | "multi" (list of Arrays, tuple labels).
        self._index_kind: str = "range"
        self._index_start: int = 0
        self._index_stop: int = dataset.total_rows
        self._index_values: pa.Array | None = None
        self._index_levels: list[pa.Array] | None = None'''
    s = replace_exact(s, old, new, tag="init")

    # 3. index property (no filter): handle multi
    old = '''        if not self._row_filters:
            if self._index_kind == "range":
                return RangeIndex(self._index_start, self._index_stop)
            assert self._index_values is not None
            return MaterializedIndex(self._index_values)'''
    new = '''        if not self._row_filters:
            if self._index_kind == "range":
                return RangeIndex(self._index_start, self._index_stop)
            if self._index_kind == "multi":
                assert self._index_levels is not None
                return MaterializedMultiIndex(self._index_levels)
            assert self._index_values is not None
            return MaterializedIndex(self._index_values)'''
    s = replace_exact(s, old, new, tag="index-prop-no-filter")

    # 4. index property (filtered): handle multi
    old = '''        if self._index_kind == "range":
            values = positions + self._index_start
            return MaterializedIndex(pa.array(values, type=pa.int64()))
        assert self._index_values is not None
        values = self._index_values.take(
            pa.array(positions, type=pa.int64())
        )
        return MaterializedIndex(values)'''
    new = '''        if self._index_kind == "range":
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
        return MaterializedIndex(values)'''
    s = replace_exact(s, old, new, tag="index-prop-filtered")

    # 5. set_index signature
    old = '''    def set_index(
        self,
        col: str,
        drop: bool = True,
        inplace: bool = False,
    ) -> "CadwDataFrame":'''
    new = '''    def set_index(
        self,
        col: "str | Sequence[str]",
        drop: bool = True,
        inplace: bool = False,
    ) -> "CadwDataFrame":'''
    s = replace_exact(s, old, new, tag="set-index-sig")

    # 6. set_index body
    old = '''        if not isinstance(col, str):
            raise TypeError(
                f"set_index: col must be str; got {type(col).__name__}"
            )
        if col not in self._columns:
            raise KeyError(
                f"set_index: column {col!r} not in {self._columns}"
            )
        if not isinstance(drop, bool):
            raise TypeError("set_index: drop must be bool")

        # v0.4.0d: материализуем col для **всего** dataset (не filtered).
        # Это O(N) scan, но `_index_values` всегда full → filtered.index =
        # subset через positions. §13.4/59.
        chunks: list[pa.Array] = []
        for b in self._dataset.iter_batches(columns=[col]):
            chunks.append(b.column(col))
        if not chunks:
            values: pa.Array | pa.ChunkedArray = pa.array(
                [], type=self._dataset.schema.field(col).type
            )
        elif len(chunks) == 1:
            values = chunks[0]
        else:
            values = pa.chunked_array(chunks)

        new_columns = list(self._columns)
        if drop:
            new_columns.remove(col)

        if inplace:
            self._columns = new_columns
            self._index_kind = "materialized"
            self._index_values = (
                values.combine_chunks()
                if isinstance(values, pa.ChunkedArray)
                else values
            )
            self._index_start = 0
            self._index_stop = len(self._index_values)
            return self

        new = CadwDataFrame.__new__(CadwDataFrame)
        new._dataset = self._dataset
        new._memory_guard = self._memory_guard
        new._spill_dir = self._spill_dir
        new._columns = new_columns
        new._row_filters = list(self._row_filters)
        new._index_kind = "materialized"
        new._index_values = (
            values.combine_chunks()
            if isinstance(values, pa.ChunkedArray)
            else values
        )
        new._index_start = 0
        new._index_stop = len(new._index_values)
        return new'''
    new = '''        # v0.6.1a: str or list[str] (MultiIndex if len >= 2).
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
        return new'''
    s = replace_exact(s, old, new, tag="set-index-body")

    # 7. reset_index: multi branch
    old = '''        # materialized
        assert self._index_values is not None
        values = self._index_values'''
    new = '''        if self._index_kind == "multi":
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
        values = self._index_values'''
    s = replace_exact(s, old, new, tag="reset-index-multi")

    # 8. _with_filter: copy _index_levels
    old = '''        new._index_kind = self._index_kind
        new._index_start = self._index_start
        new._index_stop = self._index_stop
        new._index_values = self._index_values
        return new'''
    new = '''        new._index_kind = self._index_kind
        new._index_start = self._index_start
        new._index_stop = self._index_stop
        new._index_values = self._index_values
        new._index_levels = self._index_levels
        return new'''
    s = replace_exact(s, old, new, tag="with-filter")

    # 9. __getitem__(list) selection: copy _index_levels
    old = '''            new._row_filters = list(self._row_filters)
            new._index_kind = self._index_kind
            new._index_start = self._index_start
            new._index_stop = self._index_stop
            new._index_values = self._index_values
            return new'''
    new = '''            new._row_filters = list(self._row_filters)
            new._index_kind = self._index_kind
            new._index_start = self._index_start
            new._index_stop = self._index_stop
            new._index_values = self._index_values
            new._index_levels = self._index_levels
            return new'''
    s = replace_exact(s, old, new, tag="getitem-list-copy")

    p.write_text(s)
    compile_check(p)
    print("  OK: dataframe.py patched")


# ============ cadw/dataset/__init__.py ============

def patch_dataset_init():
    print("\n=== cadw/dataset/__init__.py ===")
    p = Path("cadw/dataset/__init__.py")
    s = p.read_text()
    if "MaterializedMultiIndex" in s:
        print("  already present — skip")
        return
    old = "from .index import CadwIndex, RangeIndex, MaterializedIndex"
    new = ("from .index import (\n"
           "    CadwIndex,\n"
           "    RangeIndex,\n"
           "    MaterializedIndex,\n"
           "    MaterializedMultiIndex,\n"
           ")")
    s = replace_exact(s, old, new, tag="dataset-init-import")
    old_all = '''__all__ = [
    "Dataset",
    "CadwIndex",
    "RangeIndex",
    "MaterializedIndex",'''
    new_all = '''__all__ = [
    "Dataset",
    "CadwIndex",
    "RangeIndex",
    "MaterializedIndex",
    "MaterializedMultiIndex",'''
    s = replace_exact(s, old_all, new_all, tag="dataset-init-all")
    p.write_text(s)
    compile_check(p)


# ============ cadw/__init__.py ============

def patch_top_init():
    print("\n=== cadw/__init__.py ===")
    p = Path("cadw/__init__.py")
    s = p.read_text()
    if "MaterializedMultiIndex" in s:
        print("  already present — skip")
        return
    old = "from .dataset import (\n    Dataset,\n    CadwIndex, RangeIndex, MaterializedIndex,"
    new = "from .dataset import (\n    Dataset,\n    CadwIndex, RangeIndex, MaterializedIndex, MaterializedMultiIndex,"
    s = replace_exact(s, old, new, tag="top-init-import")
    old_all = '"CadwIndex",\n    "RangeIndex",\n    "MaterializedIndex",'
    new_all = '"CadwIndex",\n    "RangeIndex",\n    "MaterializedIndex",\n    "MaterializedMultiIndex",'
    s = replace_exact(s, old_all, new_all, tag="top-init-all")
    p.write_text(s)
    compile_check(p)


# ============ cadw/indexers.py: _rebuild_index, _table_to_cadwdf, loc multi error ============

def patch_indexers_py():
    print("\n=== cadw/indexers.py ===")
    p = Path("cadw/indexers.py")
    s = p.read_text()

    # 1. _rebuild_index — multi-aware
    old = '''def _rebuild_index(parent: "CadwDataFrame", raw_positions: list[int]):
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
    return MaterializedIndex(pa.array(vals, type=src.type))'''
    new = '''def _rebuild_index(parent: "CadwDataFrame", raw_positions: list[int]):
    """Rebuild index for iloc[list] result (v0.6.1a: multi-aware).

    - RangeIndex → MaterializedIndex([start+p, ...]).
    - MaterializedIndex → subset.
    - MaterializedMultiIndex → per-level subset.
    """
    from cadw.dataset.index import MaterializedIndex, MaterializedMultiIndex

    if parent._index_kind == "range":
        vals = [parent._index_start + p for p in raw_positions]
        return MaterializedIndex(pa.array(vals, type=pa.int64()))

    if parent._index_kind == "multi":
        assert parent._index_levels is not None
        idx = pa.array(raw_positions, type=pa.int64())
        new_levels = [lv.take(idx) for lv in parent._index_levels]
        return MaterializedMultiIndex(new_levels)

    src = parent._index_values
    vals = [src[p] for p in raw_positions]
    return MaterializedIndex(pa.array(vals, type=src.type))'''
    s = replace_exact(s, old, new, tag="rebuild-index-multi")

    # 2. _table_to_cadwdf — multi-aware
    old = '''    if new_index is not None:
        new._index_kind = "materialized"
        new._index_values = new_index.to_arrow()
        new._index_start = 0
        new._index_stop = len(new_index)
    return new'''
    new = '''    if new_index is not None:
        from cadw.dataset.index import MaterializedMultiIndex
        if isinstance(new_index, MaterializedMultiIndex):
            new._index_kind = "multi"
            new._index_values = None
            new._index_levels = list(new_index._levels)
        else:
            new._index_kind = "materialized"
            new._index_values = new_index.to_arrow()
            new._index_levels = None
        new._index_start = 0
        new._index_stop = len(new_index)
    return new'''
    s = replace_exact(s, old, new, tag="table-to-cadwdf-multi")

    # 3. _LocIndexer._materialized: clearer error for multi
    old = '''    def _materialized(self):
        from cadw.dataset.index import MaterializedIndex
        p = self._parent
        if p._index_kind == "range":
            raise TypeError(
                "loc: requires MaterializedIndex (after set_index). "
                "RangeIndex — use iloc."
            )
        # v0.4.0d: с фильтрами тоже работаем — filtered.index даёт
        # MaterializedIndex-subset, поверх него get_indexer работает.
        if p._row_filters:'''
    new = '''    def _materialized(self):
        from cadw.dataset.index import MaterializedIndex
        p = self._parent
        if p._index_kind == "range":
            raise TypeError(
                "loc: requires MaterializedIndex (after set_index). "
                "RangeIndex — use iloc."
            )
        if p._index_kind == "multi":
            raise NotImplementedError(
                "loc: MultiIndex not supported in v0.6.1a. "
                "Use reset_index + loc, or iloc. v0.6.1b adds loc[tuple]."
            )
        # v0.4.0d: с фильтрами тоже работаем — filtered.index даёт
        # MaterializedIndex-subset, поверх него get_indexer работает.
        if p._row_filters:'''
    s = replace_exact(s, old, new, tag="loc-multi-error")

    p.write_text(s)
    compile_check(p)


# ============ tests ============

TESTS_SRC = '''"""v0.6.1a: MaterializedMultiIndex + set_index(list) + reset_index explode."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.dataset.index import (
    RangeIndex,
    MaterializedIndex,
    MaterializedMultiIndex,
)


def make_df(tmp_path, name="L", data=None, chunk=3):
    if data is None:
        data = {
            "a": ["x", "x", "y", "y", "z"],
            "b": [1, 2, 1, 2, 1],
            "v": [10, 20, 30, 40, 50],
        }
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- MaterializedMultiIndex ----------

def test_multi_index_nlevels():
    idx = MaterializedMultiIndex([
        pa.array(["a", "b"]), pa.array([1, 2]),
    ])
    assert idx.nlevels == 2 and len(idx) == 2


def test_multi_index_getitem_int():
    idx = MaterializedMultiIndex([
        pa.array(["a", "b", "c"]), pa.array([1, 2, 3]),
    ])
    assert idx[0] == ("a", 1) and idx[-1] == ("c", 3)


def test_multi_index_getitem_slice():
    idx = MaterializedMultiIndex([
        pa.array(["a", "b", "c"]), pa.array([1, 2, 3]),
    ])
    sub = idx[1:3]
    assert isinstance(sub, MaterializedMultiIndex)
    assert list(sub) == [("b", 2), ("c", 3)]


def test_multi_index_iter():
    idx = MaterializedMultiIndex([
        pa.array(["a", "b"]), pa.array([1, 2]),
    ])
    assert list(idx) == [("a", 1), ("b", 2)]


def test_multi_index_to_arrow():
    idx = MaterializedMultiIndex([
        pa.array(["a", "b"]), pa.array([1, 2]),
    ])
    arr = idx.to_arrow()
    assert pa.types.is_struct(arr.type)
    assert arr.type.num_fields == 2
    assert arr[0].as_py() == {"level_0": "a", "level_1": 1}


def test_multi_index_equals():
    a = MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1, 2])])
    b = MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1, 2])])
    c = MaterializedMultiIndex([pa.array(["a", "c"]), pa.array([1, 2])])
    assert a.equals(b) and not a.equals(c)


def test_multi_index_has_duplicates():
    a = MaterializedMultiIndex([pa.array(["a", "b", "a"]), pa.array([1, 2, 1])])
    assert a.has_duplicates()
    b = MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1, 2])])
    assert not b.has_duplicates()


def test_multi_index_get_indexer():
    idx = MaterializedMultiIndex([
        pa.array(["a", "b", "c"]), pa.array([1, 2, 3]),
    ])
    assert idx.get_indexer(("b", 2)) == [1]
    assert idx.get_indexer(("z", 99)) == [-1]
    assert idx.get_indexer([("a", 1), ("c", 3)]) == [0, 2]


def test_multi_index_monotonic():
    a = MaterializedMultiIndex([
        pa.array(["a", "a", "b"]), pa.array([1, 2, 1]),
    ])
    assert a.is_monotonic_increasing()
    assert not a.is_monotonic_decreasing()


def test_multi_index_requires_two_levels():
    with pytest.raises(ValueError, match="at least 2"):
        MaterializedMultiIndex([pa.array(["a"])])


def test_multi_index_length_mismatch():
    with pytest.raises(ValueError, match="level 1"):
        MaterializedMultiIndex([pa.array(["a", "b"]), pa.array([1])])


# ---------- set_index multi ----------

def test_set_index_multi_drop_true(tmp_path):
    df = make_df(tmp_path)
    new = df.set_index(["a", "b"], drop=True)
    assert isinstance(new.index, MaterializedMultiIndex)
    assert new.index.nlevels == 2
    assert list(new.index) == [
        ("x", 1), ("x", 2), ("y", 1), ("y", 2), ("z", 1),
    ]
    assert "a" not in new.columns and "b" not in new.columns
    assert "v" in new.columns
    assert "a" in df.columns  # original unchanged


def test_set_index_multi_drop_false(tmp_path):
    df = make_df(tmp_path)
    new = df.set_index(["a", "b"], drop=False)
    assert isinstance(new.index, MaterializedMultiIndex)
    assert "a" in new.columns and "b" in new.columns


def test_set_index_multi_inplace(tmp_path):
    df = make_df(tmp_path)
    result = df.set_index(["a", "b"], drop=True, inplace=True)
    assert result is df
    assert isinstance(df.index, MaterializedMultiIndex)


def test_set_index_single_still_works(tmp_path):
    df = make_df(tmp_path)
    new = df.set_index("a", drop=False)
    assert isinstance(new.index, MaterializedIndex)
    assert list(new.index) == ["x", "x", "y", "y", "z"]


# ---------- reset_index multi ----------

def test_reset_index_multi_drop_true(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index(["a", "b"], drop=True)
    restored = with_idx.reset_index(drop=True)
    assert isinstance(restored.index, RangeIndex)
    assert restored.index.stop == 5


def test_reset_index_multi_drop_false(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index(["a", "b"], drop=True)
    restored = with_idx.reset_index(drop=False)
    assert isinstance(restored.index, RangeIndex)
    assert "level_0" in restored.columns and "level_1" in restored.columns
    out = tmp_path / "o.parquet"
    restored.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["level_0"] for r in rows] == ["x", "x", "y", "y", "z"]
    assert [r["level_1"] for r in rows] == [1, 2, 1, 2, 1]


def test_reset_index_multi_collision(tmp_path):
    df = make_df(tmp_path, data={
        "a": ["x", "y"], "b": [1, 2],
        "level_0": [100, 200], "v": [10, 20],
    })
    with_idx = df.set_index(["a", "b"], drop=True)
    restored = with_idx.reset_index(drop=False)
    assert "level_0" in restored.columns
    assert "level_0_0" in restored.columns
    out = tmp_path / "o.parquet"
    restored.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["level_0_0"] for r in rows] == ["x", "y"]


# ---------- filter + iloc + parquet ----------

def test_index_multi_with_filter(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index(["a", "b"], drop=False)
    filtered = with_idx[with_idx["v"] > 15]
    idx = filtered.index
    assert isinstance(idx, MaterializedMultiIndex)
    assert list(idx) == [("x", 2), ("y", 1), ("y", 2), ("z", 1)]


def test_iloc_list_preserves_multi_index(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index(["a", "b"], drop=False)
    sub = with_idx.iloc[[0, 2, 4]]
    assert isinstance(sub.index, MaterializedMultiIndex)
    assert list(sub.index) == [("x", 1), ("y", 1), ("z", 1)]


def test_multi_index_via_parquet(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index(["a", "b"], drop=True)
    restored = with_idx.reset_index(drop=False)
    out = tmp_path / "o.parquet"
    restored.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 5
    assert "level_0" in rows[0] and "level_1" in rows[0]
'''


def write_tests():
    print("\n=== tests/test_multiindex.py ===")
    p = Path("tests/test_multiindex.py")
    p.write_text(TESTS_SRC)
    compile_check(p)


# ============ main ============

def main():
    ts_str = ts()
    branch = "patch/v061a-multi-index"

    print("=== v0.6.1a: MultiIndex ===")
    ensure_branch(branch)
    require_clean()

    backup_dir = Path(f".backups/backup-pre-v061a-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v061a-{ts_str}",
        "-m", "before patch v0.6.1a MultiIndex"])

    patch_index_py()
    patch_dataframe_py()
    patch_dataset_init()
    patch_top_init()
    patch_indexers_py()
    write_tests()

    print("\n=== pytest tests/test_multiindex.py ===")
    r = subprocess.run(["uv", "run", "pytest", "tests/test_multiindex.py", "-v"])
    if r.returncode != 0:
        print("\n*** FAILURE — WIP commit for debugging ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.6.1a (test red)"], check=False)
        sys.exit(1)

    print("\n=== pytest full fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest full slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.6.1a: MaterializedMultiIndex + set_index(list) + reset_index explode"])

    print("\nv0.6.1a applied. Next: v0.6.1b (loc[tuple]).")


if __name__ == "__main__":
    main()