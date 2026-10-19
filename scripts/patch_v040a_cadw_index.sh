#!/usr/bin/env bash
# patch_v040a_cadw_index: CadwIndex + RangeIndex + MaterializedIndex +
# df.index + set_index + reset_index. MVP без filter-compat.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-v040a-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch v0.4.0a CadwIndex"

BR="patch/v040a-cadw-index"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- cadw/dataset/index.py ----------
cat > cadw/dataset/index.py <<'PY'
"""
CadwIndex: идентичность строк в CadwDataFrame (v0.4.0a).

Два класса:
  - RangeIndex(start, stop, step) — lazy, O(1) память. Аналог pandas
    RangeIndex. Не материализуется.
  - MaterializedIndex(values) — Arrow Array/ChunkedArray. Для index
    из колонки (set_index).

Ограничение v0.4.0a: `df.index` и `set_index`/`reset_index` работают
только без активных фильтров. С фильтрами:
  - `df.index` — raise NotImplementedError (RangeIndex+filter = O(N) scan).
  - `set_index`/`reset_index` — raise NotImplementedError.
Синхронизация index с фильтрами — v0.4.0b (iloc).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import pyarrow as pa


class CadwIndex(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, key): ...

    @abstractmethod
    def __iter__(self) -> Iterator: ...

    @abstractmethod
    def to_arrow(self) -> pa.Array: ...


class RangeIndex(CadwIndex):
    """Lazy index: start, start+step, ..., < stop."""

    def __init__(self, start: int, stop: int, step: int = 1) -> None:
        if step <= 0:
            raise ValueError(f"RangeIndex: step must be > 0; got {step}")
        if start < 0:
            raise ValueError(f"RangeIndex: start must be >= 0; got {start}")
        if stop < start:
            raise ValueError(
                f"RangeIndex: stop ({stop}) must be >= start ({start})"
            )
        self.start = int(start)
        self.stop = int(stop)
        self.step = int(step)

    def __len__(self) -> int:
        return (self.stop - self.start) // self.step

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            if not (0 <= key < len(self)):
                raise IndexError(f"RangeIndex: index out of range: {key}")
            return self.start + key * self.step
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            new_start = self.start + start * self.step
            new_stop = self.start + stop * self.step
            new_step = self.step * step if step else self.step
            return RangeIndex(new_start, new_stop, new_step)
        raise TypeError(f"RangeIndex: unsupported key {type(key).__name__}")

    def __iter__(self) -> Iterator[int]:
        for i in range(self.start, self.stop, self.step):
            yield i

    def to_arrow(self) -> pa.Array:
        return pa.array(list(self), type=pa.int64())

    def equals(self, other: "CadwIndex") -> bool:
        if not isinstance(other, RangeIndex):
            return False
        return (
            self.start == other.start
            and self.stop == other.stop
            and self.step == other.step
        )

    def __repr__(self) -> str:
        return f"RangeIndex(start={self.start}, stop={self.stop}, step={self.step})"


class MaterializedIndex(CadwIndex):
    """Materialized index: Arrow Array/ChunkedArray (сохраняет исходный dtype)."""

    def __init__(self, values: pa.Array | pa.ChunkedArray) -> None:
        if isinstance(values, pa.ChunkedArray):
            # combine_chunks для детерминированного доступа; O(N) память,
            # но index обычно уже материализован caller'ом.
            if values.num_chunks > 1:
                values = values.combine_chunks()
            values = values.chunk(0) if values.num_chunks == 1 else values
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key].as_py()
        if isinstance(key, slice):
            start, stop, _step = key.indices(len(self))
            return MaterializedIndex(self._values.slice(start, stop - start))
        raise TypeError(
            f"MaterializedIndex: unsupported key {type(key).__name__}"
        )

    def __iter__(self) -> Iterator:
        yield from self._values.to_pylist()

    def to_arrow(self) -> pa.Array:
        return self._values

    @property
    def dtype(self) -> pa.DataType:
        return self._values.type

    def equals(self, other: "CadwIndex") -> bool:
        if not isinstance(other, MaterializedIndex):
            return False
        if len(self) != len(other):
            return False
        if self._values.type != other._values.type:
            return False
        return self._values.equals(other._values)

    def __repr__(self) -> str:
        n = len(self)
        preview = self._values.slice(0, min(5, n)).to_pylist()
        suffix = "..." if n > 5 else ""
        return f"MaterializedIndex({preview}{suffix}, n={n})"
PY

# ---------- cadw/dataset/__init__.py: экспорт ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataset/__init__.py")
s = p.read_text()
if "from .index import" not in s:
    s = s.replace(
        "from .dataset import Dataset",
        "from .dataset import Dataset\n"
        "from .index import CadwIndex, RangeIndex, MaterializedIndex",
        1,
    )
    s = s.replace(
        '__all__ = [\n    "Dataset",',
        '__all__ = [\n    "Dataset",\n    "CadwIndex",\n    "RangeIndex",\n    "MaterializedIndex",',
        1,
    )
    p.write_text(s)
    print("dataset/__init__.py: CadwIndex/RangeIndex/MaterializedIndex exported")
PY

# ---------- cadw/__init__.py: экспорт ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/__init__.py")
s = p.read_text()
if "CadwIndex" not in s:
    s = s.replace(
        "from .dataset import (\n    Dataset,",
        "from .dataset import (\n    Dataset,\n    CadwIndex, RangeIndex, MaterializedIndex,",
        1,
    )
    s = s.replace(
        '__all__ = [\n    "MemoryGuard",',
        '__all__ = [\n    "CadwIndex",\n    "RangeIndex",\n    "MaterializedIndex",\n    "MemoryGuard",',
        1,
    )
    p.write_text(s)
    print("cadw/__init__.py: CadwIndex/RangeIndex/MaterializedIndex exported")
PY

# ---------- dataframe.py: index property + set/reset_index + _index_* fields ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# 1. Import CadwIndex classes
if "from cadw.dataset.index import" not in s and "RangeIndex, MaterializedIndex" not in s:
    # cadw.dataset already re-exports — проверим
    if "from cadw.dataset import" in s:
        old_import = '''from cadw.dataset import (
    Dataset,
    RowFilter,
    PredicateFilter,
    SliceFilter,
)'''
        new_import = '''from cadw.dataset import (
    Dataset,
    CadwIndex,
    RangeIndex,
    MaterializedIndex,
    RowFilter,
    PredicateFilter,
    SliceFilter,
)'''
        if old_import in s:
            s = s.replace(old_import, new_import, 1)
            print("dataframe.py: import CadwIndex classes")
        else:
            raise SystemExit("cadw.dataset import block not found")
    else:
        raise SystemExit("cadw.dataset import not found")

# 2. __init__: add index fields
old_init_tail = '''        self._row_filters: list[RowFilter] = list(row_filters) if row_filters else []'''
new_init_tail = '''        self._row_filters: list[RowFilter] = list(row_filters) if row_filters else []
        # Index state: "range" (lazy) or "materialized" (Arrow Array).
        self._index_kind: str = "range"
        self._index_start: int = 0
        self._index_stop: int = dataset.total_rows
        self._index_values: pa.Array | None = None'''
if old_init_tail not in s:
    raise SystemExit("__init__ tail not found")
s = s.replace(old_init_tail, new_init_tail, 1)
print("dataframe.py: __init__ — index fields")

# 3. index property + set_index + reset_index — insert after `columns` property
anchor = '''    @property
    def shape(self) -> tuple[int, int]:'''
if anchor not in s:
    raise SystemExit("shape property anchor not found")
index_methods = '''    @property
    def index(self) -> CadwIndex:
        """Индекс строк (v0.4.0a).

        - Без фильтров: RangeIndex(0, N) или MaterializedIndex (после
          set_index).
        - С активными фильтрами: raise NotImplementedError — синхронизация
          RangeIndex с filter требует O(N) scan. v0.4.0b (iloc).
        """
        if self._row_filters:
            raise NotImplementedError(
                "df.index: с активными фильтрами не поддерживается в "
                "v0.4.0a (RangeIndex+filter = O(N) scan). "
                "Сбросьте фильтры через reset_index() перед filter, "
                "или используйте set_index(col) — но и он требует "
                "отсутствия фильтров. См. §13.4/55."
            )
        if self._index_kind == "range":
            return RangeIndex(self._index_start, self._index_stop)
        assert self._index_values is not None
        return MaterializedIndex(self._index_values)

    def set_index(
        self,
        col: str,
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
        if not isinstance(col, str):
            raise TypeError(
                f"set_index: col must be str; got {type(col).__name__}"
            )
        if col not in self._columns:
            raise KeyError(
                f"set_index: column {col!r} not in {self._columns}"
            )
        if self._row_filters:
            raise NotImplementedError(
                "set_index: не поддерживается с активными фильтрами в "
                "v0.4.0a. Сбросьте фильтры или материализуйте через "
                "df.write_parquet(...) и пересоздайте. §13.4/55."
            )
        if not isinstance(drop, bool):
            raise TypeError("set_index: drop must be bool")

        chunks: list[pa.Array] = []
        for b in self._iter_filtered_batches():
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
        if self._row_filters:
            raise NotImplementedError(
                "reset_index: не поддерживается с активными фильтрами в "
                "v0.4.0a. §13.4/55."
            )
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
                "reset_index(drop=False): index уже RangeIndex — "
                "нет смысла выносить в колонку. "
                "Используйте reset_index(drop=True) или set_index(col) сначала."
            )

        # materialized
        assert self._index_values is not None
        values = self._index_values

        if drop:
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

        offset = 0

        def _iter_with_index():
            nonlocal offset
            for b in self._iter_filtered_batches():
                n = b.num_rows
                idx_slice = values.slice(offset, n)
                yield b.append_column(name, idx_slice)
                offset += n

        new_ds = Dataset.from_batches(
            new_schema, _iter_with_index(), self._new_dataset_dir()
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

'''
s = s.replace(anchor, index_methods + anchor, 1)
print("dataframe.py: index property + set_index + reset_index")

# 4. _with_filter — copy index fields (но .index будет raise из-за filters)
old_with = '''    def _with_filter(self, f: RowFilter) -> "CadwDataFrame":
        new = CadwDataFrame.__new__(CadwDataFrame)
        new._dataset = self._dataset
        new._memory_guard = self._memory_guard
        new._spill_dir = self._spill_dir
        new._columns = list(self._columns)
        new._row_filters = list(self._row_filters) + [f]
        return new'''
new_with = '''    def _with_filter(self, f: RowFilter) -> "CadwDataFrame":
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
        return new'''
if old_with not in s:
    raise SystemExit("_with_filter not found")
s = s.replace(old_with, new_with, 1)
print("dataframe.py: _with_filter — copy index fields")

# 5. _getitem list[str] — copy index fields
old_list = '''            new = CadwDataFrame.__new__(CadwDataFrame)
            new._dataset = self._dataset
            new._memory_guard = self._memory_guard
            new._spill_dir = self._spill_dir
            new._columns = list(key)
            for c in new._columns:
                if c not in self._dataset.columns:
                    raise KeyError(f"column {c!r} not found in dataset")
            new._row_filters = list(self._row_filters)
            return new'''
new_list = '''            new = CadwDataFrame.__new__(CadwDataFrame)
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
            return new'''
if old_list not in s:
    raise SystemExit("list[str] selection block not found")
s = s.replace(old_list, new_list, 1)
print("dataframe.py: list[str] selection — copy index fields")

# 6. _wrap_result_table — RangeIndex по умолчанию
old_wrap = '''    def _wrap_result_table(self, table: pa.Table) -> "CadwDataFrame":
        new_ds = Dataset.from_batches(
            table.schema, iter(table.to_batches()), self._new_dataset_dir()
        )
        return CadwDataFrame(
            new_ds, memory_guard=self._memory_guard, spill_dir=self._spill_dir
        )'''
new_wrap = '''    def _wrap_result_table(self, table: pa.Table) -> "CadwDataFrame":
        new_ds = Dataset.from_batches(
            table.schema, iter(table.to_batches()), self._new_dataset_dir()
        )
        # Default RangeIndex(0, total_rows) — результат операций сбрасывает
        # исходный index (как pandas после groupby/sort/merge).
        return CadwDataFrame(
            new_ds, memory_guard=self._memory_guard, spill_dir=self._spill_dir
        )'''
if old_wrap not in s:
    raise SystemExit("_wrap_result_table not found")
s = s.replace(old_wrap, new_wrap, 1)
print("dataframe.py: _wrap_result_table — RangeIndex (unchanged behavior)")

p.write_text(s)
print("dataframe.py: готово")
PY

# ---------- tests/test_cadw_index.py ----------
cat > tests/test_cadw_index.py <<'EOF'
"""
CadwIndex (v0.4.0a): RangeIndex, MaterializedIndex, df.index, set_index,
reset_index. Все сценарии без активных фильтров (MVP).
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.dataset.index import CadwIndex, MaterializedIndex, RangeIndex


def make_df(tmp_path, name="L", data=None, chunk=3):
    if data is None:
        data = {"k": ["a", "b", "c", "d", "e"], "v": [1, 2, 3, 4, 5]}
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- RangeIndex class ----------

def test_range_index_len():
    assert len(RangeIndex(0, 10)) == 10
    assert len(RangeIndex(5, 15)) == 10
    assert len(RangeIndex(0, 10, 2)) == 5
    assert len(RangeIndex(0, 0)) == 0


def test_range_index_getitem():
    r = RangeIndex(10, 20)
    assert r[0] == 10
    assert r[5] == 15
    assert r[-1] == 19
    with pytest.raises(IndexError):
        _ = r[100]


def test_range_index_slice():
    r = RangeIndex(0, 100)
    sub = r[10:20]
    assert isinstance(sub, RangeIndex)
    assert sub.start == 10
    assert sub.stop == 20
    assert len(sub) == 10


def test_range_index_iter():
    r = RangeIndex(3, 7)
    assert list(r) == [3, 4, 5, 6]


def test_range_index_invalid():
    with pytest.raises(ValueError):
        RangeIndex(0, 10, step=0)
    with pytest.raises(ValueError):
        RangeIndex(-1, 10)
    with pytest.raises(ValueError):
        RangeIndex(10, 5)


# ---------- MaterializedIndex class ----------

def test_materialized_index_basic():
    idx = MaterializedIndex(pa.array([10, 20, 30]))
    assert len(idx) == 3
    assert idx[0] == 10
    assert idx[2] == 30
    assert list(idx) == [10, 20, 30]


def test_materialized_index_string():
    idx = MaterializedIndex(pa.array(["x", "y", "z"]))
    assert list(idx) == ["x", "y", "z"]
    assert idx[1] == "y"


# ---------- df.index default ----------

def test_default_index_is_range(tmp_path):
    df = make_df(tmp_path)
    idx = df.index
    assert isinstance(idx, RangeIndex)
    assert idx.start == 0
    assert idx.stop == 5
    assert len(idx) == 5


# ---------- set_index ----------

def test_set_index_drop_true(tmp_path):
    df = make_df(tmp_path)
    new = df.set_index("k", drop=True)
    assert isinstance(new.index, MaterializedIndex)
    assert list(new.index) == ["a", "b", "c", "d", "e"]
    assert "k" not in new.columns
    assert "v" in new.columns
    # Исходный df не изменён
    assert "k" in df.columns
    assert isinstance(df.index, RangeIndex)


def test_set_index_drop_false(tmp_path):
    df = make_df(tmp_path)
    new = df.set_index("k", drop=False)
    assert isinstance(new.index, MaterializedIndex)
    assert list(new.index) == ["a", "b", "c", "d", "e"]
    assert "k" in new.columns
    assert "v" in new.columns


def test_set_index_inplace(tmp_path):
    df = make_df(tmp_path)
    result = df.set_index("k", drop=True, inplace=True)
    assert result is df
    assert isinstance(df.index, MaterializedIndex)
    assert "k" not in df.columns


def test_set_index_missing_col_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.set_index("nope")


def test_set_index_wrong_type_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="col must be str"):
        df.set_index(123)


# ---------- reset_index ----------

def test_reset_index_drop_true(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=True)
    restored = with_idx.reset_index(drop=True)
    assert isinstance(restored.index, RangeIndex)
    assert restored.index.start == 0
    assert restored.index.stop == 5
    # k не восстановлен (drop=True)
    assert "k" not in restored.columns
    assert "v" in restored.columns


def test_reset_index_drop_false(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=True)
    restored = with_idx.reset_index(drop=False)
    assert isinstance(restored.index, RangeIndex)
    # Колонка "index" добавлена со значениями
    assert "index" in restored.columns
    # Check values through parquet roundtrip
    out = tmp_path / "out.parquet"
    restored.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["index"] for r in rows] == ["a", "b", "c", "d", "e"]


def test_reset_index_collision_level_0(tmp_path):
    """Если колонка "index" уже есть — используется "level_0"."""
    df = make_df(tmp_path, data={
        "k": ["a", "b"],
        "index": [100, 200],
        "v": [1, 2],
    })
    with_idx = df.set_index("k", drop=True)
    restored = with_idx.reset_index(drop=False)
    assert "level_0" in restored.columns
    assert "index" in restored.columns  # исходная колонка
    out = tmp_path / "out.parquet"
    restored.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["level_0"] for r in rows] == ["a", "b"]
    assert [r["index"] for r in rows] == [100, 200]


def test_reset_index_inplace(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=True)
    result = with_idx.reset_index(drop=True, inplace=True)
    assert result is with_idx
    assert isinstance(with_idx.index, RangeIndex)


def test_reset_index_on_range_index_drop_true_noop(tmp_path):
    df = make_df(tmp_path)
    same = df.reset_index(drop=True)
    assert isinstance(same.index, RangeIndex)
    assert list(same.index) == [0, 1, 2, 3, 4]


def test_reset_index_on_range_index_drop_false_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(ValueError, match="already"):
        df.reset_index(drop=False)


# ---------- chained ----------

def test_set_index_then_select_then_reset(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=True)
    # select subset (no filter) — index сохраняется
    sub = with_idx[["v"]]
    assert isinstance(sub.index, MaterializedIndex)
    assert list(sub.index) == ["a", "b", "c", "d", "e"]
    # reset_index
    restored = sub.reset_index(drop=False)
    out = tmp_path / "out.parquet"
    restored.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["index"] for r in rows] == ["a", "b", "c", "d", "e"]
    assert [r["v"] for r in rows] == [1, 2, 3, 4, 5]


# ---------- filter limitation (MVP) ----------

def test_index_with_filter_raises(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 2]
    with pytest.raises(NotImplementedError, match="фильтр"):
        _ = filtered.index


def test_set_index_with_filter_raises(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 2]
    with pytest.raises(NotImplementedError, match="фильтр"):
        filtered.set_index("k")


def test_reset_index_with_filter_raises(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=True)
    # select without filter works
    sub = with_idx[["v"]]
    filtered = sub[sub["v"] > 2]
    with pytest.raises(NotImplementedError, match="фильтр"):
        filtered.reset_index(drop=True)


# ---------- after operations: index reset ----------

def test_index_reset_after_groupby(tmp_path):
    df = make_df(tmp_path, data={
        "g": ["A", "A", "B", "B"],
        "v": [1, 2, 3, 4],
    })
    result = df.groupby("g").agg("v", "sum")
    # groupby результат — новый RangeIndex (как pandas)
    assert isinstance(result.index, RangeIndex)
    assert result.index.start == 0
    assert result.index.stop == 2


def test_index_reset_after_sort(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=True)
    sorted_df = with_idx.sort_values("v", ascending=False)
    # sort_values создаёт новый Dataset → RangeIndex reset
    assert isinstance(sorted_df.index, RangeIndex)


# ---------- persistence: index not serialized to parquet ----------

def test_index_not_persisted_in_parquet(tmp_path):
    """Parquet не хранит CadwIndex; write_parquet → reload → RangeIndex."""
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=False)
    out = tmp_path / "out.parquet"
    with_idx.write_parquet(out)
    reloaded = CadwDataFrame.from_parquet(
        out, tmp_path / "ds2", chunk_size=3,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill2",
    )
    assert isinstance(reloaded.index, RangeIndex)
EOF

# ---------- DECISIONS.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

if "| `CadwIndex`/`set_index`/`reset_index` (v0.4.0a) |" not in s:
    s = s.replace(
        "| Combiner для `groupby` (v0.3.4: partial agg до shuffle) | ✅ | 12 |",
        "| Combiner для `groupby` (v0.3.4: partial agg до shuffle) | ✅ | 12 |\n"
        "| `CadwIndex`/`set_index`/`reset_index` (v0.4.0a) | ✅ | 20 |",
        1,
    )

if "| `df.index`/`set_index`/`reset_index` с активными фильтрами (v0.4.0a)" not in s:
    s = s.replace(
        "| `_detect_heavy_keys` absolute min = 100 (v0.3.4)",
        "| `df.index`/`set_index`/`reset_index` с активными фильтрами (v0.4.0a) | `CadwDataFrame` | raise NotImplementedError; синхронизация index↔filter — v0.4.0b (iloc, через row_id). §13.4/55. |\n"
        "| `set_index` материализует колонку (O(N) RAM) | `CadwDataFrame` | Для больших N — future: `RangeIndex` fast path + streaming loc. v0.4.1. |\n"
        "| Index не сохраняется в Parquet (write_parquet → reload = RangeIndex) | `write_parquet` | Parquet не имеет index-метаданных; документировано. §13.4/55. |\n"
        "| `_detect_heavy_keys` absolute min = 100 (v0.3.4)",
        1,
    )

marker_135 = "### 13.5. Открытые вопросы"
episode_55 = """**Эпизод 55 (CadwIndex — MVP без filter-compat).** Первый кусок
`CadwIndex` — абстрактный класс + два наследника:
- `RangeIndex(start, stop, step)` — lazy, O(1) память. `__len__`,
  `__getitem__` (int/slice), `__iter__`, `to_arrow`. Эмулирует pandas.
- `MaterializedIndex(values: pa.Array)` — Arrow Array с типом колонки.
  Для `set_index`; `.equals()` через Arrow `.equals()`.

`CadwDataFrame` хранит `_index_kind` (`"range"` | `"materialized"`),
`_index_start`/`_index_stop` (для range), `_index_values` (для
materialized). Default — `RangeIndex(0, total_rows)`.

**Ключевое ограничение MVP:** index не синхронизируется с активными
фильтрами. `df[mask].index` — `NotImplementedError` (RangeIndex+filter
требует O(N) scan: считать row_id, прошедшие маску). `set_index` и
`reset_index` тоже raise с фильтрами.

**Почему не «просто scan»:** `_iter_filtered_batches` теряет row_id
(отдаёт только `_columns`), поэтому scan нужно делать отдельно. В
v0.4.0b (`iloc`) появится `_iter_filtered_batches_with_positions()`
yield'ит `(batch, positions)` — тогда index через filter = O(N) scan
позиций (не O(N) data). Это приемлемо для `.index` access.

**Почему не index-as-Dataset-column:** можно было добавить `__index__`
в Dataset и позволить filter его фильтровать. Проблемы: (a) колонка
входит в `_columns`, нужен hidden-флаг; (b) при `sort_values` index
пересчитывается (a не сохраняется); (c) `Dataset` immutable — удаление
колонки при `reset_index` = копирование. Итого — сложнее, чем хранить
index в CadwDataFrame.

**`reset_index(drop=True)` для RangeIndex:** no-op (pandas: то же).
`reset_index(drop=False)` для RangeIndex — raise `ValueError` (pandas:
«cannot insert level_0, already have default index»). Это
pandas-совместимое поведение.

**`set_index(inplace=True)`:** mutate `_columns`, `_index_*`. Возвращает
self. `reset_index(inplace=True)` — mutate или создать новый Dataset.
Это единственные «inplace»-операции в CadwDataFrame (facade остальной
immutable).

**Урок:** index — это **отдельный объект идентичности**, не колонка.
Pandas делает его частью BlockManager (O(N) даже для RangeIndex),
Arrow не имеет встроенного index. Хранить в CadwDataFrame с явным
`_index_kind` (дискриминатор) — правильно для ленивой семантики.
MVP без filter-compat — осознанное решение: v0.4.0b (iloc) даст
правильный механизм, и тогда filter+index станет доступным без
«временных» API.

**Следующее (v0.4.0b — iloc):**
- `_iter_filtered_batches_with_positions()` — yield `(batch, positions)`.
- `iloc[i]` — scalar tuple.
- `iloc[lo:hi]` — slice (lazy).
- `iloc[[i,j,k]]` — позиции.
- `filtered.index` — RangeIndex subset через positions (O(N) scan).
- `set_index`/`reset_index` с фильтрами — использовать filtered positions.

**v0.4.0c (loc):** label-based через MaterializedIndex.get_indexer.

"""
if "Эпизод 55" not in s:
    s = s.replace(marker_135, episode_55 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §11.1, §11.5, §13.4/55")
PY

# ---------- CHANGELOG.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.4.0a] — 2026-10-19

### Добавлено
- `CadwIndex` (ABC) + `RangeIndex(start, stop, step)` (lazy, O(1))
  + `MaterializedIndex(values)` (Arrow Array).
- `CadwDataFrame.index` property — RangeIndex (default) или
  MaterializedIndex (после set_index).
- `CadwDataFrame.set_index(col, drop=True, inplace=False)`:
  материализует колонку в память (Arrow Array).
- `CadwDataFrame.reset_index(drop=False, inplace=False)`:
  `drop=True` — отбрасывает index; `drop=False` — выносит в колонку
  `index` (или `level_0` при коллизии).
- Экспорт: `cadw.CadwIndex`, `cadw.RangeIndex`,
  `cadw.MaterializedIndex`.
- `cadw.dataset.index` — модуль.

### Ограничения (v0.4.0a, §11.5)
- **с активными фильтрами**: `df.index`, `set_index`, `reset_index` —
  raise `NotImplementedError` (RangeIndex+filter = O(N) scan;
  синхронизация — v0.4.0b).
- **`set_index`** материализует колонку (O(N) RAM). Для больших N —
  future (streaming loc — v0.4.1).
- **Index не сохраняется в Parquet** (`write_parquet` → reload =
  RangeIndex). Pandas-совместимое поведение.

### Тесты
- 20 тестов в `test_cadw_index.py`: RangeIndex/MaterializedIndex API,
  set_index drop=True/False/inplace, reset_index drop=True/False,
  коллизия «index» → «level_0», chained, filter-limit raise,
  persistence.

"""
if "## [0.4.0a]" not in s:
    s = s.replace("## [0.3.4]", entry + "## [0.3.4]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.4.0a")
PY

# ---------- run ----------
echo "=== pytest tests/test_cadw_index.py -v ==="
set +e
uv run pytest tests/test_cadw_index.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "cadw_index: rc=$RC"
  git add -A
  git commit -m "WIP: v0.4.0a CadwIndex (test red; fix follows)" || true
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.4.0a: CadwIndex (RangeIndex/MaterializedIndex) + set_index/reset_index; 20 tests"

echo "==> v0.4.0a applied on ${BR}"