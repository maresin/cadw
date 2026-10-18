#!/usr/bin/env bash
# patch_03c: RowFilter + ленивые __getitem__ (slice, mask, column selection).
set -euo pipefail

require_clean_tree() {
  local bt bu
  bt="$(git status --porcelain --untracked-files=no)"
  [[ -z "$bt" ]] || { echo "ERROR: dirty tracked:"; echo "$bt"; return 1; }
  bu=""
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    case "$f" in
      scripts/*|.backups/*|__pycache__/*|*/__pycache__/*|*.pyc|.pytest_cache/*) : ;;
      *) bu+="  $f"$'\n' ;;
    esac
  done < <(git ls-files --others --exclude-standard)
  [[ -z "$bu" ]] || { echo "ERROR: untracked outside scripts/.backups/caches:"; echo "$bu"; return 1; }
  return 0
}
require_clean_tree || exit 1

CUR="$(git rev-parse --abbrev-ref HEAD)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-03c-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_03c"

BR="patch/03c-filters"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "switch or delete ${BR}"; exit 1; }
  echo "Resuming on ${BR}"
else
  git checkout -b "${BR}"
fi

# ---------- row_filter.py ----------
cat > cadw/dataset/row_filter.py <<'PY'
"""
RowFilter — ленивые фильтры строк.

Контракт:
  iter_filtered(batch_iter, schema) -> Iterator[RecordBatch]:
    применяет фильтр к потоку батчей, не материализуя.

  columns_required() -> set[str]:
    колонки, нужные фильтру (для selection перед итерацией).

Два вида:
  - MaskFilter: per-batch булев массив (Predicate, And, Or, Not).
  - SliceFilter: сквозной позиционный срез (не mask, нужна stateful итерация).

And/Or определены только для MaskFilter: комбинировать Slice с mask
нужно через chaining (`df[10:50][mask]`), а не через `&`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


class RowFilter(ABC):
    @abstractmethod
    def iter_filtered(self, batch_iter, schema) -> Iterator[pa.RecordBatch]: ...

    @abstractmethod
    def columns_required(self) -> set[str]: ...

    def __and__(self, other: "RowFilter") -> "AndFilter":
        return AndFilter(self, other)

    def __or__(self, other: "RowFilter") -> "OrFilter":
        return OrFilter(self, other)

    def __invert__(self) -> "NotFilter":
        return NotFilter(self)


class MaskFilter(RowFilter):
    @abstractmethod
    def mask_for_batch(self, batch: pa.RecordBatch) -> pa.Array: ...

    def iter_filtered(self, batch_iter, schema):
        for b in batch_iter:
            mask = self.mask_for_batch(b)
            if pc.any(mask).as_py():
                yield b.filter(mask)


class PredicateFilter(MaskFilter):
    _OPS = {
        "gt": pc.greater,
        "ge": pc.greater_equal,
        "lt": pc.less,
        "le": pc.less_equal,
        "eq": pc.equal,
        "ne": pc.not_equal,
    }

    def __init__(self, column: str, op: str, value) -> None:
        if op not in self._OPS:
            raise ValueError(f"unsupported op {op!r}; expected one of {list(self._OPS)}")
        self.column = column
        self.op = op
        self.value = value

    def mask_for_batch(self, batch):
        arr = batch.column(self.column)
        scalar = pa.scalar(self.value, type=arr.type)
        return self._OPS[self.op](arr, scalar)

    def columns_required(self):
        return {self.column}

    def __repr__(self) -> str:
        return f"PredicateFilter({self.column!r} {self.op} {self.value!r})"


class AndFilter(MaskFilter):
    def __init__(self, left, right) -> None:
        if not isinstance(left, MaskFilter):
            raise TypeError(f"AndFilter left: MaskFilter required, got {type(left).__name__}")
        if not isinstance(right, MaskFilter):
            raise TypeError(f"AndFilter right: MaskFilter required, got {type(right).__name__}")
        self.left, self.right = left, right

    def mask_for_batch(self, batch):
        return pc.and_(self.left.mask_for_batch(batch), self.right.mask_for_batch(batch))

    def columns_required(self):
        return self.left.columns_required() | self.right.columns_required()

    def __repr__(self) -> str:
        return f"AndFilter({self.left!r}, {self.right!r})"


class OrFilter(MaskFilter):
    def __init__(self, left, right) -> None:
        if not isinstance(left, MaskFilter):
            raise TypeError(f"OrFilter left: MaskFilter required, got {type(left).__name__}")
        if not isinstance(right, MaskFilter):
            raise TypeError(f"OrFilter right: MaskFilter required, got {type(right).__name__}")
        self.left, self.right = left, right

    def mask_for_batch(self, batch):
        return pc.or_(self.left.mask_for_batch(batch), self.right.mask_for_batch(batch))

    def columns_required(self):
        return self.left.columns_required() | self.right.columns_required()

    def __repr__(self) -> str:
        return f"OrFilter({self.left!r}, {self.right!r})"


class NotFilter(MaskFilter):
    def __init__(self, inner) -> None:
        if not isinstance(inner, MaskFilter):
            raise TypeError(f"NotFilter: MaskFilter required, got {type(inner).__name__}")
        self.inner = inner

    def mask_for_batch(self, batch):
        return pc.invert(self.inner.mask_for_batch(batch))

    def columns_required(self):
        return self.inner.columns_required()

    def __repr__(self) -> str:
        return f"NotFilter({self.inner!r})"


class SliceFilter(RowFilter):
    def __init__(self, start, stop, step) -> None:
        if step is None:
            step = 1
        if step <= 0:
            raise ValueError("step must be positive")
        self.start = start if start is not None else 0
        self.stop = stop
        self.step = step

    def iter_filtered(self, batch_iter, schema):
        pos = 0
        for b in batch_iter:
            n = b.num_rows
            positions = np.arange(pos, pos + n, dtype=np.int64)
            mask = positions >= self.start
            if self.stop is not None:
                mask &= positions < self.stop
            if self.step > 1:
                mask &= ((positions - self.start) % self.step == 0)
            if mask.any():
                yield b.filter(pa.array(mask, type=pa.bool_()))
            pos += n
            if self.stop is not None and pos >= self.stop:
                return

    def columns_required(self):
        return set()

    def __repr__(self) -> str:
        return f"SliceFilter({self.start}, {self.stop}, {self.step})"
PY

# ---------- dataset/__init__.py ----------
cat > cadw/dataset/__init__.py <<'PY'
from .dataset import Dataset
from .row_filter import (
    RowFilter,
    MaskFilter,
    PredicateFilter,
    AndFilter,
    OrFilter,
    NotFilter,
    SliceFilter,
)

__all__ = [
    "Dataset",
    "RowFilter",
    "MaskFilter",
    "PredicateFilter",
    "AndFilter",
    "OrFilter",
    "NotFilter",
    "SliceFilter",
]
PY

# ---------- dataframe.py (rewrite: filters integrated) ----------
cat > cadw/dataframe.py <<'PY'
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
from typing import Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from cadw.core.memory_guard import MemoryGuard
from cadw.dataset import (
    Dataset,
    RowFilter,
    PredicateFilter,
    SliceFilter,
)
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ParquetBatchWriter
from cadw.ops.groupby import groupby_aggregate
from cadw.ops.sort import external_sort


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

    def to_batches(self, batch_size: int | None = None) -> Iterator[pa.RecordBatch]:
        """Поток батчей с учётом фильтров. batch_size=None — как в Dataset."""
        if not self._row_filters and batch_size is None:
            yield from self._dataset.iter_batches(columns=self._columns)
            return
        if not self._row_filters:
            # re-chunk без фильтров
            yield from self._dataset.iter_batches(
                batch_size=batch_size, columns=self._columns
            )
            return
        # С фильтрами: фильтруем поток, потом re-chunk на выходе
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
                    yield pa.Table.from_batches(accum).combine_chunks().to_batches()[0] \
                        if len(accum) == 1 else _concat_batches(accum)
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

    def _groupby_impl(self, by_cols, agg_columns, agg_funcs) -> "CadwDataFrame":
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

    def agg(self, agg_columns, agg_funcs) -> CadwDataFrame:
        agg_cols = [agg_columns] if isinstance(agg_columns, str) else list(agg_columns)
        funcs = [agg_funcs] if isinstance(agg_funcs, str) else list(agg_funcs)
        return self._parent._groupby_impl(self._by, agg_cols, funcs)


def _concat_batches(batches: list[pa.RecordBatch]) -> pa.RecordBatch:
    if len(batches) == 1:
        return batches[0]
    return pa.Table.from_batches(batches).combine_chunks().to_batches()[0]
PY

# ---------- export ----------
python3 - <<'PY'
p = "cadw/__init__.py"
s = open(p).read()
if "from .dataset import" in s and "RowFilter" not in s:
    s = s.replace(
        "from .dataset import Dataset",
        "from .dataset import (\n    Dataset,\n    RowFilter, PredicateFilter, AndFilter, OrFilter, NotFilter, SliceFilter,\n)",
        1,
    )
    s = s.replace(
        '"Dataset", "CadwDataFrame",',
        '"Dataset", "CadwDataFrame",\n    "RowFilter", "PredicateFilter", "AndFilter", "OrFilter", "NotFilter", "SliceFilter",',
        1,
    )
    open(p, "w").write(s)
    print("patched cadw/__init__.py: filters exported")
PY

# ---------- tests ----------
cat > tests/test_row_filter.py <<'EOF'
"""Тесты RowFilter (03c)."""
import pyarrow as pa
import pytest

from cadw.dataset import (
    PredicateFilter, SliceFilter, AndFilter, OrFilter, NotFilter, MaskFilter,
)


def make_batches(n: int = 20, chunk: int = 5):
    t = pa.table({"val": list(range(n))})
    for i in range(0, n, chunk):
        yield t.slice(i, chunk).to_batches()[0]


def collect(filter_, n=20, chunk=5):
    it = filter_.iter_filtered(make_batches(n, chunk), None)
    return [b.column("val").to_pylist() for b in it]


# ---------- PredicateFilter ----------

def test_predicate_gt():
    f = PredicateFilter("val", "gt", 15)
    chunks = collect(f)
    assert chunks == [[16, 17, 18, 19]]


def test_predicate_eq():
    f = PredicateFilter("val", "eq", 7)
    assert collect(f) == [[7]]


def test_predicate_returns_empty_if_none_match():
    f = PredicateFilter("val", "gt", 100)
    assert collect(f) == []


def test_predicate_columns_required():
    assert PredicateFilter("val", "gt", 0).columns_required() == {"val"}


# ---------- SliceFilter ----------

def test_slice_basic():
    f = SliceFilter(2, 8, 1)
    chunks = collect(f)
    flat = [v for c in chunks for v in c]
    assert flat == [2, 3, 4, 5, 6, 7]


def test_slice_with_step():
    f = SliceFilter(0, 20, 3)
    flat = [v for c in collect(f) for v in c]
    assert flat == [0, 3, 6, 9, 12, 15, 18]


def test_slice_across_batches():
    f = SliceFilter(3, 12, 1)
    flat = [v for c in collect(f, n=20, chunk=4) for v in c]
    assert flat == [3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_slice_stop_beyond_end():
    f = SliceFilter(15, 100, 1)
    flat = [v for c in collect(f) for v in c]
    assert flat == [15, 16, 17, 18, 19]


def test_slice_columns_required_empty():
    assert SliceFilter(0, 10, 1).columns_required() == set()


# ---------- And/Or/Not ----------

def test_and_filter():
    f = AndFilter(
        PredicateFilter("val", "ge", 5),
        PredicateFilter("val", "lt", 10),
    )
    flat = [v for c in collect(f) for v in c]
    assert flat == [5, 6, 7, 8, 9]


def test_or_filter():
    f = OrFilter(
        PredicateFilter("val", "lt", 3),
        PredicateFilter("val", "gt", 17),
    )
    flat = [v for c in collect(f) for v in c]
    assert flat == [0, 1, 2, 18, 19]


def test_not_filter():
    f = NotFilter(PredicateFilter("val", "lt", 17))
    flat = [v for c in collect(f) for v in c]
    assert flat == [17, 18, 19]


def test_and_with_slice_raises():
    with pytest.raises(TypeError):
        PredicateFilter("val", "gt", 0) & SliceFilter(0, 10, 1)


def test_columns_required_union():
    f = AndFilter(
        PredicateFilter("a", "gt", 0),
        PredicateFilter("b", "lt", 5),
    )
    assert f.columns_required() == {"a", "b"}
EOF

cat > tests/test_dataframe_filters.py <<'EOF'
"""Тесты ленивых __getitem__ в CadwDataFrame (03c)."""
import inspect
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cadw.dataframe as df_mod
from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path: Path, n: int = 20) -> CadwDataFrame:
    t = pa.table({
        "cat": [f"g{i % 5}" for i in range(n)],
        "val": list(range(n)),
    })
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=5,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )


def read_parquet_vals(path):
    return pq.read_table(path).to_pylist()


# ---------- column selection ----------

def test_select_columns(tmp_path):
    df = make_df(tmp_path)
    sub = df[["val"]]
    assert sub.columns == ["val"]
    assert sub.shape == (20, 1)
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    t = pq.read_table(out)
    assert t.schema.names == ["val"]


def test_select_columns_missing_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError):
        df[["nope"]]


# ---------- slice ----------

def test_slice_basic(tmp_path):
    df = make_df(tmp_path)
    sliced = df[5:10]
    out = tmp_path / "o.parquet"
    sliced.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [5, 6, 7, 8, 9]


def test_slice_with_step(tmp_path):
    df = make_df(tmp_path)
    sliced = df[0:20:3]
    out = tmp_path / "o.parquet"
    sliced.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [0, 3, 6, 9, 12, 15, 18]


# ---------- predicate filters ----------

def test_filter_gt(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["val"] > 15]
    out = tmp_path / "o.parquet"
    filtered.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [16, 17, 18, 19]


def test_filter_eq(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["val"] == 7]
    out = tmp_path / "o.parquet"
    filtered.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [7]


def test_filter_and(tmp_path):
    df = make_df(tmp_path)
    filtered = df[(df["val"] >= 5) & (df["val"] < 10)]
    out = tmp_path / "o.parquet"
    filtered.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [5, 6, 7, 8, 9]


def test_filter_or(tmp_path):
    df = make_df(tmp_path)
    filtered = df[(df["val"] < 3) | (df["val"] > 17)]
    out = tmp_path / "o.parquet"
    filtered.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [0, 1, 2, 18, 19]


def test_filter_not(tmp_path):
    df = make_df(tmp_path)
    filtered = df[~(df["val"] < 17)]
    out = tmp_path / "o.parquet"
    filtered.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [17, 18, 19]


# ---------- chained filters ----------

def test_slice_then_filter(tmp_path):
    df = make_df(tmp_path)
    result = df[5:15][df["val"] > 10]
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [11, 12, 13, 14]


def test_filter_then_slice(tmp_path):
    df = make_df(tmp_path)
    result = df[df["val"] > 10][1:4]
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [12, 13, 14]


def test_filter_then_select(tmp_path):
    df = make_df(tmp_path)
    result = df[df["val"] > 15][["val"]]
    assert result.columns == ["val"]
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    t = pq.read_table(out)
    assert t.column("val").to_pylist() == [16, 17, 18, 19]


def test_filter_on_non_selected_column(tmp_path):
    """Фильтр по колонке не из selection работает: filter-колонка добавляется, потом select."""
    df = make_df(tmp_path)
    result = df[["cat"]][df["val"] > 15]
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = read_parquet_vals(out)
    assert len(rows) == 4
    assert set(rows[0].keys()) == {"cat"}


# ---------- head/count/shape ----------

def test_head_after_filter(tmp_path):
    df = make_df(tmp_path)
    h = df[df["val"] > 15].head(2)
    assert h.num_rows == 2
    assert h.column("val").to_pylist() == [16, 17]


def test_count_after_filter(tmp_path):
    df = make_df(tmp_path)
    assert df[df["val"] > 15].count() == 4


def test_shape_is_upper_bound(tmp_path):
    df = make_df(tmp_path)
    assert df[df["val"] > 15].shape[0] == 20  # upper bound


# ---------- filter + groupby/sort ----------

def test_filter_then_groupby(tmp_path):
    df = make_df(tmp_path)
    result = df[df["val"] > 5].groupby("cat").agg("val", "count")
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    rows = {r["cat"]: r["val_count"] for r in read_parquet_vals(out)}
    # значения >5: 6,7,8,9,10,11,12,13,14,15,16,17,18,19
    # cat: g(val%5). 6→g1, 7→g2, 8→g3, 9→g4, 10→g0, 11→g1, 12→g2, ...
    assert sum(rows.values()) == 14


def test_filter_then_sort(tmp_path):
    df = make_df(tmp_path)
    result = df[df["val"] > 15].sort_values("val", ascending=False)
    out = tmp_path / "o.parquet"
    result.write_parquet(out)
    vals = [r["val"] for r in read_parquet_vals(out)]
    assert vals == [19, 18, 17, 16]


# ---------- facade guards ----------

def test_facade_has_no_as_py_ast():
    import ast
    src = inspect.getsource(df_mod)
    tree = ast.parse(src)
    bad = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "as_py"
    ]
    assert not bad, "facade calls .as_py on data"


def test_facade_has_no_row_loop_ast():
    import ast
    src = inspect.getsource(df_mod)
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        it = node.iter
        if (isinstance(it, ast.Call)
                and isinstance(it.func, ast.Name)
                and it.func.id == "range"):
            for arg in it.args:
                if isinstance(arg, ast.Attribute) and arg.attr == "num_rows":
                    offenders.append(ast.unparse(node.target))
    assert not offenders
EOF

# ---------- DECISIONS update ----------
python3 - <<'PY'
p = "DECISIONS.md"
s = open(p).read()
if "| `RowFilter` (03c)" not in s:
    s = s.replace(
        "| `CadwDataFrame` (03b, минимальный фасад) | ✅ | 17 |",
        "| `CadwDataFrame` (03b, минимальный фасад) | ✅ | 17 |\n"
        "| `RowFilter` + ленивые `__getitem__` (03c) | ✅ | 25 |",
        1,
    )
if "03b — CadwDataFrame ✅" not in s and "03c ✅" not in s:
    s = s.replace(
        "### 12.2. Приоритет 1 — фасад (03a — Dataset ✅)",
        "### 12.2. Приоритет 1 — фасад (03a ✅, 03b ✅, 03c ✅)",
        1,
    )
open(p, "w").write(s)
print("DECISIONS §11.1 / §12.2 updated")
PY

set +e
uv run pytest -q
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
    echo "ERROR: tests failing (rc=${RC})"
    exit 1
fi

git add -A
git commit -m "03c: RowFilter (slice/predicate/and/or/not) + lazy __getitem__ (slice/list/str/mask) + tests (25)"

echo "==> sha256"
sha256sum cadw/dataset/row_filter.py cadw/dataframe.py
echo "==> 03c applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"
echo "    Hard:   git reset --hard ${BK}"