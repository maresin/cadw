"""
CadwDataFrame — публичный фасад.

Контракт «честный фасад»:
  - не реализует алгоритмы обработки данных;
  - не содержит построчных циклов (.as_py() на данных);
  - группировку, сортировку, unique делегирует примитивам
    (groupby_aggregate, external_sort, hash_shuffle);
  - хранит только: Dataset + memory_guard + spill_dir + columns (selection).

См. DECISIONS.md §10.3, §12.2.
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from cadw.core.memory_guard import MemoryGuard
from cadw.dataset import Dataset
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ParquetBatchWriter
from cadw.ops.groupby import groupby_aggregate
from cadw.ops.sort import external_sort


class CadwDataFrame:
    # ---------------- construction ----------------

    def __init__(
        self,
        dataset: Dataset,
        *,
        memory_guard: MemoryGuard | None = None,
        spill_dir: str | Path | None = None,
        columns: Sequence[str] | None = None,
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
        return (self._dataset.total_rows, len(self._columns))

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    # ---------------- reading ----------------

    def to_batches(
        self, batch_size: int | None = None
    ) -> Iterator[pa.RecordBatch]:
        """Поток батчей выбранных колонок. Не материализует."""
        return self._dataset.iter_batches(
            batch_size=batch_size, columns=self._columns
        )

    def head(self, n: int = 5) -> pa.Table:
        """Первые n строк как pa.Table (без .as_py())."""
        if n <= 0 or self._dataset.total_rows == 0:
            schema = pa.schema(
                [self._dataset.schema.field(c) for c in self._columns]
            )
            return pa.Table.from_batches([], schema=schema)
        first = True
        acc: list[pa.RecordBatch] = []
        remaining = n
        for b in self._dataset.iter_batches(
            batch_size=min(n, 1000), columns=self._columns
        ):
            take = min(b.num_rows, remaining)
            acc.append(b.slice(0, take))
            remaining -= take
            first = False
            if remaining <= 0:
                break
        if first:
            schema = pa.schema(
                [self._dataset.schema.field(c) for c in self._columns]
            )
            return pa.Table.from_batches([], schema=schema)
        return pa.Table.from_batches(acc).slice(0, n)

    # ---------------- groupby ----------------

    def groupby(self, by: str | Sequence[str]) -> "_GroupBy":
        by_cols = [by] if isinstance(by, str) else list(by)
        return _GroupBy(self, by_cols)

    def _groupby_impl(
        self,
        by_cols: Sequence[str],
        agg_columns: Sequence[str],
        agg_funcs: Sequence[str],
    ) -> "CadwDataFrame":
        reader_iter = self._dataset.iter_batches(columns=None)
        result = groupby_aggregate(
            reader_iter,
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
        """Уникальные значения колонки как одноколоночный CadwDataFrame.

        Реализация — композиция примитивов: hash_shuffle по колонке
        (через groupby_aggregate с count=1 как побочным результатом),
        затем отбрасываем count. Никаких построчных .as_py().
        """
        if column not in self._columns:
            raise KeyError(f"column {column!r} not found")
        reader_iter = self._dataset.iter_batches(columns=[column])
        result = groupby_aggregate(
            reader_iter,
            [column],
            [column],
            ["count"],
            self._memory_guard,
            self._spill_dir / "unique",
            log=False,
        )
        values_only = result.select([column])
        return self._wrap_result_table(values_only)

    # ---------------- sort_values ----------------

    def sort_values(
        self,
        by: str | Sequence[str],
        *,
        ascending: bool | Sequence[bool] = True,
    ) -> "CadwDataFrame":
        by_cols = [by] if isinstance(by, str) else list(by)
        for c in by_cols:
            if c not in self._columns:
                raise KeyError(f"sort_values: column {c!r} not found")
        path = external_sort(
            self._dataset.iter_batches(columns=None),
            key_columns=by_cols,
            value_columns=self._columns,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "sort",
            ascending=ascending,
            log=False,
        )
        schema = pa.schema(
            [self._dataset.schema.field(c) for c in self._columns]
        )

        def _iter() -> Iterable[pa.RecordBatch]:
            reader = ArrowIPCBatchReader(str(path), self._columns)
            for b in reader:
                yield b

        new_ds = Dataset.from_batches(schema, _iter(), self._new_dataset_dir())
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    # ---------------- write ----------------

    def write_parquet(
        self, path: str | Path, *, compression: str = "snappy"
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema(
            [self._dataset.schema.field(c) for c in self._columns]
        )
        writer = ParquetBatchWriter(
            str(path), schema, compression=compression
        )
        try:
            for b in self._dataset.iter_batches(columns=self._columns):
                writer.write_batch(b)
        finally:
            writer.close()
        return path

    # ---------------- helpers ----------------

    def _wrap_result_table(self, table: pa.Table) -> "CadwDataFrame":
        new_ds = Dataset.from_batches(
            table.schema,
            iter(table.to_batches()),
            self._new_dataset_dir(),
        )
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )

    def _new_dataset_dir(self) -> Path:
        p = self._spill_dir / "datasets" / f"ds_{uuid.uuid4().hex[:8]}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def __repr__(self) -> str:
        return (
            f"CadwDataFrame(rows={self._dataset.total_rows}, "
            f"columns={self._columns})"
        )


class _GroupBy:
    def __init__(self, parent: CadwDataFrame, by_cols: Sequence[str]) -> None:
        self._parent = parent
        self._by = list(by_cols)

    def agg(
        self,
        agg_columns: str | Sequence[str],
        agg_funcs: str | Sequence[str],
    ) -> CadwDataFrame:
        agg_cols = (
            [agg_columns] if isinstance(agg_columns, str) else list(agg_columns)
        )
        funcs = [agg_funcs] if isinstance(agg_funcs, str) else list(agg_funcs)
        return self._parent._groupby_impl(self._by, agg_cols, funcs)
