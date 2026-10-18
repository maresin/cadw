"""
Dataset — базовое хранилище CADW.

Инвариант: у всех колонок одинаковое число stored batches и одинаковые
num_rows в каждом batch[i] (chunk boundaries). Нарушение ловится при записи.

Layout:
    base_dir/
    ├── columns/
    │   ├── <col1>.arrow     # Arrow IPC File, N record batches
    │   └── <col2>.arrow
    └── dataset_meta.json

Arrow IPC File (не Stream) выбран потому, что позволяет random-access
к batch[i] через ipc.open_file(path).get_batch(i).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

_COLUMNS_DIR = "columns"
_META_FILE = "dataset_meta.json"


class Dataset:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self._columns_dir = self.base_dir / _COLUMNS_DIR
        self._meta_path = self.base_dir / _META_FILE
        self._schema: pa.Schema | None = None
        self._column_paths: dict[str, Path] = {}
        self._chunk_boundaries: list[int] = []
        self._total_rows: int = 0
        self._load_meta_if_exists()

    # ---------------- factories ----------------

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str | Path,
        base_dir: str | Path,
        chunk_size: int = 10_000,
    ) -> "Dataset":
        parquet_path = Path(parquet_path)
        pf = pq.ParquetFile(parquet_path)
        schema = pf.schema_arrow

        ds = cls(base_dir)
        ds._init_empty(schema)

        writers: dict[str, tuple[ipc.RecordBatchFileWriter, pa.OSFile]] = {}
        try:
            for field in schema:
                p = ds._columns_dir / f"{field.name}.arrow"
                sink = pa.OSFile(str(p), "wb")
                w = ipc.new_file(sink, pa.schema([field]))
                writers[field.name] = (w, sink)
                ds._column_paths[field.name] = p

            for rb in pf.iter_batches(batch_size=chunk_size):
                for field in schema:
                    single = pa.RecordBatch.from_arrays(
                        [rb.column(field.name)], names=[field.name]
                    )
                    writers[field.name][0].write_batch(single)
                ds._chunk_boundaries.append(rb.num_rows)
                ds._total_rows += rb.num_rows
        finally:
            for w, sink in writers.values():
                w.close()
                sink.close()

        ds._save_meta()
        return ds

    @classmethod
    def from_batches(
        cls,
        schema: pa.Schema,
        batches_iter: Iterable[pa.RecordBatch],
        base_dir: str | Path,
    ) -> "Dataset":
        ds = cls(base_dir)
        ds._init_empty(schema)

        writers: dict[str, tuple[ipc.RecordBatchFileWriter, pa.OSFile]] = {}
        try:
            for field in schema:
                p = ds._columns_dir / f"{field.name}.arrow"
                sink = pa.OSFile(str(p), "wb")
                w = ipc.new_file(sink, pa.schema([field]))
                writers[field.name] = (w, sink)
                ds._column_paths[field.name] = p

            for rb in batches_iter:
                if rb.schema != schema:
                    rb = rb.select(schema.names).cast(schema)
                for field in schema:
                    single = pa.RecordBatch.from_arrays(
                        [rb.column(field.name)], names=[field.name]
                    )
                    writers[field.name][0].write_batch(single)
                ds._chunk_boundaries.append(rb.num_rows)
                ds._total_rows += rb.num_rows
        finally:
            for w, sink in writers.values():
                w.close()
                sink.close()

        ds._save_meta()
        return ds

    # ---------------- write ----------------

    def add_column(self, name: str, batches_iter: Iterable[pa.RecordBatch]) -> None:
        """Добавить колонку. batches_iter должен давать batches с num_rows,
        совпадающими с self._chunk_boundaries (i-й batch → boundary[i]).
        """
        if self._schema is None:
            raise RuntimeError("Dataset is empty; use from_batches/from_parquet")
        if name in self._column_paths:
            raise ValueError(f"column {name!r} already exists")

        target = self._columns_dir / f"{name}.arrow"
        tmp = target.with_suffix(".tmp.arrow")
        sink = pa.OSFile(str(tmp), "wb")
        writer: ipc.RecordBatchFileWriter | None = None
        field: pa.Field | None = None
        i = 0
        try:
            for rb in batches_iter:
                col = rb.column(0) if rb.num_columns == 1 else rb.column(name)
                if writer is None:
                    field = pa.field(name, col.type)
                    writer = ipc.new_file(sink, pa.schema([field]))
                if i >= len(self._chunk_boundaries):
                    raise ValueError(
                        f"add_column: too many batches "
                        f"(expected {len(self._chunk_boundaries)}, got more)"
                    )
                expected = self._chunk_boundaries[i]
                if rb.num_rows != expected:
                    raise ValueError(
                        f"add_column: batch {i} has {rb.num_rows} rows, "
                        f"expected {expected} (chunk boundaries must match)"
                    )
                single = pa.RecordBatch.from_arrays([col], names=[name])
                writer.write_batch(single)
                i += 1
        finally:
            if writer is not None:
                writer.close()
            sink.close()

        if i != len(self._chunk_boundaries):
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"add_column: got {i} batches, expected {len(self._chunk_boundaries)}"
            )

        tmp.rename(target)
        self._column_paths[name] = target
        self._schema = pa.schema(list(self._schema) + [field])
        self._save_meta()

    # ---------------- read ----------------

    def iter_batches(
        self,
        batch_size: int | None = None,
        columns: Sequence[str] | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Yield RecordBatch'и.

        batch_size=None — yield stored batches as-is (одна итерация = одна
        итерация по chunk_boundaries).
        batch_size=N — re-chunk. N > stored: concat; N < stored: split.
        columns — подмножество (порядок важен).
        """
        if not self._column_paths:
            return
        cols = list(columns) if columns is not None else list(self._column_paths.keys())
        for c in cols:
            if c not in self._column_paths:
                raise KeyError(f"column {c!r} not found")

        readers = {c: ipc.open_file(self._column_paths[c]) for c in cols}
        n_batches = readers[cols[0]].num_record_batches

        if batch_size is None:
            for i in range(n_batches):
                arrays = [readers[c].get_batch(i).column(c) for c in cols]
                yield pa.RecordBatch.from_arrays(arrays, names=cols)
            return

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        accum: dict[str, list[pa.Array]] = {c: [] for c in cols}
        accum_n = 0
        for i in range(n_batches):
            stored = {c: readers[c].get_batch(i) for c in cols}
            n = stored[cols[0]].num_rows
            j = 0
            while j < n:
                take = min(batch_size - accum_n, n - j)
                for c in cols:
                    accum[c].append(stored[c].column(c).slice(j, take))
                accum_n += take
                j += take
                if accum_n == batch_size:
                    arrays = [pa.concat_arrays(accum[c]) for c in cols]
                    yield pa.RecordBatch.from_arrays(arrays, names=cols)
                    accum = {c: [] for c in cols}
                    accum_n = 0
        if accum_n > 0:
            arrays = [pa.concat_arrays(accum[c]) for c in cols]
            yield pa.RecordBatch.from_arrays(arrays, names=cols)

    # ---------------- properties ----------------

    @property
    def columns(self) -> list[str]:
        return list(self._column_paths.keys())

    @property
    def schema(self) -> pa.Schema | None:
        return self._schema

    @property
    def total_rows(self) -> int:
        return self._total_rows

    @property
    def chunk_boundaries(self) -> list[int]:
        return list(self._chunk_boundaries)

    # ---------------- helpers ----------------

    def _init_empty(self, schema: pa.Schema) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._columns_dir.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._column_paths = {}
        self._chunk_boundaries = []
        self._total_rows = 0

    def _save_meta(self) -> None:
        assert self._schema is not None
        # PyArrow 26: сериализация схемы — Schema.serialize() (не
        # pa.ipc.serialize_schema, которого в этом API нет).
        schema_bytes = self._schema.serialize().to_pybytes()
        meta = {
            "schema_b64": base64.b64encode(schema_bytes).decode(),
            "column_files": {k: v.name for k, v in self._column_paths.items()},
            "chunk_boundaries": self._chunk_boundaries,
            "total_rows": self._total_rows,
        }
        self._meta_path.write_text(json.dumps(meta, indent=2))

    def _load_meta_if_exists(self) -> None:
        if not self._meta_path.exists():
            return
        meta = json.loads(self._meta_path.read_text())
        schema_bytes = base64.b64decode(meta["schema_b64"])
        self._schema = pa.ipc.read_schema(pa.BufferReader(schema_bytes))
        self._column_paths = {
            k: self._columns_dir / v for k, v in meta["column_files"].items()
        }
        self._chunk_boundaries = list(meta["chunk_boundaries"])
        self._total_rows = int(meta["total_rows"])

    def __repr__(self) -> str:
        return (
            f"Dataset(columns={self.columns}, rows={self._total_rows}, "
            f"batches={len(self._chunk_boundaries)})"
        )
