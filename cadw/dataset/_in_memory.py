"""In-memory Dataset backing for CadwDataFrame.from_table (v0.9.6b1).

Read-only view: base_dir is unused; add_column raises
NotImplementedError. Operations requiring new disk datasets (astype,
rename, encode_categorical, groupby, etc.) still write to disk via
Dataset.from_batches — same behaviour as disk-backed CadwDataFrame.

sort_values on in-memory CadwDataFrame uses pa.Table.sort_by directly
(CadwDataFrame.sort_values fast path). For small per-group tables this
is ~100-1000× faster than external_sort: avoids disk round-trip for
every group. Critical for _per_group_apply (v0.9.6b2 will move to
CADW-native fn contract; 100k groups × external_sort = 100-500s
overhead just for file creation).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import pyarrow as pa

from cadw.dataset.dataset import Dataset


class _InMemoryDataset(Dataset):
    """In-memory Dataset backed by a single pa.Table.

    Read-only. No disk I/O. Iterates table.to_batches() directly.

    Not a drop-in for disk Dataset: add_column raises
    NotImplementedError (in-memory df is a view, not a mutable
    store). Use it for per-group tables in _per_group_apply and for
    ad-hoc in-memory DataFrame creation (tests, small transforms).
    """

    def __init__(self, table: pa.Table) -> None:
        if not isinstance(table, pa.Table):
            raise TypeError(
                f"_InMemoryDataset requires pa.Table; "
                f"got {type(table).__name__}"
            )
        # Do NOT call super().__init__ — it would load meta from disk.
        self._table = table
        self.base_dir = Path("")  # sentinel — never used
        self._columns_dir = Path("")
        self._meta_path = Path("")
        self._schema = table.schema
        self._column_paths = {}  # not used
        self._chunk_boundaries = []
        self._total_rows = table.num_rows

    def iter_batches(
        self,
        batch_size: int | None = None,
        columns: Sequence[str] | None = None,
    ) -> Iterator[pa.RecordBatch]:
        t = self._table.select(columns) if columns is not None else self._table
        if batch_size is None:
            yield from t.to_batches()
            return
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        accum: list[pa.Array] = []
        accum_n = 0
        for b in t.to_batches():
            n = b.num_rows
            j = 0
            while j < n:
                take = min(batch_size - accum_n, n - j)
                if not accum:
                    accum = [c.slice(j, take) for c in b.columns]
                else:
                    accum = [
                        pa.concat_arrays([a, b.column(i).slice(j, take)])
                        for i, a in enumerate(accum)
                    ]
                accum_n += take
                j += take
                if accum_n == batch_size:
                    yield pa.RecordBatch.from_arrays(
                        accum, names=t.schema.names,
                    )
                    accum = []
                    accum_n = 0
        if accum_n > 0:
            yield pa.RecordBatch.from_arrays(accum, names=t.schema.names)

    def add_column(self, name: str, batches_iter) -> None:
        raise NotImplementedError(
            "_InMemoryDataset is read-only. Operations requiring new "
            "columns should write to disk first (v0.9.6b1 design)."
        )

    @property
    def columns(self) -> list[str]:
        return list(self._table.schema.names)

    @property
    def schema(self) -> pa.Schema:
        return self._table.schema

    @property
    def total_rows(self) -> int:
        return self._table.num_rows

    @property
    def chunk_boundaries(self) -> list[int]:
        return [b.num_rows for b in self._table.to_batches()]


__all__ = ["_InMemoryDataset"]

