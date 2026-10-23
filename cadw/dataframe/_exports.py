# cadw/dataframe/_exports.py
# v0.9.26h (B.1 step 2): _ExportsMixin + concat.
# Extracted verbatim from _core.py; no logic change.

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from cadw.core.logging import get_logger
from cadw.dataframe_helpers import _convert_batch
from cadw.core.memory_guard import MemoryGuard
from cadw.dataset import Dataset
from cadw.dataset._in_memory import _InMemoryDataset
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import (
    ArrowIPCBatchWriter,
    ParquetBatchWriter,
)

logger = get_logger(__name__)


class _ExportsMixin:
    """Export/IO mixin for CadwDataFrame (v0.9.26h).

    Methods: to_batches, write_parquet, from_table,
    from_parquet, from_dataset.
    """

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



    @classmethod
    def from_table(
        cls,
        table: pa.Table,
        *,
        memory_guard: MemoryGuard | None = None,
        spill_dir: str | Path | None = None,
    ) -> "CadwDataFrame":
        """In-memory CadwDataFrame backed by pa.Table. No disk I/O.

        v0.9.6b1. Read-only view: add_column raises
        NotImplementedError. Operations that produce a new
        Dataset (astype, rename, groupby, etc.) still write to
        disk (via Dataset.from_batches) — but this is rare on
        small in-memory groups and acceptable. sort_values on
        the in-memory df uses pa.Table.sort_by directly (fast
        path) — critical for per-group apply.

        Peak RAM: 0 (no copy — table ref shared). Caller owns
        table and any pre-materialization cost.
        """
        from cadw.dataset._in_memory import _InMemoryDataset
        if not isinstance(table, pa.Table):
            raise TypeError(
                f"from_table requires pa.Table; "
                f"got {type(table).__name__}"
            )
        return cls(
            _InMemoryDataset(table),
            memory_guard=memory_guard,
            spill_dir=spill_dir,
        )



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




def concat(
    dfs,
    axis: int = 0,
) -> "CadwDataFrame":
    # Concatenate CadwDataFrames along rows (axis=0) or columns (axis=1).
    #
    # axis=0 (rows):
    #   Schemas must match exactly (same column names/types, same order).
    #   Streams via chain(iter_batches); total rows = sum(inputs).
    #
    # axis=1 (columns):
    #   Row counts must match. Column names must be unique across inputs.
    #   Per-batch zip: all inputs iterate at fixed batch_size=10000 to
    #   align chunk boundaries (independent of stored chunk layout).
    #
    # Preconditions:
    #   - Row-level filters are applied BEFORE concat (iter_filtered).
    #   - axis=1 positional alignment is valid only when filters preserve
    #     original row order (always true in CADW: filters don't reorder).
    #
    # Raises:
    #   ValueError: empty dfs, invalid axis, schema/row-count mismatch,
    #     duplicate column names (axis=1).
    #   TypeError: non-CadwDataFrame input.
    from cadw.dataframe import CadwDataFrame as _CDF
    if not dfs:
        raise ValueError("concat: at least one df required")
    if axis not in (0, 1):
        raise ValueError(
            f"concat: axis must be 0 or 1; got {axis!r}"
        )
    dfs = list(dfs)
    for d in dfs:
        if not isinstance(d, _CDF):
            raise TypeError(
                f"concat: all dfs must be CadwDataFrame; "
                f"got {type(d).__name__}"
            )

    first = dfs[0]

    if axis == 0:
        out_schema = pa.schema(
            [first._dataset.schema.field(c) for c in first._columns]
        )
        for i, d in enumerate(dfs[1:], 1):
            d_schema = pa.schema(
                [d._dataset.schema.field(c) for c in d._columns]
            )
            if d_schema != out_schema:
                raise ValueError(
                    f"concat(axis=0): schema mismatch between df[0] "
                    f"and df[{i}]; expected {out_schema.names}/"
                    f"{out_schema.types}, got {d_schema.names}/"
                    f"{d_schema.types}"
                )

        def _iter_rows():
            for d in dfs:
                yield from d._iter_arrow_batches()

        new_ds = Dataset.from_batches(
            out_schema, _iter_rows(), first._new_dataset_dir(),
        )
        return _CDF(
            new_ds,
            memory_guard=first._memory_guard,
            spill_dir=first._spill_dir,
        )

    # axis == 1
    n_first = first.count()
    for i, d in enumerate(dfs[1:], 1):
        n_i = d.count()
        if n_i != n_first:
            raise ValueError(
                f"concat(axis=1): row count mismatch: df[0]="
                f"{n_first}, df[{i}]={n_i}"
            )

    seen = set()
    out_fields = []
    for d in dfs:
        for c in d._columns:
            if c in seen:
                raise ValueError(
                    f"concat(axis=1): duplicate column {c!r} "
                    f"across inputs"
                )
            seen.add(c)
            out_fields.append(d._dataset.schema.field(c))
    out_schema = pa.schema(out_fields)
    all_names = list(out_schema.names)

    def _iter_cols():
        iters = [d._iter_arrow_batches(batch_size=10000) for d in dfs]
        for batches in zip(*iters):
            n = batches[0].num_rows
            for b in batches[1:]:
                if b.num_rows != n:
                    raise RuntimeError(
                        f"concat(axis=1): batch size mismatch at "
                        f"streaming: {n} vs {b.num_rows}"
                    )
            arrays = []
            for b in batches:
                for j in range(b.num_columns):
                    arrays.append(b.column(j))
            yield pa.RecordBatch.from_arrays(arrays, names=all_names)

    new_ds = Dataset.from_batches(
        out_schema, _iter_cols(), first._new_dataset_dir(),
    )
    return _CDF(
        new_ds,
        memory_guard=first._memory_guard,
        spill_dir=first._spill_dir,
    )
