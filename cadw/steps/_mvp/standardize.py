"""Standardize — narrow column standardization (reference MVP step).

Independent copy of cadw.scalers.StandardScaler logic. Streaming-capable:
fit_iter (1-pass Welford) + transform_batch (per-batch).
"""
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from cadw.steps.base import Transformer


class Standardize(Transformer):
    is_barrier = False
    cadw_function = None

    def __init__(self, columns):
        self._columns = ([columns] if isinstance(columns, str)
                         else list(columns))
        self._mean: dict[str, float] = {}
        self._std: dict[str, float] = {}

    def _welford(self, factory):
        n = {c: 0 for c in self._columns}
        mean = {c: 0.0 for c in self._columns}
        M2 = {c: 0.0 for c in self._columns}
        for b in factory():
            for c in self._columns:
                arr = np.asarray(
                    b.column(c).to_numpy(zero_copy_only=False),
                    dtype=np.float64)
                k = arr.shape[0]
                if k == 0:
                    continue
                n[c] += k
                delta = arr - mean[c]
                mean[c] += delta.sum() / n[c]
                M2[c] += ((arr - mean[c]) * delta).sum()
        self._mean = mean
        self._std = {
            c: (float(np.sqrt(M2[c] / n[c])) if n[c] > 0 else 1.0)
            for c in self._columns}
        self._std = {c: s if s > 0 else 1.0 for c, s in self._std.items()}

    def fit(self, df, target=None):
        self._welford(lambda: df._iter_filtered_batches())
        return self

    def fit_iter(self, factory, template_df):
        self._welford(factory)
        return self

    def transform(self, df):
        from cadw.dataset import Dataset
        from cadw.dataframe import CadwDataFrame
        # v0.9.241 (L7): use df._columns view, not df._dataset.schema.
        # df[cols] is a column-view over a shared dataset (schema NOT
        # projected); _iter_filtered_batches() yields only df._columns.
        # Building output schema from df._dataset.schema would mismatch
        # (e.g. 4 fields vs 1-col batches) -> "Schema and number of arrays
        # unequal". Same pattern as cadw/scalers.py::_output_schema.
        cols = list(df._columns)
        schema = pa.schema([df._dataset.schema.field(c) for c in cols])
        def _iter():
            for b in df._iter_filtered_batches():
                yield self._apply_batch(b, schema)
        new_ds = Dataset.from_batches(schema, _iter(), df._new_dataset_dir())
        return CadwDataFrame(
            new_ds, memory_guard=df._memory_guard,
            spill_dir=df._spill_dir)

    def transform_batch(self, batch):
        return self._apply_batch(batch, batch.schema)

    def _apply_batch(self, batch, schema):
        arrays = []
        for c in batch.schema.names:
            if c in self._columns:
                arr = batch.column(c)
                m = pa.scalar(self._mean[c], type=pa.float64())
                s = pa.scalar(self._std[c], type=pa.float64())
                arr = pc.divide(pc.subtract(arr, m), s)
                arrays.append(arr)
            else:
                arrays.append(batch.column(c))
        return pa.RecordBatch.from_arrays(arrays, schema=schema)

    def output_schema(self, schema):
        return schema

    def estimate_state_bytes(self, dc):
        return 2 * len(self._columns) * 8 + 256

    def state_bytes(self):
        if not self._mean:
            return 0
        return 2 * len(self._mean) * 8 + 256

    def to_dict(self):
        return {"columns": list(self._columns),
                "state": ({"mean": self._mean, "std": self._std}
                          if self._mean else None)}

    @classmethod
    def from_dict(cls, d):
        s = cls(d["columns"])
        st = d.get("state")
        if st:
            s._mean = st["mean"]; s._std = st["std"]
        return s
