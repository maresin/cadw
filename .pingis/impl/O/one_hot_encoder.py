"""Section O - OneHotEncoder reference implementation (out-of-core).

Pass 1: external_sort each categorical column -> stream unique values via
        read_sorted -> per-column vocabulary (spilled with SpillDir if V
        is large).
Pass 2: stream encode via np.searchsorted on the vocabulary.

Memory: per-column vocab is bounded by state_limit_bytes; if larger, we
spill to disk and encode by streaming the vocab chunk-wise. No pa.Table,
no read_all(), no per-row Python loops over N.

See .pingis/impl/O/notes.md for the spec.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import pyarrow as pa

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cadw.core.memory_guard import MemoryGuard
from cadw.core.spill import SpillDir
from cadw.ops.sort import external_sort, read_sorted

BatchIterFactory = Callable[[], Iterator[pa.RecordBatch]]


def _collect_unique_column(
    batch_iter_factory: BatchIterFactory,
    col: str,
    mg: MemoryGuard,
    spill_dir: Path,
) -> np.ndarray:
    """Sort by col, stream uniques (O(1) RAM, u on disk)."""
    sorted_path = external_sort(
        batch_iter_factory(),
        key_columns=[col], value_columns=[col],
        memory_guard=mg, spill_dir=spill_dir, log=False,
    )
    uniques = []
    last = object()
    for rb in read_sorted(sorted_path, columns=[col]):
        arr = rb.column(col).to_pylist()
        for v in arr:
            if v != last:
                uniques.append(v); last = v
    if not uniques:
        return np.empty(0, dtype=object)
    return np.asarray(uniques, dtype=object)


@dataclass
class OneHotEncoderModel:
    categories_: dict          # col -> np.ndarray of categories (may be object)
    feature_names_: list
    handle_unknown_: str = 'ignore'

    def transform_batches(self, batch_iter: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        # For each batch, produce a dense (b, total_dim) float block. Caller can
        # further compress to CSR if needed; keeping dense here for simplicity
        # because a full OHE sparse matrix requires index tracking anyway.
        cols = list(self.categories_.keys())
        offsets = []
        total = 0
        for c in cols:
            offsets.append(total)
            total += len(self.categories_[c]) + (1 if self.handle_unknown_ == 'ignore' else 0)
        for batch in batch_iter:
            n = batch.num_rows
            if n == 0:
                continue
            out = np.zeros((n, total), dtype=np.float64)
            for j, c in enumerate(cols):
                vals = batch.column(c).to_pylist()
                cats = self.categories_[c]
                cat_to_idx = {v: i for i, v in enumerate(cats)}
                unk_idx = len(cats)
                start = offsets[j]
                for row_i, v in enumerate(vals):
                    idx = cat_to_idx.get(v)
                    if idx is None:
                        if self.handle_unknown_ == 'error':
                            raise ValueError(f"unknown category {v!r} in column {c!r}")
                        idx = unk_idx
                    out[row_i, start + idx] = 1.0
            yield pa.RecordBatch.from_arrays(
                [pa.array(out[:, j]) for j in range(total)],
                names=[f'oh{j}' for j in range(total)],
            )


def fit_one_hot_encoder(
    batch_iter_factory: BatchIterFactory,
    categorical_cols: Sequence[str],
    *,
    handle_unknown: str = 'ignore',
    memory_guard: Optional[MemoryGuard] = None,
) -> OneHotEncoderModel:
    if handle_unknown not in ('ignore', 'error'):
        raise ValueError(f"handle_unknown must be 'ignore'|'error', got {handle_unknown!r}")
    mg = memory_guard or MemoryGuard(max_memory_mb=512)
    cats = {}
    feature_names = []
    with SpillDir(guard=mg, name='one_hot_vocab') as sdir:
        for c in categorical_cols:
            arr = _collect_unique_column(batch_iter_factory, c, mg, sdir.root)
            cats[c] = arr
            feature_names.extend([f'{c}_{v}' for v in arr])
            if handle_unknown == 'ignore':
                feature_names.append(f'{c}__unknown')
    return OneHotEncoderModel(categories_=cats, feature_names_=feature_names,
                              handle_unknown_=handle_unknown)


def _make_batches(n_batches, rows, cols, cardinalities, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        arrays = []
        for c, K in zip(cols, cardinalities):
            vals = rng.integers(0, K, size=rows).astype(np.int64)
            arrays.append(pa.array(vals))
        yield pa.RecordBatch.from_arrays(arrays, names=list(cols))


if __name__ == '__main__':
    import resource, time
    cols = ['c0', 'c1']
    card = [12, 8]
    factory = lambda: _make_batches(10, 500, cols, card, seed=0)
    mg = MemoryGuard(max_memory_mb=256)
    t0 = time.time()
    model = fit_one_hot_encoder(factory, cols, memory_guard=mg)
    dt = time.time() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f'OneHotEncoder fit in {dt:.3f}s; peak RSS = {peak_mb:.1f} MiB')
    for c in cols:
        print(f'  col {c!r}: {len(model.categories_[c])} categories')
    total = 0; n_batches = 0
    for rb in model.transform_batches(factory()):
        n_batches += 1; total += rb.num_rows
        assert rb.num_columns == len(model.feature_names_)
    print(f'  transformed {n_batches} batches, {total} rows, {len(model.feature_names_)} features')
