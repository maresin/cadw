"""OneHotEncoder — out-of-core (v0.9.143-batch5, DEC-058).

Promoted from .pingis/impl/O/one_hot_encoder.py.
See O_eda.yaml::OneHotEncoder card.

Fit: external_sort each categorical column -> stream unique categories
(spilled if large). State = sum(len(cats[c])) Python-object values +
per-column ndarray pointer arrays. bounded_state: true (total_categories
user param via data); refuses at guard if state_limit exceeded.

transform_batches: per-batch dense (n, total) f64, streaming. RAM
O(batch * total_categories).
"""
from __future__ import annotations

import sys as _sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded
from cadw.core.spill import SpillDir
from cadw.ops.sort import external_sort, read_sorted


_HEADER_BYTES = 256


def _estimate_category_bytes(v) -> int:
    """Conservative per-category bytes (numpy pointer + Python object)."""
    return 8 + _sys.getsizeof(v)


def _collect_unique_column(
    df,
    col: str,
    mg: MemoryGuard,
    spill_dir: Path,
    *,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """External sort by col, stream unique categories (O(1) RAM to spill)."""
    sorted_path = external_sort(
        df._iter_filtered_batches(batch_size=batch_size),
        key_columns=[col], value_columns=[col],
        memory_guard=mg, spill_dir=spill_dir, log=False,
    )
    uniques = []
    last = object()
    cumulative_bytes = 0
    for rb in read_sorted(sorted_path, columns=[col]):
        arr = rb.column(col).to_pylist()
        for v in arr:
            if v != last:
                # G2/G5: guard before list append (unbounded vocab)
                cumulative_bytes += _estimate_category_bytes(v)
                mg.check_state(cumulative_bytes + 8 * len(uniques),
                               name="one_hot_encoder.vocab")
                uniques.append(v)
                last = v
    if not uniques:
        return np.empty(0, dtype=object)
    return np.asarray(uniques, dtype=object)


@dataclass
class OneHotEncoderModel:
    categories_: dict
    feature_names_: list
    handle_unknown_: str = "ignore"

    def state_nbytes(self) -> int:
        total = _sys.getsizeof(self.categories_)  # dict header
        for col, arr in self.categories_.items():
            total += _sys.getsizeof(arr)  # ndarray header + pointer array
            total += _sys.getsizeof(col)  # str key
            for v in arr:
                total += _sys.getsizeof(v)
        return int(total + _HEADER_BYTES)

    def transform_batches(
        self,
        df, *,
        batch_size: int | None = None,
        memory_guard=None,
    ) -> Iterator[pa.RecordBatch]:
        """Yield per-batch dense one-hot (n, total) f64 blocks.

        handle_unknown='ignore' -> extra unknown column per categorical col.
        handle_unknown='error'  -> raise ValueError on unseen value.
        """
        mg = memory_guard or df._memory_guard
        cols = list(self.categories_.keys())
        offsets = []
        total = 0
        for c in cols:
            offsets.append(total)
            total += len(self.categories_[c]) + (
                1 if self.handle_unknown_ == "ignore" else 0)
        cat_to_idx = [
            {v: i for i, v in enumerate(self.categories_[c])} for c in cols
        ]
        unk_idx = [len(self.categories_[c]) for c in cols]
        for batch in df._iter_filtered_batches(batch_size=batch_size):
            n = batch.num_rows
            if n == 0:
                continue
            # guard before np.zeros((n, total))
            mg.check_state(n * total * 8, name="one_hot_encoder.transform")
            out = np.zeros((n, total), dtype=np.float64)
            for j, c in enumerate(cols):
                vals = batch.column(c).to_pylist()
                lookup = cat_to_idx[j]
                unk = unk_idx[j]
                start = offsets[j]
                for row_i, v in enumerate(vals):
                    idx = lookup.get(v)
                    if idx is None:
                        if self.handle_unknown_ == "error":
                            raise ValueError(
                                f"unknown category {v!r} in column {c!r}")
                        idx = unk
                    out[row_i, start + idx] = 1.0
            yield pa.RecordBatch.from_arrays(
                [pa.array(out[:, j], type=pa.float64())
                 for j in range(total)],
                names=[f"oh{j}" for j in range(total)],
            )


def fit_one_hot_encoder(
    df, *,
    categorical_cols: Sequence[str],
    handle_unknown: str = "ignore",
    memory_guard=None,
    batch_size: int | None = None,
) -> OneHotEncoderModel:
    """Fit OHE on categorical_cols (external_sort per column -> vocab).

    State = total_categories Python objects + per-column pointer arrays.
    Refuses at guard if vocab exceeds state_limit.
    """
    if handle_unknown not in ("ignore", "error"):
        raise ValueError(
            f"handle_unknown must be 'ignore'|'error', got {handle_unknown!r}")
    if not categorical_cols:
        raise ValueError("categorical_cols must be non-empty")
    mg = memory_guard or MemoryGuard(max_memory_mb=512)

    cats = {}
    feature_names = []
    with SpillDir(guard=mg, name="one_hot_vocab") as sdir:
        for c in categorical_cols:
            arr = _collect_unique_column(
                df, c, mg, sdir.root, batch_size=batch_size)
            cats[c] = arr
            feature_names.extend([f"{c}_{v}" for v in arr])
            if handle_unknown == "ignore":
                feature_names.append(f"{c}__unknown")
    return OneHotEncoderModel(
        categories_=cats, feature_names_=feature_names,
        handle_unknown_=handle_unknown,
    )
