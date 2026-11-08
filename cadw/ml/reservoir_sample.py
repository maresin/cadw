"""ReservoirSample — uniform sample of k rows, one pass (v0.9.143-batch5).

Promoted from .pingis/impl/O/reservoir.py (DEC-058).
See O_eda.yaml::ReservoirSample card.

State = k'*d*f64 (reservoir matrix) + 256. Bounded by k (user param).
Fit: single streaming pass via cadw.ops.sampling.reservoir_sample.

bounded_state: true (k*d user params). n-independent (Algorithm R/L
treats each arriving item independently; state does not grow with n).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.sampling import reservoir_sample


_HEADER_BYTES = 256


@dataclass
class ReservoirSampleModel:
    sample_: np.ndarray           # (k', d) float64
    feature_names_: list[str]
    k_: int                       # requested k
    n_seen_: int
    seed_: int

    def state_nbytes(self) -> int:
        return int(self.sample_.nbytes + _HEADER_BYTES)

    def sample_batches(
        self, batch_size: int = 4096,
    ) -> Iterator[pa.RecordBatch]:
        """Yield sample as RecordBatch chunks (streaming consumption)."""
        d = self.sample_.shape[1]
        for i in range(0, self.sample_.shape[0], batch_size):
            chunk = self.sample_[i:i + batch_size]
            yield pa.RecordBatch.from_arrays(
                [pa.array(chunk[:, j], type=pa.float64()) for j in range(d)],
                names=self.feature_names_,
            )


def fit_reservoir_sample(
    df, *,
    feature_cols: Sequence[str] | None = None,
    k: int = 1000,
    seed: int = 0,
    memory_guard=None,
    batch_size: int | None = None,
) -> ReservoirSampleModel:
    """Uniform reservoir sample of k rows (one streaming pass).

    feature_cols=None -> all columns (numeric only; non-numeric raises
    TypeError from to_numpy).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    mg = memory_guard or MemoryGuard(max_memory_mb=512)

    # infer columns from first batch
    try:
        first = next(df._iter_filtered_batches(batch_size=1000))
    except StopIteration:
        raise ValueError("fit_reservoir_sample: empty input")
    if first.num_rows == 0:
        raise ValueError("fit_reservoir_sample: empty first batch")
    if feature_cols is None:
        feature_cols = list(first.schema.names)
    feature_cols = list(feature_cols)
    d = len(feature_cols)
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")

    sample, n_seen = reservoir_sample(
        df._iter_filtered_batches(batch_size=batch_size),
        feature_cols, k,
        seed=seed, memory_guard=mg,
    )
    return ReservoirSampleModel(
        sample_=sample,
        feature_names_=feature_cols,
        k_=k, n_seen_=n_seen, seed_=seed,
    )
