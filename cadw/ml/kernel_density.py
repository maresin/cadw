"""Kernel Density Estimation (Gaussian kernel) — out-of-core.

Promoted from internal audit prototype (DEC-058).
See N_density.yaml::KernelDensity card.

Fit: reservoir-sample K centers in ONE streaming pass (vectorized
per batch). bandwidth from Silverman's rule (or user-supplied).
Score: log p(x) = logsumexp_i(-||x-c_i||^2/(2h^2)) - log K
                  - d*log(h*sqrt(2*pi)).

bounded_state: true (K*d user params). Center matrix spills to SpillDir
when K*d*8 > state_limit_bytes; scoring streams chunks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.core.spill import (
    SpillDir, array_schema, array_to_record_batch, record_batch_to_numpy,
    spill_array_writer, spill_array_reader,
)
from cadw.ops.sampling import batch_to_features, reservoir_sample


_HEADER_BYTES = 128
_SPILLED_HEADER_BYTES = 256


def _silverman_bandwidth(X: np.ndarray) -> float:
    """Silverman's rule of thumb, averaged over dims (inline, P3)."""
    n, d = X.shape
    if n < 2:
        return 1.0
    std = float(np.nanstd(X, axis=0).mean())
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    return float(std * n ** (-1.0 / (d + 4.0)))


@dataclass
class KernelDensityModel:
    centers_: Optional[np.ndarray]          # None when spilled
    centers_path_: Optional[Path]          # set when spilled
    bandwidth_: float
    n_features_: int
    feature_names_: list
    n_centers_: int
    n_seen_: int
    _spill_dir: Optional[SpillDir] = None

    def state_nbytes(self) -> int:
        if self.centers_ is not None:
            return int(self.centers_.nbytes + _HEADER_BYTES)
        return int(_SPILLED_HEADER_BYTES)

    def _iter_center_chunks(
        self, chunk_rows: int = 4096,
    ) -> Iterator[np.ndarray]:
        if self.centers_ is not None:
            for i in range(0, self.centers_.shape[0], chunk_rows):
                yield self.centers_[i:i + chunk_rows]
            return
        with spill_array_reader(self.centers_path_) as r:
            buf = []
            cur = 0
            for rb in r:
                arr = record_batch_to_numpy(rb)
                buf.append(arr); cur += arr.shape[0]
                if cur >= chunk_rows:
                    yield np.vstack(buf); buf = []; cur = 0
            if buf:
                yield np.vstack(buf)

    def _log_density_batch(
        self, X: np.ndarray, *, memory_guard: Optional[MemoryGuard] = None,
    ) -> np.ndarray:
        n_q, d = X.shape
        if d != self.n_features_:
            raise ValueError(
                f"X has {d} features, model expects {self.n_features_}")
        h = self.bandwidth_
        log_kernel_const = -d * np.log(h * np.sqrt(2.0 * np.pi))
        logsumexp = np.full(n_q, -np.inf, dtype=np.float64)
        for C in self._iter_center_chunks():
            if memory_guard is not None:
                memory_guard.check_state(
                    n_q * C.shape[0] * 8, name="kernel_density.logpdf_chunk")
            x2 = (X * X).sum(axis=1)[:, None]
            c2 = (C * C).sum(axis=1)[None, :]
            d2 = x2 + c2 - 2.0 * X @ C.T
            d2 = np.maximum(d2, 0.0)
            lk = -d2 / (2.0 * h * h)
            chunk_lse = np.logaddexp.reduce(lk, axis=1)
            logsumexp = np.logaddexp(logsumexp, chunk_lse)
        return logsumexp - np.log(self.n_centers_) + log_kernel_const

    def score_samples(
        self,
        df, *,
        batch_size: int | None = None,
        memory_guard=None,
    ) -> Iterator[np.ndarray]:
        """Stream per-batch log-densities (shape (n_batch,)).

        Caller controls iteration; RAM O(batch + batch*K_chunk). Streaming.
        """
        mg = memory_guard or df._memory_guard
        cols = list(self.feature_names_)
        for batch in df._iter_filtered_batches(batch_size=batch_size):
            X = batch_to_features(batch, cols)
            if X.shape[0] == 0:
                yield np.empty(0, dtype=np.float64)
                continue
            yield self._log_density_batch(X, memory_guard=mg)

    def score(
        self, df, *, batch_size: int | None = None, memory_guard=None,
    ) -> float:
        """Mean log-likelihood over df (streaming; O(batch) RAM)."""
        total = 0.0; n = 0
        for arr in self.score_samples(
                df, batch_size=batch_size, memory_guard=memory_guard):
            total += float(arr.sum()); n += int(arr.size)
        return total / n if n else float("nan")

    def cleanup(self) -> None:
        if self._spill_dir is not None:
            self._spill_dir.__exit__(None, None, None)
            self._spill_dir = None


def fit_kernel_density(
    df, *,
    feature_cols: Sequence[str] | None = None,
    n_centers: int = 4096,
    bandwidth: Optional[float] = None,
    memory_guard=None,
    batch_size: int | None = None,
    seed: int = 0,
    spill_dir: Path | None = None,
) -> KernelDensityModel:
    """Fit Gaussian KDE on feature_cols.

    feature_cols=None -> all columns.
    n_centers: reservoir size (user param). bandwidth=None -> Silverman.
    Centers spill to disk when K*d*8 > state_limit_bytes.
    """
    if n_centers < 1:
        raise ValueError(f"n_centers must be >= 1, got {n_centers}")
    mg = memory_guard or MemoryGuard(max_memory_mb=512)

    # infer columns from first batch
    try:
        first = next(df._iter_filtered_batches(batch_size=1000))
    except StopIteration:
        raise ValueError("fit_kernel_density: empty input")
    if first.num_rows == 0:
        raise ValueError("fit_kernel_density: empty first batch")
    if feature_cols is None:
        feature_cols = list(first.schema.names)
    feature_cols = list(feature_cols)
    d = len(feature_cols)
    if d < 1:
        raise ValueError(f"fit_kernel_density: d must be >= 1, got {d}")

    centers, n_seen = reservoir_sample(
        df._iter_filtered_batches(batch_size=batch_size),
        feature_cols, n_centers, seed=seed, memory_guard=mg,
    )
    K = centers.shape[0]
    h = float(bandwidth) if bandwidth is not None else _silverman_bandwidth(centers)
    if h <= 0:
        raise ValueError(f"bandwidth must be > 0, got {h}")

    centers_bytes = 8 * K * d
    need_spill = centers_bytes > mg.state_limit_bytes

    if not need_spill:
        mg.check_state(centers_bytes, name="kernel_density.centers")
        return KernelDensityModel(
            centers_=centers, centers_path_=None, bandwidth_=h,
            n_features_=d, feature_names_=feature_cols,
            n_centers_=K, n_seen_=n_seen,
        )

    # spill
    if spill_dir is not None:
        sdir = SpillDir(guard=mg, name="kde_centers", path=spill_dir)
    else:
        sdir = SpillDir(guard=mg, name="kde_centers")
    sdir.__enter__()
    path = sdir.path("centers.arrow")
    schema = array_schema((K, d), np.float64)
    try:
        with spill_array_writer(path, schema) as w:
            chunk = 4096
            for i in range(0, K, chunk):
                w.write_batch(array_to_record_batch(centers[i:i + chunk]))
    except BaseException:
        sdir.__exit__(None, None, None)
        raise

    return KernelDensityModel(
        centers_=None, centers_path_=path, bandwidth_=h,
        n_features_=d, feature_names_=feature_cols,
        n_centers_=K, n_seen_=n_seen, _spill_dir=sdir,
    )
