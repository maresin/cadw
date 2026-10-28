"""Section N - KernelDensity reference implementation (out-of-core).

fit: reservoir-sample K centers in ONE streaming pass (Algorithm R,
     vectorized per batch). K*d*8 is bounded by MemoryGuard.check_state;
     if K is too large, centers are spilled to disk with SpillDir and
     streamed chunk-by-chunk at score time.

score: for each query batch, log p(x) = logsumexp_i(-||x-c_i||^2/(2h^2))
       - log K - d*log(h*sqrt(2*pi)). When centers are spilled, the
       logsumexp is accumulated chunk-wise (never all centers in RAM).

No pa.Table, no read_all(), no per-row Python loops over N.
See .pingis/impl/N/notes.md for spec and .pingis/impl/PRIMITIVES_USED.md
for primitives reused.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import pyarrow as pa

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cadw.core.memory_guard import MemoryGuard
from cadw.core.spill import (
    SpillDir, array_schema, array_to_record_batch, record_batch_to_numpy,
    spill_array_writer, spill_array_reader,
)

BatchIterFactory = Callable[[], Iterator[pa.RecordBatch]]



def _batch_to_X(batch: pa.RecordBatch, feature_cols) -> np.ndarray:
    cols = []
    for c in feature_cols:
        arr = batch.column(c).to_numpy(zero_copy_only=False)
        cols.append(np.asarray(arr, dtype=np.float64))
    X = np.column_stack(cols)
    mask = ~np.isnan(X).any(axis=1)
    return X[mask]


def _reservoir_k(batch_iter: Iterator[pa.RecordBatch], feature_cols, K: int, seed: int = 0):
    """Algorithm R reservoir sample of K rows, vectorized per batch.

    For each incoming batch of size b, we treat it as b consecutive
    stream items i = n_seen+1 .. n_seen+b. We only need to replace slots
    where an item i <= K OR rand(0..i-1) < K. Simplification valid because
    the arrival order is arbitrary and the sample is uniform: for each
    i > K, the probability that item i lands in the reservoir is K/i.
    We implement via per-item RNG draws only when b is small; for large
    batches we use vectorized uniform comparison against K/seen_i.
    """
    rng = np.random.default_rng(seed)
    reservoir = None
    n_seen = 0
    for batch in batch_iter:
        X = _batch_to_X(batch, feature_cols)
        if X.shape[0] == 0:
            continue
        b, d = X.shape
        if reservoir is None:
            reservoir = np.empty((K, d), dtype=np.float64)
        # first fill
        take = min(K - n_seen, b)
        if take > 0:
            reservoir[n_seen:n_seen + take] = X[:take]
            n_seen += take
            X = X[take:]
            b = X.shape[0]
        if b == 0:
            continue
        # items i = n_seen+1 .. n_seen+b (1-indexed stream positions)
        stream_idx = np.arange(n_seen + 1, n_seen + b + 1, dtype=np.float64)
        # Probability of acceptance for each item i > K: K/i
        probs = K / stream_idx
        accept = rng.random(b) < probs
        n_acc = int(accept.sum())
        if n_acc > 0:
            x_acc = X[accept]
            slots = rng.integers(0, K, size=n_acc)
            reservoir[slots] = x_acc
        n_seen += b
    if reservoir is None or n_seen == 0:
        raise ValueError("KernelDensity.fit: empty input")
    n_kept = min(n_seen, K)
    return reservoir[:n_kept].copy(), n_seen


def _silverman_bandwidth(X: np.ndarray) -> float:
    n, d = X.shape
    if n < 2:
        return 1.0
    std = float(np.nanstd(X, axis=0).mean())
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    # Silverman's rule of thumb, averaged over dims
    return float(std * n ** (-1.0 / (d + 4.0)))


@dataclass
class KernelDensityModel:
    centers_: Optional[np.ndarray]              # None when spilled
    centers_path_: Optional[Path]              # set when spilled
    bandwidth_: float
    n_features_: int
    feature_names_: list
    n_centers_: int
    n_seen_: int
    _spill_dir: Optional[SpillDir] = None

    def _iter_center_chunks(self, chunk_rows: int = 4096) -> Iterator[np.ndarray]:
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

    def _log_density_batch(self, X: np.ndarray) -> np.ndarray:
        # log p(x) = logsumexp_i(-||x-c_i||^2 / (2h^2)) - log K - d*log(h*sqrt(2pi))
        n_q, d = X.shape
        if d != self.n_features_:
            raise ValueError(f"X has {d} features, model expects {self.n_features_}")
        h = self.bandwidth_
        log_kernel_const = -d * np.log(h * np.sqrt(2.0 * np.pi))
        # accumulate logsumexp over center chunks
        logsumexp = np.full(n_q, -np.inf, dtype=np.float64)
        for C in self._iter_center_chunks():
            # squared distance matrix (n_q, K_chunk)
            # use ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x.c
            x2 = (X * X).sum(axis=1)[:, None]
            c2 = (C * C).sum(axis=1)[None, :]
            d2 = x2 + c2 - 2.0 * X @ C.T
            d2 = np.maximum(d2, 0.0)
            lk = -d2 / (2.0 * h * h)
            chunk_lse = np.logaddexp.reduce(lk, axis=1)
            logsumexp = np.logaddexp(logsumexp, chunk_lse)
        log_density = logsumexp - np.log(self.n_centers_) + log_kernel_const
        return log_density

    def score_samples(self, batch_iter: Iterator[pa.RecordBatch], feature_cols) -> Iterator[np.ndarray]:
        for batch in batch_iter:
            X = _batch_to_X(batch, feature_cols)
            if X.shape[0] == 0:
                yield np.empty(0, dtype=np.float64)
                continue
            yield self._log_density_batch(X)

    def score(self, batch_iter, feature_cols) -> float:
        total = 0.0; n = 0
        for arr in self.score_samples(batch_iter, feature_cols):
            total += float(arr.sum()); n += int(arr.size)
        return total / n if n else float("nan")

    def cleanup(self) -> None:
        if self._spill_dir is not None:
            self._spill_dir.__exit__(None, None, None)
            self._spill_dir = None


def fit_kernel_density(
    batch_iter_factory: BatchIterFactory,
    feature_cols,
    *,
    n_centers: int = 4096,
    bandwidth: Optional[float] = None,
    memory_guard: Optional[MemoryGuard] = None,
    seed: int = 0,
    spill_dir: Optional[Path] = None,
) -> KernelDensityModel:
    if n_centers < 1:
        raise ValueError(f"n_centers must be >= 1, got {n_centers}")
    mg = memory_guard or MemoryGuard(max_memory_mb=512)
    feature_cols = list(feature_cols)
    d = len(feature_cols)

    centers, n_seen = _reservoir_k(batch_iter_factory(), feature_cols, n_centers, seed=seed)
    K = centers.shape[0]
    h = float(bandwidth) if bandwidth is not None else _silverman_bandwidth(centers)
    if h <= 0:
        raise ValueError(f"bandwidth must be > 0, got {h}")

    centers_bytes = 8 * K * d
    need_spill = centers_bytes > mg.state_limit_bytes

    if not need_spill:
        mg.check_state(centers_bytes, name="KernelDensity.centers")
        return KernelDensityModel(
            centers_=centers, centers_path_=None, bandwidth_=h,
            n_features_=d, feature_names_=feature_cols,
            n_centers_=K, n_seen_=n_seen,
        )

    # spill centers
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


def _make_batches(n_batches: int, rows: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        X = rng.normal(size=(rows, d))
        yield pa.RecordBatch.from_arrays(
            [pa.array(X[:, j]) for j in range(d)],
            names=[f"x{j}" for j in range(d)],
        )


if __name__ == "__main__":
    import resource, time
    d = 8
    cols = [f"x{j}" for j in range(d)]
    factory = lambda: _make_batches(20, 5000, d, seed=0)
    mg = MemoryGuard(max_memory_mb=256)
    t0 = time.time()
    model = fit_kernel_density(factory, cols, n_centers=2048, memory_guard=mg, seed=1)
    ll = model.score(factory(), cols)
    dt = time.time() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"kde fit+score in {dt:.3f}s; peak RSS = {peak_mb:.1f} MiB")
    print(f"n_centers={model.n_centers_}  n_seen={model.n_seen_}  h={model.bandwidth_:.4f}  mean_log_lik={ll:.4f}")
    model.cleanup()
