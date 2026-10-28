"""Section O - ReservoirSample (Vitter Algorithm R/L, out-of-core).

Single streaming pass. State = O(K * d) reservoir matrix, K user-chosen.
For large batches we use an acceptance-probability scheme (Algorithm L
in spirit: expected O(K * log(N/K)) RNG draws instead of O(N)): for
stream position i > K the probability that item i enters the reservoir
is K/i; draws are vectorized per batch.

Memory guard: MemoryGuard.check_state(8*K*d) before allocating.
No pa.Table, no read_all(), no per-row Python loops.

See .pingis/impl/O/notes.md for the spec.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import pyarrow as pa

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cadw.core.memory_guard import MemoryGuard


def _batch_to_X(batch: pa.RecordBatch, feature_cols: Sequence[str]) -> np.ndarray:
    cols = []
    for c in feature_cols:
        arr = batch.column(c).to_numpy(zero_copy_only=False)
        cols.append(np.asarray(arr, dtype=np.float64))
    return np.column_stack(cols)


@dataclass
class ReservoirSampler:
    k: int
    feature_cols: Sequence[str]
    d: int
    _R: Optional[np.ndarray] = None
    _n_seen: int = 0
    _rng: Optional[np.random.Generator] = None
    _fitted: bool = False

    def __init__(self, k: int, feature_cols: Sequence[str], *, random_state: int = 0):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)
        self.feature_cols = list(feature_cols)
        self.d = len(self.feature_cols)
        self._R = None
        self._n_seen = 0
        self._rng = np.random.default_rng(random_state)
        self._fitted = False

    def add(self, batch_iter: Iterator[pa.RecordBatch], *,
            memory_guard: Optional[MemoryGuard] = None) -> "ReservoirSampler":
        mg = memory_guard
        if mg is not None:
            mg.check_state(8 * self.k * self.d, name="Reservoir.state")
        K = self.k
        if self._R is None:
            self._R = np.empty((K, self.d), dtype=np.float64)
        R = self._R; rng = self._rng; n_seen = self._n_seen
        for batch in batch_iter:
            X = _batch_to_X(batch, self.feature_cols)
            if X.shape[0] == 0:
                continue
            b = X.shape[0]
            take = min(K - n_seen, b)
            if take > 0:
                R[n_seen:n_seen + take] = X[:take]
                n_seen += take
                X = X[take:]; b = X.shape[0]
            if b == 0:
                continue
            stream_idx = np.arange(n_seen + 1, n_seen + b + 1, dtype=np.float64)
            probs = K / stream_idx
            accept = rng.random(b) < probs
            n_acc = int(accept.sum())
            if n_acc > 0:
                slots = rng.integers(0, K, size=n_acc)
                R[slots] = X[accept]
            n_seen += b
        self._n_seen = n_seen
        self._fitted = True
        return self

    @property
    def n_seen_(self) -> int:
        return self._n_seen

    def sample_array(self) -> np.ndarray:
        if not self._fitted or self._R is None:
            raise RuntimeError("call add() first")
        n_kept = min(self._n_seen, self.k)
        return self._R[:n_kept].copy()

    def sample(self) -> pa.RecordBatch:
        arr = self.sample_array()
        n, d = arr.shape
        return pa.RecordBatch.from_arrays(
            [pa.array(arr[:, j]) for j in range(d)], names=list(self.feature_cols),
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
    d = 16
    cols = [f"x{j}" for j in range(d)]
    factory = lambda: _make_batches(50, 10000, d, seed=0)
    mg = MemoryGuard(max_memory_mb=256)
    t0 = time.time()
    s = ReservoirSampler(k=5000, feature_cols=cols, random_state=7)
    s.add(factory(), memory_guard=mg)
    dt = time.time() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    arr = s.sample_array()
    print(f"reservoir sample in {dt:.3f}s; peak RSS = {peak_mb:.1f} MiB")
    print(f"n_seen={s.n_seen_}  shape={arr.shape}  mean={arr.mean():.4f}  std={arr.std():.4f}")
