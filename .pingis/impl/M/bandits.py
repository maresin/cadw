"""Section M - bandit reference implementations (out-of-core).

Three algorithms sharing a single ArmStatsOp reducer:
  * EpsilonGreedyPolicy    (M1.EpsilonGreedy)
  * UCB1Policy             (M1.UCB1)
  * ThompsonSamplingPolicy (M1.ThompsonSampling)

All impls accept `batch_iter_factory` (callable returning a fresh iterator)
so they can stream the same data multiple times without buffering. State
is O(A) where A = number of arms (bounded, checked by MemoryGuard). No
pa.Table, no read_all(), no per-row Python loops over N.

See .pingis/impl/M/notes.md for the spec and .pingis/impl/PRIMITIVES_USED.md
for the cadw primitives reused here.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import pyarrow as pa

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.reduce import ReduceOp, reduce

BatchIterFactory = Callable[[], Iterator[pa.RecordBatch]]


class ArmStatsOp(ReduceOp):
    """Per-arm (sums, counts[, successes]) in a single streaming pass.

    Batches must contain: `arm` (int), `reward` (float, or 0/1 for TS).
    Rows with NaN reward or out-of-range arm are skipped.
    """

    def __init__(self, n_arms: int, *, track_bernoulli: bool = False) -> None:
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1, got {n_arms}")
        self.n_arms = int(n_arms)
        self.track_bernoulli = bool(track_bernoulli)

    def estimate_state_nbytes(self) -> int:
        n = 8 * self.n_arms * 3 if self.track_bernoulli else 8 * self.n_arms * 2
        return int(n + 128)

    def init(self):
        A = self.n_arms
        sums = np.zeros(A, dtype=np.float64)
        counts = np.zeros(A, dtype=np.int64)
        succ = np.zeros(A, dtype=np.int64) if self.track_bernoulli else None
        return (sums, counts, succ)

    def update(self, state, batch: pa.RecordBatch):
        sums, counts, succ = state
        if batch.num_rows == 0:
            return state
        arms = np.asarray(batch.column("arm").to_numpy(zero_copy_only=False),
                          dtype=np.int64)
        rewards = np.asarray(batch.column("reward").to_numpy(zero_copy_only=False),
                             dtype=np.float64)
        mask = ~np.isnan(rewards) & (arms >= 0) & (arms < self.n_arms)
        if not mask.all():
            arms = arms[mask]; rewards = rewards[mask]
        if arms.size == 0:
            return state
        np.add.at(sums, arms, rewards)
        np.add.at(counts, arms, 1)
        if succ is not None:
            np.add.at(succ, arms, (rewards > 0.5).astype(np.int64))
        return (sums, counts, succ)

    def merge(self, a, b):
        s = a[0] + b[0]; c = a[1] + b[1]
        if a[2] is None:
            return (s, c, None)
        return (s, c, a[2] + b[2])

    def state_nbytes(self, state) -> int:
        sums, counts, succ = state
        n = sums.nbytes + counts.nbytes
        if succ is not None:
            n += succ.nbytes
        return int(n + 128)

    def finalize(self, state):
        sums, counts, succ = state
        return {"sums": sums, "counts": counts, "successes": succ}


def _fit_common(batch_iter_factory, n_arms, memory_guard, *, track_bernoulli=False):
    mg = memory_guard or MemoryGuard(max_memory_mb=512)
    op = ArmStatsOp(n_arms, track_bernoulli=track_bernoulli)
    out = reduce(batch_iter_factory(), op, mg, log=False)
    sums = out["sums"]; counts = out["counts"]; succ = out["successes"]
    q = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return q, counts.astype(np.int64), (None if succ is None else succ.astype(np.int64))


@dataclass
class EpsilonGreedyPolicy:
    n_arms: int
    q_: np.ndarray
    n_: np.ndarray
    eps0: float = 0.1
    decay: float = 1.0
    total_pulls: int = 0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def current_eps(self) -> float:
        return float(self.eps0 / np.sqrt(1.0 + self.total_pulls / max(self.decay, 1e-9)))

    def select(self, context=None) -> int:
        if self.total_pulls < self.n_arms:
            return int(self.total_pulls)
        if self.rng.random() < self.current_eps():
            return int(self.rng.integers(0, self.n_arms))
        return int(np.argmax(self.q_))

    def update(self, arm: int, reward: float) -> None:
        if arm < 0 or arm >= self.n_arms:
            raise IndexError(f"arm={arm} out of range [0,{self.n_arms})")
        if np.isnan(reward):
            return
        self.total_pulls += 1
        self.n_[arm] += 1
        self.q_[arm] += (reward - self.q_[arm]) / self.n_[arm]

    @classmethod
    def fit_from_log(cls, batch_iter_factory, n_arms, *, eps0=0.1, decay=1.0,
                     memory_guard=None, seed=0):
        q, n, _ = _fit_common(batch_iter_factory, n_arms, memory_guard)
        return cls(n_arms=n_arms, q_=q, n_=n, eps0=float(eps0), decay=float(decay),
                   total_pulls=int(n.sum()), rng=np.random.default_rng(seed))


@dataclass
class UCB1Policy:
    n_arms: int
    q_: np.ndarray
    n_: np.ndarray
    c: float = 1.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def ucb_scores(self) -> np.ndarray:
        total = max(int(self.n_.sum()), 1)
        bonus = self.c * np.sqrt(2.0 * np.log(total + 1.0) / np.maximum(self.n_, 1))
        scores = self.q_ + bonus
        unexplored = self.n_ == 0
        if unexplored.any():
            scores = scores.copy(); scores[unexplored] = np.inf
        return scores

    def select(self, context=None) -> int:
        return int(np.argmax(self.ucb_scores()))

    def update(self, arm: int, reward: float) -> None:
        if arm < 0 or arm >= self.n_arms:
            raise IndexError(f"arm={arm} out of range [0,{self.n_arms})")
        if np.isnan(reward):
            return
        self.n_[arm] += 1
        self.q_[arm] += (reward - self.q_[arm]) / self.n_[arm]

    @classmethod
    def fit_from_log(cls, batch_iter_factory, n_arms, *, c=1.0,
                     memory_guard=None, seed=0):
        q, n, _ = _fit_common(batch_iter_factory, n_arms, memory_guard)
        return cls(n_arms=n_arms, q_=q, n_=n, c=float(c),
                   rng=np.random.default_rng(seed))


@dataclass
class ThompsonSamplingPolicy:
    n_arms: int
    q_: np.ndarray
    n_: np.ndarray
    alpha_: np.ndarray
    beta_: np.ndarray
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def select(self, context=None) -> int:
        theta = self.rng.beta(self.alpha_, self.beta_)
        return int(np.argmax(theta))

    def update(self, arm: int, reward: float) -> None:
        if arm < 0 or arm >= self.n_arms:
            raise IndexError(f"arm={arm} out of range [0,{self.n_arms})")
        if np.isnan(reward):
            return
        self.n_[arm] += 1
        success = 1 if reward > 0.5 else 0
        self.alpha_[arm] += success
        self.beta_[arm] += (1 - success)
        self.q_[arm] += (reward - self.q_[arm]) / self.n_[arm]

    @classmethod
    def fit_from_log(cls, batch_iter_factory, n_arms, *, memory_guard=None, seed=0):
        q, n, succ = _fit_common(batch_iter_factory, n_arms, memory_guard, track_bernoulli=True)
        alpha = 1.0 + succ.astype(np.float64)
        beta = 1.0 + (n - succ).astype(np.float64)
        return cls(n_arms=n_arms, q_=q, n_=n, alpha_=alpha, beta_=beta,
                   rng=np.random.default_rng(seed))


def _make_batches(n_batches: int, rows_per_batch: int, n_arms: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    true_p = np.linspace(0.2, 0.8, n_arms)
    for _ in range(n_batches):
        arms = rng.integers(0, n_arms, size=rows_per_batch)
        rewards = (rng.random(rows_per_batch) < true_p[arms]).astype(np.float64)
        yield pa.RecordBatch.from_arrays(
            [pa.array(arms.astype(np.int64)), pa.array(rewards)],
            names=["arm", "reward"],
        )


if __name__ == "__main__":
    import resource, time
    n_arms = 10
    factory = lambda: _make_batches(20, 5000, n_arms)
    mg = MemoryGuard(max_memory_mb=256)
    t0 = time.time()
    eg = EpsilonGreedyPolicy.fit_from_log(factory, n_arms, memory_guard=mg, seed=1)
    ucb = UCB1Policy.fit_from_log(factory, n_arms, memory_guard=mg, seed=1)
    ts = ThompsonSamplingPolicy.fit_from_log(factory, n_arms, memory_guard=mg, seed=1)
    dt = time.time() - t0
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"fit ok in {dt:.3f}s; peak RSS = {peak_kb/1024:.1f} MiB")
    print("EG   q_ =", np.round(eg.q_, 3).tolist(), "best=", int(np.argmax(eg.q_)))
    print("UCB  q_ =", np.round(ucb.q_, 3).tolist(), "best=", ucb.select())
    print("TS   alpha=", np.round(ts.alpha_, 2).tolist(), "best=", ts.select())
