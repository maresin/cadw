"""Bandits (M_bandits) — out-of-core via Reduce (v0.9.143, DEC-058).

Promoted from internal audit prototype. Three policies share a single
ArmStatsOp reducer (cadw.ops.stats.arm_stats).

State is O(n_arms) independent of n (bounded_state: true). Fit is
single-pass streaming; online select/update after fit is O(n_arms).

Reward semantics:
  - EG / UCB1: mean reward (any real value)
  - TS: Bernoulli threshold (reward > 0.5 -> success), Beta posterior
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.stats import ArmStatsOp, arm_stats


_HEADER_BYTES = 128  # dataclass instance overhead; rng state not counted
                     # (O(1) ~2.5 KB, not scaling with n).


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

@dataclass
class EpsilonGreedyPolicy:
    """Epsilon-greedy with sqrt-decay epsilon.

    After fit, `select()` returns argmax(q_) with prob 1 - eps, random arm
    with prob eps (eps decays as eps0 / sqrt(1 + pulls/decay)). First
    n_arms pulls are forced exploration (one per arm).
    """
    n_arms: int
    q_: np.ndarray
    n_: np.ndarray
    eps0: float = 0.1
    decay: float = 1.0
    total_pulls: int = 0
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0))

    def state_nbytes(self) -> int:
        return int(self.q_.nbytes + self.n_.nbytes + _HEADER_BYTES)

    def current_eps(self) -> float:
        return float(self.eps0
                     / np.sqrt(1.0 + self.total_pulls / max(self.decay, 1e-9)))

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


@dataclass
class UCB1Policy:
    """UCB1: argmax(q_ + c * sqrt(2 ln(T+1) / n_))."""
    n_arms: int
    q_: np.ndarray
    n_: np.ndarray
    c: float = 1.0
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0))

    def state_nbytes(self) -> int:
        return int(self.q_.nbytes + self.n_.nbytes + _HEADER_BYTES)

    def ucb_scores(self) -> np.ndarray:
        total = max(int(self.n_.sum()), 1)
        bonus = self.c * np.sqrt(
            2.0 * np.log(total + 1.0) / np.maximum(self.n_, 1))
        scores = self.q_ + bonus
        unexplored = self.n_ == 0
        if unexplored.any():
            scores = scores.copy()
            scores[unexplored] = np.inf
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


@dataclass
class ThompsonSamplingPolicy:
    """Thompson Sampling with Beta(1 + succ, 1 + fail) per arm.

    Assumes Bernoulli rewards. For non-Bernoulli, use EG/UCB1 (mean reward).
    """
    n_arms: int
    q_: np.ndarray
    n_: np.ndarray
    alpha_: np.ndarray
    beta_: np.ndarray
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0))

    def state_nbytes(self) -> int:
        return int(self.q_.nbytes + self.n_.nbytes
                   + self.alpha_.nbytes + self.beta_.nbytes + _HEADER_BYTES)

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


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------

def _infer_n_arms(df, arm_col: str, *, sample_bs: int = 1000) -> int:
    """Sample first batch, n_arms = max(arm)+1. Warn if sparse.

    Rare arms may be missed in the sample; pass n_arms explicitly to
    guarantee coverage.
    """
    try:
        first = next(df._iter_filtered_batches(batch_size=sample_bs))
    except StopIteration:
        raise ValueError("empty dataframe: cannot infer n_arms")
    arr = first.column(arm_col).to_numpy(zero_copy_only=False)
    if len(arr) == 0:
        raise ValueError("empty first batch: cannot infer n_arms")
    n_arms = int(np.max(arr)) + 1
    seen = len(np.unique(arr))
    if seen < n_arms * 0.5:
        warnings.warn(
            f"n_arms inferred as {n_arms} from first batch of "
            f"{first.num_rows} rows, but only {seen} distinct arms seen. "
            f"Rare arms may be missing. Pass n_arms explicitly if known.",
            stacklevel=4,
        )
    return n_arms


def _resolve_batch_size(mg, n_arms: int, track_bernoulli: bool) -> int:
    """Auto-derive batch_size from state estimate via MemoryGuard."""
    state_est = ArmStatsOp(n_arms, track_bernoulli).estimate_state_nbytes()
    try:
        bs = mg.get_batch_rows(expansion_factor=3.0, state_bytes=state_est)
    except TypeError:
        bs = mg.get_batch_rows(expansion_factor=3.0)
    return int(bs) if bs else 10_000


def _fit_bandit_stats(
    df, *, arm_col: str, reward_col: str,
    n_arms, track_bernoulli: bool,
    memory_guard, batch_size,
):
    """Single streaming pass via arm_stats. Returns (q, n, succ, n_arms, mg)."""
    mg = memory_guard or MemoryGuard(max_memory_mb=512)
    if n_arms is None:
        n_arms = _infer_n_arms(df, arm_col)
    n_arms = int(n_arms)
    if n_arms < 1:
        raise ValueError(f"n_arms must be >= 1, got {n_arms}")

    if batch_size is None:
        batch_size = _resolve_batch_size(mg, n_arms, track_bernoulli)

    def _iter():
        yield from df._iter_filtered_batches(batch_size=batch_size)

    q, n, succ = arm_stats(
        _iter(), n_arms,
        track_bernoulli=track_bernoulli,
        memory_guard=mg,
    )
    return q, n, succ, n_arms, mg


# ---------------------------------------------------------------------------
# Public fit_* (3 cards)
# ---------------------------------------------------------------------------

def fit_epsilon_greedy(
    df, *, arm_col: str, reward_col: str,
    n_arms: int | None = None,
    eps0: float = 0.1, decay: float = 1.0,
    memory_guard=None, batch_size: int | None = None, seed: int = 0,
) -> EpsilonGreedyPolicy:
    """Fit EpsilonGreedy. n_arms=None -> sample-infer (warn if sparse)."""
    q, n, _, n_arms, _ = _fit_bandit_stats(
        df, arm_col=arm_col, reward_col=reward_col, n_arms=n_arms,
        track_bernoulli=False, memory_guard=memory_guard,
        batch_size=batch_size,
    )
    return EpsilonGreedyPolicy(
        n_arms=n_arms, q_=q, n_=n,
        eps0=float(eps0), decay=float(decay),
        total_pulls=int(n.sum()),
        rng=np.random.default_rng(seed),
    )


def fit_ucb1(
    df, *, arm_col: str, reward_col: str,
    n_arms: int | None = None, c: float = 1.0,
    memory_guard=None, batch_size: int | None = None, seed: int = 0,
) -> UCB1Policy:
    """Fit UCB1. n_arms=None -> sample-infer (warn if sparse)."""
    q, n, _, n_arms, _ = _fit_bandit_stats(
        df, arm_col=arm_col, reward_col=reward_col, n_arms=n_arms,
        track_bernoulli=False, memory_guard=memory_guard,
        batch_size=batch_size,
    )
    return UCB1Policy(
        n_arms=n_arms, q_=q, n_=n, c=float(c),
        rng=np.random.default_rng(seed),
    )


def fit_thompson_sampling(
    df, *, arm_col: str, reward_col: str,
    n_arms: int | None = None,
    memory_guard=None, batch_size: int | None = None, seed: int = 0,
) -> ThompsonSamplingPolicy:
    """Fit TS (Bernoulli rewards: reward>0.5 -> success)."""
    q, n, succ, n_arms, _ = _fit_bandit_stats(
        df, arm_col=arm_col, reward_col=reward_col, n_arms=n_arms,
        track_bernoulli=True, memory_guard=memory_guard,
        batch_size=batch_size,
    )
    alpha = 1.0 + succ.astype(np.float64)
    beta = 1.0 + (n - succ).astype(np.float64)
    return ThompsonSamplingPolicy(
        n_arms=n_arms, q_=q, n_=n, alpha_=alpha, beta_=beta,
        rng=np.random.default_rng(seed),
    )
