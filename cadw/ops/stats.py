"""Statistical primitives (v0.9.134 DEC-054, v0.9.143 DEC-058).

Per-row normalization + autocorrelation (numpy) + arm statistics
(ReduceOp). No I/O.

History:
- v0.9.134: created. Moved znormalize_rows from cadw.ops.numeric;
  extracted acf from cadw.ts.transforms.AutoStationarizer._acf (forward
  extraction, AP10 exception per DEC-056: 5+ prospective consumers —
  binseg, pelt, bocpd, matrix_profile, stray).
- v0.9.143: added ArmStatsOp + arm_stats. Moved from
  .pingis/impl/M/bandits.py (promoted per DEC-058). Consumers:
  cadw.ml.bandits.fit_epsilon_greedy / fit_ucb1 / fit_thompson_sampling
  (3 -> P1 per PRIMITIVES_BACKLOG L15).
- v0.9.148: added normalize_weights (L17, P1). Consumers: ML-kNN,
  SMOTE (weighted sampling), +2 in O_eda.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa

from cadw.ops.reduce import ReduceOp, reduce as _reduce


def znormalize_rows(X: np.ndarray) -> np.ndarray:
    """Per-row z-normalization with zero-std guard.

    Each row standardized independently: mu and sd per row.
    Rows with sd < 1e-12 -> sd forced to 1 (output = X - mu, i.e. zeros).
    """
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def acf(y: np.ndarray, lag: int) -> float:
    """Autocorrelation at given lag (Pearson-style, normalized by var).

    Returns 0.0 if n <= lag or if variance < 1e-12 (degenerate series).
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n <= lag:
        return 0.0
    mean = float(y.mean())
    c0 = float(((y - mean) ** 2).mean())
    if c0 < 1e-12:
        return 0.0
    c = float(((y[:-lag] - mean) * (y[lag:] - mean)).mean())
    return c / c0


# ============================================================================
# Arm statistics (v0.9.143, DEC-058 promotion from .pingis/impl/M/bandits.py)
# ============================================================================
#
# Pattern: Reduce (AI_CONSTITUTION §3.1). State O(n_arms), transient
# O(batch). Consumers: cadw.ml.bandits (3 fit_* functions, P1 L15).
#
# Formula (from code, AP23):
#   state_nbytes = n_arms * f64 * (2 or 3) + 128
#   peak         = state_nbytes + c_batch * batch * row_size
#   c_batch measured in tests/test_bandits_verification.py (CW-3).
# ============================================================================

_ARM_STATS_HEADER_BYTES = 128


class ArmStatsOp(ReduceOp):
    """Per-arm (sums, counts[, successes]) in a single streaming pass.

    Batches must contain columns ``arm`` (integer) and ``reward`` (float).
    Rows with NaN reward or out-of-range arm are silently skipped.

    track_bernoulli=True  -> also per-arm successes (reward > 0.5 counts
                              as 1). Used by Thompson Sampling.
    track_bernoulli=False -> mean-reward path. Used by EG / UCB1.

    Guard: ``reduce()`` calls ``estimate_state_nbytes()`` and
    ``check_batch`` per iteration (AI_CONSTITUTION §3.1).
    """

    def __init__(self, n_arms: int, *, track_bernoulli: bool = False) -> None:
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1, got {n_arms}")
        self.n_arms = int(n_arms)
        self.track_bernoulli = bool(track_bernoulli)

    def estimate_state_nbytes(self) -> int:
        n = self.n_arms * 8 * (3 if self.track_bernoulli else 2)
        return int(n + _ARM_STATS_HEADER_BYTES)

    def init(self):
        A = self.n_arms
        sums = np.zeros(A, dtype=np.float64)
        counts = np.zeros(A, dtype=np.int64)
        succ = (np.zeros(A, dtype=np.int64)
                if self.track_bernoulli else None)
        return (sums, counts, succ)

    def update(self, state, batch: pa.RecordBatch):
        sums, counts, succ = state
        if batch.num_rows == 0:
            return state
        arms = np.asarray(
            batch.column("arm").to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        rewards = np.asarray(
            batch.column("reward").to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        mask = ~np.isnan(rewards) & (arms >= 0) & (arms < self.n_arms)
        if not mask.all():
            arms = arms[mask]
            rewards = rewards[mask]
        if arms.size == 0:
            return state
        np.add.at(sums, arms, rewards)
        np.add.at(counts, arms, 1)
        if succ is not None:
            np.add.at(succ, arms, (rewards > 0.5).astype(np.int64))
        return (sums, counts, succ)

    def merge(self, a, b):
        s = a[0] + b[0]
        c = a[1] + b[1]
        if a[2] is None:
            return (s, c, None)
        return (s, c, a[2] + b[2])

    def state_nbytes(self, state) -> int:
        sums, counts, succ = state
        n = sums.nbytes + counts.nbytes
        if succ is not None:
            n += succ.nbytes
        return int(n + _ARM_STATS_HEADER_BYTES)

    def finalize(self, state):
        sums, counts, succ = state
        return {"sums": sums, "counts": counts, "successes": succ}


def arm_stats(
    batch_iter,
    n_arms: int,
    *,
    track_bernoulli: bool = False,
    memory_guard=None,
):
    """Single-pass per-arm aggregation via Reduce.

    Returns (q, n, successes_or_None):
      q          = mean reward per arm (0.0 where count == 0)
      n          = pull count per arm (int64)
      successes  = per-arm successes (int64) if track_bernoulli else None

    OOC: state O(n_arms), transient O(batch). See AI_CONSTITUTION §3.1.
    """
    from cadw.core.memory_guard import MemoryGuard
    mg = memory_guard or MemoryGuard(max_memory_mb=512)
    op = ArmStatsOp(n_arms, track_bernoulli=track_bernoulli)
    out = _reduce(batch_iter, op, mg, log=False)
    sums = out["sums"]
    counts = out["counts"]
    succ = out["successes"]
    q = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    n = counts.astype(np.int64)
    succ_out = None if succ is None else succ.astype(np.int64)
    return q, n, succ_out


# ============================================================================
# Weight normalization (v0.9.148, P1 L17)
# ============================================================================

def normalize_weights(
    w: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize non-negative weights so that sum(w) == 1.

    Args:
      w: weights, any shape, non-negative.
      eps: floor for sum(w); if sum(w) < eps, return zeros (all-zero
           input maps to all-zero output, not NaN).

    Returns:
      w / max(sum(w), eps), same shape as input.

    Raises:
      ValueError: if w has negative entries (invalid for probability /
                  sampling weights).

    Handles:
      - All-zero input -> zeros (sum stays 0)
      - Empty input -> empty
      - Shape preserved exactly

    Consumers:
      - ML-kNN (weighted class voting)
      - SMOTE (interpolation weights between x_i and its kNN)
    """
    w = np.asarray(w, dtype=np.float64)
    if w.size == 0:
        return w.copy()
    if (w < 0).any():
        raise ValueError(
            "normalize_weights: negative entries not allowed")
    s = float(w.sum())
    if s < eps:
        return np.zeros_like(w)
    return w / s
