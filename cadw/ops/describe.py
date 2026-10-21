"""Fused single-pass describe (v0.9.7b, R5: 109× speedup benchmark).

Single Reduce pass per column group: Welford (mean/M2/std) +
min/max + reservoir (Algorithm R) for quantiles. Replaces 5 Reduce
passes + 1 external_sort per column.

Reference (bench_r5.py, N=2M, 2 cols): current 2.47s / 19.3 MB;
fused 0.02s / 1.6 MB (109×, 12.1× RSS). Speedup: single pass over
data (no 5× re-scan) and no external_sort per column (reservoir
for quantiles).

Semantics (§11.5 documented):
  - count/mean/std/min/max: exact (Welford + min/max).
  - quantiles (25%/50%/75%): exact if N ≤ reservoir_size (all
    values kept); approximate (reservoir) above. Default
    reservoir_size=10000. For exact large-N, use df.quantile().
  - Output dtype float64 for all fields (int input also → float;
    3.0 == 3 in Python, existing tests pass).
  - Empty / all-null column → NaN for all stat fields; count=0.
"""
from __future__ import annotations

import random
from typing import Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.reduce import ReduceOp, reduce


_DEFAULT_RESERVOIR = 10_000
_DEFAULT_SEED = 42
_STAT_KEYS = ("count", "mean", "std", "min", "25%", "50%", "75%", "max")


class DescribeOp(ReduceOp):
    """Per-column Welford (mean/M2/std) + min/max + reservoir.

    State: list of d dicts, each {n, mean, M2, min, max,
    reservoir (list), seen}. Bounded by d × (5 × 8 + reservoir_size
    × 8) bytes; independent of N.
    """

    def __init__(
        self,
        columns: Sequence[str],
        *,
        reservoir_size: int = _DEFAULT_RESERVOIR,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        if not isinstance(reservoir_size, int) or reservoir_size < 1:
            raise ValueError(
                f"DescribeOp: reservoir_size must be int >= 1; "
                f"got {reservoir_size!r}"
            )
        self.columns = list(columns)
        self.d = len(self.columns)
        self.reservoir_size = int(reservoir_size)
        self.seed = int(seed)
        self._batch_count = 0

    def estimate_state_nbytes(self) -> int:
        # Per column: 5 float64 (n not counted; mean/M2/min/max) =
        # 5 × 8 = 40 B + reservoir (reservoir_size × 8 B).
        return self.d * (40 + self.reservoir_size * 8)

    def init(self):
        return [
            {
                "n": 0, "mean": 0.0, "M2": 0.0,
                "min": None, "max": None,
                "reservoir": [], "seen": 0,
            }
            for _ in range(self.d)
        ]

    def update(self, state, batch):
        if batch.num_rows == 0:
            return state
        rng = random.Random(self.seed + self._batch_count)
        self._batch_count += 1
        for j, col in enumerate(self.columns):
            s = state[j]
            arr_f = pc.cast(batch.column(col), pa.float64())
            x_np = arr_f.to_numpy(zero_copy_only=False)
            mask = ~np.isnan(x_np)
            x_np = x_np[mask]
            n_b = int(len(x_np))
            if n_b == 0:
                continue
            mean_b = float(x_np.mean())
            M2_b = float(((x_np - mean_b) ** 2).sum())
            n_a = int(s["n"])
            if n_a == 0:
                s["mean"] = mean_b
                s["M2"] = M2_b
            else:
                n_total = n_a + n_b
                delta = mean_b - s["mean"]
                s["mean"] = s["mean"] + delta * n_b / n_total
                s["M2"] = (
                    s["M2"] + M2_b
                    + delta * delta * n_a * n_b / n_total
                )
            s["n"] = n_a + n_b
            mn_b = float(x_np.min())
            mx_b = float(x_np.max())
            s["min"] = mn_b if s["min"] is None else min(s["min"], mn_b)
            s["max"] = mx_b if s["max"] is None else max(s["max"], mx_b)
            # Reservoir (Algorithm R)
            for v in x_np.tolist():
                s["seen"] += 1
                if len(s["reservoir"]) < self.reservoir_size:
                    s["reservoir"].append(v)
                else:
                    jj = rng.randint(0, s["seen"] - 1)
                    if jj < self.reservoir_size:
                        s["reservoir"][jj] = v
        return state

    def merge(self, a, b):
        """Best-effort merge (reduce is sequential; merge rarely
        called). Welford merge + reservoir pool + uniform downsample.
        """
        rng = random.Random(self.seed)
        out = []
        for sa, sb in zip(a, b):
            n_a, n_b = int(sa["n"]), int(sb["n"])
            n = n_a + n_b
            if n_a == 0:
                out.append(dict(sb))
                continue
            if n_b == 0:
                out.append(dict(sa))
                continue
            delta = sb["mean"] - sa["mean"]
            mean = sa["mean"] + delta * n_b / n
            M2 = (sa["M2"] + sb["M2"]
                  + delta * delta * n_a * n_b / n)
            mn = sa["min"] if sa["min"] is not None else sb["min"]
            if sa["min"] is not None and sb["min"] is not None:
                mn = min(sa["min"], sb["min"])
            mx = sa["max"] if sa["max"] is not None else sb["max"]
            if sa["max"] is not None and sb["max"] is not None:
                mx = max(sa["max"], sb["max"])
            pool = list(sa["reservoir"]) + list(sb["reservoir"])
            if len(pool) > self.reservoir_size:
                rng.shuffle(pool)
                pool = pool[: self.reservoir_size]
            out.append({
                "n": n, "mean": mean, "M2": M2,
                "min": mn, "max": mx,
                "reservoir": pool,
                "seen": sa["seen"] + sb["seen"],
            })
        return out

    def state_nbytes(self, state):
        return sum(
            40 + len(s["reservoir"]) * 8
            for s in state
        )

    def finalize(self, state):
        result = {}
        for j, col in enumerate(self.columns):
            s = state[j]
            n = int(s["n"])
            if n == 0:
                result[col] = {k: float("nan") for k in _STAT_KEYS}
                result[col]["count"] = 0
                continue
            if n > 1:
                std = float(np.sqrt(s["M2"] / (n - 1)))
            else:
                std = float("nan")
            res = np.asarray(s["reservoir"], dtype=np.float64)
            if len(res) == 0:
                q25 = q50 = q75 = float("nan")
            else:
                q25, q50, q75 = np.percentile(res, [25, 50, 75])
            result[col] = {
                "count": n,
                "mean": float(s["mean"]),
                "std": std,
                "min": float(s["min"]) if s["min"] is not None else float("nan"),
                "25%": float(q25),
                "50%": float(q50),
                "75%": float(q75),
                "max": float(s["max"]) if s["max"] is not None else float("nan"),
            }
        return result


def describe_columns(
    batch_iter: Iterable[pa.RecordBatch],
    columns: Sequence[str],
    memory_guard: MemoryGuard,
    *,
    reservoir_size: int = _DEFAULT_RESERVOIR,
) -> dict:
    """One Reduce pass over batch_iter; returns {col: {stat: value}}.

    Empty columns list → {}. Independent of N; state bounded by
    d × reservoir_size (checked pre-init by reduce()).
    """
    if not columns:
        return {}
    op = DescribeOp(columns, reservoir_size=reservoir_size)
    return reduce(batch_iter, op, memory_guard, log=False)


__all__ = ["DescribeOp", "describe_columns"]

