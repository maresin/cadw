# cadw/ops/reduce.py
"""
Reduce: свёртка потока батчей в один аккумулятор (без shuffle, без IO).

Контракт (DECISIONS.md §5):
  - состояние не зависит от N (числа строк);
  - update(state, batch) — векторизованно, без построчных циклов;
  - merge(a, b) — ассоциативно (для параллельного reduce и HashShuffle);
  - finalize(state) — итоговое значение;
  - state_nbytes(state) — консервативная оценка для MemoryGuard.check_state.

Reduce — единственный примитив без записи на диск.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard

logger = get_logger("ops.reduce")

State = Any


class ReduceOp(ABC):
    @abstractmethod
    def init(self) -> State: ...
    @abstractmethod
    def update(self, state: State, batch: pa.RecordBatch) -> State: ...
    @abstractmethod
    def merge(self, a: State, b: State) -> State: ...
    @abstractmethod
    def finalize(self, state: State) -> Any: ...
    @abstractmethod
    def state_nbytes(self, state: State) -> int: ...

    def estimate_state_nbytes(self) -> "int | None":
        """Upper bound on state size BEFORE init() (v0.5.1, P8 fix).

        Called by `reduce()` for `MemoryGuard.check_state` *before*
        `op.init()`, so the state must be estimable without allocation.

        - Built-in ops override with known sizes (no allocation, O(1)).
        - User ops should override — otherwise `MemoryGuard.
          estimate_state_size` falls back to 1 MiB hardcode (P8 in
          _FOR_PINGIS/REPORT.md, DECISIONS §9.2.4).
        - Returning None means "use MemoryGuard fallback".

        Contract: `estimate_state_nbytes() >= state_nbytes(init())`.
        """
        return None


def reduce(
    batch_iter: Iterator[pa.RecordBatch],
    op: ReduceOp,
    memory_guard: MemoryGuard,
    *,
    log: bool = True,
) -> Any:
    """Один последовательный проход. Пик RAM ≤ batch + state + O(1)."""
    # Pre-flight: оценка размера state ДО op.init() (must-fix, §11.4).
    # CovarianceOp(d) = O(d²) — np.zeros((d,d)) до check_state → OOM для
    # d=10^5 (80 GB). HistogramOp(bins) — np.zeros(bins) — то же.
    # estimate_state_size знает про CovarianceOp (8+8d+8d²) и
    # HistogramOp (8*bins+16); для остальных — консервативный fallback.
    # v0.5.1 (P8): prefer op.estimate_state_nbytes() when overridden;
    # fall back to MemoryGuard.estimate_state_size for unknown ops.
    _estimated = op.estimate_state_nbytes()
    if _estimated is None:
        _estimated = memory_guard.estimate_state_size(
            type(op).__name__,
            schema=None,
            d=getattr(op, "d", None),
            bins=getattr(op, "bins", 255),
        )
    memory_guard.check_state(_estimated, name=f"{type(op).__name__}.estimated")

    state = op.init()
    batch_id = 0
    for batch in batch_iter:
        memory_guard.check_batch(batch, name=f"{type(op).__name__}.batch")
        state = op.update(state, batch)
        memory_guard.check_state(op.state_nbytes(state), name=type(op).__name__)
        if log:
            memory_guard.log_step(
                "reduce",
                batch_id=batch_id,
                n_rows=batch.num_rows,
                batch_bytes=batch.nbytes,
                state_bytes=op.state_nbytes(state),
            )
        batch_id += 1
    return op.finalize(state)


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

class CountOp(ReduceOp):

    def estimate_state_nbytes(self) -> int:
        return 16
    def __init__(self, column: str) -> None:
        self.column = column

    def init(self) -> State:
        return (0, 0)

    def update(self, state: State, batch: pa.RecordBatch) -> State:
        arr = batch.column(self.column)
        n = len(arr)
        return (state[0] + (n - arr.null_count), state[1] + arr.null_count)

    def merge(self, a: State, b: State) -> State:
        return (a[0] + b[0], a[1] + b[1])

    def finalize(self, state: State) -> int:
        return state[0]

    def state_nbytes(self, state: State) -> int:
        return 16


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------

class SumOp(ReduceOp):

    def estimate_state_nbytes(self) -> int:
        return 16
    """Сумма. Состояние — float64 (прототип; overflow int64 не обрабатывается)."""

    def __init__(self, column: str) -> None:
        self.column = column

    def init(self) -> State:
        return (0.0, 0)

    def update(self, state: State, batch: pa.RecordBatch) -> State:
        arr = batch.column(self.column)
        n_valid = len(arr) - arr.null_count
        if n_valid == 0:
            return state
        s = pc.sum(arr, skip_nulls=True).as_py()
        return (state[0] + float(s if s is not None else 0.0), state[1] + n_valid)

    def merge(self, a: State, b: State) -> State:
        return (a[0] + b[0], a[1] + b[1])

    def finalize(self, state: State) -> Any:
        return state[0] if state[1] > 0 else None

    def state_nbytes(self, state: State) -> int:
        return 16


# ---------------------------------------------------------------------------
# min / max
# ---------------------------------------------------------------------------

class MinMaxOp(ReduceOp):

    def estimate_state_nbytes(self) -> int:
        return 16
    def __init__(self, column: str, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.column = column
        self.mode = mode

    def init(self) -> State:
        return (None, False)

    def update(self, state: State, batch: pa.RecordBatch) -> State:
        arr = batch.column(self.column)
        n = len(arr)
        if n == 0 or arr.null_count == n:
            return state
        mm = pc.min_max(arr)
        v = mm["min"].as_py() if self.mode == "min" else mm["max"].as_py()
        if v is None:
            return state
        cur, valid = state
        if not valid:
            return (v, True)
        if self.mode == "min":
            return (v if v < cur else cur, True)
        return (v if v > cur else cur, True)

    def merge(self, a: State, b: State) -> State:
        if not a[1]:
            return b
        if not b[1]:
            return a
        if self.mode == "min":
            return (a[0] if a[0] <= b[0] else b[0], True)
        return (a[0] if a[0] >= b[0] else b[0], True)

    def finalize(self, state: State) -> Any:
        return state[0] if state[1] else None

    def state_nbytes(self, state: State) -> int:
        return 16


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------

class MeanOp(ReduceOp):

    def estimate_state_nbytes(self) -> int:
        return 16
    def __init__(self, column: str) -> None:
        self.column = column

    def init(self) -> State:
        return (0.0, 0)

    def update(self, state: State, batch: pa.RecordBatch) -> State:
        arr = batch.column(self.column)
        n_valid = len(arr) - arr.null_count
        if n_valid == 0:
            return state
        s = pc.sum(arr, skip_nulls=True).as_py()
        return (state[0] + float(s if s is not None else 0.0), state[1] + n_valid)

    def merge(self, a: State, b: State) -> State:
        return (a[0] + b[0], a[1] + b[1])

    def finalize(self, state: State) -> float:
        return state[0] / state[1] if state[1] > 0 else float("nan")

    def state_nbytes(self, state: State) -> int:
        return 16


# ---------------------------------------------------------------------------
# var / std (Welford)
# ---------------------------------------------------------------------------

class StdOp(ReduceOp):

    def estimate_state_nbytes(self) -> int:
        return 24
    """Параллельный Welford (Chan et al.). Численно устойчив к большому сдвигу."""

    def __init__(self, column: str, ddof: int = 1) -> None:
        self.column = column
        self.ddof = ddof

    def init(self) -> State:
        return (0, 0.0, 0.0)

    def update(self, state: State, batch: pa.RecordBatch) -> State:
        arr = batch.column(self.column)
        n_b = len(arr) - arr.null_count
        if n_b == 0:
            return state
        mean_b = pc.mean(arr, skip_nulls=True).as_py()
        diff = pc.subtract(arr, pa.scalar(mean_b, type=arr.type))
        M2_b = pc.sum(pc.multiply(diff, diff), skip_nulls=True).as_py() or 0.0

        n_a, mean_a, M2_a = state
        n = n_a + n_b
        delta = mean_b - mean_a
        mean = mean_a + delta * (n_b / n)
        M2 = M2_a + M2_b + delta * delta * (n_a * n_b / n)
        return (n, mean, M2)

    def merge(self, a: State, b: State) -> State:
        n_a, mean_a, M2_a = a
        n_b, mean_b, M2_b = b
        n = n_a + n_b
        if n == 0:
            return (0, 0.0, 0.0)
        delta = mean_b - mean_a
        mean = mean_a + delta * (n_b / n)
        M2 = M2_a + M2_b + delta * delta * (n_a * n_b / n)
        return (n, mean, M2)

    def finalize(self, state: State) -> float:
        n, _, M2 = state
        if n - self.ddof <= 0:
            return float("nan")
        return float(np.sqrt(M2 / (n - self.ddof)))

    def state_nbytes(self, state: State) -> int:
        return 24


# ---------------------------------------------------------------------------
# covariance / correlation (матрица d×d)
# ---------------------------------------------------------------------------

class CovarianceOp(ReduceOp):

    def estimate_state_nbytes(self) -> int:
        return 8 + 8 * self.d + 8 * self.d * self.d
    """Ковариационная матрица d×d. Состояние O(d²) — см. DECISIONS.md §5.4."""

    def __init__(self, columns: list[str]) -> None:
        if not columns:
            raise ValueError("Нужна хотя бы одна колонка")
        self.columns = list(columns)
        self.d = len(self.columns)

    def init(self) -> State:
        return (
            0,
            np.zeros(self.d, dtype=np.float64),
            np.zeros((self.d, self.d), dtype=np.float64),
        )

    def update(self, state: State, batch: pa.RecordBatch) -> State:
        X = np.column_stack([
            batch.column(c).to_numpy(zero_copy_only=False) for c in self.columns
        ]).astype(np.float64, copy=False)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        n_b = X.shape[0]
        if n_b == 0:
            return state
        n_a, sum_x_a, sum_xx_a = state
        return (n_a + n_b, sum_x_a + X.sum(axis=0), sum_xx_a + X.T @ X)

    def merge(self, a: State, b: State) -> State:
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def finalize(self, state: State) -> np.ndarray:
        n, sum_x, sum_xx = state
        if n < 2:
            return np.full((self.d, self.d), np.nan)
        mean = sum_x / n
        return (sum_xx - n * np.outer(mean, mean)) / (n - 1)

    def state_nbytes(self, state: State) -> int:
        return 8 + 8 * self.d + 8 * self.d * self.d


# ---------------------------------------------------------------------------
# histogram (binning для LightGBM)
# ---------------------------------------------------------------------------

class HistogramOp(ReduceOp):

    def estimate_state_nbytes(self) -> int:
        return 8 * self.bins + 16
    """Гистограмма. Требует фиксированных min_val/max_val (двухпроходный режим)."""

    def __init__(self, column: str, bins: int = 255,
                 min_val: float | None = None, max_val: float | None = None) -> None:
        if min_val is None or max_val is None:
            raise ValueError("Нужны min_val и max_val (двухпроходный режим)")
        if max_val <= min_val:
            raise ValueError("max_val должен быть > min_val")
        self.column = column
        self.bins = int(bins)
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def init(self) -> State:
        return np.zeros(self.bins, dtype=np.int64)

    def update(self, state: State, batch: pa.RecordBatch) -> State:
        arr = batch.column(self.column).to_numpy(zero_copy_only=False)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return state
        counts, _ = np.histogram(arr, bins=self.bins, range=(self.min_val, self.max_val))
        return state + counts

    def merge(self, a: State, b: State) -> State:
        return a + b

    def finalize(self, state: State) -> np.ndarray:
        return state

    def state_nbytes(self, state: State) -> int:
        return self.bins * 8