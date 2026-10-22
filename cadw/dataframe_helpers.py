"""Pure helpers extracted from cadw/dataframe.py (v0.9.25f-a).

No CadwDataFrame dependency (pure module invariant — see
test_v0925f_a.py::test_helpers_module_no_cadwdataframe_dep).

Imported + re-exported by cadw.dataframe for backward compat:
`from cadw.dataframe import _RollState` still works.

Contents (10 items, 433 L total):
  - _ArrowTail (45 L): Arrow-native sliding tail buffer (v0.9.18).
  - _RollState (85 L): incremental rolling state (v0.6.2c).
  - _ExpandingState (65 L): cumulative window state (v0.8.0).
  - _EwmState (48 L): EWMA state (v0.8.0).
  - _group_boundaries_arrow (41 L): Arrow-native group boundaries
    (v0.9.21, §1.4).
  - _convert_batch (31 L): format='arrow'|'pandas'|'numpy'|'torch'.
  - _concat_batches (13 L): per-column concat (v0.9.10).
  - _empty_value_counts_table (13 L): empty schema (v0.9.23g-fix2).
  - _reservoir_positions (41 L): Algorithm R reservoir.
  - _weighted_reservoir_positions (51 L): Efraimidis-Spirakis A-Res.

This module can be imported independently (no cadw.dataframe dep);
useful for tests that want just the state classes without the full
facade.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa


class _ArrowTail:
    """v0.9.18 (AUDIT §1.1): Arrow-native sliding tail buffer of k rows.

    Replaces `deque(maxlen=k)` of Python scalars + per-batch
    `col_vals.to_pylist()` with `deque` of Arrow slices. Peak
    memory: one small Arrow array of length <= k per column,
    not `batch_size * python_obj_size`.

    Correctness: preserves Arrow types and nulls exactly.
    Cross-batch identity: same Arrow buffer passed to push()
    may be held (via slice ref) until next push; if batch buffer
    pressure matters, call sites should copy after yield. For
    shift semantics, holding one prior batch alive is acceptable
    (was already the case with deque of Python values, which held
    full batches as Python objects).
    """

    __slots__ = ("_k", "_tail")

    def __init__(self, k: int) -> None:
        self._k = k
        self._tail = None  # pa.Array of length <= k, or None

    def push(self, arr) -> None:
        """Append Arrow array; trim to last k values."""
        n = len(arr)
        if n == 0:
            return
        if self._tail is None:
            combined = arr
        else:
            combined = pa.concat_arrays([self._tail, arr])
        if len(combined) > self._k:
            combined = combined.slice(len(combined) - self._k)
        # Force a small copy so we do not retain parent batch buffer
        if len(combined) > 0:
            # pa.array(combined.to_pylist(), type=...) is O(k) — tiny;
            # forces fresh buffer, releases parent.
            self._tail = pa.array(combined.to_pylist(), type=combined.type)
        else:
            self._tail = None

    def get(self):
        """Return current tail pa.Array of length <= k, or None."""
        return self._tail


class _RollState:
    """Incremental rolling aggregation state (v0.6.2c).

    - sum/mean/count: running accumulators (O(1) push).
    - min/max: monotonic deque (O(1) amortized push).

    Nulls are skipped (pandas default). Count = non-null in window.
    """

    __slots__ = (
        "_window", "_min_periods", "_agg", "_buf", "_idx",
        "_sum", "_count", "_dq",
    )

    def __init__(self, window: int, min_periods: int, agg: str) -> None:
        from collections import deque as _dq_mod
        self._window = window
        self._min_periods = min_periods
        self._agg = agg
        self._buf = _dq_mod()  # (idx, is_null, value)
        self._idx = 0
        self._sum = 0.0
        self._count = 0  # non-null in current window
        self._dq = _dq_mod()  # for min/max: (idx, value)

    def push(self, v):
        window = self._window
        i = self._idx
        # 1. Add current
        is_null = v is None
        if not is_null:
            self._count += 1
            if self._agg in ("sum", "mean"):
                self._sum += float(v)
        self._buf.append((i, is_null, v))
        # v0.8.4-fix (S1): null + non-empty _dq → `_dq[-1][1] >= None`
        # raises TypeError. Nulls skip monotone-deque maintenance entirely
        # (they don't contribute to min/max; window slot still consumed).
        if self._agg == "min":
            if not is_null:
                while self._dq and self._dq[-1][1] >= v:
                    self._dq.pop()
                self._dq.append((i, v))
        elif self._agg == "max":
            if not is_null:
                while self._dq and self._dq[-1][1] <= v:
                    self._dq.pop()
                self._dq.append((i, v))

        # 2. Evict if len > window
        while len(self._buf) > window:
            old_i, old_null, old_v = self._buf.popleft()
            if not old_null:
                self._count -= 1
                if self._agg in ("sum", "mean"):
                    self._sum -= float(old_v)
            if self._agg in ("min", "max") and self._dq and self._dq[0][0] == old_i:
                self._dq.popleft()

        self._idx += 1

        # 3. Compute result
        if self._count < self._min_periods:
            return None
        if self._agg == "count":
            return self._count
        if self._agg == "sum":
            return self._sum
        if self._agg == "mean":
            return self._sum / self._count
        if self._agg in ("min", "max"):
            return self._dq[0][1] if self._dq else None
        raise RuntimeError(f"_RollState: unknown agg {self._agg!r}")

    def reset(self) -> None:
        """Reset state for per-group rolling (v0.8.10).

        Called on group change. Clears deque/history/accumulators; next
        push starts fresh (min_periods applies from group start).
        """
        self._buf.clear()
        self._idx = 0
        self._sum = 0.0
        self._count = 0
        self._dq.clear()


class _ExpandingState:
    """Cumulative (growing) window aggregation state (v0.8.0).

    O(1) per row. Unlike rolling, no deque needed:
    - sum/mean/count: scalar accumulators.
    - min/max: current min/max (monotone by construction).

    Nulls: skip (count/sum/min/max not updated). Output for a null row is
    null — the input row stays null-visible (unlike rolling, where a null
    row with min_periods=1 may inherit prior output).
    """

    __slots__ = ("_agg", "_min_periods", "_count", "_sum",
                 "_prod", "_min", "_max")

    def __init__(self, agg: str, min_periods: int) -> None:
        self._agg = agg
        self._min_periods = min_periods
        self._count = 0
        self._sum = 0.0
        self._prod = 1.0  # v0.9.23e: cumprod
        self._min = None
        self._max = None

    def push(self, v):
        if v is None:
            return None
        self._count += 1
        x = float(v)
        if self._agg == "count":
            result = self._count
        elif self._agg == "sum":
            self._sum += x
            result = self._sum
        elif self._agg == "prod":  # v0.9.23e
            self._prod *= x
            result = self._prod
        elif self._agg == "mean":
            self._sum += x
            result = self._sum / self._count
        elif self._agg == "min":
            self._min = x if self._min is None else min(self._min, x)
            result = self._min
        elif self._agg == "max":
            self._max = x if self._max is None else max(self._max, x)
            result = self._max
        else:
            raise RuntimeError(f"_ExpandingState: unknown agg {self._agg!r}")
        if self._count < self._min_periods:
            return None
        return result

    def reset(self) -> None:
        """v0.9.23e-fix1: reset cumulative state (used by
        _per_group_cumulative on group change).

        Resets _count/_sum/_prod/_min/_max to initial values.
        Keeps _agg and _min_periods unchanged (config, not state).
        Mirrors _RollState.reset() (v0.8.10).
        """
        self._count = 0
        self._sum = 0.0
        self._prod = 1.0
        self._min = None
        self._max = None


class _EwmState:
    """Exponentially weighted mean state (v0.8.0). O(1) per row.

    adjust=True:  s_num = (1-α)·s_num + x_t,  s_den = (1-α)·s_den + 1
                  y_t = s_num / s_den. Init s_num=x_0, s_den=1.
    adjust=False: y_t = (1-α)·y_{t-1} + α·x_t. Init y_0 = x_0.

    Nulls: skip (state frozen). Output for null row is null.
    min_periods counts non-null rows.
    """

    __slots__ = ("_alpha", "_adjust", "_min_periods", "_count",
                 "_s_num", "_s_den", "_ewm")

    def __init__(self, alpha: float, adjust: bool, min_periods: int) -> None:
        self._alpha = alpha
        self._adjust = adjust
        self._min_periods = min_periods
        self._count = 0
        self._s_num = 0.0
        self._s_den = 0.0
        self._ewm = None

    def push(self, v):
        if v is None:
            return None
        self._count += 1
        x = float(v)
        if self._adjust:
            if self._count == 1:
                self._s_num = x
                self._s_den = 1.0
            else:
                self._s_num = (1.0 - self._alpha) * self._s_num + x
                self._s_den = (1.0 - self._alpha) * self._s_den + 1.0
            result = self._s_num / self._s_den
        else:
            if self._count == 1:
                self._ewm = x
            else:
                self._ewm = (
                    (1.0 - self._alpha) * self._ewm
                    + self._alpha * x
                )
            result = self._ewm
        if self._count < self._min_periods:
            return None
        return result


def _group_boundaries_arrow(batch, group_cols):
    """v0.9.21 (§1.4): Arrow-native group boundary detection.

    Returns list [0, b1, b2, ..., n] where boundaries[k] is the
    row index at which group k starts (0 = first, n = past end).

    Assumes input is sorted by group_cols (caller contract; group
    boundaries = value changes between consecutive rows).

    Peak: O(n) transient bool mask per group col. No n × K_cols
    Python objects (prior `to_pylist()` + per-row tuple alloc).

    Null semantics: null==null → same group (SQL-style; matches
    Python tuple compare used before v0.9.21). NULL vs non-null
    → different group (Arrow Kleene equal returns null → treated
    as different via fill_null(False)).
    """
    import pyarrow.compute as _pc
    n = batch.num_rows
    if n <= 1:
        return [0, n] if n > 0 else [0]
    same = None
    for gc in group_cols:
        col = batch.column(gc)
        prev_slice = col.slice(0, n - 1)
        curr_slice = col.slice(1)
        eq = _pc.equal(curr_slice, prev_slice)
        both_null = _pc.and_(
            _pc.is_null(curr_slice), _pc.is_null(prev_slice)
        )
        col_same = _pc.or_(
            _pc.fill_null(eq, pa.scalar(False, type=pa.bool_())),
            both_null,
        )
        if same is None:
            same = col_same
        else:
            same = _pc.and_(same, col_same)
    same_np = same.to_numpy(zero_copy_only=False)
    changes = np.flatnonzero(~same_np) + 1
    return [0] + changes.tolist() + [n]


def _convert_batch(batch: pa.RecordBatch, format: str):
    """Конвертация одного батча. Векторизованно, без .as_py()."""
    if format == "arrow":
        return batch
    if format == "pandas":
        # v0.9.6e1: pandas is optional. Arrow's to_pandas() raises
        # ModuleNotFoundError without pandas; convert to a clear
        # install-hint error.
        try:
            return batch.to_pandas()
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(
                "to_batches(format='pandas') requires pandas. "
                "Install with: uv add 'cadw-framework[pandas]' "
                "(or: pip install pandas). For pandas-free usage, "
                "use format='arrow' | 'numpy' | 'torch'."
            ) from e
    if format == "numpy":
        return {
            name: batch.column(i).to_numpy(zero_copy_only=False)
            for i, name in enumerate(batch.schema.names)
        }
    if format == "torch":
        import torch  # ленивый импорт
        return {
            name: torch.from_numpy(
                batch.column(i).to_numpy(zero_copy_only=False)
            )
            for i, name in enumerate(batch.schema.names)
        }
    raise ValueError(f"unsupported format {format!r}")


def _concat_batches(batches: list[pa.RecordBatch]) -> pa.RecordBatch:
    if len(batches) == 1:
        return batches[0]
    # v0.9.10 (U2): per-column concat_arrays (1x alloc per col)
    # vs from_batches + combine_chunks + to_batches (3x alloc).
    # Bounded by batch_size (10k rows in _iter_arrow_batches).
    arrays = [
        pa.concat_arrays([b.column(i) for b in batches])
        for i in range(batches[0].num_columns)
    ]
    return pa.RecordBatch.from_arrays(
        arrays, names=batches[0].schema.names,
    )


def _empty_value_counts_table(column, col_type, *, normalize: bool):
    """Empty pa.Table with value_counts output schema (0 rows).

    Schema: [column (col_type), "count" (int64)] or
    [column (col_type), "fraction" (float64)].
    """
    import pyarrow as _pa

    second_name = "fraction" if normalize else "count"
    second_type = _pa.float64() if normalize else _pa.int64()
    return _pa.Table.from_pylist([], schema=_pa.schema([
        (column, col_type), (second_name, second_type),
    ]))


def _reservoir_positions(batches, n, N, rng, memory_guard):
    """Algorithm R reservoir sampling, vectorized per batch.

    Returns list of n positions in [0, N) without replacement.
    Positions are filtered-view 0-based (match iloc).

    Algorithm: reservoir = [0, 1, ..., n-1]. For each row at position
    i >= n, with probability n/(i+1) replace random reservoir slot
    with i. Vectorized per batch: generate all random decisions at
    once, only per-replacement Python loop (~n*log(N/n) iterations).
    """
    import numpy as np

    positions = list(range(n))
    if N <= n:
        return positions

    start = 0
    for b in batches:
        if memory_guard is not None and b.num_rows > 0:
            memory_guard.check_batch(b, name="sample.reservoir_input")
        m = b.num_rows
        end = start + m
        # Skip rows [0, n) — always in reservoir.
        lo = max(start, n)
        if lo >= end:
            start = end
            continue
        row_indices = np.arange(lo, end, dtype=np.int64)
        u = rng.random(len(row_indices))
        # Probability for row i (0-indexed) = n / (i+1)
        probs = n / (row_indices + 1.0)
        replace_mask = u < probs
        replace_rows = row_indices[replace_mask]
        if len(replace_rows) > 0:
            rand_slots = rng.integers(0, n, size=len(replace_rows))
            for slot, row_pos in zip(rand_slots.tolist(),
                                     replace_rows.tolist()):
                positions[slot] = row_pos
        start = end
    return positions


def _weighted_reservoir_positions(
    batches, n, weights_col, rng, memory_guard,
):
    """Efraimidis-Spirakis A-Res weighted sampling without replacement.

    For each row with weight w > 0, compute key = u^(1/w), u~U(0,1).
    Keep top-n keys by value (min-heap of size n). Null / non-positive
    weights are skipped. Peak O(n), time O(N log n).

    Returns positions in filtered-view 0-based order.
    """
    import heapq
    import numpy as np

    heap: list = []  # min-heap of (key, position)
    pos_offset = 0
    for b in batches:
        if memory_guard is not None and b.num_rows > 0:
            memory_guard.check_batch(
                b, name="sample.weighted_input", extra_factor=3.5,
            )
        w_col = b.column(weights_col)
        w_np = w_col.to_numpy(zero_copy_only=False)
        m = len(w_np)
        if m == 0:
            continue
        # Valid = weight > 0 and not NaN (for float dtype).
        if w_np.dtype.kind == "f":
            valid = (w_np > 0) & ~np.isnan(w_np)
        else:
            valid = w_np > 0
        idx = np.where(valid)[0]
        if len(idx) == 0:
            pos_offset += m
            continue
        # Vectorized key generation for valid rows.
        u = rng.random(len(idx))
        keys = u ** (1.0 / w_np[idx])
        for i_local, k in zip(idx.tolist(), keys.tolist()):
            pos = pos_offset + int(i_local)
            if len(heap) < n:
                heapq.heappush(heap, (float(k), pos))
            elif float(k) > heap[0][0]:
                heapq.heapreplace(heap, (float(k), pos))
        pos_offset += m
    if not heap:
        raise ValueError(
            "sample: no valid weights (all null or <= 0); cannot "
            "draw a weighted sample"
        )
    return [pos for _, pos in heap]
