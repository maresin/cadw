"""P8 fix (v0.5.1): ReduceOp.estimate_state_nbytes + MemoryGuard integration."""
import pyarrow as pa
import pytest

from cadw import MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ops.reduce import (
    reduce, ReduceOp, CountOp, SumOp, MeanOp, MinMaxOp, StdOp,
    CovarianceOp, HistogramOp,
)


def make_iter(vals, batch_size=3):
    for i in range(0, len(vals), batch_size):
        yield pa.RecordBatch.from_arrays(
            [pa.array(vals[i:i+batch_size], type=pa.float64())],
            names=["x"],
        )


# ---------- built-in ops have estimate_state_nbytes ----------

@pytest.mark.parametrize("op, expected", [
    (CountOp("x"), 16),
    (SumOp("x"), 16),
    (MeanOp("x"), 16),
    (MinMaxOp("x"), 16),
    (StdOp("x"), 24),
    (CovarianceOp(["a", "b", "c"]), 8 + 8*3 + 8*3*3),
    (HistogramOp("x", bins=10, min_val=0, max_val=1), 8*10 + 16),
])
def test_builtin_ops_have_estimate(op, expected):
    est = op.estimate_state_nbytes()
    assert est is not None
    assert est == expected


def test_covariance_estimate_matches_actual_state():
    """Estimate must be >= actual state_nbytes(init())."""
    op = CovarianceOp(["a", "b", "c", "d"])
    assert op.estimate_state_nbytes() >= op.state_nbytes(op.init())


def test_histogram_estimate_matches_actual_state():
    op = HistogramOp("x", bins=255, min_val=0, max_val=1)
    assert op.estimate_state_nbytes() >= op.state_nbytes(op.init())


# ---------- custom op with estimate override ----------

class CustomOp(ReduceOp):
    def __init__(self, d):
        self.d = d
    def init(self):
        return 0.0  # placeholder
    def update(self, state, batch):
        return state
    def merge(self, a, b):
        return a + b
    def finalize(self, state):
        return state
    def state_nbytes(self, state):
        return 8 * self.d * self.d
    def estimate_state_nbytes(self):
        return 8 * self.d * self.d


def test_custom_op_estimate_overridden():
    """Custom op with estimate_state_nbytes — no hardcode fallback."""
    op = CustomOp(d=100)  # 80 KB
    assert op.estimate_state_nbytes() == 80000


def test_reduce_uses_estimate_from_op(tmp_path):
    """reduce() should call op.estimate_state_nbytes(), not estimate_state_size."""
    # 100 MB budget; op state = 80 KB — easily fits
    guard = MemoryGuard(max_memory_mb=100)
    op = CustomOp(d=100)
    result = reduce(make_iter([1.0]*10), op, guard, log=False)
    assert result == 0.0  # placeholder


def test_reduce_custom_op_rejects_oversized_state():
    """Custom op with state > budget → MemoryLimitExceeded BEFORE op.init()."""
    # Custom op with state = 100 MB, budget = 10 MB
    class HugeOp(ReduceOp):
        def init(self):
            raise AssertionError("init() should NOT be called — check first")
        def update(self, state, batch):
            return state
        def merge(self, a, b):
            return a
        def finalize(self, state):
            return state
        def state_nbytes(self, state):
            return 100 * 1024 * 1024
        def estimate_state_nbytes(self):
            return 100 * 1024 * 1024

    guard = MemoryGuard(max_memory_mb=10)
    op = HugeOp()
    with pytest.raises(MemoryLimitExceeded):
        reduce(make_iter([1.0]), op, guard, log=False)


# ---------- fallback for op without override ----------

class LegacyOp(ReduceOp):
    """Does not override estimate_state_nbytes → falls back to hardcode."""
    def init(self):
        return 0.0
    def update(self, state, batch):
        return state
    def merge(self, a, b):
        return a
    def finalize(self, state):
        return state
    def state_nbytes(self, state):
        return 8


def test_legacy_op_falls_back_to_estimate_state_size(tmp_path):
    """Op without override uses estimate_state_size (1 MiB fallback)."""
    guard = MemoryGuard(max_memory_mb=100)  # 1 MiB > 100 MB? No, 1 < 100, OK
    op = LegacyOp()
    assert op.estimate_state_nbytes() is None
    # reduce() should use 1 MiB fallback (fits under 100 MB budget)
    reduce(make_iter([1.0]*5), op, guard, log=False)


def test_legacy_op_with_tight_budget_uses_1mib_fallback():
    """Legacy op (no override) → 1 MiB fallback → fits under 2 MB but not 0.5 MB."""
    # 500 KB budget → 1 MiB fallback > 500 KB * 0.5 = 250 KB → raise
    guard = MemoryGuard(max_memory_bytes=500_000)  # 500 KB
    op = LegacyOp()
    with pytest.raises(MemoryLimitExceeded):
        reduce(make_iter([1.0]*5), op, guard, log=False)
