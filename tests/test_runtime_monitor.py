"""v0.9.19: closed-loop memory measurement tests."""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded


# ============ defaults: no-op ============


def test_defaults_off():
    """Both flags default False → zero behavior change."""
    g = MemoryGuard(max_memory_mb=10)
    assert g._runtime_monitor is False
    assert g._calibrate is False
    # Lazy baselines not initialized
    assert g._pool is None
    assert g._process is None


def test_calibrate_off_yields_only():
    """calibrate=False + runtime_monitor=False → no measurement."""
    g = MemoryGuard(max_memory_mb=10)
    with g.calibrate("test_op", 1000):
        pass
    assert g.calibration_samples == {}
    assert g._pool is None  # never touched


def test_basic_paths_still_work_when_off():
    """Existing check_batch/check_state unchanged."""
    g = MemoryGuard(max_memory_mb=10)
    batch = pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], names=["x"])
    g.check_batch(batch)
    g.check_state(16)


# ============ calibrate: sample recording ============


def test_calibrate_records_sample():
    """calibrate=True → sample appended per call."""
    g = MemoryGuard(max_memory_mb=10, calibrate=True)
    with g.calibrate("test_op", 5000):
        # Allocate some memory to produce non-zero delta
        x = [0] * 10_000
        del x
    samples = g.calibration_samples
    assert "test_op" in samples
    assert len(samples["test_op"]) == 1
    s = samples["test_op"][0]
    assert s["predicted"] == 5000
    # v0.9.20-fix3 (§13.4/235): entry has peak_pool/peak_rss (deltas
    # above op-start, comparable to formula predicted). v0.9.19 fields
    # ("arrow"/"rss") are stale.
    assert "peak_pool" in s and "peak_rss" in s
    assert "net_pool_delta" in s and "net_rss_delta" in s
    assert s["sample_mode"] == "delta"


def test_calibrate_accumulates_samples():
    """Multiple calibrate() calls → list of samples."""
    g = MemoryGuard(max_memory_mb=10, calibrate=True)
    for predicted in (100, 200, 300):
        with g.calibrate("op", predicted):
            pass
    samples = g.calibration_samples["op"]
    assert len(samples) == 3
    assert [s["predicted"] for s in samples] == [100, 200, 300]


def test_calibrate_records_unknown_predicted():
    """predicted=0 → sample recorded, no drift warning."""
    g = MemoryGuard(max_memory_mb=10, calibrate=True)
    with g.calibrate("op_no_predicted", 0):
        pass
    samples = g.calibration_samples["op_no_predicted"]
    assert len(samples) == 1
    assert samples[0]["predicted"] == 0


def test_reset_calibration():
    """reset_calibration clears accumulated samples."""
    g = MemoryGuard(max_memory_mb=10, calibrate=True)
    with g.calibrate("op", 100):
        pass
    assert len(g.calibration_samples["op"]) == 1
    g.reset_calibration()
    assert g.calibration_samples == {}


def test_calibration_samples_is_copy():
    """Mutating returned dict doesn't affect internal state."""
    g = MemoryGuard(max_memory_mb=10, calibrate=True)
    with g.calibrate("op", 100):
        pass
    samples = g.calibration_samples
    samples["op"].clear()
    assert len(g.calibration_samples["op"]) == 1  # unchanged


# ============ calibrate: drift warning ============


def test_calibrate_drift_warns(caplog):
    """actual/predicted > 2.0 → WARNING logged."""
    caplog.set_level(logging.WARNING, logger="cadw.memory_guard")
    g = MemoryGuard(max_memory_mb=10, calibrate=True)
    # Force huge delta by monkeypatching _get_pool_delta
    pool_deltas = [0, 100_000_000]  # 0 on entry, 100 MB on exit
    call_idx = [0]
    def fake_pool_delta():
        v = pool_deltas[min(call_idx[0], len(pool_deltas) - 1)]
        call_idx[0] += 1
        return v
    g._get_pool_delta = fake_pool_delta
    # Stub pool with controllable bytes_allocated
    pool_values = [0, 100_000_000]
    pidx = [0]
    class FakePool:
        def bytes_allocated(self):
            v = pool_values[min(pidx[0], len(pool_values) - 1)]
            pidx[0] += 1
            return v
    g._pool = FakePool()
    g._pool_baseline = 0
    # Stub process
    class FakeProcess:
        def memory_info(self):
            return MagicMock(rss=0)
    g._process = FakeProcess()
    g._rss_baseline = 0
    with g.calibrate("big_op", 1000):  # predicted 1 KB, actual 100 MB
        pass
    warnings = [r for r in caplog.records if "DRIFT" in r.message]
    assert warnings, f"expected drift warning; got {caplog.records}"


# ============ runtime_monitor: RSS hard raise ============


def test_runtime_monitor_raises_on_hard_rss_drift():
    """RSS delta > rss_hard_factor × budget → MemoryLimitExceeded."""
    g = MemoryGuard(max_memory_mb=1, runtime_monitor=True)  # 1 MB budget
    # Fake pool + process: entry 0, exit 100 MB (>> 1.8 MB hard limit)
    pool_values = [0, 0]  # pool delta 0 (irrelevant for hard check)
    pidx = [0]
    class FakePool:
        def bytes_allocated(self):
            v = pool_values[min(pidx[0], len(pool_values) - 1)]
            pidx[0] += 1
            return v
    g._pool = FakePool()
    g._pool_baseline = 0
    rss_values = [0, 100_000_000]
    ridx = [0]
    class FakeProcess:
        def memory_info(self):
            v = rss_values[min(ridx[0], len(rss_values) - 1)]
            ridx[0] += 1
            return MagicMock(rss=v)
    g._process = FakeProcess()
    g._rss_baseline = 0
    with pytest.raises(MemoryLimitExceeded, match="RSS delta"):
        with g.calibrate("huge_op", 0):
            pass


def test_runtime_monitor_soft_warn_no_raise(caplog):
    """RSS delta between soft and hard → warning, no raise."""
    caplog.set_level(logging.WARNING, logger="cadw.memory_guard")
    g = MemoryGuard(max_memory_mb=10, runtime_monitor=True)  # 10 MB
    # 15 MB delta: > 1.2×10=12 MB (soft), < 1.8×10=18 MB (hard)
    pool_values = [0, 0]
    pidx = [0]
    class FakePool:
        def bytes_allocated(self):
            v = pool_values[min(pidx[0], len(pool_values) - 1)]
            pidx[0] += 1
            return v
    g._pool = FakePool()
    g._pool_baseline = 0
    rss_values = [0, 15_000_000]
    ridx = [0]
    class FakeProcess:
        def memory_info(self):
            v = rss_values[min(ridx[0], len(rss_values) - 1)]
            ridx[0] += 1
            return MagicMock(rss=v)
    g._process = FakeProcess()
    g._rss_baseline = 0
    with g.calibrate("medium_op", 0):
        pass
    soft_warnings = [
        r for r in caplog.records if "soft-drift" in r.message
    ]
    assert soft_warnings, f"expected soft warning; got {caplog.records}"


def test_runtime_monitor_no_drift_no_raise():
    """RSS delta < soft → no raise, no warning."""
    g = MemoryGuard(max_memory_mb=100, runtime_monitor=True)
    pool_values = [0, 0]
    pidx = [0]
    class FakePool:
        def bytes_allocated(self):
            v = pool_values[min(pidx[0], len(pool_values) - 1)]
            pidx[0] += 1
            return v
    g._pool = FakePool()
    g._pool_baseline = 0
    rss_values = [0, 1_000_000]  # 1 MB delta < 120 MB soft limit
    ridx = [0]
    class FakeProcess:
        def memory_info(self):
            v = rss_values[min(ridx[0], len(rss_values) - 1)]
            ridx[0] += 1
            return MagicMock(rss=v)
    g._process = FakeProcess()
    g._rss_baseline = 0
    with g.calibrate("small_op", 0):
        pass  # no raise


# ============ reduce() integration ============


def test_reduce_calibration_integration():
    """reduce() with calibrate=True records a sample for the op."""
    from cadw.ops.reduce import CountOp, reduce
    batches = [
        pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], names=["x"]),
    ]
    g = MemoryGuard(max_memory_mb=10, calibrate=True)
    result = reduce(iter(batches), CountOp("x"), g, log=False)
    assert result == 3
    samples = g.calibration_samples
    assert "CountOp" in samples
    assert len(samples["CountOp"]) == 1


def test_reduce_no_calibration_by_default():
    """reduce() with calibrate=False → no samples."""
    from cadw.ops.reduce import CountOp, reduce
    batches = [
        pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], names=["x"]),
    ]
    g = MemoryGuard(max_memory_mb=10)  # no flags
    result = reduce(iter(batches), CountOp("x"), g, log=False)
    assert result == 3
    assert g.calibration_samples == {}


def test_reduce_result_correct_with_calibration():
    """Calibration wrapper does not change reduce() result."""
    from cadw.ops.reduce import MeanOp, reduce
    batches = [
        pa.RecordBatch.from_arrays(
            [pa.array([1.0, 2.0, 3.0], type=pa.float64())],
            names=["x"],
        ),
    ]
    g = MemoryGuard(max_memory_mb=10, calibrate=True, runtime_monitor=True)
    result = reduce(iter(batches), MeanOp("x"), g, log=False)
    assert result == pytest.approx(2.0)


# ============ AST guards ============


def test_calibrate_in_memory_guard_module():
    """AST: MemoryGuard has calibrate, _record_calibration, etc."""
    from cadw.core import memory_guard as mg_mod
    src = __import__("inspect").getsource(mg_mod)
    for name in (
        "def calibrate(",
        "def _record_calibration(",
        "def _get_pool_delta(",
        "def _get_rss_delta(",
        "def reset_calibration(",
        "def calibration_samples(",
    ):
        assert name in src, f"missing {name}"


def test_reduce_wrapped_in_calibrate():
    """AST: reduce() body is inside `with memory_guard.calibrate(...)`."""
    import ast as _ast
    import inspect as _inspect
    import textwrap as _tw
    from cadw.ops.reduce import reduce
    src = _tw.dedent(_inspect.getsource(reduce))
    tree = _ast.parse(src)
    has_with_calibrate = False
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.With):
            continue
        for item in node.items:
            ctx = item.context_expr
            if (isinstance(ctx, _ast.Call)
                    and isinstance(ctx.func, _ast.Attribute)
                    and ctx.func.attr == "calibrate"):
                has_with_calibrate = True
                break
        if has_with_calibrate:
            break
    assert has_with_calibrate, (
        "reduce() must wrap body in memory_guard.calibrate(...)"
    )


# ============ bench script: compile regression (§13.4/230) ============


def test_bench_v0919_calibration_compiles():
    """v0.9.19-fix1: bench script must always be syntactically valid.

    Initial v0.9.19 broke this by splitting an f-string literal across
    two write_content() list elements without a closing quote on the
    first (§13.4/230: string-list authoring — each element must be a
    complete syntactic unit for f-strings). assert_compiles() in the
    patch script caught it, but a compilation guard in test suite
    prevents reintroduction via manual edits.
    """
    import py_compile
    from pathlib import Path as _P
    bench = (
        _P(__file__).resolve().parent.parent
        / "scripts" / "bench_v0919_calibration.py"
    )
    assert bench.exists(), f"bench script missing: {bench}"
    # doraise=True → raises PyCompileError on syntax errors
    py_compile.compile(str(bench), doraise=True)


def test_bench_v0919_calibration_has_main():
    """Sanity: bench has a `def main():` and `if __name__` guard."""
    from pathlib import Path as _P
    bench = (
        _P(__file__).resolve().parent.parent
        / "scripts" / "bench_v0919_calibration.py"
    )
    src = bench.read_text()
    assert "def main():" in src
    assert 'if __name__ == "__main__":' in src


# ============ v0.9.19-fix2: calibrate prologue contract ============


def test_calibrate_does_not_call_get_delta():
    """calibrate() must NOT call _get_pool_delta/_get_rss_delta.

    §13.4/231: those are caller-facing measurement APIs that read
    from self._pool / self._process on every call. calibrate() must
    do direct lazy init instead (only when baselines are None).
    Otherwise it consumes 1 extra read from each start/end measurement
    → delta collapses to 0 (root cause of v0.9.19-fix1 pytest failure).
    """
    g = MemoryGuard(max_memory_mb=10, calibrate=True)
    calls = {"pool": 0, "rss": 0}
    orig_pool = g._get_pool_delta
    orig_rss = g._get_rss_delta
    def spy_pool():
        calls["pool"] += 1
        return orig_pool()
    def spy_rss():
        calls["rss"] += 1
        return orig_rss()
    g._get_pool_delta = spy_pool
    g._get_rss_delta = spy_rss
    # Stub pool/process: no real reads, controlled return values.
    class _FakePool:
        def bytes_allocated(self):
            return 0
    class _FakeProcess:
        def memory_info(self):
            return MagicMock(rss=0)
    g._pool = _FakePool()
    g._pool_baseline = 0
    g._process = _FakeProcess()
    g._rss_baseline = 0
    with g.calibrate("op", 100):
        pass
    assert calls["pool"] == 0, (
        f"calibrate() called _get_pool_delta {calls['pool']} times "
        f"(expected 0; would consume fake reads and collapse delta)"
    )
    assert calls["rss"] == 0, (
        f"calibrate() called _get_rss_delta {calls['rss']} times "
        f"(expected 0; would consume fake reads and collapse delta)"
    )


# ============ v0.9.20-fix1: bench decomposition ============


def test_bench_has_decomposed_columns():
    """v0.9.20-fix1: bench prints pred_batch/pred_state/C_batch columns.

    User feedback: 'why only batch size, no aux object sizes?'
    Fix decomposes reduce (batch + state), wires sort/shuffle/groupby
    models for pred_total, and adds C_batch = peak/batch (N× batch
    memory transparency for ops with significant non-batch state).
    """
    from pathlib import Path as _P
    bench = (
        _P(__file__).resolve().parent.parent
        / "scripts" / "bench_v0919_calibration.py"
    )
    src = bench.read_text()
    # Header column names
    assert "pred_batch" in src, "missing pred_batch column"
    assert "pred_state" in src, "missing pred_state column"
    assert "pred_total" in src, "missing pred_total column"
    assert "C_batch" in src, "missing C_batch column"
    # Model wiring for sort/shuffle/groupby
    assert '_model_predicted("sort"' in src, "sort model not wired"
    assert '_model_predicted("hash_shuffle"' in src, (
        "hash_shuffle model not wired"
    )
    assert '_model_predicted("groupby"' in src, (
        "groupby model not wired"
    )


def test_bench_wires_models_via_solve_estimate():
    """v0.9.20-fix1: bench uses model.solve()+estimate() for pred_total.

    §9.2.2-4 formulas exist; bench must call them, not hardcode 0.
    """
    from pathlib import Path as _P
    bench = (
        _P(__file__).resolve().parent.parent
        / "scripts" / "bench_v0919_calibration.py"
    )
    src = bench.read_text()
    assert "model.solve(" in src
    assert "model.estimate(" in src
    # data_chars_from_sample imported for DataChars construction
    assert "data_chars_from_sample" in src
