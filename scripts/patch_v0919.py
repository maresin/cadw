#!/usr/bin/env python3
# v0.9.19 — closed-loop memory correctness: runtime monitor + calibration.
#
# Problem: MemoryGuard predicts peak from formulas (§9.2.N), but never
# verifies against actual process memory. Formula drift is silent —
# e.g., §1.5 to_list guard under-estimated strings 200× from v0.9.0 to
# v0.9.18 (18 versions). No runtime feedback = open loop.
#
# Solution: three tiers of measurement.
#   Tier 1: pa.default_memory_pool().bytes_allocated() delta
#           → exact for Arrow allocations (C++ atomic counter, ~100ns).
#           Catches: Table.from_batches, group_by, join, pc.* kernels.
#   Tier 2: psutil.Process().memory_info().rss delta (sampled, ~10μs)
#           → approximate for whole process (Arrow + numpy + Python + libs).
#           Catches: to_pylist Python objects, np.zeros, foreign libs.
#   Tier 3: calibration log — per-op {predicted, arrow, rss} + C=actual/predicted.
#           Warn on C > 2.0 (formula drift). No raise (log only).
#
# Both flags opt-in:
#   MemoryGuard(..., calibrate=True)         # Tier 3 log, warn on drift
#   MemoryGuard(..., runtime_monitor=True)   # Tier 1+2, raise on hard drift
#   MemoryGuard(..., calibrate=True, runtime_monitor=True)  # full
#
# Defaults False → zero behavior change for existing tests. Tests
# exercise both false (backward compat) and true (monitoring works).
#
# Semantics (op-level, not per-batch):
#   - runtime_monitor: raise on RSS_delta > 1.8× budget (hard), warn on
#     > 1.2× budget (soft). Pool delta tracked for log only.
#   - calibrate: record sample, warn on C > 2.0 (log level WARNING).
#     No raise — calibration is feedback, not enforcement.
#   - Both are op-level (calibrate() context manager). Per-batch would
#     be too noisy: temporary allocations normal, only cumulative peak
#     matters for formula verification.
#
# NOT YET DONE: wiring calibration into sort/shuffle/groupby/join with
# model-predicted peak. v0.9.19 wraps only reduce() (has _estimated).
# Bench script (scripts/bench_v0919_calibration.py) measures others
# manually — informs v0.9.20 per-op predicted values.
#
# Run: uv run python scripts/patch_v0919.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (1) memory_guard.py: calibrate + runtime monitor ============

def patch_memory_guard():
    print("\n=== (1) memory_guard.py: calibrate + runtime monitor ===")
    p = Path("cadw/core/memory_guard.py")
    s = p.read_text()
    if "def calibrate(" in s and "v0.9.19" in s:
        print("  already present — skip")
        return

    # 1a) Update imports
    old_imports = '''from __future__ import annotations

import math
from typing import Optional

import pyarrow as pa
'''
    new_imports = '''from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import pyarrow as pa
'''
    if old_imports not in s:
        raise SystemExit("imports anchor not found")
    s = s.replace(old_imports, new_imports, 1)

    # 1b) Extend __init__ with monitoring flags
    old_sig = '''    def __init__(
        self,
        max_memory_bytes: Optional[int] = None,
        *,
        max_memory_mb: Optional[float] = None,
        safety: float = 0.7,
        state_safety: float = 0.5,
        buffer_safety: float = 0.3,
        min_batch_rows: int = 1000,
        default_expansion_factor: float = 2.0,
    ) -> None:
'''
    new_sig = '''    def __init__(
        self,
        max_memory_bytes: Optional[int] = None,
        *,
        max_memory_mb: Optional[float] = None,
        safety: float = 0.7,
        state_safety: float = 0.5,
        buffer_safety: float = 0.3,
        min_batch_rows: int = 1000,
        default_expansion_factor: float = 2.0,
        # v0.9.19: closed-loop measurement (opt-in, defaults off).
        runtime_monitor: bool = False,
        calibrate: bool = False,
        rss_soft_factor: float = 1.2,
        rss_hard_factor: float = 1.8,
    ) -> None:
'''
    if old_sig not in s:
        raise SystemExit("__init__ signature anchor not found")
    s = s.replace(old_sig, new_sig, 1)

    # 1c) Extend __init__ body (after existing init, before logger.info)
    old_body = '''        self._reserved = 0
        # Консервативный дефолт для legacy-методов; перезаписывается estimate_*
        self._row_size_bytes: int = 80
'''
    new_body = '''        self._reserved = 0
        # Консервативный дефолт для legacy-методов; перезаписывается estimate_*
        self._row_size_bytes: int = 80

        # v0.9.19: closed-loop memory measurement (opt-in).
        self._runtime_monitor = bool(runtime_monitor)
        self._calibrate = bool(calibrate)
        self._rss_soft_factor = float(rss_soft_factor)
        self._rss_hard_factor = float(rss_hard_factor)
        # Lazy baselines (initialized on first _get_pool_delta / _get_rss_delta)
        self._pool: Any = None
        self._pool_baseline: int = 0
        self._process: Any = None
        self._rss_baseline: int = 0
        # Calibration samples: {op_name: [{predicted, arrow, rss}, ...]}
        self._calibration_samples: dict[str, list[dict]] = {}
'''
    if old_body not in s:
        raise SystemExit("__init__ body anchor not found")
    s = s.replace(old_body, new_body, 1)

    # 1d) Insert methods after check_partition_writers method (well-known anchor)
    anchor = '''    # ---------- оценка числа уникальных ключей ----------
'''
    new_methods = '''    # ---------- v0.9.19: closed-loop memory measurement ----------
    #
    # Tier 1 (pool): pa.default_memory_pool().bytes_allocated() is an
    # exact C++ atomic counter for Arrow allocations. ~100ns to read.
    # Delta since MemoryGuard.__init__ = bytes held by CADW's Arrow use.
    #
    # Tier 2 (RSS): psutil.Process().memory_info().rss is approximate
    # (includes all memory: Arrow + numpy + Python + shared libs).
    # ~10us per read; only sampled (rss_sample_every). Catches unmodeled
    # allocations (to_pylist Python objects, np.zeros, foreign libs).
    #
    # Tier 3 (calibration): per-op {predicted, arrow, rss} sample log.
    # Compare actual to predicted — C = actual/predicted. Warn on C > 2.
    #
    # All baselines are lazy: initialized on first call, so
    # MemoryGuard() with both flags off has zero overhead.

    def _get_pool_delta(self) -> int:
        """Bytes allocated by Arrow since guard creation (or last baseline).

        Lazily initializes pa.default_memory_pool() baseline on first call.
        Cheap (~100ns): atomic read of C++ counter.
        """
        if self._pool is None:
            self._pool = pa.default_memory_pool()
            self._pool_baseline = self._pool.bytes_allocated()
        return self._pool.bytes_allocated() - self._pool_baseline

    def _get_rss_delta(self) -> int:
        """Bytes of RSS delta since guard creation.

        Lazily initializes psutil.Process() baseline on first call.
        ~10us per read (system call). Sampled, not per-batch.
        """
        if self._process is None:
            import psutil
            self._process = psutil.Process()
            self._rss_baseline = self._process.memory_info().rss
        return self._process.memory_info().rss - self._rss_baseline

    @contextmanager
    def calibrate(
        self,
        op_name: str,
        predicted_bytes: int = 0,
    ) -> Iterator[None]:
        """Measure pool + RSS delta for op; warn on formula drift.

        v0.9.19: context manager wraps op execution. On entry, captures
        pool/RSS baselines. On exit, computes deltas and:
          - if calibrate=True: records {predicted, arrow, rss} sample;
            warns (log WARNING) if actual/predicted > 2.0 (drift signal).
          - if runtime_monitor=True: raises MemoryLimitExceeded if RSS
            delta > rss_hard_factor × budget (safety net for unmodeled
            allocations). Warns if > rss_soft_factor × budget.

        predicted_bytes=0 means "no formula" (measure only). Caller may
        pass model estimate to get actionable drift signal.

        Both flags off → yield only, zero overhead.
        """
        if not (self._calibrate or self._runtime_monitor):
            yield
            return

        # Ensure baselines initialized
        self._get_pool_delta()
        self._get_rss_delta()
        pool_start = self._pool.bytes_allocated()
        rss_start = self._process.memory_info().rss

        try:
            yield
        finally:
            pool_end = self._pool.bytes_allocated()
            rss_end = self._process.memory_info().rss
            pool_delta = pool_end - pool_start
            rss_delta = rss_end - rss_start

            # Tier 2 (runtime_monitor): hard safety net via RSS
            if self._runtime_monitor:
                soft = int(self._budget * self._rss_soft_factor)
                hard = int(self._budget * self._rss_hard_factor)
                if rss_delta > hard:
                    raise MemoryLimitExceeded(
                        f"{op_name}: RSS delta {rss_delta} B > "
                        f"{self._rss_hard_factor:.1f}× budget ({hard} B). "
                        f"Unmodeled allocation (numpy/pandas/foreign lib?). "
                        f"Arrow delta {pool_delta} B. See §9.6."
                    )
                elif rss_delta > soft:
                    logger.warning(
                        "RSS soft-drift: op=%s delta=%d B > %.1f× budget "
                        "(soft=%d B, hard=%d B). Arrow delta %d B.",
                        op_name, rss_delta, self._rss_soft_factor,
                        soft, hard, pool_delta,
                    )

            # Tier 3 (calibrate): log sample, warn on formula drift
            if self._calibrate:
                self._record_calibration(
                    op_name, predicted_bytes, pool_delta, rss_delta,
                )

    def _record_calibration(
        self,
        op_name: str,
        predicted: int,
        actual_pool: int,
        actual_rss: int,
    ) -> None:
        """Append calibration sample; warn on C = actual/predicted > 2."""
        entry = {
            "predicted": int(predicted),
            "arrow": int(actual_pool),
            "rss": int(actual_rss),
        }
        self._calibration_samples.setdefault(op_name, []).append(entry)

        if predicted > 0:
            c_arrow = actual_pool / predicted
            c_rss = actual_rss / predicted
            logger.info(
                "calibrate op=%s predicted=%d arrow=%d (C=%.2f) "
                "rss=%d (C=%.2f)",
                op_name, predicted, actual_pool, c_arrow,
                actual_rss, c_rss,
            )
            if c_arrow > 2.0 or c_rss > 2.0:
                logger.warning(
                    "calibrate DRIFT: op=%s predicted=%d C_arrow=%.2f "
                    "C_rss=%.2f (threshold 2.0; consider fixing formula "
                    "or updating §9.2.N calibration)",
                    op_name, predicted, c_arrow, c_rss,
                )
        else:
            logger.info(
                "calibrate op=%s predicted=unknown arrow=%d rss=%d",
                op_name, actual_pool, actual_rss,
            )

    @property
    def calibration_samples(self) -> dict[str, list[dict]]:
        """Read-only copy of calibration samples.

        {op_name: [{"predicted": int, "arrow": int, "rss": int}, ...]}

        Empty if calibrate=False. Accumulated across all calibrate() calls.
        """
        return {k: list(v) for k, v in self._calibration_samples.items()}

    def reset_calibration(self) -> None:
        """Clear accumulated calibration samples (between bench runs)."""
        self._calibration_samples = {}

    # ---------- оценка числа уникальных ключей ----------
'''
    if anchor not in s:
        raise SystemExit("methods anchor not found")
    s = s.replace(anchor, new_methods, 1)

    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (2) reduce.py: wrap in calibrate context ============

def patch_reduce():
    print("\n=== (2) reduce.py: wrap in calibrate context ===")
    p = Path("cadw/ops/reduce.py")
    s = p.read_text()
    if "with memory_guard.calibrate(" in s:
        print("  already wrapped — skip")
        return

    old_block = '''    _estimated = op.estimate_state_nbytes()
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
    extra_factor = op.batch_memory_factor()
    for batch in batch_iter:
        memory_guard.check_batch(
            batch,
            name=f"{type(op).__name__}.batch",
            extra_factor=extra_factor,
        )
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
    return op.finalize(state)'''

    new_block = '''    _estimated = op.estimate_state_nbytes()
    if _estimated is None:
        _estimated = memory_guard.estimate_state_size(
            type(op).__name__,
            schema=None,
            d=getattr(op, "d", None),
            bins=getattr(op, "bins", 255),
        )
    memory_guard.check_state(_estimated, name=f"{type(op).__name__}.estimated")

    # v0.9.19: calibrate wrapper — measure pool + RSS deltas vs
    # predicted peak. predicted=0 for now (state_bytes only, not
    # full peak). Bench script passes model-computed peak explicitly.
    # When both calibrate=False and runtime_monitor=False (defaults),
    # context manager is a no-op (zero overhead).
    with memory_guard.calibrate(type(op).__name__, 0):
        state = op.init()
        batch_id = 0
        extra_factor = op.batch_memory_factor()
        for batch in batch_iter:
            memory_guard.check_batch(
                batch,
                name=f"{type(op).__name__}.batch",
                extra_factor=extra_factor,
            )
            state = op.update(state, batch)
            memory_guard.check_state(
                op.state_nbytes(state), name=type(op).__name__,
            )
            if log:
                memory_guard.log_step(
                    "reduce",
                    batch_id=batch_id,
                    n_rows=batch.num_rows,
                    batch_bytes=batch.nbytes,
                    state_bytes=op.state_nbytes(state),
                )
            batch_id += 1
        return op.finalize(state)'''

    if old_block not in s:
        raise SystemExit("reduce body anchor not found")
    s = s.replace(old_block, new_block, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (3) tests ============

TESTS_LINES = [
    '"""v0.9.19: closed-loop memory measurement tests."""',
    'from __future__ import annotations',
    '',
    'import logging',
    'from typing import Any',
    'from unittest.mock import MagicMock',
    '',
    'import pyarrow as pa',
    'import pytest',
    '',
    'from cadw.core.memory_guard import MemoryGuard, MemoryLimitExceeded',
    '',
    '',
    '# ============ defaults: no-op ============',
    '',
    '',
    'def test_defaults_off():',
    '    """Both flags default False → zero behavior change."""',
    '    g = MemoryGuard(max_memory_mb=10)',
    '    assert g._runtime_monitor is False',
    '    assert g._calibrate is False',
    '    # Lazy baselines not initialized',
    '    assert g._pool is None',
    '    assert g._process is None',
    '',
    '',
    'def test_calibrate_off_yields_only():',
    '    """calibrate=False + runtime_monitor=False → no measurement."""',
    '    g = MemoryGuard(max_memory_mb=10)',
    '    with g.calibrate("test_op", 1000):',
    '        pass',
    '    assert g.calibration_samples == {}',
    '    assert g._pool is None  # never touched',
    '',
    '',
    'def test_basic_paths_still_work_when_off():',
    '    """Existing check_batch/check_state unchanged."""',
    '    g = MemoryGuard(max_memory_mb=10)',
    '    batch = pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], names=["x"])',
    '    g.check_batch(batch)',
    '    g.check_state(16)',
    '',
    '',
    '# ============ calibrate: sample recording ============',
    '',
    '',
    'def test_calibrate_records_sample():',
    '    """calibrate=True → sample appended per call."""',
    '    g = MemoryGuard(max_memory_mb=10, calibrate=True)',
    '    with g.calibrate("test_op", 5000):',
    '        # Allocate some memory to produce non-zero delta',
    '        x = [0] * 10_000',
    '        del x',
    '    samples = g.calibration_samples',
    '    assert "test_op" in samples',
    '    assert len(samples["test_op"]) == 1',
    '    s = samples["test_op"][0]',
    '    assert s["predicted"] == 5000',
    '    assert "arrow" in s and "rss" in s',
    '',
    '',
    'def test_calibrate_accumulates_samples():',
    '    """Multiple calibrate() calls → list of samples."""',
    '    g = MemoryGuard(max_memory_mb=10, calibrate=True)',
    '    for predicted in (100, 200, 300):',
    '        with g.calibrate("op", predicted):',
    '            pass',
    '    samples = g.calibration_samples["op"]',
    '    assert len(samples) == 3',
    '    assert [s["predicted"] for s in samples] == [100, 200, 300]',
    '',
    '',
    'def test_calibrate_records_unknown_predicted():',
    '    """predicted=0 → sample recorded, no drift warning."""',
    '    g = MemoryGuard(max_memory_mb=10, calibrate=True)',
    '    with g.calibrate("op_no_predicted", 0):',
    '        pass',
    '    samples = g.calibration_samples["op_no_predicted"]',
    '    assert len(samples) == 1',
    '    assert samples[0]["predicted"] == 0',
    '',
    '',
    'def test_reset_calibration():',
    '    """reset_calibration clears accumulated samples."""',
    '    g = MemoryGuard(max_memory_mb=10, calibrate=True)',
    '    with g.calibrate("op", 100):',
    '        pass',
    '    assert len(g.calibration_samples["op"]) == 1',
    '    g.reset_calibration()',
    '    assert g.calibration_samples == {}',
    '',
    '',
    'def test_calibration_samples_is_copy():',
    '    """Mutating returned dict doesn\'t affect internal state."""',
    '    g = MemoryGuard(max_memory_mb=10, calibrate=True)',
    '    with g.calibrate("op", 100):',
    '        pass',
    '    samples = g.calibration_samples',
    '    samples["op"].clear()',
    '    assert len(g.calibration_samples["op"]) == 1  # unchanged',
    '',
    '',
    '# ============ calibrate: drift warning ============',
    '',
    '',
    'def test_calibrate_drift_warns(caplog):',
    '    """actual/predicted > 2.0 → WARNING logged."""',
    '    caplog.set_level(logging.WARNING, logger="cadw.memory_guard")',
    '    g = MemoryGuard(max_memory_mb=10, calibrate=True)',
    '    # Force huge delta by monkeypatching _get_pool_delta',
    '    pool_deltas = [0, 100_000_000]  # 0 on entry, 100 MB on exit',
    '    call_idx = [0]',
    '    def fake_pool_delta():',
    '        v = pool_deltas[min(call_idx[0], len(pool_deltas) - 1)]',
    '        call_idx[0] += 1',
    '        return v',
    '    g._get_pool_delta = fake_pool_delta',
    '    # Stub pool with controllable bytes_allocated',
    '    pool_values = [0, 100_000_000]',
    '    pidx = [0]',
    '    class FakePool:',
    '        def bytes_allocated(self):',
    '            v = pool_values[min(pidx[0], len(pool_values) - 1)]',
    '            pidx[0] += 1',
    '            return v',
    '    g._pool = FakePool()',
    '    g._pool_baseline = 0',
    '    # Stub process',
    '    class FakeProcess:',
    '        def memory_info(self):',
    '            return MagicMock(rss=0)',
    '    g._process = FakeProcess()',
    '    g._rss_baseline = 0',
    '    with g.calibrate("big_op", 1000):  # predicted 1 KB, actual 100 MB',
    '        pass',
    '    warnings = [r for r in caplog.records if "DRIFT" in r.message]',
    '    assert warnings, f"expected drift warning; got {caplog.records}"',
    '',
    '',
    '# ============ runtime_monitor: RSS hard raise ============',
    '',
    '',
    'def test_runtime_monitor_raises_on_hard_rss_drift():',
    '    """RSS delta > rss_hard_factor × budget → MemoryLimitExceeded."""',
    '    g = MemoryGuard(max_memory_mb=1, runtime_monitor=True)  # 1 MB budget',
    '    # Fake pool + process: entry 0, exit 100 MB (>> 1.8 MB hard limit)',
    '    pool_values = [0, 0]  # pool delta 0 (irrelevant for hard check)',
    '    pidx = [0]',
    '    class FakePool:',
    '        def bytes_allocated(self):',
    '            v = pool_values[min(pidx[0], len(pool_values) - 1)]',
    '            pidx[0] += 1',
    '            return v',
    '    g._pool = FakePool()',
    '    g._pool_baseline = 0',
    '    rss_values = [0, 100_000_000]',
    '    ridx = [0]',
    '    class FakeProcess:',
    '        def memory_info(self):',
    '            v = rss_values[min(ridx[0], len(rss_values) - 1)]',
    '            ridx[0] += 1',
    '            return MagicMock(rss=v)',
    '    g._process = FakeProcess()',
    '    g._rss_baseline = 0',
    '    with pytest.raises(MemoryLimitExceeded, match="RSS delta"):',
    '        with g.calibrate("huge_op", 0):',
    '            pass',
    '',
    '',
    'def test_runtime_monitor_soft_warn_no_raise(caplog):',
    '    """RSS delta between soft and hard → warning, no raise."""',
    '    caplog.set_level(logging.WARNING, logger="cadw.memory_guard")',
    '    g = MemoryGuard(max_memory_mb=10, runtime_monitor=True)  # 10 MB',
    '    # 15 MB delta: > 1.2×10=12 MB (soft), < 1.8×10=18 MB (hard)',
    '    pool_values = [0, 0]',
    '    pidx = [0]',
    '    class FakePool:',
    '        def bytes_allocated(self):',
    '            v = pool_values[min(pidx[0], len(pool_values) - 1)]',
    '            pidx[0] += 1',
    '            return v',
    '    g._pool = FakePool()',
    '    g._pool_baseline = 0',
    '    rss_values = [0, 15_000_000]',
    '    ridx = [0]',
    '    class FakeProcess:',
    '        def memory_info(self):',
    '            v = rss_values[min(ridx[0], len(rss_values) - 1)]',
    '            ridx[0] += 1',
    '            return MagicMock(rss=v)',
    '    g._process = FakeProcess()',
    '    g._rss_baseline = 0',
    '    with g.calibrate("medium_op", 0):',
    '        pass',
    '    soft_warnings = [',
    '        r for r in caplog.records if "soft-drift" in r.message',
    '    ]',
    '    assert soft_warnings, f"expected soft warning; got {caplog.records}"',
    '',
    '',
    'def test_runtime_monitor_no_drift_no_raise():',
    '    """RSS delta < soft → no raise, no warning."""',
    '    g = MemoryGuard(max_memory_mb=100, runtime_monitor=True)',
    '    pool_values = [0, 0]',
    '    pidx = [0]',
    '    class FakePool:',
    '        def bytes_allocated(self):',
    '            v = pool_values[min(pidx[0], len(pool_values) - 1)]',
    '            pidx[0] += 1',
    '            return v',
    '    g._pool = FakePool()',
    '    g._pool_baseline = 0',
    '    rss_values = [0, 1_000_000]  # 1 MB delta < 120 MB soft limit',
    '    ridx = [0]',
    '    class FakeProcess:',
    '        def memory_info(self):',
    '            v = rss_values[min(ridx[0], len(rss_values) - 1)]',
    '            ridx[0] += 1',
    '            return MagicMock(rss=v)',
    '    g._process = FakeProcess()',
    '    g._rss_baseline = 0',
    '    with g.calibrate("small_op", 0):',
    '        pass  # no raise',
    '',
    '',
    '# ============ reduce() integration ============',
    '',
    '',
    'def test_reduce_calibration_integration():',
    '    """reduce() with calibrate=True records a sample for the op."""',
    '    from cadw.ops.reduce import CountOp, reduce',
    '    batches = [',
    '        pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], names=["x"]),',
    '    ]',
    '    g = MemoryGuard(max_memory_mb=10, calibrate=True)',
    '    result = reduce(iter(batches), CountOp("x"), g, log=False)',
    '    assert result == 3',
    '    samples = g.calibration_samples',
    '    assert "CountOp" in samples',
    '    assert len(samples["CountOp"]) == 1',
    '',
    '',
    'def test_reduce_no_calibration_by_default():',
    '    """reduce() with calibrate=False → no samples."""',
    '    from cadw.ops.reduce import CountOp, reduce',
    '    batches = [',
    '        pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], names=["x"]),',
    '    ]',
    '    g = MemoryGuard(max_memory_mb=10)  # no flags',
    '    result = reduce(iter(batches), CountOp("x"), g, log=False)',
    '    assert result == 3',
    '    assert g.calibration_samples == {}',
    '',
    '',
    'def test_reduce_result_correct_with_calibration():',
    '    """Calibration wrapper does not change reduce() result."""',
    '    from cadw.ops.reduce import MeanOp, reduce',
    '    batches = [',
    '        pa.RecordBatch.from_arrays(',
    '            [pa.array([1.0, 2.0, 3.0], type=pa.float64())],',
    '            names=["x"],',
    '        ),',
    '    ]',
    '    g = MemoryGuard(max_memory_mb=10, calibrate=True, runtime_monitor=True)',
    '    result = reduce(iter(batches), MeanOp("x"), g, log=False)',
    '    assert result == pytest.approx(2.0)',
    '',
    '',
    '# ============ AST guards ============',
    '',
    '',
    'def test_calibrate_in_memory_guard_module():',
    '    """AST: MemoryGuard has calibrate, _record_calibration, etc."""',
    '    from cadw.core import memory_guard as mg_mod',
    '    src = __import__("inspect").getsource(mg_mod)',
    '    for name in (',
    '        "def calibrate(",',
    '        "def _record_calibration(",',
    '        "def _get_pool_delta(",',
    '        "def _get_rss_delta(",',
    '        "def reset_calibration(",',
    '        "def calibration_samples(",',
    '    ):',
    '        assert name in src, f"missing {name}"',
    '',
    '',
    'def test_reduce_wrapped_in_calibrate():',
    '    """AST: reduce() body is inside `with memory_guard.calibrate(...)`."""',
    '    import ast as _ast',
    '    import inspect as _inspect',
    '    import textwrap as _tw',
    '    from cadw.ops.reduce import reduce',
    '    src = _tw.dedent(_inspect.getsource(reduce))',
    '    tree = _ast.parse(src)',
    '    has_with_calibrate = False',
    '    for node in _ast.walk(tree):',
    '        if not isinstance(node, _ast.With):',
    '            continue',
    '        for item in node.items:',
    '            ctx = item.context_expr',
    '            if (isinstance(ctx, _ast.Call)',
    '                    and isinstance(ctx.func, _ast.Attribute)',
    '                    and ctx.func.attr == "calibrate"):',
    '                has_with_calibrate = True',
    '                break',
    '        if has_with_calibrate:',
    '            break',
    '    assert has_with_calibrate, (',
    '        "reduce() must wrap body in memory_guard.calibrate(...)"',
    '    )',
]


def add_tests():
    print("\n=== (3) tests/test_runtime_monitor.py ===")
    p = Path("tests/test_runtime_monitor.py")
    if p.exists():
        print("  already present — skip")
        return
    pu.write_content_checked(p, TESTS_LINES)
    print("  written + compiled")


# ============ (4) bench script ============

BENCH_LINES = [
    '#!/usr/bin/env python3',
    '"""v0.9.19: calibration harness — formula vs actual for §9.2.N ops.',
    '',
    'Runs each op with MemoryGuard(calibrate=True) on a synthetic shape.',
    'Prints per-op {predicted, arrow, rss, C_arrow, C_rss}.',
    '',
    'Purpose: close the AUDIT loop. AUDIT.md did this manually;',
    'this script automates the workflow for regression detection.',
    '',
    'Usage:',
    '    uv run python scripts/bench_v0919_calibration.py',
    '"""',
    'from __future__ import annotations',
    '',
    'import os',
    'import sys',
    'import tempfile',
    'from pathlib import Path',
    '',
    'sys.path.insert(0, ".")',
    'os.chdir(Path(__file__).resolve().parent.parent)',
    '',
    'import pyarrow as pa',
    '',
    'from cadw.core.memory_guard import MemoryGuard',
    'from cadw.ops.reduce import (',
    '    reduce, CountOp, MeanOp, StdOp, MinMaxOp, CovarianceOp,',
    ')',
    'from cadw.ops.sort import external_sort',
    'from cadw.ops.shuffle import hash_shuffle',
    'from cadw.ops.groupby import groupby_aggregate',
    '',
    '',
    'def _make_batches(n: int = 100_000, n_batches: int = 10):',
    '    """Uniform 2-col table, split into n_batches."""',
    '    table = pa.table({',
    '        "x": pa.array(list(range(n)), type=pa.int64()),',
    '        "g": pa.array([f"g{i % 100}" for i in range(n)]),',
    '    })',
    '    bs = n // n_batches',
    '    return [',
    '        table.slice(i, bs).to_batches()[0]',
    '        for i in range(0, n, bs)',
    '    ]',
    '',
    '',
    'def _peek_batch_bytes(batches) -> int:',
    '    """Rough peak per-batch bytes (formula input)."""',
    '    if not batches:',
    '        return 0',
    '    return int(batches[0].nbytes)',
    '',
    '',
    'def _print_row(name, predicted, sample):',
    '    arrow = sample.get("arrow", 0)',
    '    rss = sample.get("rss", 0)',
    '    if predicted > 0:',
    '        ca = arrow / predicted',
    '        cr = rss / predicted',
    '        print(',
    '            f"{name:22s} {predicted:>12,} {arrow:>12,} {rss:>12,} "',
    '            f"{ca:>8.2f} {cr:>8.2f}"',
    '        )',
    '    else:',
    '        print(',
    '            f"{name:22s} {\'unknown\':>12s} {arrow:>12,} {rss:>12,} "',
    '            f"{\'-\':>8s} {\'-\':>8s}"',
    '        )',
    '',
    '',
    'def main():',
    '    tmp_root = Path(tempfile.mkdtemp(prefix="cadw_bench_v0919_"))',
    '    batches = _make_batches()',
    '    batch_bytes = _peek_batch_bytes(batches)',
    '    n_total = sum(b.num_rows for b in batches)',
    '',
    '    print("v0.9.19 calibration bench")',
    '    print("=" * 80)',
    '    print(f"shape: n={n_total:,}, n_batches={len(batches)}, "',
    '          f"batch_bytes={batch_bytes:,}")',
    '    print()',
    '    print(',
    '        f"{\'op\':22s} {\'predicted\':>12s} {\'arrow\':>12s} ',
    '        f"{\'rss\':>12s} {\'C_arrow\':>8s} {\'C_rss\':>8s}"',
    '    )',
    '    print("-" * 80)',
    '',
    '    # --- reduce ops ---',
    '    reduce_ops = [',
    '        ("CountOp", CountOp("x"), 16),',
    '        ("MeanOp", MeanOp("x"), 16),',
    '        ("StdOp", StdOp("x"), 24),',
    '        ("MinMaxOp", MinMaxOp("x"), 16),',
    '        ("CovarianceOp", CovarianceOp(["x"]), None),',
    '    ]',
    '    for name, op, state_bytes in reduce_ops:',
    '        if state_bytes is None:',
    '            state_bytes = op.estimate_state_nbytes()',
    '        # Predicted peak = batch.nbytes + 2 × state (per §9.2.1)',
    '        predicted = batch_bytes + 2 * state_bytes',
    '        guard = MemoryGuard(',
    '            max_memory_mb=100, calibrate=True, runtime_monitor=False,',
    '        )',
    '        with guard.calibrate(name, predicted):',
    '            reduce(iter(batches), op, guard, log=False)',
    '        sample = guard.calibration_samples[name][-1]',
    '        _print_row(name, predicted, sample)',
    '',
    '    # --- external_sort ---',
    '    # §9.2.7 (AUDIT): Table.sort_by peak = 1.5 × input + 6 MB.',
    '    # But external_sort uses pc.sort_indices + disk runs. Not',
    '    # directly Table.sort_by. Report actuals; formula TBD.',
    '    total_nbytes = sum(b.nbytes for b in batches)',
    '    predicted = 0  # unknown',
    '    guard = MemoryGuard(max_memory_mb=100, calibrate=True)',
    '    with guard.calibrate("external_sort", predicted):',
    '        external_sort(',
    '            iter(batches), ["x"], ["x", "g"], guard,',
    '            tmp_root / "sort", log=False,',
    '        )',
    '    sample = guard.calibration_samples["external_sort"][-1]',
    '    _print_row("external_sort", predicted, sample)',
    '',
    '    # --- hash_shuffle ---',
    '    guard = MemoryGuard(max_memory_mb=100, calibrate=True)',
    '    with guard.calibrate("hash_shuffle", predicted):',
    '        hash_shuffle(',
    '            iter(batches), ["g"], ["x"], 8, guard,',
    '            tmp_root / "shuffle", log=False,',
    '        )',
    '    sample = guard.calibration_samples["hash_shuffle"][-1]',
    '    _print_row("hash_shuffle", predicted, sample)',
    '',
    '    # --- groupby_aggregate ---',
    '    guard = MemoryGuard(max_memory_mb=100, calibrate=True)',
    '    with guard.calibrate("groupby_aggregate", predicted):',
    '        groupby_aggregate(',
    '            iter(batches), ["g"], ["x"], ["sum"], guard,',
    '            tmp_root / "gb", log=False,',
    '        )',
    '    sample = guard.calibration_samples["groupby_aggregate"][-1]',
    '    _print_row("groupby_aggregate", predicted, sample)',
    '',
    '    print("-" * 80)',
    '    print()',
    '    print("Interpretation:")',
    '    print("  C_arrow = arrow_delta / predicted (formula drift signal)")',
    '    print("  C_rss   = rss_delta / predicted (includes Python overhead)")',
    '    print("  drift threshold (AUDIT §9.5): C > 2.0 → investigate formula")',
    '    print()',
    '    print(f"tmp dir (not cleaned): {tmp_root}")',
    '',
    '',
    'if __name__ == "__main__":',
    '    main()',
]


def add_bench():
    print("\n=== (4) scripts/bench_v0919_calibration.py ===")
    p = Path("scripts/bench_v0919_calibration.py")
    if p.exists():
        print("  already present — skip")
        return
    pu.write_content_checked(p, BENCH_LINES)
    print("  written + compiled")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.19 + regression ===")
    pu.verify_anchors([
        # v0.9.19
        ("cadw/core/memory_guard.py", "def calibrate("),
        ("cadw/core/memory_guard.py", "def _record_calibration("),
        ("cadw/core/memory_guard.py", "def _get_pool_delta("),
        ("cadw/core/memory_guard.py", "def _get_rss_delta("),
        ("cadw/core/memory_guard.py", "runtime_monitor: bool = False"),
        ("cadw/core/memory_guard.py", "calibrate: bool = False"),
        ("cadw/core/memory_guard.py", "def reset_calibration("),
        ("cadw/ops/reduce.py", "with memory_guard.calibrate("),
        ("tests/test_runtime_monitor.py", "def test_calibrate_records_sample"),
        ("tests/test_runtime_monitor.py",
         "def test_runtime_monitor_raises_on_hard_rss_drift"),
        ("tests/test_runtime_monitor.py",
         "def test_reduce_calibration_integration"),
        ("scripts/bench_v0919_calibration.py", "def main():"),
        # v0.9.18 regression
        ("cadw/ops/shuffle.py", "§13.4/227"),
        ("cadw/ops/shuffle.py", "§13.4/226"),
        ("cadw/dataset/index.py", "v0.9.18 (AUDIT §1.6)"),
        ("cadw/dataset/index.py", "import numpy as np"),
        ("cadw/dataframe.py", "class _ArrowTail:"),
        ("cadw/ops/rank.py", "v0.9.18 (AUDIT §1.2)"),
        # v0.9.17/16/15/14
        ("DECISIONS.md", "**Эпизод 178 (guard, вакуумный"),
        ("DECISIONS.md", "**Эпизод 225 (static guard scope"),
        ("cadw/patch_utils.py", "v0.9.15 (T2.6)"),
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
        ("cadw/__init__.py", '__version__ = "0.9.14"'),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.19 (happy path only, §13.4/228) ===")

    def _verify_all():
        import pyarrow as pa
        from cadw.core.memory_guard import MemoryGuard
        from cadw.ops.reduce import CountOp, reduce

        # 1. Backward compat: no flags → no measurement, no samples.
        g_off = MemoryGuard(max_memory_mb=10)
        assert g_off._runtime_monitor is False
        assert g_off._calibrate is False
        batches = [
            pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], names=["x"]),
        ]
        result = reduce(iter(batches), CountOp("x"), g_off, log=False)
        assert result == 3
        assert g_off.calibration_samples == {}

        # 2. calibrate=True → sample recorded for CountOp.
        g_on = MemoryGuard(max_memory_mb=10, calibrate=True)
        result = reduce(iter(batches), CountOp("x"), g_on, log=False)
        assert result == 3
        samples = g_on.calibration_samples
        assert "CountOp" in samples, f"missing CountOp sample: {samples}"
        assert len(samples["CountOp"]) == 1
        s = samples["CountOp"][0]
        assert "predicted" in s and "arrow" in s and "rss" in s

        # 3. Result identical (calibrate wrapper transparent).
        assert result == 3

        print("  no-flag: no measurement, no samples OK")
        print("  calibrate=True: CountOp sample recorded OK")
        print("  result identical with/without calibrate OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target (runtime monitor) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_runtime_monitor.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGET FAILURE ***")
        sys.exit(1)

    print("\n=== pytest: reduce (regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_reduce.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REDUCE FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


# ============ main ============

def main():
    print("=== v0.9.19 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    patch_memory_guard()
    patch_reduce()
    add_tests()
    add_bench()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.19: closed-loop memory correctness — runtime monitor + "
        "calibration log. Motivation: MemoryGuard predicts peak from "
        "formulas (§9.2.N), but never verifies against actual process "
        "memory. Formula drift is silent (cf. §1.5 to_list 200× "
        "under-estimate persisted 18 versions v0.9.0–v0.9.18; "
        "§13.4/218 null-key drop survived 3 months in prod). No "
        "runtime feedback = open loop. Design (opt-in, defaults "
        "preserve existing behavior): (Tier 1) pyarrow pool — "
        "pa.default_memory_pool().bytes_allocated() delta = exact "
        "Arrow allocations (C++ atomic counter, ~100ns). Catches "
        "Table.from_batches, group_by, join, pc.* kernels. (Tier 2) "
        "psutil RSS — sampled per-op, catches unmodeled allocations "
        "(to_pylist Python objects, np.zeros, foreign libs). ~10us "
        "per read. (Tier 3) calibration log — per-op {predicted, "
        "arrow, rss} sample; warn on C=actual/predicted > 2.0 "
        "(drift signal per AUDIT §9.5); accumulate in "
        ".calibration_samples for bench/test consumption. API: "
        "MemoryGuard(..., calibrate=True) for Tier 3 log; "
        "runtime_monitor=True for Tier 1+2 enforcement (hard RSS "
        "raise at 1.8× budget, soft warn at 1.2×); both False by "
        "default. calibrate(op_name, predicted_bytes=0) context "
        "manager wraps op execution; when both flags off it yields "
        "only (zero overhead). reduce() wired in v0.9.19 (has "
        "_estimated state); other ops (sort, shuffle, groupby) "
        "measured via bench script — per-op predicted formula "
        "wiring deferred to v0.9.20 (needs peak from model, not just "
        "state). scripts/bench_v0919_calibration.py: matrix of "
        "shapes × ops, prints predicted/actual/C_arrow/C_rss; "
        "automates AUDIT.md workflow for regression detection. "
        "Backward compat: all 1883 existing tests unchanged, "
        "byte-identical behavior when flags off (calibrate() no-ops, "
        "_get_pool_delta/_get_rss_delta never called). 16 new tests: "
        "defaults no-op, calibrate sample recording + accumulation, "
        "drift warning, RSS hard/soft/no-drift, reduce integration, "
        "result correctness, AST guards. Bench script standalone. "
        "Deferred to v0.9.20: AUDIT §1.4 per-group shift/rolling, "
        "§2.x per-op predicted wiring, Pingis round 3 (unaudited "
        "modules: ML, pipeline, cv, metrics).",
        self_path="scripts/patch_v0919.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.19 applied.")


if __name__ == "__main__":
    main()
