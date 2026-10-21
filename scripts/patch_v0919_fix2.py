#!/usr/bin/env python3
# v0.9.19-fix2 — calibrate() prologue must not consume _get_*_delta reads.
# v0.9.19 COMPLETE.
#
# v0.9.19 initial + fix1 files are on disk UNCOMMITTED (pytest failed
# at fix1 → sys.exit before commit). Current disk state:
#   cadw/core/memory_guard.py       ✓ v0.9.19 (calibrate + monitor)
#   cadw/ops/reduce.py              ✓ v0.9.19 (calibrate wrapper)
#   tests/test_runtime_monitor.py   ✓ v0.9.19 + fix1 (18 tests)
#   scripts/bench_v0919_*.py        ✓ v0.9.19 + fix1 (fixed f-string)
#   git commit                       ✗ never reached
#
# v0.9.19-fix1 pytest (target) 2 failures:
#   test_runtime_monitor_raises_on_hard_rss_drift   DID NOT RAISE
#   test_runtime_monitor_soft_warn_no_raise         no soft-drift warn
#
# Root: calibrate() prologue did:
#   self._get_pool_delta()      # reads self._pool.bytes_allocated()
#   self._get_rss_delta()       # reads self._process.memory_info()
#   pool_start = self._pool.bytes_allocated()  # reads again
#   rss_start = self._process.memory_info()    # reads again
#
# _get_pool_delta/_get_rss_delta are caller-facing measurement APIs
# that read from self._pool/_process on EVERY call (even when baseline
# already initialized). Prologue consumed 1 read from each fake, so
# pool_start/end and rss_start/end both read the LAST fake value →
# delta = 0.
#
# Trace for test_runtime_monitor_raises_on_hard_rss_drift:
#   FakePool: pool_values=[0, 0], 2 reads; FakeProcess: rss_values=[0, 100M]
#   prologue _get_pool_delta: pidx 0→1, returns 0
#   prologue _get_rss_delta:  ridx 0→1, returns 0
#   pool_start: pidx 1→2, returns pool_values[1]=0 (CLAMPED)
#   rss_start:  ridx 1→2, returns rss_values[1]=100M (CLAMPED)
#   pool_end:   pidx 2→3, returns pool_values[1]=0 (CLAMPED)
#   rss_end:    ridx 2→3, returns rss_values[1]=100M (CLAMPED)
#   pool_delta = 0 - 0 = 0
#   rss_delta  = 100M - 100M = 0   → NO raise
#
# §13.4/231 (new): lazy-baseline helper methods (like _get_pool_delta,
# _get_rss_delta) are CALLER-FACING APIs. They read state on every
# call. calibrate() must NOT call them; instead it must do direct
# lazy init (only when self._pool/self._process is None). Otherwise
# it consumes reads that start/end measurements need, and does
# redundant work when already initialized. Contract: calibrate() is
# exactly 1 start-read + 1 end-read per fake; helper methods stay
# for external callers.
#
# Fix2:
#   (a) calibrate() prologue: direct lazy init, no _get_*_delta calls.
#   (b) Defensive regression test:
#       test_calibrate_does_not_call_get_delta — spies on
#       _get_pool_delta/_get_rss_delta, asserts calls==0 inside
#       calibrate().
#   (c) verify + dogfood + pytest + atomic v0.9.19 commit
#       (initial + fix1 + fix2 since none committed).
#
# Run: uv run python scripts/patch_v0919_fix2.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) fix calibrate prologue ============

def fix_calibrate_prologue():
    print("\n=== (a) calibrate() prologue: no _get_*_delta calls ===")
    p = Path("cadw/core/memory_guard.py")
    s = p.read_text()
    if "# v0.9.19-fix2" in s and "direct lazy init" in s:
        print("  already fixed — skip")
        return

    old_block = '''        # Ensure baselines initialized
        self._get_pool_delta()
        self._get_rss_delta()
        pool_start = self._pool.bytes_allocated()
        rss_start = self._process.memory_info().rss'''

    new_block = '''        # v0.9.19-fix2: direct lazy init — do NOT call _get_*_delta.
        # Those are caller-facing measurement APIs that read from
        # self._pool / self._process on EVERY call, even when baseline
        # is already set. Calling them here consumed 1 extra read from
        # each of pool/process, breaking start/end measurement:
        # start=clamped[last], end=clamped[last] → delta=0. §13.4/231.
        if self._pool is None:
            self._pool = pa.default_memory_pool()
            self._pool_baseline = self._pool.bytes_allocated()
        if self._process is None:
            import psutil
            self._process = psutil.Process()
            self._rss_baseline = self._process.memory_info().rss

        pool_start = self._pool.bytes_allocated()
        rss_start = self._process.memory_info().rss'''

    if old_block not in s:
        raise SystemExit("calibrate prologue anchor not found")
    s = s.replace(old_block, new_block, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (b) defensive regression test ============

def add_defensive_test():
    print("\n=== (b) defensive test: calibrate does not call _get_*_delta ===")
    p = Path("tests/test_runtime_monitor.py")
    s = p.read_text()
    if "def test_calibrate_does_not_call_get_delta" in s:
        print("  already present — skip")
        return

    extra = '''

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
'''

    p.write_text(s + extra, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.19 + fix1 + fix2 + regression ===")
    pu.verify_anchors([
        # fix2
        ("cadw/core/memory_guard.py", "# v0.9.19-fix2"),
        ("cadw/core/memory_guard.py", "direct lazy init"),
        ("tests/test_runtime_monitor.py",
         "def test_calibrate_does_not_call_get_delta"),
        # v0.9.19 (initial + fix1)
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
        ("tests/test_runtime_monitor.py",
         "def test_bench_v0919_calibration_compiles"),
        ("scripts/bench_v0919_calibration.py",
         "f\"{'op':22s} {'predicted':>12s} {'arrow':>12s} \""),
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


# ============ dogfood (happy path only, §13.4/228) ============

def dogfood():
    print("\n=== dogfood: v0.9.19-fix2 (happy path only, §13.4/228) ===")

    def _verify_all():
        import pyarrow as pa
        from cadw.core.memory_guard import MemoryGuard
        from cadw.ops.reduce import CountOp, reduce

        # 1. Backward compat
        g_off = MemoryGuard(max_memory_mb=10)
        assert g_off._runtime_monitor is False
        assert g_off._calibrate is False
        batches = [
            pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], names=["x"]),
        ]
        result = reduce(iter(batches), CountOp("x"), g_off, log=False)
        assert result == 3
        assert g_off.calibration_samples == {}

        # 2. Calibrate records sample
        g_on = MemoryGuard(max_memory_mb=10, calibrate=True)
        result = reduce(iter(batches), CountOp("x"), g_on, log=False)
        assert result == 3
        assert "CountOp" in g_on.calibration_samples

        # 3. Bench compiles (v0.9.19-fix1)
        import py_compile
        from pathlib import Path as _P
        bench = _P("scripts/bench_v0919_calibration.py")
        assert bench.exists()
        py_compile.compile(str(bench), doraise=True)

        # 4. calibrate() prologue: no _get_*_delta calls (v0.9.19-fix2)
        from unittest.mock import MagicMock as _MM
        g_spy = MemoryGuard(max_memory_mb=10, calibrate=True)
        calls = {"pool": 0, "rss": 0}
        g_spy._get_pool_delta = lambda: calls.__setitem__("pool", calls["pool"] + 1) or 0
        g_spy._get_rss_delta = lambda: calls.__setitem__("rss", calls["rss"] + 1) or 0
        class _FP:
            def bytes_allocated(self):
                return 0
        class _FPr:
            def memory_info(self):
                return _MM(rss=0)
        g_spy._pool = _FP()
        g_spy._pool_baseline = 0
        g_spy._process = _FPr()
        g_spy._rss_baseline = 0
        with g_spy.calibrate("op", 100):
            pass
        assert calls["pool"] == 0, f"_get_pool_delta called {calls['pool']}x"
        assert calls["rss"] == 0, f"_get_rss_delta called {calls['rss']}x"

        print("  no-flag: no samples OK")
        print("  calibrate: CountOp sample recorded OK")
        print("  bench compiles OK")
        print("  calibrate prologue: _get_*_delta not called OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target (runtime monitor + bench compile) ===")
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
    print("=== v0.9.19-fix2 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_calibrate_prologue()
    add_defensive_test()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.19: closed-loop memory correctness — runtime monitor + "
        "calibration log (initial + fix1 + fix2, atomic — none of "
        "prior attempts committed). Design: MemoryGuard(calibrate=True, "
        "runtime_monitor=True, ...) adds three tiers. Tier 1: pyarrow "
        "pool delta (exact Arrow allocations, ~100ns, catches "
        "Table.from_batches/group_by/join/pc.* kernels). Tier 2: psutil "
        "RSS delta (sampled per-op, catches to_pylist Python objects, "
        "np.zeros, foreign libs). Tier 3: calibration log per op "
        "{predicted, arrow, rss}, warn on C=actual/predicted > 2.0 "
        "(drift signal). calibrate(op_name, predicted_bytes=0) context "
        "manager wraps op; when both flags off it yields only (zero "
        "overhead). reduce() wired (has _estimated state); other ops "
        "via bench script (per-op predicted wiring deferred v0.9.20). "
        "scripts/bench_v0919_calibration.py: matrix of shapes × ops, "
        "prints C_arrow/C_rss. Motivation: formulas (§9.2.N) can "
        "silently drift (cf. §1.5 to_list 200× under-estimate 18 "
        "versions; §13.4/218 null-key drop 3 months prod). No runtime "
        "feedback = open loop. Fix1 (bench compile): initial v0.9.19 "
        "died at add_bench → assert_compiles on line 83 of bench — "
        "print-headers f-string split across two write_content() list "
        "elements without closing quote. §13.4/230 (new): write_content"
        "(lines) requires each element to be a complete syntactic unit "
        "for f-strings; cannot split mid-literal. Fixed + added "
        "test_bench_v0919_calibration_compiles (py_compile) and "
        "_has_main (smoke). Fix2 (calibrate prologue): fix1's 2 pytest "
        "target failures — calibrate() called _get_pool_delta/"
        "_get_rss_delta in prologue to \"ensure baselines init\". Those "
        "are caller-facing measurement APIs that read from "
        "self._pool/_process on EVERY call, even when baseline set. "
        "Fake with 2 reads [0, 100M]: prologue consumed read 0 for each, "
        "pool_start/end both read clamped value[1], rss_delta = 100M - "
        "100M = 0 → no raise. §13.4/231 (new): calibrate() must do "
        "direct lazy init (only if self._pool/self._process is None), "
        "not call _get_*_delta. Fix: prologue checks `if X is None:` "
        "directly; 1 read for start + 1 for end. Added defensive test "
        "test_calibrate_does_not_call_get_delta — spies on helpers, "
        "asserts 0 calls inside calibrate (regression guard). "
        "(Backward compat) 1885 existing tests unchanged; zero overhead "
        "when both flags off (calibrate() yields only). 19 new tests: "
        "17 runtime_monitor (defaults no-op, sample recording, "
        "accumulation, drift warning, RSS hard/soft/no-drift, reduce "
        "integration, AST guards, prologue contract) + 2 bench "
        "compile/smoke. Deferred to v0.9.20: AUDIT §1.4 per-group "
        "shift/rolling numpy, §2.x per-op predicted wiring, Pingis "
        "round 3 (unaudited modules: ML, pipeline, cv, metrics), "
        "§13.4/226-231 DECISIONS append.",
        self_path="scripts/patch_v0919_fix2.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.19-fix2 applied. v0.9.19 COMPLETE.")


if __name__ == "__main__":
    main()
