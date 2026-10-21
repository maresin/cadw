#!/usr/bin/env python3
# v0.9.19-fix1 — bench script unterminated f-string + regression test.
# v0.9.19 COMPLETE.
#
# v0.9.19 initial died at add_bench() → write_content_checked() →
# assert_compiles → SystemExit. Line 83 of bench script:
#   f"{'op':22s} {'predicted':>12s} {'arrow':>12s}   ← unterminated
#   f"{'rss':>12s} {'C_arrow':>8s} {'C_rss':>8s}"
#
# Root: in BENCH_LINES (list of lines for write_content), the
# print-headers f-string was split across two elements:
#   '        f"{\'op\':22s} {\'predicted\':>12s} {\'arrow\':>12s} ',
#                                                      ↑ trailing space, NO closing "
#   '        f"{\'rss\':>12s} {\'C_arrow\':>8s} {\'C_rss\':>8s}"',
#
# write_content joins elements with \n — each becomes a separate line.
# Result: two f-string opens, one close, only one valid. SyntaxError.
#
# §13.4/230 (new): string-list authoring. write_content(lines: list[str])
# creates one file-line per element. Each element must be a COMPLETE
# syntactic unit for any multi-part construct (f-string, string
# concatenation, backslash continuation). Cannot split `f"...` mid-
# literal across elements. Options: single-line f-string, explicit
# `\` continuation, or extract to intermediate variable.
#
# Side effect: v0.9.19 died BEFORE commit. Files on disk uncommitted:
#   memory_guard.py       ✓ applied (v0.9.19)
#   reduce.py             ✓ applied (v0.9.19)
#   test_runtime_monitor  ✓ written (v0.9.19, 16 tests)
#   bench_v0919_*.py      ✗ broken (syntax error)
# Fix1 completes v0.9.19 as single atomic commit.
#
# Fix1:
#   (a) bench script: fix unterminated f-string (add closing ")
#   (b) test_runtime_monitor.py: add test_bench_v0919_calibration_compiles
#       (regression: bench must always be syntactically valid)
#   (c) verify (v0.9.19 + fix1) + dogfood + pytest + commit (atomic
#       v0.9.19 = initial + fix1, since initial never committed)
#
# Run: uv run python scripts/patch_v0919_fix1.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) fix unterminated f-string in bench ============

def fix_bench_fstring():
    print("\n=== (a) bench: fix unterminated f-string (line 83) ===")
    p = Path("scripts/bench_v0919_calibration.py")
    if not p.exists():
        raise SystemExit("bench script missing (v0.9.19 partial state?)")
    s = p.read_text()
    # Idempotency: fixed version ends with `{'arrow':>12s} "` (closing quote)
    if "f\"{'op':22s} {'predicted':>12s} {'arrow':>12s} \"" in s:
        print("  already fixed — skip")
        return

    # Broken fragment: trailing space after `>12s}`, then newline, no `"`
    # NOTE: use chr(10) to avoid ambiguity with embedded newlines
    nl = chr(10)
    broken = (
        "    print(" + nl +
        "        f\"{'op':22s} {'predicted':>12s} {'arrow':>12s} " + nl +
        "        f\"{'rss':>12s} {'C_arrow':>8s} {'C_rss':>8s}\"" + nl +
        "    )"
    )
    fixed = (
        "    print(" + nl +
        "        f\"{'op':22s} {'predicted':>12s} {'arrow':>12s} \"" + nl +
        "        f\"{'rss':>12s} {'C_arrow':>8s} {'C_rss':>8s}\"" + nl +
        "    )"
    )

    if broken not in s:
        # Show context for debugging
        print("  BROKEN FRAGMENT NOT FOUND — showing line 80-90:")
        lines = s.split(nl)
        for i in range(max(0, 78), min(len(lines), 92)):
            print(f"    L{i+1:4d}: {lines[i]!r}")
        raise SystemExit("bench broken fragment anchor not found")

    s = s.replace(broken, fixed, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK (compiles)")


# ============ (b) regression test ============

def add_bench_compile_test():
    print("\n=== (b) test_runtime_monitor: bench compile regression ===")
    p = Path("tests/test_runtime_monitor.py")
    s = p.read_text()
    if "def test_bench_v0919_calibration_compiles" in s:
        print("  already present — skip")
        return

    # Append after last test
    extra = '''

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
'''

    p.write_text(s + extra, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.19 + fix1 + regression ===")
    pu.verify_anchors([
        # fix1 (a) bench compiles
        ("scripts/bench_v0919_calibration.py",
         "f\"{'op':22s} {'predicted':>12s} {'arrow':>12s} \""),
        # fix1 (b) regression test
        ("tests/test_runtime_monitor.py",
         "def test_bench_v0919_calibration_compiles"),
        ("tests/test_runtime_monitor.py",
         "def test_bench_v0919_calibration_has_main"),
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


# ============ dogfood (happy path only, §13.4/228) ============

def dogfood():
    print("\n=== dogfood: v0.9.19-fix1 (happy path only, §13.4/228) ===")

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

        # 3. Bench compiles (v0.9.19-fix1 core deliverable)
        import py_compile
        from pathlib import Path as _P
        bench = _P("scripts/bench_v0919_calibration.py")
        assert bench.exists()
        py_compile.compile(str(bench), doraise=True)

        print("  no-flag: no samples OK")
        print("  calibrate: CountOp sample recorded OK")
        print("  bench compiles OK")

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
    print("=== v0.9.19-fix1 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_bench_fstring()
    add_bench_compile_test()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.19: closed-loop memory correctness — runtime monitor + "
        "calibration log (v0.9.19 initial + v0.9.19-fix1, atomic "
        "since initial never committed). (Motivation) MemoryGuard "
        "predicts peak from formulas (§9.2.N), but never verifies "
        "against actual process memory. Formula drift is silent "
        "(cf. §1.5 to_list 200× under-estimate persisted 18 "
        "versions v0.9.0–v0.9.18; §13.4/218 null-key drop survived "
        "3 months in prod). No runtime feedback = open loop. "
        "(Design, opt-in, defaults preserve existing behavior) "
        "Tier 1: pyarrow pool — pa.default_memory_pool()."
        "bytes_allocated() delta = exact Arrow allocations (C++ "
        "atomic counter, ~100ns). Catches Table.from_batches, "
        "group_by, join, pc.* kernels. Tier 2: psutil RSS — "
        "sampled per-op, catches unmodeled allocations "
        "(to_pylist Python objects, np.zeros, foreign libs). ~10us "
        "per read. Tier 3: calibration log — per-op {predicted, "
        "arrow, rss} sample; warn on C=actual/predicted > 2.0 "
        "(drift signal per AUDIT §9.5); accumulate in "
        ".calibration_samples for bench/test consumption. "
        "(API) MemoryGuard(..., calibrate=True) for Tier 3 log; "
        "runtime_monitor=True for Tier 1+2 enforcement (hard RSS "
        "raise at 1.8× budget, soft warn at 1.2×); both False by "
        "default. calibrate(op_name, predicted_bytes=0) context "
        "manager wraps op execution; when both flags off it yields "
        "only (zero overhead). reduce() wired in v0.9.19 (has "
        "_estimated state); other ops (sort, shuffle, groupby) "
        "measured via bench script — per-op predicted formula "
        "wiring deferred to v0.9.20 (needs peak from model, not "
        "just state). (Bench) scripts/bench_v0919_calibration.py: "
        "matrix of shapes × ops, prints predicted/actual/C_arrow/"
        "C_rss; automates AUDIT.md workflow for regression "
        "detection. (Fix1) v0.9.19 initial patch broke bench "
        "script: in BENCH_LINES (write_content list-of-lines), the "
        "print-headers f-string was split across two elements "
        "without closing quote on first. Result: two f-string "
        "opens, one close. SyntaxError at line 83. Patch died at "
        "add_bench → assert_compiles → SystemExit BEFORE commit. "
        "Files on disk uncommitted. §13.4/230 (new): string-list "
        "authoring — write_content(lines: list[str]) creates one "
        "file-line per element. Each element must be a COMPLETE "
        "syntactic unit for any multi-part construct (f-string, "
        "string concatenation, backslash continuation). Cannot "
        "split `f\"...` mid-literal. Fix: single-line f-string, "
        "explicit `\\` continuation, or extract to intermediate "
        "variable. (Fix1 changes) (a) bench script: add closing `\"` "
        "after `{'arrow':>12s} `; (b) test_runtime_monitor.py: add "
        "test_bench_v0919_calibration_compiles (py_compile, catches "
        "future manual edits that break syntax) + "
        "test_bench_v0919_calibration_has_main (smoke: def main + "
        "__main__ guard); (c) verify (26 anchors) + dogfood + "
        "pytest + commit — atomic v0.9.19 (initial + fix1). "
        "(Backward compat) All 1883 existing tests unchanged, "
        "byte-identical behavior when flags off (calibrate() "
        "no-ops, _get_pool_delta/_get_rss_delta never called). "
        "18 new tests total: 16 runtime_monitor (defaults, sample "
        "recording, accumulation, drift warning, RSS hard/soft/"
        "no-drift, reduce integration, AST guards) + 2 bench "
        "compile/smoke. Deferred to v0.9.20: AUDIT §1.4 per-group "
        "shift/rolling, §2.x per-op predicted wiring, Pingis "
        "round 3 (unaudited modules: ML, pipeline, cv, metrics), "
        "§13.4/226–230 append to DECISIONS.",
        self_path="scripts/patch_v0919_fix1.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.19-fix1 applied. v0.9.19 COMPLETE.")


if __name__ == "__main__":
    main()
