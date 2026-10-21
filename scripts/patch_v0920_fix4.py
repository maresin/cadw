#!/usr/bin/env python3
# v0.9.20-fix4 — whitelist MemoryGuard.calibrate (threading.Thread.join).
# v0.9.20 COMPLETE.
#
# v0.9.20-fix3 verify+dogfood+target-pytest green, but FAST pytest 1
# failure:
#   test_no_unguarded_high_risk_arrow_calls:
#     cadw/core/memory_guard.py:540: `join()` in `MemoryGuard.calibrate`
#     without preceding check_state/check_batch/stat
#
# Root: AST guard HIGH_RISK_ATTRS includes "join" (for pa.Table.join).
# v0.9.20 added `sampler_thread.join(timeout=2.0)` — that's
# threading.Thread.join (stdlib, safe), NOT Table.join (C++ hash join,
# unbounded output materialization). AST matcher is attribute-name only
# → false positive.
#
# §13.4/236 (new, refines §13.4/229): generic method names (join,
# update, filter, compute) in HIGH_RISK_ATTRS give false positives on
# stdlib/threading. Options:
#   (a) whitelist per qualname with rationale — used in project
#   (b) receiver-type filter via AST — needs data flow, complex
#   (c) remove from HIGH_RISK_ATTRS — HIDES real Table.join violations
# Choose (a): whitelist MemoryGuard.calibrate with clear rationale
# "threading.Thread.join for sampler cleanup, not Table.join".
#
# Also relevant: since v0.9.19 introduced threading in calibrate,
# future threading usage in the same method must remain whitelisted
# by rationale. Rule: whitelist is BOTH permission AND documentation.
#
# Fix4:
#   (a) tests/test_arrow_call_memory_guard.py: add whitelist entry
#       "cadw/core/memory_guard.py:MemoryGuard.calibrate" with rationale.
#   (b) Test: test_calibrate_thread_join_whitelisted — AST-checks
#       whitelist key exists + rationale contains "threading.Thread.join"
#       (prevents silently removing the key and hiding Table.join).
#   (c) verify + dogfood + pytest fast + atomic v0.9.20 commit.
#
# Run: uv run python scripts/patch_v0920_fix4.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) whitelist MemoryGuard.calibrate ============

def whitelist_calibrate():
    print("\n=== (a) whitelist MemoryGuard.calibrate (threading.Thread.join) ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    key = '"cadw/core/memory_guard.py:MemoryGuard.calibrate"'
    if key in s:
        print("  already whitelisted — skip")
        return

    # Insert after _string_hash_column whitelist entry
    anchor = '''    "cadw/ops/shuffle.py:_string_hash_column":
        "dict_encode O(K + batch) where K = uniques-in-batch; caller "
        "hash_shuffle._process check_batch(extra_factor=3.0) covers "
        "peak ≤ 3×batch.nbytes; §13.4/226/227 null edges handled",'''

    new_entry = '''    "cadw/ops/shuffle.py:_string_hash_column":
        "dict_encode O(K + batch) where K = uniques-in-batch; caller "
        "hash_shuffle._process check_batch(extra_factor=3.0) covers "
        "peak ≤ 3×batch.nbytes; §13.4/226/227 null edges handled",
    # v0.9.20-fix4 (§13.4/236): MemoryGuard.calibrate calls
    # sampler_thread.join(timeout=2.0) — that's threading.Thread.join
    # (stdlib, safe daemon cleanup), NOT pa.Table.join (C++ hash join,
    # unbounded output materialization). AST guard matches by attribute
    # name only; Thread.join is safe. Sampler thread: daemon, 10ms
    # interval, join has 2s timeout. Rationale MUST keep the
    # "threading.Thread.join" phrase — test_calibrate_thread_join_whitelisted
    # asserts it to prevent silent removal (hiding real Table.join).
    "cadw/core/memory_guard.py:MemoryGuard.calibrate":
        "threading.Thread.join(timeout=2.0) for peak sampler daemon "
        "cleanup; NOT pa.Table.join; sampler bounded 10ms interval, "
        "2s timeout, daemon thread",'''

    if anchor not in s:
        raise SystemExit("whitelist anchor not found")
    s = s.replace(anchor, new_entry, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (b) regression test ============

def add_whitelist_regression_test():
    print("\n=== (b) test_arrow_call_memory_guard: calibrate whitelist test ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    if "def test_calibrate_thread_join_whitelisted" in s:
        print("  already present — skip")
        return

    extra = '''

# ============ v0.9.20-fix4 (§13.4/236): calibrate whitelist ============


def test_calibrate_thread_join_whitelisted():
    """MemoryGuard.calibrate whitelisted for threading.Thread.join.

    v0.9.20 introduced `sampler_thread.join(timeout=2.0)` in calibrate
    for peak sampler cleanup. AST guard matches by attribute name only
    ("join"), so this benign stdlib call is flagged. Whitelist entry
    exists with rationale; this test asserts:
      (a) key present in WHITELISTED_FUNCS
      (b) rationale mentions "threading.Thread.join" — proves the
          entry is for the benign stdlib call, not papering over a
          real pa.Table.join violation elsewhere in calibrate.
    """
    key = "cadw/core/memory_guard.py:MemoryGuard.calibrate"
    assert key in WHITELISTED_FUNCS, (
        f"{key!r} missing from whitelist — MemoryGuard.calibrate has "
        f"threading.Thread.join(); AST guard false-positives without "
        f"whitelist. See §13.4/236."
    )
    rationale = WHITELISTED_FUNCS[key]
    assert "threading.Thread.join" in rationale, (
        f"{key!r} rationale must mention 'threading.Thread.join' to "
        f"distinguish from pa.Table.join. Rationale: {rationale!r}"
    )
    assert "NOT pa.Table.join" in rationale or "not pa.Table.join" in rationale, (
        f"{key!r} rationale must state NOT pa.Table.join "
        f"(otherwise may hide real C++ hash join violations). "
        f"Rationale: {rationale!r}"
    )


def test_calibrate_uses_thread_join_not_table_join():
    """AST: calibrate calls .join() on sampler_thread, not on pa.Table.

    Verifies that the whitelist is genuine: calibrate really uses
    threading.Thread.join, not pa.Table.join somewhere.
    """
    import ast as _ast
    import inspect as _inspect
    import textwrap as _tw
    from cadw.core.memory_guard import MemoryGuard

    src = _tw.dedent(_inspect.getsource(MemoryGuard.calibrate))
    tree = _ast.parse(src)
    join_receivers = []
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Call)
                and isinstance(node.func, _ast.Attribute)
                and node.func.attr == "join"):
            recv = node.func.value
            if isinstance(recv, _ast.Name):
                join_receivers.append(recv.id)
            else:
                join_receivers.append(_ast.unparse(recv))
    assert join_receivers, "expected at least one .join() call in calibrate"
    for recv in join_receivers:
        assert "sampler" in recv.lower() or "thread" in recv.lower(), (
            f"calibrate .join() receiver {recv!r} is not thread-like — "
            f"whitelist may hide real Table.join. Receivers: {join_receivers}"
        )
'''

    p.write_text(s + extra, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.20 + fix1 + fix2 + fix3 + fix4 ===")
    pu.verify_anchors([
        # fix4 (a)
        ("tests/test_arrow_call_memory_guard.py",
         '"cadw/core/memory_guard.py:MemoryGuard.calibrate"'),
        ("tests/test_arrow_call_memory_guard.py",
         "threading.Thread.join(timeout=2.0) for peak sampler"),
        # fix4 (b)
        ("tests/test_arrow_call_memory_guard.py",
         "def test_calibrate_thread_join_whitelisted"),
        ("tests/test_arrow_call_memory_guard.py",
         "def test_calibrate_uses_thread_join_not_table_join"),
        # fix3
        ("cadw/core/memory_guard.py", "# v0.9.20-fix3"),
        ("cadw/core/memory_guard.py", "peak_pool_delta: int,"),
        ("tests/test_runtime_monitor.py",
         'assert "peak_pool" in s and "peak_rss" in s'),
        # fix2
        ("scripts/bench_v0919_calibration.py",
         '_model_predicted("hash_shuffle", batches, ["g"], available)'),
        # fix1
        ("DECISIONS.md", "**Эпизод 226 (nullable int to_numpy"),
        ("DECISIONS.md", "**Эпизод 232 (metric unit must match"),
        ("scripts/bench_v0919_calibration.py", "pred_batch"),
        ("scripts/bench_v0919_calibration.py", "C_batch"),
        ("tests/test_changelog.py", "def test_decisions_226_232_order"),
        # v0.9.20 (a)
        ("cadw/core/memory_guard.py", "sample_peak: bool = False"),
        ("cadw/core/memory_guard.py", "peak_pool_box"),
        # v0.9.19/18/17/16 regression
        ("cadw/core/memory_guard.py", "def calibrate("),
        ("cadw/ops/reduce.py", "with memory_guard.calibrate("),
        ("cadw/ops/shuffle.py", "§13.4/227"),
        ("cadw/dataset/index.py", "v0.9.18 (AUDIT §1.6)"),
        ("DECISIONS.md", "**Эпизод 178 (guard, вакуумный"),
        ("DECISIONS.md", "**Эпизод 225 (static guard scope"),
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.20-fix4 (happy path only) ===")

    def _verify_all():
        # 1. Whitelist entry present with required rationale phrases
        from tests.test_arrow_call_memory_guard import WHITELISTED_FUNCS
        key = "cadw/core/memory_guard.py:MemoryGuard.calibrate"
        assert key in WHITELISTED_FUNCS, f"whitelist missing {key}"
        r = WHITELISTED_FUNCS[key]
        assert "threading.Thread.join" in r
        assert "NOT pa.Table.join" in r or "not pa.Table.join" in r

        # 2. AST guard now passes (no false-positive)
        from tests.test_arrow_call_memory_guard import (
            _collect_violations,
        )
        violations = _collect_violations()
        assert not violations, (
            f"AST guard still has violations: {violations}"
        )

        print("  whitelist MemoryGuard.calibrate with threading.Thread.join rationale OK")
        print("  AST guard: no violations OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target (guard + runtime) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_runtime_monitor.py",
         "tests/test_changelog.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGET FAILURE ***")
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
    print("=== v0.9.20-fix4 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    whitelist_calibrate()
    add_whitelist_regression_test()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.20: peak-polling in calibrate() + DECISIONS §13.4/226-232 "
        "+ bench decomposition + calibrate whitelist (initial + fix1 + "
        "fix2 + fix3 + fix4, atomic — v0.9.20 chain: initial died at "
        "append_decisions assert; fix1 died at verify anchor (multi-line "
        "hash_shuffle call); fix2 died at pytest (1 stale test); fix3 "
        "fixed real peak_abs vs peak_delta bug; fix4 died at pytest "
        "fast (AST guard false-positive on threading.Thread.join). "
        "(Motivation) v0.9.19 bench C_arrow=0 for reduce ops: net "
        "pool delta ≈ 0 for streaming (batches pre-allocated at guard "
        "creation, Arrow temp buffers released before end). Formula "
        "§9.2.1 predicts delta above op-start (batch + state); net "
        "delta useless for validation. §13.4/232: metric unit must "
        "match formula unit. (Design a) calibrate(sample_peak=True, "
        "sample_interval_ms=10) — background thread polls pool+RSS, "
        "tracks max. Zero overhead when calibrate=False. "
        "(§13.4/235 fix3) peak_pool/peak_rss entry fields are DELTAS "
        "above op-start, not absolute. v0.9.20 initial passed "
        "peak_pool_abs/peak_rss_abs → C_rss=32210 (161 MB abs RSS / "
        "5 KB predicted). Fix: pass peak_delta; both sides delta. "
        "Same class as §13.4/232 opposite direction. (Design b) bench "
        "decomposition: reduce pred_batch + pred_state; sort/shuffle/"
        "groupby wired via model.solve()+estimate(); C_batch = "
        "peak/batch. Columns: op, pred_batch, pred_state, pred_total, "
        "batch, peak_pool, peak_rss, C_peak, C_batch. (DECISIONS) "
        "§13.4/226-232 append: 226-232 (nullable int to_numpy → "
        "float64, dict_encode all-null → K=0, dogfood happy-path only, "
        "numpy == coerces label, write_content syntactic units, helper "
        "methods caller-facing, metric unit formula unit). (Fix chain) "
        "(1) append_decisions anchor = 'Gap 190–210' (present) with "
        "fallback to §13.5; gap EXISTS → insert before gap → wrong "
        "order (§13.4/233: check ALL anchors, pick LATEST). (2) verify "
        "anchor '_model_predicted(\"hash_shuffle\"' multi-line call "
        "(§13.4/234: contiguous-text anchor needs single-line). "
        "(3) peak_abs vs predicted (§13.4/235: formula is delta, "
        "not absolute). (4) AST guard 'join' in HIGH_RISK_ATTRS "
        "matches threading.Thread.join too (§13.4/236: generic method "
        "names need per-qualname whitelist with rationale, NOT removal "
        "from HIGH_RISK_ATTRS — removal would hide real pa.Table.join). "
        "Whitelist MemoryGuard.calibrate with rationale mentioning "
        "'threading.Thread.join' AND 'NOT pa.Table.join'; regression "
        "test asserts both phrases AND that calibrate .join() receivers "
        "are thread-like (sampler_thread/thread), not pa.Table. "
        "(Backward compat) 1904 existing tests unchanged; zero overhead "
        "when calibrate=False. 11 new tests: 6 v0.9.20 (peak mode, "
        "delta mode, spike capture, thread joined, monitor peak, bench "
        "sample_peak) + 3 fix1 (decisions order, bench decomposition, "
        "bench wiring) + 2 fix4 (calibrate whitelist, calibrate "
        "threading.join receivers). Deferred to v0.9.21: AUDIT §1.4 "
        "per-group shift/rolling; per-op predicted wiring in modules "
        "(not just bench); Pingis round 3; DECISIONS §13.4/233-236 "
        "append.",
        self_path="scripts/patch_v0920_fix4.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.20-fix4 applied. v0.9.20 COMPLETE.")


if __name__ == "__main__":
    main()
