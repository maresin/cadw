#!/usr/bin/env python3
# v0.9.20-fix2 — bench hash_shuffle call: multi-line → single-line.
# v0.9.20 COMPLETE.
#
# v0.9.20-fix1 files on disk UNCOMMITTED (verify failed):
#   cadw/core/memory_guard.py       ✓ (v0.9.20 initial a)
#   scripts/bench_v0919_calibration.py ✓ v2 but hash_shuffle call multi-line
#   DECISIONS.md                    ✓ 226-232 moved (correct order)
#   tests/test_changelog.py         ✓ 226-232 order test
#   tests/test_runtime_monitor.py   ✓ 2 decomposition tests
#   git commit                       ✗ never reached
#
# Root cause: verify anchor '_model_predicted("hash_shuffle"' expects
# single-line call, but BENCH_V2 has it multi-line:
#   pred_shuf = _model_predicted(
#       "hash_shuffle", batches, ["g"], available,
#   )
# Sort and groupby happen to be single-line → they pass.
#
# §13.4/234 (new): anchor на contiguous text (func("arg")) не работает
# если func и arg на разных строках. Bench style inconsistency —
# sort/groupby single-line, hash_shuffle multi-line. Fix: normalize
# bench — all three single-line. Prevents similar future regressions.
#
# Fix2:
#   (a) bench: hash_shuffle call multi-line → single-line (normalize
#       with sort/groupby).
#   (b) verify + dogfood + pytest + atomic v0.9.20 commit.
#
# Run: uv run python scripts/patch_v0920_fix2.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) bench: normalize hash_shuffle call ============

def normalize_bench_hash_shuffle():
    print("\n=== (a) bench: hash_shuffle call single-line ===")
    p = Path("scripts/bench_v0919_calibration.py")
    s = p.read_text()
    single = (
        '    pred_shuf = _model_predicted'
        '("hash_shuffle", batches, ["g"], available)'
    )
    if single in s:
        print("  already single-line — skip")
        return
    multi = (
        '    pred_shuf = _model_predicted(\n'
        '        "hash_shuffle", batches, ["g"], available,\n'
        '    )'
    )
    if multi not in s:
        # Show context for debugging
        lines = s.split("\n")
        for i, ln in enumerate(lines):
            if "hash_shuffle" in ln and "pred" in ln:
                for j in range(max(0, i-1), min(len(lines), i+3)):
                    print(f"    L{j+1}: {lines[j]!r}")
        raise SystemExit("multi-line hash_shuffle anchor not found")
    s = s.replace(multi, single, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK (single-line now)")

    # Also verify sort and groupby are already single-line
    assert '_model_predicted("sort", batches, [], available)' in s
    assert '_model_predicted("groupby", batches, ["g"], available)' in s
    print("  sort + groupby confirmed single-line")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.20 + fix1 + fix2 + regression ===")
    pu.verify_anchors([
        # fix2 (a) bench single-line
        ("scripts/bench_v0919_calibration.py",
         '_model_predicted("hash_shuffle", batches, ["g"], available)'),
        # v0.9.20-fix1 (a) DECISIONS content
        ("DECISIONS.md", "**Эпизод 226 (nullable int to_numpy"),
        ("DECISIONS.md", "**Эпизод 232 (metric unit must match"),
        # v0.9.20-fix1 (b) bench decomposition columns
        ("scripts/bench_v0919_calibration.py", "pred_batch"),
        ("scripts/bench_v0919_calibration.py", "pred_state"),
        ("scripts/bench_v0919_calibration.py", "pred_total"),
        ("scripts/bench_v0919_calibration.py", "C_batch"),
        ("scripts/bench_v0919_calibration.py", '_model_predicted("sort"'),
        ("scripts/bench_v0919_calibration.py", '_model_predicted("groupby"'),
        # v0.9.20-fix1 (c)(d) tests
        ("tests/test_changelog.py", "def test_decisions_226_232_order"),
        ("tests/test_runtime_monitor.py",
         "def test_bench_has_decomposed_columns"),
        # v0.9.20 (a) calibrate peak
        ("cadw/core/memory_guard.py", "sample_peak: bool = False"),
        ("cadw/core/memory_guard.py", "peak_pool_box"),
        # v0.9.19 regression
        ("cadw/core/memory_guard.py", "def calibrate("),
        ("cadw/ops/reduce.py", "with memory_guard.calibrate("),
        # v0.9.18/17/16
        ("cadw/ops/shuffle.py", "§13.4/227"),
        ("cadw/dataset/index.py", "v0.9.18 (AUDIT §1.6)"),
        ("DECISIONS.md", "**Эпизод 178 (guard, вакуумный"),
        ("DECISIONS.md", "**Эпизод 225 (static guard scope"),
        ("CHANGELOG.md", "## [0.9.14] — 2026-10-21"),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.20-fix2 (happy path only) ===")

    def _verify_all():
        from pathlib import Path as _P

        # 1. DECISIONS order
        text = _P("DECISIONS.md").read_text()
        idx_225 = text.find("**Эпизод 225 ")
        idx_226 = text.find("**Эпизод 226 ")
        idx_232 = text.find("**Эпизод 232 ")
        idx_135 = text.find("### 13.5. Открытые вопросы")
        assert 0 <= idx_225 < idx_226 < idx_232 < idx_135, (
            f"DECISIONS order broken: 225={idx_225} 226={idx_226} "
            f"232={idx_232} §13.5={idx_135}"
        )

        # 2. Bench compiles + all three _model_predicted single-line
        import py_compile
        bench = _P("scripts/bench_v0919_calibration.py")
        py_compile.compile(str(bench), doraise=True)
        src = bench.read_text()
        assert "pred_batch" in src
        assert "pred_state" in src
        assert "C_batch" in src
        assert 'def _dc_from_batches' in src
        assert 'def _model_predicted' in src
        # All three single-line (fix2)
        assert '_model_predicted("sort", batches, [], available)' in src
        assert '_model_predicted("hash_shuffle", batches, ["g"], available)' in src
        assert '_model_predicted("groupby", batches, ["g"], available)' in src

        print("  DECISIONS order 225 < 226 < 232 < §13.5 OK")
        print("  bench compiles + all 3 _model_predicted single-line OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py", "tests/test_runtime_monitor.py", "-q"],
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
    print("=== v0.9.20-fix2 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    normalize_bench_hash_shuffle()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.20: peak-polling in calibrate() + DECISIONS §13.4/226-232 "
        "append + bench decomposition (initial + fix1 + fix2, atomic — "
        "v0.9.20 initial died at append_decisions assert; fix1 died at "
        "verify anchor '_model_predicted(\"hash_shuffle\"' because "
        "bench had that call multi-line while sort/groupby single-line). "
        "(Motivation) v0.9.19 bench C_arrow=0 for reduce — net pool "
        "delta = end − start; batches pre-allocated at guard creation, "
        "Arrow temp buffers released before end → net ≈ 0. Formula "
        "§9.2.1 predicts absolute peak; net useless for validation. "
        "§13.4/232: metric unit must match formula unit (peak vs peak). "
        "(Design a) `calibrate(sample_peak=True, sample_interval_ms=10)` "
        "— background thread polls pool+RSS, tracks max. Records "
        "peak_pool/peak_rss (absolute, comparable to formula) AND "
        "net_pool_delta (leak). Zero overhead when calibrate=False. "
        "(Design b) Bench decomposition — user feedback: 'why only "
        "batch, no aux object sizes?'. Reduce ops: pred_batch + "
        "pred_state decomposition; sort/shuffle/groupby wired via "
        "model.solve()+estimate() for pred_total; C_batch = peak/batch "
        "(N× batch memory transparency). Columns: op, pred_batch, "
        "pred_state, pred_total, batch, peak_pool, peak_rss, C_peak, "
        "C_batch. (DECISIONS) §13.4/226-232 appended: 226 (nullable "
        "int to_numpy → float64), 227 (dict_encode all-null → K=0), "
        "228 (dogfood = happy path only), 229 (numpy == coerces label "
        "to dtype), 230 (write_content lines — complete syntactic "
        "units), 231 (helper methods are caller-facing), 232 (metric "
        "unit must match formula unit). (Fix chain) Initial: "
        "append_decisions used anchor = '**Gap 190–210' (present "
        "v0.9.17) with fallback to §13.5; gap-note EXISTS → 226-232 "
        "inserted BEFORE it → order became 189..226-232..gap..211-225. "
        "Assert 225<226 failed. §13.4/233: append with fallback anchor "
        "must check ALL candidates; pick LATEST by position. Fix1: "
        "extract + move 226-232 before §13.5; bench v2 with decomposition. "
        "Fix1 died at verify: anchor '_model_predicted(\"hash_shuffle\"' "
        "expected single-line but bench had multi-line "
        "_model_predicted(\\n \"hash_shuffle\"...). §13.4/234: anchor на "
        "contiguous text (func(\"arg\")) fails if func and arg on "
        "different lines; style inconsistency between sort/groupby "
        "(single) and hash_shuffle (multi). Fix2: normalize all three "
        "to single-line. (Backward compat) 1902 existing tests "
        "unchanged; zero overhead when calibrate=False. New tests: "
        "6 v0.9.20 (peak mode, delta mode, spike capture, thread "
        "joined, monitor peak, bench sample_peak) + 3 fix1 (decisions "
        "order, bench decomposition, bench model wiring) = 9 new. "
        "Deferred to v0.9.21: AUDIT §1.4 per-group shift/rolling; "
        "per-op predicted wiring in sort/shuffle/groupby modules (not "
        "just bench); Pingis round 3 (ML/pipeline/cv/metrics); "
        "DECISIONS §13.4/233-234 append.",
        self_path="scripts/patch_v0920_fix2.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.20-fix2 applied. v0.9.20 COMPLETE.")


if __name__ == "__main__":
    main()
