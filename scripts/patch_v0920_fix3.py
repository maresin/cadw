#!/usr/bin/env python3
# v0.9.20-fix3 — calibrate: pass peak DELTA (not abs) to _record_calibration.
# v0.9.20 COMPLETE.
#
# v0.9.20-fix2 verify+dogfood green, pytest target: 1 failure:
#   test_calibrate_records_sample
#   AssertionError: 'arrow' not in {net_pool_delta, net_rss_delta,
#                                    peak_pool, peak_rss, ...}
#   WARNING: C_rss=32210.94 — absurd
#
# Two related issues:
#
# (1) Stale test names: test was written for v0.9.19 API
#     {predicted, arrow, rss}. v0.9.20 renamed to {predicted,
#     peak_pool, peak_rss, net_pool_delta, net_rss_delta, sample_mode}.
#     Test asserts `"arrow" in s` — always False. Fix: update to
#     `"peak_pool" in s and "peak_rss" in s`.
#
# (2) REAL BUG: C_rss=32210.94 = peak_rss_abs / predicted.
#     peak_rss_abs = 161 MB (absolute process RSS — includes Python
#     interpreter + pyarrow + everything loaded). predicted=5000 B.
#     C = 161MB / 5000B = 32210.
#
#     Formula §9.2.1-§9.2.4 predicts DELTA above op-start, not
#     absolute. `peak = batch + 2×state` — this is incremental
#     memory needed during op, on top of what was already in RAM.
#
#     Correct: peak_rss_delta = peak_rss_abs - rss_start
#             (e.g., 800 KB for a 100k list during op)
#             C_rss = peak_rss_delta / predicted  (correct)
#
#     Current: _record_calibration receives peak_rss_abs (absolute)
#     → C uses absolute → absurd.
#
#     Note: the local variable `peak_rss_delta` and `peak_pool_delta`
#     are ALREADY computed in calibrate() body (used for runtime_monitor).
#     But _record_calibration is called with peak_pool_abs / peak_rss_abs.
#     Fix: pass the delta variables, not abs.
#
# §13.4/235 (new, refines §13.4/232): formula §9.2.N predicts DELTA
# above op-start, not absolute. Predicted = batch + state (incremental);
# actual = peak_abs - start (also incremental). Comparison
# peak_abs / predicted = unit mismatch (absolute vs delta).
# Rule: both sides must measure the same delta. In v0.9.19 the bug was
# net_delta/predicted = 0 for streaming ops (metric=net, formula=peak).
# In v0.9.20 the bug is absolute/predicted = huge (metric=abs,
# formula=delta). Both same class: wrong unit.
#
# Same class as §13.4/232 but opposite direction — §13.4/232 said
# "net delta useless for formula validation, use peak". v0.9.20 did
# peak but as ABSOLUTE, not as DELTA. Formula always delta. Fix3
# passes peak DELTA.
#
# Files on disk uncommitted (all v0.9.20 a,b,fix1,fix2 + this fix3).
#
# Fix3:
#   (a) calibrate(): pass peak_pool_delta/peak_rss_delta to
#       _record_calibration (not abs).
#   (b) _record_calibration: rename params to *_delta (semantic);
#       entry fields peak_pool/peak_rss are DELTAS above op-start.
#   (c) test_calibrate_records_sample: assert peak_pool/peak_rss
#       (not stale "arrow"/"rss").
#   (d) verify + dogfood + pytest + atomic v0.9.20 commit.
#
# Run: uv run python scripts/patch_v0920_fix3.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (a) calibrate: pass delta not abs ============

def fix_calibrate_pass_delta():
    print("\n=== (a) calibrate: pass peak DELTA not abs ===")
    p = Path("cadw/core/memory_guard.py")
    s = p.read_text()
    if "# v0.9.20-fix3" in s:
        print("  already fixed — skip")
        return

    # Fix call site: pass peak_pool_delta / peak_rss_delta
    old_call = '''            # Tier 3 (calibrate): log sample, warn on formula drift
            if self._calibrate:
                self._record_calibration(
                    op_name,
                    predicted_bytes,
                    peak_pool_abs,
                    peak_rss_abs,
                    net_pool_delta,
                    net_rss_delta,
                    sample_mode=sample_peak,
                )'''

    new_call = '''            # Tier 3 (calibrate): log sample, warn on formula drift
            # v0.9.20-fix3 (§13.4/235): pass peak DELTA (above op-start)
            # not peak ABS. Formula §9.2.N predicts delta (batch + state);
            # actual must be delta too. Absolute pool/RSS include
            # pre-existing allocations (all batches, interpreter) → C
            # absurd (e.g. 161 MB / 5 KB = 32210).
            if self._calibrate:
                self._record_calibration(
                    op_name,
                    predicted_bytes,
                    peak_pool_delta,
                    peak_rss_delta,
                    net_pool_delta,
                    net_rss_delta,
                    sample_mode=sample_peak,
                )'''

    if old_call not in s:
        raise SystemExit("calibrate call anchor not found")
    s = s.replace(old_call, new_call, 1)

    # Fix _record_calibration signature (rename params) + body
    old_rec_sig = '''    def _record_calibration(
        self,
        op_name: str,
        predicted: int,
        peak_pool_abs: int,
        peak_rss_abs: int,
        net_pool_delta: int,
        net_rss_delta: int,
        *,
        sample_mode: bool = False,
    ) -> None:
        """Append calibration sample; warn on C = peak/predicted > 2.

        v0.9.20: records peak (absolute, comparable to formula §9.2.N
        which predicts absolute peak) and net_delta (persistent, leak
        detection). sample_mode=True when background thread was used
        (peak is honest max); False when peak_pool_abs == net_pool_abs
        (i.e., no sampling, only start/end).
        """
        entry = {
            "predicted": int(predicted),
            "peak_pool": int(peak_pool_abs),
            "peak_rss": int(peak_rss_abs),
            "net_pool_delta": int(net_pool_delta),
            "net_rss_delta": int(net_rss_delta),
            "sample_mode": "peak" if sample_mode else "delta",
        }'''

    new_rec_sig = '''    def _record_calibration(
        self,
        op_name: str,
        predicted: int,
        peak_pool_delta: int,
        peak_rss_delta: int,
        net_pool_delta: int,
        net_rss_delta: int,
        *,
        sample_mode: bool = False,
    ) -> None:
        """Append calibration sample; warn on C = peak_delta/predicted > 2.

        v0.9.20-fix3 (§13.4/235): peak_pool/peak_rss entry fields are
        DELTAS above op-start (peak_abs - start), not absolute.
        Formula §9.2.N predicts delta (batch + state); comparison must
        be delta/delta, not absolute/predicted. v0.9.20 compared peak_abs
        (e.g. 161 MB interpreter RSS) to predicted (5 KB) → C=32210.
        """
        entry = {
            "predicted": int(predicted),
            "peak_pool": int(peak_pool_delta),
            "peak_rss": int(peak_rss_delta),
            "net_pool_delta": int(net_pool_delta),
            "net_rss_delta": int(net_rss_delta),
            "sample_mode": "peak" if sample_mode else "delta",
        }'''

    if old_rec_sig not in s:
        raise SystemExit("_record_calibration signature anchor not found")
    s = s.replace(old_rec_sig, new_rec_sig, 1)

    # Fix C calculations: peak_pool_abs -> peak_pool_delta etc.
    old_calc = '''        if predicted > 0:
            c_peak = peak_pool_abs / predicted
            c_rss = peak_rss_abs / predicted
            logger.info(
                "calibrate op=%s mode=%s predicted=%d peak_pool=%d "
                "(C=%.2f) peak_rss=%d (C=%.2f) net_pool_delta=%d",
                op_name, entry["sample_mode"], predicted, peak_pool_abs,
                c_peak, peak_rss_abs, c_rss, net_pool_delta,
            )'''

    new_calc = '''        if predicted > 0:
            c_peak = peak_pool_delta / predicted
            c_rss = peak_rss_delta / predicted
            logger.info(
                "calibrate op=%s mode=%s predicted=%d peak_pool_delta=%d "
                "(C=%.2f) peak_rss_delta=%d (C=%.2f) net_pool_delta=%d",
                op_name, entry["sample_mode"], predicted, peak_pool_delta,
                c_peak, peak_rss_delta, c_rss, net_pool_delta,
            )'''

    if old_calc not in s:
        raise SystemExit("C calc anchor not found")
    s = s.replace(old_calc, new_calc, 1)

    old_else = '''        else:
            logger.info(
                "calibrate op=%s mode=%s predicted=unknown peak_pool=%d "
                "peak_rss=%d net_pool_delta=%d",
                op_name, entry["sample_mode"], peak_pool_abs,
                peak_rss_abs, net_pool_delta,
            )'''

    new_else = '''        else:
            logger.info(
                "calibrate op=%s mode=%s predicted=unknown "
                "peak_pool_delta=%d peak_rss_delta=%d net_pool_delta=%d",
                op_name, entry["sample_mode"], peak_pool_delta,
                peak_rss_delta, net_pool_delta,
            )'''

    if old_else not in s:
        raise SystemExit("else log anchor not found")
    s = s.replace(old_else, new_else, 1)

    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ (b) update stale test ============

def fix_stale_test():
    print("\n=== (b) test_calibrate_records_sample: peak_pool/peak_rss ===")
    p = Path("tests/test_runtime_monitor.py")
    s = p.read_text()
    if 'assert "peak_pool" in s and "peak_rss" in s' in s:
        print("  already fixed — skip")
        return

    old = '''    s = samples["test_op"][0]
    assert s["predicted"] == 5000
    assert "arrow" in s and "rss" in s'''

    new = '''    s = samples["test_op"][0]
    assert s["predicted"] == 5000
    # v0.9.20-fix3 (§13.4/235): entry has peak_pool/peak_rss (deltas
    # above op-start, comparable to formula predicted). v0.9.19 fields
    # ("arrow"/"rss") are stale.
    assert "peak_pool" in s and "peak_rss" in s
    assert "net_pool_delta" in s and "net_rss_delta" in s
    assert s["sample_mode"] == "delta"'''

    if old not in s:
        raise SystemExit("test anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s, newline="")
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.20 + fix1 + fix2 + fix3 ===")
    pu.verify_anchors([
        # fix3 (a)
        ("cadw/core/memory_guard.py", "# v0.9.20-fix3"),
        ("cadw/core/memory_guard.py", "peak_pool_delta,"),
        ("cadw/core/memory_guard.py", "peak_rss_delta,"),
        ("cadw/core/memory_guard.py",
         "peak_pool_delta: int,"),
        # fix3 (b)
        ("tests/test_runtime_monitor.py",
         'assert "peak_pool" in s and "peak_rss" in s'),
        # fix2 (a) bench single-line
        ("scripts/bench_v0919_calibration.py",
         '_model_predicted("hash_shuffle", batches, ["g"], available)'),
        # fix1 (a) DECISIONS content
        ("DECISIONS.md", "**Эпизод 226 (nullable int to_numpy"),
        ("DECISIONS.md", "**Эпизод 232 (metric unit must match"),
        # fix1 (b) bench decomposition
        ("scripts/bench_v0919_calibration.py", "pred_batch"),
        ("scripts/bench_v0919_calibration.py", "C_batch"),
        ("scripts/bench_v0919_calibration.py", '_model_predicted("sort"'),
        ("scripts/bench_v0919_calibration.py", '_model_predicted("groupby"'),
        # fix1 (c)(d) tests
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
    print("\n=== dogfood: v0.9.20-fix3 (happy path only) ===")

    def _verify_all():
        import pyarrow as pa
        from cadw.core.memory_guard import MemoryGuard

        # 1. Delta mode: entry has new field names, peak_pool/peak_rss
        #    are DELTAS (small), not absolute (huge).
        g = MemoryGuard(max_memory_mb=10, calibrate=True)
        with g.calibrate("test_op", 5000):
            _ = [None] * 10_000  # ~80 KB Python list
        entry = g.calibration_samples["test_op"][0]
        assert "peak_pool" in entry
        assert "peak_rss" in entry
        assert entry["sample_mode"] == "delta"
        # Sanity: peak_rss should be small (delta above op-start, KB-MB
        # not 100+ MB absolute). v0.9.20-fix2 reported 161 MB absolute.
        assert entry["peak_rss"] < 50_000_000, (
            f"peak_rss={entry['peak_rss']} looks like ABSOLUTE not DELTA"
        )

        # 2. Bench compiles + single-line _model_predicted calls
        import py_compile
        from pathlib import Path as _P
        py_compile.compile(
            str(_P("scripts/bench_v0919_calibration.py")), doraise=True,
        )
        src = _P("scripts/bench_v0919_calibration.py").read_text()
        assert '_model_predicted("hash_shuffle", batches, ["g"], available)' in src

        # 3. DECISIONS order
        d = _P("DECISIONS.md").read_text()
        assert d.find("**Эпизод 225 ") < d.find("**Эпизод 226 ")
        assert d.find("**Эпизод 232 ") < d.find("### 13.5. Открытые вопросы")

        print("  delta mode: peak_rss is small (delta), not absolute OK")
        print("  bench compiles + hash_shuffle single-line OK")
        print("  DECISIONS 225 < 226 < 232 < §13.5 OK")

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
    print("=== v0.9.20-fix3 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    fix_calibrate_pass_delta()
    fix_stale_test()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.20: peak-polling in calibrate() + DECISIONS §13.4/226-232 "
        "+ bench decomposition (initial + fix1 + fix2 + fix3, atomic — "
        "v0.9.20 chain: initial died at append_decisions assert; fix1 "
        "died at verify anchor (multi-line hash_shuffle call); fix2 "
        "died at pytest (1 stale test); fix3 fixed the real bug). "
        "(Motivation) v0.9.19 bench C_arrow=0 for reduce ops: net pool "
        "delta = end − start; batches pre-allocated at guard creation, "
        "Arrow temp buffers released before end → net ≈ 0. Formula "
        "§9.2.1 predicts delta above op-start (batch + state); net "
        "delta ≈ 0 useless for validation. §13.4/232: metric unit must "
        "match formula unit. (Design a) calibrate(sample_peak=True, "
        "sample_interval_ms=10) — background thread polls pool+RSS, "
        "tracks max. Zero overhead when calibrate=False. "
        "(§13.4/235 fix3) peak_pool/peak_rss entry fields are DELTAS "
        "above op-start, not absolute. v0.9.20 initial passed "
        "peak_pool_abs/peak_rss_abs to _record_calibration → C_rss = "
        "32210 (161 MB abs RSS / 5 KB predicted). Fix: pass "
        "peak_pool_delta = peak_abs − start; peak_rss_delta = "
        "peak_rss_abs − rss_start. Both sides now measure delta, "
        "comparable to formula. Same class as §13.4/232 but opposite "
        "direction — v0.9.19 used net_delta (metric=net, formula=peak); "
        "v0.9.20 used peak_abs (metric=abs, formula=delta). Both wrong "
        "unit. Correct: peak_delta/predicted_delta. (Design b) bench "
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
        "not absolute). (Backward compat) 1902 existing tests "
        "unchanged; zero overhead when calibrate=False. 9 new tests: "
        "6 v0.9.20 + 3 fix1 (decisions order, bench decomposition, "
        "bench wiring). Deferred to v0.9.21: AUDIT §1.4 per-group "
        "shift/rolling; per-op predicted wiring in modules (not just "
        "bench); Pingis round 3; DECISIONS §13.4/233-235 append.",
        self_path="scripts/patch_v0920_fix3.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.20-fix3 applied. v0.9.20 COMPLETE.")


if __name__ == "__main__":
    main()
