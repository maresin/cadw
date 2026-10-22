#!/usr/bin/env python3
# v0.9.25d-b1-fix3 — revert b1's accidental bump of pivot.stream_sorted.
#
# STATE (on-disk, HEAD=v0.9.25d-a; b1+fix1+fix2 uncommitted):
#   cadw/dataframe.py: 4 bumps 7.0 (intentional — Tier B)
#   cadw/ops/metrics.py: 1 bump 7.0 (intentional — Tier B)
#   cadw/ops/pivot.py: scan_columns 7.0 (fix1 — correct),
#                      stream_sorted 7.0 (b1 — ACCIDENTAL)
#   tests/test_v0925d_a.py: test_low_factor_zero_after_bumps +
#                           categories sanity updated
#
# FAILURE (fix2's pytest gate):
#   tests/test_extra_factor_sites.py::test_check_batch_has_extra_factor[
#     pivot.stream_sorted-5.0]
#   Asserts exact literal 5.0 (v0.9.22b contract); sees 7.0.
#
# ROOT: b1's _bump_factor per-function heuristic (bug §13.4/253)
#   picked max(check_batch_before_any_pylist, key=lineno) = stream_sorted
#   (line 88), NOT scan_columns (line 41). Bumped stream_sorted
#   5.0 → 7.0 by accident. Was marked "harmless over-guard" in b1's
#   commit message — INCORRECT: any literal value change breaks the
#   hardcoded-SITES test.
#
# FIX: revert stream_sorted 7.0 → 5.0 (restore v0.9.22b intent).
#   scan_columns 7.0 stays (fix1's real fix — was absent → default 1.0).
#
# §13.4/254 (NEW): "harmless over-guard" ≠ harmless when test hardcodes
#   literal values. Any check_batch factor change (even conservative)
#   breaks test_extra_factor_sites. Rule: either (a) don't bump existing
#   factor without updating SITES in same patch, or (b) redesign test
#   to assert invariant (extra_factor >= 3.0) rather than literal value
#   (backlog §11.6).
#
# Fix-cycle count this v0.9.25d-b1 effort: b1(initial)→fix1(pivot.scan)
# →fix2(categories stale test)→fix3(revert stream_sorted) = 4 cycles,
# budget was 2. Every cycle caught by a specific gate (verify_after /
# pytest) BEFORE commit — zero bad-commits. Actual cadw/ changes
# (Tier B bumps 7.0) intact. This is diagnostic-first working as
# designed, not churn.
#
# Run: uv run python scripts/patch_v0925d_b1_fix3_revert_stream.py

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    run_verify_in_subprocess, check_no_invalid_escape,
)


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25d-a; on-disk b1+fix1+fix2; "
          "stream_sorted=7.0 (needs revert) ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25d-a" not in log:
        raise SystemExit(f"expected HEAD=v0.9.25d-a; got {log[:200]!r}")
    if "v0.9.25d-b1" in log:
        raise SystemExit("v0.9.25d-b1 already committed — fix3 misplaced")

    # Diagnostic: low_factor=0 (b1+fix1+fix2 on-disk state is correct)
    import importlib.util as _iu
    spec = _iu.spec_from_file_location(
        "diag", "scripts/service/diagnose_unguarded_to_pylist.py",
    )
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.scan()
    lf = r.get("low_factor", [])
    print(f"  diagnostic: low_factor={len(lf)}, "
          f"unguarded={len(r.get('unguarded', []))}, "
          f"guarded={len(r.get('guarded', []))}")
    if lf:
        raise SystemExit(
            f"low_factor={len(lf)} != 0; fix1 not fully applied? "
            f"Sites: {[(s['file'], s['line']) for s in lf]}"
        )

    # stream_sorted at 7.0 (bug we're fixing)
    src = Path("cadw/ops/pivot.py").read_text()
    if 'name="pivot.stream_sorted", extra_factor=7.0' not in src:
        raise SystemExit(
            'expected pivot.stream_sorted at extra_factor=7.0 (b1 bug); '
            'already reverted? Inspect cadw/ops/pivot.py manually'
        )
    if 'name="pivot.scan_columns", extra_factor=7.0' not in src:
        raise SystemExit(
            'expected pivot.scan_columns at extra_factor=7.0 (fix1); '
            'fix1 not fully applied?'
        )
    print("  on-disk state correct for fix3 (stream=7.0, scan=7.0)")


# ============ 1. revert stream_sorted ============

def revert_stream_sorted():
    print("\n=== revert pivot.stream_sorted extra_factor 7.0 → 5.0 ===")
    applied = apply_or_fail(
        "cadw/ops/pivot.py",
        'b, name="pivot.stream_sorted", extra_factor=7.0,',
        'b, name="pivot.stream_sorted", extra_factor=5.0,',
        label="revert_stream_sorted",
    )
    if applied:
        assert_compiles("cadw/ops/pivot.py")
        check_no_invalid_escape("cadw/ops/pivot.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 2. verify diagnostic still 0 (both >= 3.0) ============

def verify_after():
    print("\n=== verify: diagnostic low_factor=0 after revert "
          "(5.0 >= 3.0 → guarded) ===")

    def _verify():
        import importlib.util as _iu
        from pathlib import Path as _P
        spec = _iu.spec_from_file_location(
            "diag", _P("scripts/service/diagnose_unguarded_to_pylist.py"),
        )
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        r = mod.scan()
        lf = r.get("low_factor", [])
        if lf:
            msg = "\n".join(
                f"  {s['file']}:{s['line']} {s['qualname']}"
                for s in lf
            )
            raise AssertionError(
                f"low_factor={len(lf)}, expected 0 after revert:\n{msg}"
            )
        # spot-check both pivot sites still guarded
        guarded_names = {(s["file"], s["qualname"]) for s in r["guarded"]}
        assert ("cadw/ops/pivot.py", "pivot_to_disk") in guarded_names, (
            "pivot_to_disk not in guarded after revert — classifier bug?"
        )
        print(
            f"  low_factor=0, unguarded={len(r.get('unguarded', []))}, "
            f"guarded={len(r.get('guarded', []))}, "
            f"whitelisted={len(r.get('whitelisted', []))}"
        )

    run_verify_in_subprocess(_verify)


# ============ 3. pytest ============

def run_pytest():
    print("\n=== pre-flight: test_facade_has_no_as_py ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_dataframe.py::test_facade_has_no_as_py", "-q"],
        timeout=60,
    )
    if r.returncode != 0:
        print("\n*** FACADE AS_PY VIOLATION — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: test_extra_factor_sites (the failing test) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_extra_factor_sites.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** test_extra_factor_sites STILL FAILING — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: v0925d_a + pivot + affected ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925d_a.py",
         "tests/test_pivot_arrow_native.py",
         "tests/test_pivot_table.py",
         "tests/test_pivot_table_e2e.py",
         "tests/test_shift_on_col.py",
         "tests/test_rolling_on_col.py",
         "tests/test_groupby_per_group.py",
         "tests/test_metrics.py",
         "tests/test_metrics_e2e.py",
         "-q", "-m", "not slow",
         "-W", "error::SyntaxWarning"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** TARGETED FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast (full) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 4. main ============

def main():
    print("=== v0.9.25d-b1-fix3: revert pivot.stream_sorted over-guard ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    revert_stream_sorted()
    verify_after()
    run_pytest()

    # Single commit: b1's on-disk + fix1 pivot.scan + fix2 test + fix3 revert
    commit_with_guards(
        "v0.9.25d-b1: factor bumps (Tier B) — complete "
        "(b1 + fix1 + fix2 + fix3). "
        "Tier B closes low_factor: 12 sites across 6 functions "
        "bumped to extra_factor=7.0. "
        "Final bumped sites: _per_group_shift, _per_group_rolling, "
        "_rolling_on, _per_group_apply (cadw/dataframe.py); "
        "_compute_roc_auc_binary (cadw/ops/metrics.py); "
        "pivot.scan_columns (cadw/ops/pivot.py). "
        "pivot.stream_sorted preserved at 5.0 (v0.9.22b — reverted "
        "from b1's accidental 7.0). "
        "Diagnostic after: low_factor=0, guarded=27, unguarded=23 "
        "(Tier C — v0.9.25d-b2 target), whitelisted=33, unknown=0, "
        "other-guard=0. "
        "Fix-cycle sequence (4, budget 2 — zero bad-commits, every "
        "cycle caught by gate pre-commit): "
        "b1(initial) bumped 5/6 functions, verify_after found "
        "pivot.scan_columns still 1.0 → SystemExit; "
        "fix1 bumped pivot.scan_columns 1.0→7.0, added "
        "test_low_factor_zero_after_bumps, verify_after passed, "
        "pytest failed on stale test_categories_present_sanity "
        "(asserted low_factor non-empty, b1 closes it); "
        "fix2 updated categories test (exclude low_factor), verify "
        "passed, pytest failed on test_extra_factor_sites expecting "
        "stream_sorted=5.0 but b1 accidentally bumped to 7.0; "
        "fix3 reverted stream_sorted 7.0→5.0 (b1's bump unintended), "
        "all gates pass. "
        "§13.4/253 (3rd occurrence): per-container heuristic fails "
        "when container holds multiple heterogeneous entities "
        "(_bump_factor per function fails on pivot_to_disk's 2 "
        "check_batch calls). "
        "§13.4/144 (4th occurrence): stale-test-on-refactor — "
        "fix1's scope-narrowing missed test_categories_present_sanity. "
        "§13.4/254 (NEW): 'harmless over-guard' ≠ harmless when test "
        "hardcodes literal value. Any factor change breaks "
        "test_extra_factor_sites (SITES parametrize with exact "
        "values). Rule: bump + update SITES same patch, OR redesign "
        "test to assert invariant (extra_factor >= 3.0) rather than "
        "literal — backlog §11.6. "
        "Tests: +1 (test_low_factor_zero_after_bumps, fix1); "
        "test_categories_present_sanity updated (3 categories, was 4). "
        "Fast suite: ~2297 → ~2298.",
        self_path="scripts/patch_v0925d_b1_fix3_revert_stream.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25d-b1 complete (b1 + fix1 + fix2 + fix3). "
          "Next: v0.9.25d-b2 (Tier C — 23 unguarded).")


if __name__ == "__main__":
    main()