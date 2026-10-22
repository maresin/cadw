#!/usr/bin/env python3
# v0.9.25d-b1-fix1 — complete b1 (uncommitted partial) + fix pivot.scan.
#
# PRIOR STATE (before this patch):
#   HEAD = v0.9.25d-a (7f03552). b1 was NEVER committed:
#   apply_bumps() → 6 groups "OK" (bumped 12 sites), but
#   verify_after() → 1 residual low_factor (pivot.scan_columns:141)
#   → raise SystemExit BEFORE add_tests/dogfood/commit.
#   On-disk (uncommitted, from b1's partial run):
#     cadw/dataframe.py: _per_group_shift / _per_group_rolling /
#                        _rolling_on / _per_group_apply (extra_factor=7.0)
#     cadw/ops/metrics.py: _compute_roc_auc_binary (extra_factor=7.0)
#     cadw/ops/pivot.py: pivot.stream_sorted (5.0 → 7.0)
#   Not on-disk: pivot.scan_columns (still 1.0); test added by b1.
#
# THIS FIX (v0.9.25d-b1-fix1):
#   1. Verify disk state = b1's partial (low_factor=1, pivot only).
#   2. Bump pivot.scan_columns 1.0 → 7.0.
#   3. Add test_low_factor_zero_after_bumps to tests/test_v0925d_a.py.
#   4. Dogfood: scan() → low_factor=0.
#   5. pytest + commit ALL on-disk work (b1 + fix1) as ONE commit.
#
# Root of b1's bug (§13.4/253, 3rd occurrence: v0.9.9-fix1c,
# v0.9.25c-fix5, v0.9.25d-b1):
#   b1's _bump_factor picked `max(candidates, key=lineno)` — latest
#   check_batch before ANY to_pylist (per-function). pivot_to_disk has
#   TWO check_batch calls: (a) scan_columns (before columns_seen pylist,
#   line 141), (b) stream_sorted=5.0 (before 3 pylist in streaming loop,
#   v0.9.22b). _bump_factor picked (b), bumped 5.0 → 7.0 (harmless),
#   left (a) at 1.0. Classifier re-classifies line-141 pylist as
#   low_factor because its latest-preceding check_batch is (a).
#   Correct heuristic: iterate per to_pylist (per element), not per
#   function (per container). b1's _bump_factor retired.
#
# Run: uv run python scripts/patch_v0925d_b1_fix1_pivot_scan.py

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


# ============ 0. verify disk state (b1 uncommitted + pivot missing) ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25d-a, b1's on-disk changes present, "
          "pivot.scan_columns still default ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25d-a" not in log:
        raise SystemExit(f"expected HEAD=v0.9.25d-a; got {log[:200]!r}")
    if "v0.9.25d-b1" in log:
        raise SystemExit("v0.9.25d-b1 already committed — fix1 misplaced")

    # Verify b1's on-disk changes via diagnostic: low_factor should be 1
    # (pivot only), unguarded ~23.
    import importlib.util as _iu
    spec = _iu.spec_from_file_location(
        "diag", "scripts/service/diagnose_unguarded_to_pylist.py",
    )
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.scan()
    lf = r.get("low_factor", [])
    ung = r.get("unguarded", [])
    print(f"  diagnostic: low_factor={len(lf)}, unguarded={len(ung)}, "
          f"guarded={len(r.get('guarded', []))}, "
          f"whitelisted={len(r.get('whitelisted', []))}")
    if len(lf) != 1:
        raise SystemExit(
            f"expected low_factor=1 (b1 partial: 11/12 bumped, "
            f"pivot.scan residual), got {len(lf)}. On-disk state differs "
            f"from expected. Sites: "
            f"{[(s['file'], s['line']) for s in lf]}"
        )
    site = lf[0]
    if "pivot" not in site["file"]:
        raise SystemExit(
            f"expected pivot site as remaining low_factor, got {site}"
        )
    if site["qualname"] != "pivot_to_disk":
        raise SystemExit(
            f"expected qualname='pivot_to_disk', got {site['qualname']!r}"
        )
    if 18 > len(ung) or len(ung) > 28:
        raise SystemExit(
            f"unguarded={len(ung)}, expected ~23 (Tier C, unchanged by b1)"
        )

    # pivot.scan_columns still default
    src = Path("cadw/ops/pivot.py").read_text()
    if 'name="pivot.scan_columns", extra_factor=' in src:
        raise SystemExit("pivot.scan_columns already bumped — fix1 misplaced")
    if 'name="pivot.scan_columns")' not in src:
        raise SystemExit("pivot.scan_columns anchor not found")
    print("  pivot.scan_columns still default (1.0) — OK")


# ============ 1. bump pivot.scan_columns ============

def bump():
    print("\n=== bump pivot.scan_columns 1.0 → 7.0 ===")
    applied = apply_or_fail(
        "cadw/ops/pivot.py",
        'memory_guard.check_batch(b, name="pivot.scan_columns")',
        'memory_guard.check_batch(b, name="pivot.scan_columns", '
        'extra_factor=7.0)',
        label="pivot_scan_columns_ef",
    )
    if applied:
        assert_compiles("cadw/ops/pivot.py")
        check_no_invalid_escape("cadw/ops/pivot.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 2. add test (b1's missing test) ============

NEW_TEST = '''

def test_low_factor_zero_after_bumps():
    """v0.9.25d-b1: all low_factor sites bumped to >= 3.0.

    After v0.9.25d-b1, no site has check_batch with factor < 3.0
    guarding to_pylist. Remaining unguarded (Tier C) — v0.9.25d-b2.
    """
    mod = _load_diagnostic()
    results = mod.scan()
    lf = results.get("low_factor", [])
    sites_str = ", ".join(f"{s['file']}:{s['line']}" for s in lf[:5])
    assert not lf, (
        f"low_factor sites remain: {len(lf)}. If you legitimately "
        f"want factor < 3.0 at one site, add it to WHITELIST with "
        f"rationale. Sites: {sites_str}"
    )
'''


def add_tests():
    print("\n=== add test_low_factor_zero_after_bumps ===")
    p = Path("tests/test_v0925d_a.py")
    s = p.read_text()
    if "test_low_factor_zero_after_bumps" in s:
        print("  already present — skip")
        return
    p.write_text(s + NEW_TEST)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ 3. verify ============

def verify_after():
    print("\n=== verify: diagnostic low_factor == 0 ===")

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
                f"low_factor={len(lf)}, expected 0:\n{msg}"
            )
        print(
            f"  low_factor=0, unguarded={len(r.get('unguarded', []))}, "
            f"guarded={len(r.get('guarded', []))}, "
            f"whitelisted={len(r.get('whitelisted', []))}"
        )

    run_verify_in_subprocess(_verify)


# ============ 4. pytest ============

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

    print("\n=== pytest: targeted (v0925d_a + bumped sites) ===")
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


# ============ 5. main ============

def main():
    print("=== v0.9.25d-b1-fix1: complete b1 + pivot.scan_columns ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    bump()
    add_tests()
    verify_after()
    run_pytest()

    # Single commit: b1's on-disk changes (5 bumps) + fix1 (pivot.scan + test)
    commit_with_guards(
        "v0.9.25d-b1: factor bumps (Tier B) + pivot.scan_columns. "
        "Combines b1's initial run (5/6 functions bumped, never "
        "committed because verify_after found pivot.scan_columns still "
        "low_factor) with fix1's completion (pivot.scan_columns bumped "
        "1.0 → 7.0, test_low_factor_zero_after_bumps added). "
        "Final state: low_factor=0 (12 sites across 6 functions all "
        ">= 3.0). unguarded=23 (Tier C — v0.9.25d-b2 target). "
        "Functions bumped: _per_group_shift, _per_group_rolling, "
        "_rolling_on, _per_group_apply (all dataframe.py, absent → 7.0); "
        "_compute_roc_auc_binary (ops/metrics.py, absent → 7.0); "
        "pivot_to_disk.scan_columns (ops/pivot.py, absent → 7.0); "
        "pivot_to_disk.stream_sorted (ops/pivot.py, 5.0 → 7.0, "
        "over-guard from b1's _bump_factor picking wrong check_batch — "
        "harmless). "
        "Root of b1's partial (per-container heuristic): b1's "
        "_bump_factor picked max(candidates, key=lineno) = latest "
        "check_batch before ANY to_pylist per FUNCTION. pivot_to_disk "
        "has 2 check_batch calls (scan_columns, stream_sorted); "
        "_bump_factor bumped stream_sorted 5.0→7.0 (harmless), left "
        "scan_columns at 1.0. Classifier re-classified line-141 "
        "to_pylist as low_factor (its latest-preceding check_batch "
        "is scan_columns, not stream_sorted). "
        "§13.4/253 (3rd occurrence: v0.9.9-fix1c, v0.9.25c-fix5, "
        "v0.9.25d-b1): per-container heuristic fails when container "
        "holds multiple heterogeneous entities. Correct: iterate per "
        "element (per to_pylist), not per container (per function). "
        "b1's _bump_factor retired after b1 — v0.9.25d-b2 uses "
        "check_batch INSERT mechanism (different shape). "
        "Tests: +1 (test_low_factor_zero_after_bumps). "
        "Fast suite: ~2297 → ~2298.",
        self_path="scripts/patch_v0925d_b1_fix1_pivot_scan.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25d-b1 complete (b1 + fix1 combined). "
          "Next: v0.9.25d-b2 (Tier C — 23 unguarded).")


if __name__ == "__main__":
    main()