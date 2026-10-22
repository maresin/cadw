#!/usr/bin/env python3
# v0.9.25d-b1-fix2 — stale test_categories_present_sanity (low_factor now 0).
#
# STATE BEFORE THIS PATCH:
#   HEAD = v0.9.25d-a (7f03552). v0.9.25d-b1 NEVER committed because
#   b1's verify_after found 1 residual low_factor (pivot.scan_columns)
#   and bailed. fix1 was run (applied pivot.scan bump + added
#   test_low_factor_zero_after_bumps), verify_after passed (low_factor=0),
#   BUT pytest-gate failed on stale test in the SAME file:
#     test_categories_present_sanity (v0.9.25d-a, written for low_factor=12)
#     → assert "low_factor non-empty" fails since b1+fix1 closes Tier B.
#
#   On-disk state:
#     cadw/dataframe.py: _per_group_shift / _per_group_rolling /
#                        _rolling_on / _per_group_apply (extra_factor=7.0)
#     cadw/ops/metrics.py: _compute_roc_auc_binary (extra_factor=7.0)
#     cadw/ops/pivot.py: pivot.stream_sorted (5.0 → 7.0) +
#                        pivot.scan_columns (absent → 7.0)
#     tests/test_v0925d_a.py: added test_low_factor_zero_after_bumps,
#                             test_categories_present_sanity still old
#
# THIS FIX (v0.9.25d-b1-fix2):
#   1. Verify state via diagnostic (low_factor=0, guarded>=26, etc.).
#   2. Update test_categories_present_sanity to exclude low_factor.
#      Docstring: reflect new state (b1 closes low_factor).
#   3. pytest targeted + fast.
#   4. Single commit "v0.9.25d-b1" (b1 initial + fix1 pivot.scan + fix2
#      test fix), because b1's on-disk changes were never committed.
#
# §13.4/144 (4th occurrence this session: v0.9.25c-fix5, v0.9.25c-fix6,
# v0.9.25d-b1, now here): stale-test-on-refactor. Fix-class: when a
# category's count changes (e.g. low_factor: 12 → 0), any test asserting
# "category non-empty" (or count range) must be updated in the SAME
# patch that closes the category. Grep "low_factor|== N|>= M" in tests/
# BEFORE the change if the change targets a category's count.
#
# Meta-lesson: fix1's own scope was "bump 1 site + add test" — missing
# "update tests that assume the OLD count" was a scope-narrowing that
# left a landmine. Fix1 shouldn't have been so narrow. Different from
# §13.4/253 (partial anchor): that was "replace only first of N"; this
# is "add new test but don't update tests broken by the change".
#
# Run: uv run python scripts/patch_v0925d_b1_fix2_categories.py

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


# ============ 0. verify disk state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25d-a; b1+fix1 on-disk; "
          "low_factor=0, categories test still stale ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25d-a" not in log:
        raise SystemExit(f"expected HEAD=v0.9.25d-a; got {log[:200]!r}")
    if "v0.9.25d-b1" in log:
        raise SystemExit("v0.9.25d-b1 already committed — fix2 misplaced")

    import importlib.util as _iu
    spec = _iu.spec_from_file_location(
        "diag", "scripts/service/diagnose_unguarded_to_pylist.py",
    )
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.scan()
    lf = r.get("low_factor", [])
    ung = r.get("unguarded", [])
    g = r.get("guarded", [])
    wl = r.get("whitelisted", [])
    print(f"  diagnostic: low_factor={len(lf)}, unguarded={len(ung)}, "
          f"guarded={len(g)}, whitelisted={len(wl)}")
    if lf:
        raise SystemExit(
            f"low_factor != 0 ({len(lf)}); fix1 not fully applied? "
            f"Sites: {[(s['file'], s['line']) for s in lf]}"
        )
    if not (18 <= len(ung) <= 28):
        raise SystemExit(
            f"unguarded={len(ung)}, expected ~23 (Tier C)"
        )
    if len(g) < 20:
        raise SystemExit(
            f"guarded={len(g)}, expected >= 20 (was 15 pre-b1)"
        )
    if len(wl) < 30:
        raise SystemExit(f"whitelisted={len(wl)}, expected >= 30")

    # Stale test present (expected bug)
    p = Path("tests/test_v0925d_a.py")
    s = p.read_text()
    if "for cat in (\"guarded\", \"whitelisted\", \"unguarded\", \"low_factor\"):" not in s:
        raise SystemExit(
            "stale test_categories_present_sanity not found — "
            "already updated? Re-run fix1's verify_after to confirm "
            "state."
        )
    if "test_low_factor_zero_after_bumps" not in s:
        raise SystemExit(
            "test_low_factor_zero_after_bumps missing — fix1 didn't run"
        )
    print("  state consistent with expectation — OK")


# ============ 1. update test_categories_present_sanity ============

OLD_TEST = '''def test_categories_present_sanity():
    """At least guarded / whitelisted / unguarded / low_factor non-empty.

    v0.9.25d-a leaves unguarded=23, low_factor=12 (v0.9.25d-b targets).
    """
    mod = _load_diagnostic()
    results = mod.scan()
    for cat in ("guarded", "whitelisted", "unguarded", "low_factor"):
        assert results.get(cat), (
            f"category {cat!r} unexpectedly empty — likely a classifier "
            f"bug or all sites fixed (check version)"
        )'''

NEW_TEST = '''def test_categories_present_sanity():
    """At least guarded / whitelisted / unguarded non-empty.

    v0.9.25d-b1 closes low_factor (Tier B: all check_batch factors
    >= 3.0). Remaining non-empty categories: guarded (already
    sufficient), whitelisted (Tier A + helper-based), unguarded
    (Tier C — v0.9.25d-b2 target).

    low_factor excluded from required list because b1's goal was
    to make it empty. For explicit assertion low_factor==0, see
    test_low_factor_zero_after_bumps.
    """
    mod = _load_diagnostic()
    results = mod.scan()
    for cat in ("guarded", "whitelisted", "unguarded"):
        assert results.get(cat), (
            f"category {cat!r} unexpectedly empty — likely a classifier "
            f"bug or all sites fixed (check version)"
        )'''


def update_test():
    print("\n=== update test_categories_present_sanity (exclude low_factor) ===")
    applied = apply_or_fail(
        "tests/test_v0925d_a.py",
        OLD_TEST, NEW_TEST,
        label="categories_sanity_update",
    )
    if applied:
        assert_compiles("tests/test_v0925d_a.py")
        check_no_invalid_escape("tests/test_v0925d_a.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 2. verify ============

def verify_after():
    print("\n=== verify: test_categories_present_sanity passes now ===")

    def _verify():
        import importlib.util as _iu
        from pathlib import Path as _P
        # Load test file as module — this runs module-level imports.
        spec = _iu.spec_from_file_location(
            "v0925d_a", _P("tests/test_v0925d_a.py"),
        )
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Invoke the categories test directly
        mod.test_categories_present_sanity()
        mod.test_low_factor_zero_after_bumps()
        print("  both category tests pass")

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

    print("\n=== pytest: v0925d_a + affected ===")
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
    print("=== v0.9.25d-b1-fix2: stale test_categories_present_sanity ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print(f"  branch: {r.stdout.strip()}")

    verify_state()
    update_test()
    verify_after()
    run_pytest()

    # Single commit: b1's on-disk + fix1's pivot.scan + fix2's test update
    commit_with_guards(
        "v0.9.25d-b1: factor bumps (Tier B) — complete (b1 + fix1 + fix2). "
        "b1's initial run bumped 5/6 functions, found pivot.scan_columns "
        "still low_factor via verify_after, bailed before commit. "
        "fix1 added pivot.scan bump + test_low_factor_zero_after_bumps, "
        "verify_after passed (low_factor=0), pytest gate failed on "
        "stale test_categories_present_sanity (asserted low_factor "
        "non-empty; b1's goal = low_factor empty). "
        "fix2 updates test_categories_present_sanity — removes "
        "low_factor from required-non-empty list (guarded / whitelisted / "
        "unguarded only); docstring reflects b1's completion; "
        "test_low_factor_zero_after_bumps (added in fix1) covers "
        "low_factor==0 explicitly. "
        "Final state: low_factor=0 (was 12 across 6 functions: "
        "_per_group_shift, _per_group_rolling, _rolling_on, "
        "_per_group_apply, _compute_roc_auc_binary, pivot_to_disk × 2). "
        "guarded=27 (was 15). unguarded=23 (Tier C, v0.9.25d-b2). "
        "whitelisted=33. unknown=0. other-guard=0. "
        "§13.4/253 (3rd occurrence: v0.9.9-fix1c, v0.9.25c-fix5, "
        "v0.9.25d-b1) — b1's _bump_factor per-function heuristic "
        "failed on pivot_to_disk's 2 check_batch calls; fixed by "
        "fix1 via direct apply_or_fail. "
        "§13.4/144 (4th occurrence this session: v0.9.25c-fix5, "
        "v0.9.25c-fix6, v0.9.25d-b1, v0.9.25d-b1-fix2): stale-test-"
        "on-refactor. fix1's scope-narrowing missed the category test; "
        "correct pattern: when a category's count changes, grep tests "
        "for that category's assumptions BEFORE the change. "
        "Tests: +1 (test_low_factor_zero_after_bumps, from fix1); "
        "test_categories_present_sanity updated (3 categories, not 4). "
        "Fast suite: ~2297 → ~2298.",
        self_path="scripts/patch_v0925d_b1_fix2_categories.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25d-b1 complete (b1 + fix1 + fix2 combined). "
          "Next: v0.9.25d-b2 (Tier C — 23 unguarded).")


if __name__ == "__main__":
    main()