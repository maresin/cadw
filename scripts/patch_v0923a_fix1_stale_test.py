#!/usr/bin/env python3
# v0.9.23a-fix1 — update stale SITES test after v0.9.23a rank factor bump.
#
# tests/test_extra_factor_sites.py hardcodes the three rank extra_factor
# values (5.0, 3.0, 5.0). v0.9.23a changed them to (7.0, 3.5, 7.0) —
# classic stale-test class §13.4/144 (behavior change without parallel
# test expectation update). This fix updates the parametrized SITES
# constant. The other two SITES entries (resample, pivot) are unchanged —
# 5.0 remains correct for their pylist call sites.
#
# No changes to cadw/. No output behavior change.
#
# Run: uv run python scripts/patch_v0923a_fix1_stale_test.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu
from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    run_verify_in_subprocess,
)


# ============ 1. SITES: three rank entries ============

def fix_sites():
    print("\n=== fix SITES (rank.streamable / pass1 / pass2) ===")
    p = Path("tests/test_extra_factor_sites.py")
    s = p.read_text()
    if '"rank.streamable.input", 7.0' in s:
        print("  already fixed — skip")
        return

    replacements = [
        (
            '"rank.streamable.input", 5.0),',
            '"rank.streamable.input", 7.0),',
            "SITES._compute_streamable",
        ),
        (
            '"rank.pass1.input", 3.0),',
            '"rank.pass1.input", 3.5),',
            "SITES._pass1_groups_to_disk",
        ),
        (
            '"rank.pass2.input", 5.0),',
            '"rank.pass2.input", 7.0),',
            "SITES._pass2_ranks_to_disk",
        ),
    ]
    for old, new, label in replacements:
        apply_or_fail(str(p), old, new, label=label)

    assert_compiles(p)
    print("  OK")


# ============ 2. Dogfood ============

def dogfood():
    print("\n=== dogfood: source factors == SITES expectations ===")

    def _verify():
        import inspect
        from cadw.ops import rank as rk

        # Source AST: verify actual extra_factor in three rank methods.
        expected = {
            "_compute_streamable": ("rank.streamable.input", "7.0"),
            "_pass1_groups_to_disk": ("rank.pass1.input", "3.5"),
            "_pass2_ranks_to_disk": ("rank.pass2.input", "7.0"),
        }
        for fn_name, (key, factor) in expected.items():
            src = inspect.getsource(getattr(rk, fn_name))
            found = False
            for line in src.split("\n"):
                if key in line:
                    assert f"extra_factor={factor}" in line, (
                        f"{fn_name}: expected extra_factor={factor} in line: "
                        f"{line!r}"
                    )
                    found = True
                    break
            assert found, f"{key} not found in {fn_name}"

        # SITES AST: verify the test file expectations match.
        sites_src = open("tests/test_extra_factor_sites.py").read()
        assert '"rank.streamable.input", 7.0)' in sites_src
        assert '"rank.pass1.input", 3.5)' in sites_src
        assert '"rank.pass2.input", 7.0)' in sites_src
        # Unchanged entries remain 5.0 (resample, pivot).
        assert '"resample.sorted_input", 5.0)' in sites_src
        assert '"pivot.stream_sorted", 5.0)' in sites_src

        print("  source factors and SITES align (3 rank new + 2 unchanged)")

    run_verify_in_subprocess(_verify)


# ============ 3. Tests ============

def run_pytest():
    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_extra_factor_sites.py",
         "tests/test_rank_extra_factor.py",
         "tests/test_rank.py",
         "tests/test_pylist_fixes.py",
         "-q"],
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
    print("=== v0.9.23a-fix1: stale SITES test ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    branch = r.stdout.strip()
    print(f"  branch: {branch}")
    if not branch.startswith("fix/"):
        print(f"\n*** WARNING: not on a fix/ branch ({branch}) ***")
        print("    Expected: git reset --soft master && git checkout -b fix/v0.9.23")
        print("    Continue anyway? [y/N] ", end="")
        if input().strip().lower() != "y":
            sys.exit(1)

    fix_sites()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.23a: rank extra_factor fix + stale SITES update. "
        "3 changes to cadw/ops/rank.py: _compute_streamable 5.0->7.0, "
        "_pass1_groups_to_disk 3.0->3.5, _pass2_ranks_to_disk 5.0->7.0. "
        "Derivation per AUDIT_FULL_v0922 §3.7: 3.5*K pylist factor "
        "(28 B/value vs Arrow 8 B/value). K=1 (pass1 by) -> 3.5; "
        "K=2 (streamable/pass2: by + row_id) -> 7.0. Prior 5.0 "
        "under by 29% on derivation, ~4x on measured pylist delta "
        "(1.61 MB / 80 KB batch = 20.2x at N=100k with 10% NaN). "
        "Guard-only fix, no output change. Plus: tests/"
        "test_rank_extra_factor.py new (5 tests: 3 AST + 2 "
        "functional regression); tests/test_extra_factor_sites.py "
        "SITES updated (3 rank entries; resample/pivot unchanged at "
        "5.0). Closes AUDIT_FULL_v0922 §3.7 / item 7.A.11 "
        "(partial — Arrow-native null path remains P1 structural). "
        "Fast: 1921 -> 1926 passed.",
        self_path="scripts/patch_v0923a_fix1_stale_test.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.23a-fix1 applied. v0.9.23a complete.")


if __name__ == "__main__":
    main()