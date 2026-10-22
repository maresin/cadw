#!/usr/bin/env python3
# v0.9.25j-fix5 — recovery: whitelist first, pre-verify AST guard,
#                 then re-run v0.9.25j.
#
# STATE: HEAD=v0.9.25i. Working tree REVERTED AGAIN (2nd time —
# between fix3's v0.9.25j re-run exit and fix4's verify_state).
# fix2v2 already reverted once after fix2v2 itself aborted.
#
# PATTERN: both reverts occur AFTER a patch script exits non-zero.
# Suspected external trigger: .git/hooks/pre-commit resetting, IDE
# auto-stash on failed cmd, or shell PROMPT_COMMAND cleanup.
#
# §13.4/276 (NEW): if working tree reverts after non-zero exit, the
# "verify all green, then commit" pattern is broken — the state the
# checks need is destroyed by the mechanism reporting the failure.
# Fix: reorder so git commit runs before the checks most likely to
# fail, OR eliminate the failing check from pre-commit path. Here,
# AST guard failure was the last blocker; patch whitelist FIRST,
# verify AST guard ALONE, then re-run — full pipeline goes green
# (exit 0), revert never triggers.
#
# §13.4/273 (existing): failed patches need explicit revert before
# re-run. Extends to: any external revert is equivalent to our
# explicit git checkout --.
#
# §13.4/275 (existing): AST guard matches attr names, not receiver
# types — CadwDataFrame.qcut needs whitelist because self.quantile
# is safe (delegates to guarded op via external_sort).
#
# APPROACH:
#   0. Diagnose revert source (reflog, hooks, config, stash list)
#      — informational; abort only if state irrecoverable.
#   1. Idempotent whitelist patch on test_arrow_call_memory_guard.py
#   2. Verify AST guard test ALONE passes (2s, no full suite)
#   3. Pre-clean dataframe.py + test file
#   4. Re-run v0.9.25j subprocess (full pipeline, all checks now
#      pre-satisfied)
#   5. Post-run verify commit landed (subject starts with v0.9.25j:)
#
# §13.4/264 (REPL-first): verify_state does 6 checks:
#   (1) HEAD subject-prefix == v0.9.25i
#   (2) script has all fixes (f-string, nullable=True, docstring,
#       regex)
#   (3) whitelist missing CadwDataFrame.qcut (idempotency marker)
#   (4) AST guard currently reports qcut violation
#   (5) apply_or_fail whitelist anchor intact (RobustScaler.fit
#       entry present)
#   (6) git status --short dumped for diagnostics
#
# §13.4/270 (subject-prefix), §13.4/272 (nullable), §13.4/274
# (full-syntax-unit) preserved: no reversion of these lessons.
#
# §13.4/260/261/262/265/269/271: no subprocess dogfood, no symbol
# moves, typed scalars, no copy-pasted asserts, no to_pylist
# low_factor sites (fix5 only touches test_arrow_call_memory_guard
# whitelist), f-strings single-literal.
#
# Usage:
#   uv run python scripts/patch_v0925j_fix5_recovery.py

from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    apply_or_fail, assert_compiles, check_no_invalid_escape,
)


DATAFRAME_PATH = "cadw/dataframe.py"
TESTS_PATH = "tests/test_v0925j.py"
SCRIPT_PATH = "scripts/patch_v0925j_cut_qcut_duplicated.py"
GUARD_TEST_PATH = "tests/test_arrow_call_memory_guard.py"


# ============ 0. diagnose revert source ============

def diagnose_revert():
    print("\n=== diagnostics: revert source ===")
    commands = [
        ("git status --short", ["git", "status", "--short"]),
        ("git stash list", ["git", "stash", "list"]),
        ("git reflog --oneline -15", ["git", "reflog", "--oneline", "-15"]),
        ("git config core.hooksPath", ["git", "config", "core.hooksPath"]),
    ]
    for label, cmd in commands:
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = r.stdout.strip()
        print(f"\n  $ {label}")
        if out:
            for line in out.splitlines()[:20]:
                print(f"    {line}")
        else:
            print("    (empty)")
    hooks_dir = Path(".git/hooks")
    print("\n  $ ls .git/hooks/ (non-sample)")
    if hooks_dir.exists():
        active = [
            p for p in sorted(hooks_dir.iterdir())
            if p.is_file() and not p.name.endswith(".sample")
        ]
        if active:
            for p in active:
                print(f"    {p.name}  ({p.stat().st_size} B)")
        else:
            print("    (none)")
    else:
        print("    .git/hooks/ not found")


# ============ 1. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25i; script fixes present; "
          "whitelist missing qcut; AST guard reports violation ===")

    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    m = re.match(r'^[a-f0-9]+ (v\d+\.\d+\.\d+\w*):', log)
    head_version = m.group(1) if m else None
    if head_version != "v0.9.25i":
        raise SystemExit(
            f"expected HEAD version v0.9.25i; got {head_version!r}. "
            f"If HEAD is v0.9.25j, the commit already succeeded — "
            f"no fix5 needed."
        )
    print(f"  (1) HEAD subject-prefix = v0.9.25i ✓")

    script_src = Path(SCRIPT_PATH).read_text()
    markers = {
        "_first_or_empty (f-string fix)": "_first_or_empty",
        'nullable=True (fix2v2)': (
            'pa.field("duplicated", pa.bool_(), nullable=True)'
        ),
        'regex ">= 2" (fix2v2)': 'match=">= 2"',
        "docstring (nullable=True) (fix3)": (
            "bool column (nullable=True)"
        ),
    }
    for label, marker in markers.items():
        if marker not in script_src:
            raise SystemExit(
                f"script missing fix: {label} (expected marker "
                f"{marker!r}). Re-apply fix1/fix2v2/fix3 as needed."
            )
    print(f"  (2) script has all {len(markers)} fixes ✓")

    guard_src = Path(GUARD_TEST_PATH).read_text()
    if "cadw/dataframe.py:CadwDataFrame.qcut" in guard_src:
        raise SystemExit(
            "whitelist already has CadwDataFrame.qcut — fix5 "
            "misplaced (or partial fix). Inspect."
        )
    if "cadw/scalers.py:RobustScaler.fit" not in guard_src:
        raise SystemExit(
            "anchor RobustScaler.fit entry missing — test file "
            "structure changed. Manual fix required."
        )
    print("  (3) whitelist lacks CadwDataFrame.qcut (idempotency "
          "marker) ✓")

    import importlib
    mod = importlib.import_module("tests.test_arrow_call_memory_guard")
    violations = mod._collect_violations()
    qcut_hits = [v for v in violations if "qcut" in v]
    if not qcut_hits:
        raise SystemExit(
            f"AST guard doesn't report qcut violation: "
            f"{violations}. If empty, qcut already whitelisted "
            f"(fix4 applied?) — run fix5 anyway to be idempotent; "
            f"if non-empty without qcut, different violation exists "
            f"that fix5 won't cover."
        )
    print(f"  (4) AST guard reports qcut violation "
          f"({len(qcut_hits)} hit) ✓")

    print("  (5) anchor entry present ✓")

    status = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"  (6) working tree:\n"
          f"    {status if status else '(clean)'}")

    print("  state verified")


# ============ 2. patch whitelist (idempotent) ============

WHITELIST_OLD = '''    "cadw/scalers.py:RobustScaler.fit":
        "per-column df.quantile — quantile op self-guards (§9.2.N)",
}'''

WHITELIST_NEW = '''    "cadw/scalers.py:RobustScaler.fit":
        "per-column df.quantile — quantile op self-guards (§9.2.N)",
    # --- dataframe.py qcut (v0.9.25j) ---
    "cadw/dataframe.py:CadwDataFrame.qcut":
        "self.quantile delegates to CADW's own guarded quantile op "
        "(external_sort + streaming); quantile self-guards via its "
        "own MemoryGuard (§9.2.N). AST guard cannot distinguish "
        "self.quantile from pc.quantile — both attr name 'quantile'.",
}'''


def patch_whitelist():
    print("\n=== whitelist CadwDataFrame.qcut for AST guard ===")
    applied = apply_or_fail(
        GUARD_TEST_PATH, WHITELIST_OLD, WHITELIST_NEW,
        label="qcut_whitelist",
    )
    if applied:
        assert_compiles(GUARD_TEST_PATH)
        check_no_invalid_escape(GUARD_TEST_PATH)
    print("  applied" if applied else "  already applied — skip")


# ============ 3. verify AST guard ALONE ============

def verify_ast_guard_alone():
    print("\n=== pre-verify: AST guard test alone (fast, ~2s) ===")
    import importlib
    mod = importlib.import_module("tests.test_arrow_call_memory_guard")
    importlib.reload(mod)
    violations = mod._collect_violations()
    if violations:
        print(f"  REMAINING violations ({len(violations)}):")
        for v in violations:
            print(f"    {v}")
        raise SystemExit(
            "AST guard still has violations after whitelist patch. "
            "Another unguarded high-risk call exists beyond qcut "
            "(previously masked). Diagnose before re-running v0.9.25j."
        )
    print("  _collect_violations(): 0 violations ✓")

    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** test_arrow_call_memory_guard.py FAILED ***")
        sys.exit(1)
    print("  test_arrow_call_memory_guard.py: PASS ✓")


# ============ 4. pre-clean + re-run v0.9.25j ============

def pre_clean():
    print("\n=== pre-clean: dataframe.py → HEAD, test file removed ===")
    r = subprocess.run(
        ["git", "checkout", "--", DATAFRAME_PATH],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        print(f"  reverted {DATAFRAME_PATH} → HEAD")
    else:
        print(f"  git checkout rc={r.returncode}: "
              f"{r.stderr.strip()[:120]}")

    tests_p = Path(TESTS_PATH)
    if tests_p.exists():
        tests_p.unlink()
        print(f"  removed {TESTS_PATH}")
    else:
        print(f"  {TESTS_PATH} already absent")


def rerun_v925j():
    print("\n=== re-run v0.9.25j (all checks pre-satisfied, "
          "commit expected to succeed) ===")
    r = subprocess.run(
        ["uv", "run", "python", SCRIPT_PATH],
        timeout=2400,
    )
    if r.returncode != 0:
        print(
            f"\n*** v0.9.25j re-run failed (rc={r.returncode}). "
            f"Working tree may be reverted by external trigger. "
            f"See diagnostics above for source. ***"
        )
        sys.exit(r.returncode)
    print("\n  v0.9.25j re-run: exit 0")


# ============ 5. post-run verify commit landed ============

def verify_commit():
    print("\n=== post-run: verify commit landed ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  HEAD: {log[:120]}")
    m = re.match(r'^[a-f0-9]+ (v\d+\.\d+\.\d+\w*):', log)
    version = m.group(1) if m else None
    if version != "v0.9.25j":
        raise SystemExit(
            f"expected HEAD subject v0.9.25j, got {version!r}. "
            f"Commit did not land — inspect working tree."
        )
    print("  ✓ v0.9.25j committed")

    status = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True, text=True,
    ).stdout.strip()
    if status:
        print(f"  WARN: working tree not clean after commit:\n"
              f"    {status}")
    else:
        print("  ✓ working tree clean")


# ============ main ============

def main():
    print("=== v0.9.25j-fix5: recovery (whitelist first, re-run) ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    diagnose_revert()
    verify_state()
    patch_whitelist()
    verify_ast_guard_alone()
    pre_clean()
    rerun_v925j()
    verify_commit()

    print("\n==> v0.9.25j COMPLETE. Phase C: 12/20. "
          "Next: C-4a (str.slice/cat/count).")


if __name__ == "__main__":
    main()