#!/usr/bin/env python3
# v0.9.25j-fix4 — whitelist CadwDataFrame.qcut for AST guard.
#
# STATE: HEAD=v0.9.25i. Working tree has v0.9.25j applied (methods +
# tests + nullable=True + regex ">= 2" + docstring fixed), NOT
# committed. Full fast suite: 1 failed / 2438 passed:
#
#   FAILED tests/test_arrow_call_memory_guard.py::
#     test_no_unguarded_high_risk_arrow_calls
#   cadw/dataframe.py:1786: `quantile()` in `CadwDataFrame.qcut`
#   without preceding check_state/check_batch/stat
#
# ROOT: qcut calls `self.quantile(column, q_positions)`. AST guard
# matches HIGH_RISK_ATTRS by attr NAME ("quantile"), not receiver
# type. So `self.quantile` (CADW's guarded method) is flagged same as
# `pc.quantile` (raw Arrow C++ sort-based). self.quantile delegates
# to `cadw.ops.quantile.quantile` which uses `external_sort` (guarded
# internally via MemoryGuard, §9.2.N).
#
# This is EXACTLY the pattern of `cadw/scalers.py:RobustScaler.fit` —
# already whitelisted with rationale "per-column df.quantile —
# quantile op self-guards (§9.2.N)". Add CadwDataFrame.qcut with
# matching rationale.
#
# §13.4/275 (NEW): AST guard matches attr names, not receiver types.
# pc.quantile = high-risk (raw sort); self.quantile = safe (delegates
# to guarded op). Cannot distinguish by receiver. Whitelist by
# qualname with explicit rationale — same class as §13.4/170 (K1.1
# qualname-keyed whitelist).
#
# REJECTED alternative: add check_batch before self.quantile in qcut.
# qcut doesn't know row_size × N without a scan (defeats purpose);
# self.quantile already guards internally. Useless check_state(0)
# just to satisfy AST guard is dishonest — whitelist with rationale
# is the correct signal (§13.4/170).
#
# §13.4/273 refined: fix1-fix3 patched SOURCE script and re-ran
# v0.9.25j. fix4 (unlike prior) patches the DATA (whitelist file),
# NOT the source, because working tree is CORRECT except for the
# missing whitelist entry. Direct commit is faster + cleaner than
# revert/re-run. Rule: patch source-of-truth if data is wrong;
# patch data + commit if data is correct but downstream config is
# missing.
#
# §13.4/264 (REPL-first): verify_state does 5 checks:
#   (1) HEAD subject-prefix == v0.9.25i (§13.4/270)
#   (2) dataframe.py HAS qcut (v0.9.25j applied, uncommitted)
#   (3) tests/test_v0925j.py exists (v0.9.25j applied)
#   (4) whitelist does NOT have CadwDataFrame.qcut entry
#   (5) AST guard currently reports qcut violation (bug marker)
#
# §13.4/266 (anchor check): unique anchor is the closing `}` of
# WHITELISTED_FUNCS after the RobustScaler.fit entry. That block
# appears once in test_arrow_call_memory_guard.py.
#
# §13.4/267 (mirror-check): RobustScaler.fit already whitelisted
# for self.quantile. New entry for qcut follows identical rationale
# pattern. Grep "quantile self-guards" post-fix → 2 hits (RobustScaler
# + qcut), matching sibling pattern.
#
# §13.4/268/269/271/272/274 preserved: no bfill fills all, no
# to_pylist low_factor sites added, f-strings single-literal,
# nullable=True for qcut's _bin field (already applied by v0.9.25j
# fix2v2), precise full-field-string checks.
#
# §13.4/260/261/262/265 preserved: no subprocess dogfood, no symbol
# moves, typed scalars, no copy-pasted asserts.
#
# Usage:
#   uv run python scripts/patch_v0925j_fix4_whitelist.py

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
    commit_with_guards,
)


DATAFRAME_PATH = "cadw/dataframe.py"
TESTS_PATH = "tests/test_v0925j.py"
GUARD_TEST_PATH = "tests/test_arrow_call_memory_guard.py"


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25i; working tree has v0.9.25j; "
          "whitelist missing qcut; AST guard reports violation ===")

    # (1) HEAD subject-prefix (§13.4/270)
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    m = re.match(r'^[a-f0-9]+ (v\d+\.\d+\.\d+\w*):', log)
    head_version = m.group(1) if m else None
    if head_version != "v0.9.25i":
        raise SystemExit(
            f"expected HEAD version v0.9.25i; got {head_version!r}"
        )
    print(f"  (1) HEAD subject-prefix = v0.9.25i ✓")

    # (2) dataframe.py has qcut (v0.9.25j applied, uncommitted)
    df_src = Path(DATAFRAME_PATH).read_text()
    for marker in ("def cut(self", "def qcut(self", "def duplicated(self"):
        if marker not in df_src:
            raise SystemExit(
                f"UNEXPECTED: {marker!r} missing — v0.9.25j not "
                f"applied? Run fix3 first."
            )
    print("  (2) dataframe.py has cut/qcut/duplicated ✓")

    # (3) test_v0925j.py exists
    if not Path(TESTS_PATH).exists():
        raise SystemExit(
            f"UNEXPECTED: {TESTS_PATH} missing — v0.9.25j not "
            f"applied? Run fix3 first."
        )
    print(f"  (3) {TESTS_PATH} exists ✓")

    # (4) whitelist missing CadwDataFrame.qcut
    guard_src = Path(GUARD_TEST_PATH).read_text()
    if "cadw/dataframe.py:CadwDataFrame.qcut" in guard_src:
        raise SystemExit(
            "whitelist already has CadwDataFrame.qcut — fix4 misplaced"
        )
    if "cadw/scalers.py:RobustScaler.fit" not in guard_src:
        raise SystemExit(
            "anchoring pattern (RobustScaler.fit) missing — test "
            "file structure changed"
        )
    print("  (4) whitelist lacks CadwDataFrame.qcut (bug marker) ✓")

    # (5) AST guard reports qcut violation
    import importlib
    mod = importlib.import_module("tests.test_arrow_call_memory_guard")
    violations = mod._collect_violations()
    qcut_hits = [v for v in violations if "qcut" in v]
    if not qcut_hits:
        raise SystemExit(
            f"AST guard doesn't report qcut violation: {violations}. "
            f"Bug marker missing — v0.9.25j may have already been "
            f"whitelisted or qcut's self.quantile call was refactored."
        )
    print(f"  (5) AST guard reports qcut violation: "
          f"{len(qcut_hits)} hit(s) ✓")

    print("  state verified")


# ============ 1. add whitelist entry ============

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


# ============ 2. verify AST guard passes ============

def verify_ast_guard_passes():
    print("\n=== verify AST guard now passes ===")
    import importlib
    mod = importlib.import_module("tests.test_arrow_call_memory_guard")
    # Reload to pick up patched WHITELISTED_FUNCS
    importlib.reload(mod)
    violations = mod._collect_violations()
    if violations:
        print(f"  REMAINING violations ({len(violations)}):")
        for v in violations:
            print(f"    {v}")
        raise SystemExit(
            "AST guard still has violations after whitelist patch. "
            "Inspect: maybe there's another unguarded high-risk call "
            "not previously reported (masked by qcut)."
        )
    print("  AST guard: 0 violations ✓")

    # Also run the specific test
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_arrow_call_memory_guard.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=180,
    )
    if r.returncode != 0:
        print(f"\n*** test_arrow_call_memory_guard FAILED ***")
        sys.exit(1)
    print("  test_arrow_call_memory_guard: PASS ✓")


# ============ 3. full fast suite ============

def run_fast_suite():
    print("\n=== pre-flight: full fast suite ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)


# ============ 4. commit ============

COMMIT_MSG = (
    "v0.9.25j: Phase C batch C-3b — cut + qcut + duplicated "
    "+ fix4 (AST guard whitelist for qcut). "
    "3 of 20 Pandas P0 features. Features: "
    "(1) `CadwDataFrame.cut(column, bins, *, right=True, "
    "include_lowest=True) -> (df, edges)` — bin numeric column. "
    "bins=int → k equal-width; bins=list[float] → explicit ascending "
    "edges. Appends `{column}_bin` int32 nullable. right=True → "
    "(lo, hi] (pandas default); include_lowest=True (CADW default) "
    "→ min goes to bin 0. MinMaxOp + np.digitize. "
    "(2) `CadwDataFrame.qcut(column, q, *, right=True, "
    "include_lowest=True) -> (df, edges)` — quantile-based binning "
    "via df.quantile (external_sort). Tied quantile edges deduped. "
    "All-same → ValueError. "
    "(3) `CadwDataFrame.duplicated(subset=None, keep='first') -> "
    "augmented df` — original cols + `duplicated` bool column "
    "(nullable=True). keep ∈ {'first', 'last', False}; pandas "
    "semantics. Implementation: hash_shuffle + dedup on "
    "(subset, __rid) via existing _ops_drop_duplicates + per-batch "
    "mask via pc.is_in. "
    "Fixes applied across v0.9.25j attempts (fix0–fix4): "
    "fix0: HEAD subject-prefix regex instead of substring "
    "(§13.4/270). "
    "fix1: f-string split across literals (qcut error message) — "
    "extract to var, single literal (§13.4/271). "
    "fix2v2: duplicated field nullable=False → nullable=True "
    "(§13.4/272); test regex 'at least 2' → '>= 2'. "
    "fix3: stale docstring 'bool column (nullable=False)' → "
    "'nullable=True'; fix2v2's verify_inner_compiles uses "
    "full-field-string check, not bare token (§13.4/274). "
    "fix4: AST guard flags self.quantile in qcut — high-risk attr "
    "match by NAME not receiver type; whitelist CadwDataFrame.qcut "
    "with rationale matching RobustScaler.fit (self-guards via "
    "external_sort, §9.2.N). §13.4/275 (NEW): AST guard matches "
    "attr names, not receiver types — bounded operations that "
    "self-guard require whitelist entry. §13.4/273 refined: "
    "patch-source-of-truth only if data is wrong; patch-data + "
    "commit if data is correct but downstream config missing. "
    "Tests (tests/test_v0925j.py): 30 total — cut (12), qcut (9), "
    "duplicated (9). "
    "Fast suite: 2404 → 2434 passed, 4 skipped. "
    "Next: Phase C C-4a (str.slice/cat/count)."
)


def do_commit():
    print("\n=== commit v0.9.25j + fix4 ===")
    commit_with_guards(
        COMMIT_MSG,
        self_path="scripts/patch_v0925j_fix4_whitelist.py",
        check_whitelist=True,
    )
    print("\n  committed")


# ============ main ============

def main():
    print("=== v0.9.25j-fix4: AST guard whitelist for qcut ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    patch_whitelist()
    verify_ast_guard_passes()
    run_fast_suite()
    do_commit()

    print("\n==> v0.9.25j COMPLETE (3 features + fix0-fix4). "
          "Phase C: 12/20. Next: C-4a (str extras).")


if __name__ == "__main__":
    main()