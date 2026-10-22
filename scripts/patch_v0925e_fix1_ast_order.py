#!/usr/bin/env python3
# v0.9.25e-fix1 — ast.walk is BFS, not DFS-source-order.
#
# STATE BEFORE THIS PATCH:
#   HEAD = v0.9.25d-b2. v0.9.25e NEVER committed:
#   - A.9 (remove _append_to_builders) applied on disk: ops/sort.py
#   - A.10 (fallback scaled 10%) applied on disk: core/memory_guard.py
#   - t1/t2 (test_reduce_state_bytes.py), t3 (test_memory_guard_v2.py)
#     updated on disk
#   - tests/test_v0925e.py added on disk (with buggy test)
#   - dogfood PASSED (numeric checks only, no AST check for order)
#   - pytest FAILED on test_estimate_state_size_fallback_is_ast_scaled:
#     `returns[-1]` returned `8 * bins + 16` (histogram branch), not
#     `fallback` (final source return).
#
# ROOT: `ast.walk()` yields BFS (documented), not source-order. For
# `estimate_state_size`:
#   - Level 1: [Assign, If×4, Expr, Assign_fallback, Expr, Return_fallback]
#   - Level 2: [... Return(16), Return(24), Return(dd), Return(bins)]
# BFS processes Level 1 fully, then Level 2. `Return_fallback` is Level 1
# (yielded position 9), inner Returns are Level 2 (yielded later). So
# `returns[-1]` = last-yielded = last Level-2 Return = Return(8*bins+16).
# Correct: order by source line — `max(returns, key=lambda n: n.lineno)`.
#
# §13.4/255 (NEW): ast.walk is BFS. Any code relying on "last node from
# ast.walk" for source-order-last is broken. Rule: for source-order
# traversal use `sorted(nodes, key=lambda n: n.lineno)` or a DFS helper
# (`_iter_functions` in patch_utils is DFS), NOT ast.walk + [-1].
#
# Note: dogfood did NOT catch this (numeric-only checks). pytest did
# (AST check). Two-tier verification works: dogfood for semantics,
# pytest for AST invariants. This is the intended split.
#
# Fix-cycle count v0.9.25e: initial(1) → fix1(1) = 2 cycles, budget 2.
# Zero bad-commits (both gates caught pre-commit). Actual production
# changes (A.9 + A.10) intact.
#
# Run: uv run python scripts/patch_v0925e_fix1_ast_order.py

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    apply_or_fail, assert_compiles, commit_with_guards,
    run_verify_in_subprocess, check_no_invalid_escape,
)


# ============ 0. verify state ============

def verify_state():
    print("\n=== verify: HEAD=v0.9.25d-b2, A.9/A.10 on-disk, "
          "buggy test present ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25d-b2" not in log:
        raise SystemExit(f"expected HEAD=v0.9.25d-b2; got {log[:200]!r}")
    if "v0.9.25e" in log:
        raise SystemExit("v0.9.25e already committed — fix1 misplaced")

    # A.9 on-disk
    sort_src = Path("cadw/ops/sort.py").read_text()
    if "def _append_to_builders(" in sort_src:
        raise SystemExit(
            "_append_to_builders still present — A.9 not applied"
        )

    # A.10 on-disk
    mg_src = Path("cadw/core/memory_guard.py").read_text()
    if "return 1024 * 1024" in mg_src:
        raise SystemExit(
            "1024*1024 still present — A.10 not applied"
        )
    if "fallback = int(self._state_limit() * 0.1)" not in mg_src:
        raise SystemExit(
            "A.10 scaled fallback not found — A.10 partial?"
        )

    # Buggy test present
    test_src = Path("tests/test_v0925e.py").read_text()
    if "last = returns[-1]" not in test_src:
        raise SystemExit(
            "buggy `last = returns[-1]` not found — already fixed?"
        )
    if "max(returns, key=lambda n: n.lineno)" in test_src:
        raise SystemExit(
            "test already uses source-order max — fix1 misplaced"
        )
    print("  state matches b1 initial (A.9/A.10 on-disk, buggy test present)")


# ============ 1. fix buggy AST test ============

OLD_BUG = '''    returns = [
        n for n in _ast.walk(func) if isinstance(n, _ast.Return)
    ]
    assert returns, "estimate_state_size has no return statement"
    last = returns[-1]'''

NEW_FIX = '''    returns = [
        n for n in _ast.walk(func) if isinstance(n, _ast.Return)
    ]
    assert returns, "estimate_state_size has no return statement"
    # §13.4/255: ast.walk is BFS, not source-order. Direct-child
    # Return(fallback) is yielded in Level 1; nested branch Returns
    # (Return(16), Return(24), ..., Return(8*bins+16)) are yielded
    # in Level 2 — AFTER. So `returns[-1]` picks the last Level-2
    # (histogram branch), not the fallback. Correct: order by source
    # line — `max(returns, key=lambda n: n.lineno)`.
    last = max(returns, key=lambda n: n.lineno)'''


def fix_test():
    print("\n=== fix ast.walk ordering in test_estimate_state_size_"
          "fallback_is_ast_scaled ===")
    applied = apply_or_fail(
        "tests/test_v0925e.py",
        OLD_BUG, NEW_FIX,
        label="ast_walk_to_source_order",
    )
    if applied:
        assert_compiles("tests/test_v0925e.py")
        check_no_invalid_escape("tests/test_v0925e.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 2. verify fixed test passes ============

def verify_after():
    print("\n=== verify: fixed test passes; A.9/A.10 invariants hold ===")

    def _verify():
        import ast as _ast
        import inspect as _inspect
        import textwrap as _tw
        from cadw.core.memory_guard import MemoryGuard
        from cadw.ops import sort as sort_mod

        # A.9: no _append_to_builders, no .as_py
        sort_src = _inspect.getsource(sort_mod)
        tree = _ast.parse(sort_src)
        assert not any(
            isinstance(n, _ast.FunctionDef)
            and n.name == "_append_to_builders"
            for n in _ast.walk(tree)
        )
        as_py_count = sum(
            1 for n in _ast.walk(tree)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "as_py"
        )
        assert as_py_count == 0, f"sort.py has {as_py_count} .as_py()"

        # A.10 AST: fallback is scaled (source-order last return)
        mg_src = _tw.dedent(
            _inspect.getsource(MemoryGuard.estimate_state_size)
        )
        mg_tree = _ast.parse(mg_src)
        func = mg_tree.body[0]
        returns = [
            n for n in _ast.walk(func) if isinstance(n, _ast.Return)
        ]
        last = max(returns, key=lambda n: n.lineno)
        unparsed = _ast.unparse(last.value)
        assert "fallback" in unparsed or "state_limit" in unparsed, (
            f"source-order last return = {unparsed!r}, not fallback — "
            f"A.10 pattern changed?"
        )

        # A.10 numeric: 1/10/100 MB
        for mb, expected in [(1, 52_428), (10, 524_288), (100, 5_242_880)]:
            g = MemoryGuard(max_memory_mb=mb)
            got = g.estimate_state_size("__test_unknown_op__")
            assert got == expected, (
                f"budget={mb}MB: expected {expected}, got {got}"
            )

        # Known ops unchanged
        g = MemoryGuard(max_memory_mb=10)
        assert g.estimate_state_size("count") == 16
        assert g.estimate_state_size("std") == 24
        assert g.estimate_state_size("cov", d=3) == 8 + 8*3 + 8*9
        assert g.estimate_state_size("histogram", bins=10) == 8*10 + 16

        print(
            "  AST source-order = fallback; numeric 1/10/100 MB OK; "
            "known ops unchanged"
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

    print("\n=== pytest: v0925e (the failing test) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_v0925e.py",
         "-q", "-W", "error::SyntaxWarning"],
        timeout=120,
    )
    if r.returncode != 0:
        print("\n*** v0925e TEST STILL FAILING — ABORT ***")
        sys.exit(1)

    print("\n=== pytest: targeted ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925e.py",
         "tests/test_memory_guard_v2.py",
         "tests/test_reduce_state_bytes.py",
         "tests/test_reduce.py",
         "tests/test_sort.py",
         "tests/test_merge_arrow.py",
         "tests/test_merge_arrow_bench.py",
         "tests/test_sort_merge_p_s1.py",
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
    print("=== v0.9.25e-fix1: ast.walk BFS ordering ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    fix_test()
    verify_after()
    run_pytest()

    commit_with_guards(
        "v0.9.25e: Phase A close — A.9 (dead code) + A.10 (fallback "
        "factor) + fix1 (ast.walk ordering). "
        "A.9: removed `_append_to_builders` from cadw/ops/sort.py. "
        "Was v0.5.2a P5 helper (per-row .as_py() then builder.append()); "
        "v0.9.23c replaced merge with Arrow-native batched "
        "(pc.sort_indices) leaving helper with zero callers. Per-row "
        ".as_py() anti-pattern (§13.4/88) and INVISIBLE to diagnostic "
        "(scans .to_pylist, not .as_py). AST removal by lineno, PEP8 "
        "blank-line normalization. "
        "A.10: MemoryGuard.estimate_state_size fallback 1024*1024 → "
        "int(self._state_limit() * 0.1). Prior fixed 1 MiB caused "
        "false raise at tight budgets (1 MiB > 250 KB state_limit at "
        "500 KB) and false pass at large (1 MiB < 2% of 50 MB at "
        "100 MB). New: 10% of state_limit — proportional at any scale. "
        "Warning message updated (Cyrillic → clear English + override "
        "guidance). §11.5 documented: op whose init() OOMs despite "
        "passing scaled fallback is op's responsibility. "
        "Tests updated PROACTIVELY (§13.4/144 6th occurrence): t1 "
        "(docstring 1MiB→10% scale), t2 renamed to "
        "test_legacy_op_fallback_scales_with_budget (500 KB no longer "
        "raises), t3 → test_estimate_state_size_unknown_scales ×3 "
        "(1/10/100 MB numeric). New tests/test_v0925e.py: 5 tests "
        "(A.9 AST guards ×2; A.10 AST guard + runtime scaling + known "
        "ops unchanged). "
        "fix1: my own AST test had bug — `returns[-1]` from "
        "`ast.walk()` picked `8*bins+16` (histogram branch, last in "
        "BFS Level 2), not `fallback` (source-order last, but BFS "
        "Level 1 yielded position 9). Root: ast.walk is BFS not DFS. "
        "Fix: `max(returns, key=lambda n: n.lineno)`. "
        "§13.4/255 (NEW): ast.walk is BFS. Rule: for source-order "
        "traversal use `sorted(nodes, key=lambda n: n.lineno)` or DFS "
        "helper (`_iter_functions` is DFS); NOT ast.walk + [-1]. "
        "Verification split worked: dogfood PASSED (numeric only); "
        "pytest CAUGHT (AST check). Two-tier verification = semantics "
        "+ invariants. "
        "Fix-cycle count v0.9.25e: initial + fix1 = 2 cycles, budget "
        "2. Zero bad-commits — both gates caught pre-commit. "
        "Fast suite: ~2299 → ~2305 (+5 new tests, +1 net from t3 "
        "1→3 rename). "
        "Phase A CLOSED (A.4–A.10 all done). "
        "Next: Phase B — dataframe.py split (~4h, 7 modules).",
        self_path="scripts/patch_v0925e_fix1_ast_order.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25e COMPLETE (A.9 + A.10 + fix1). "
          "Phase A CLOSED. Next: Phase B — dataframe.py split.")


if __name__ == "__main__":
    main()