#!/usr/bin/env python3
# v0.9.25e — Phase A close: A.9 (dead code) + A.10 (fallback factor).
#
# CONTEXT (v0.9.25d-b2 closed Phase A.6):
#   Phase A.6 (to_pylist memory hygiene): closed across diagnostic
#   (v0.9.25d), Tier A (v0.9.25d-a), Tier B (v0.9.25d-b1), Tier C
#   (v0.9.25d-b2). Final state: unguarded=0, low_factor=0,
#   unknown=0, other-guard=0, guarded=31, whitelisted=52 (per-site).
#
# SCOPE (v0.9.25e closes remaining Phase A items):
#
# A.9 — remove dead `_append_to_builders` from cadw/ops/sort.py.
#   Was v0.5.2a P5 helper: per-row `.as_py()` then `builder.append()`
#   to avoid Python-list buffer overhead (2017-2019 peak 3-5x).
#   v0.9.23c replaced merge with Arrow-native batched (pc.sort_indices),
#   leaving _append_to_builders with zero callers. Per-row `.as_py()`
#   is anti-pattern (§13.4/88) and INVISIBLE to diagnostic (which
#   scans `.to_pylist`, not `.as_py`). Audit caught it manually.
#   AST-based removal (target.lineno..target.end_lineno), not string
#   anchor. Collapse resulting 4-blank gaps to 2 (PEP8).
#
# A.10 — MemoryGuard.estimate_state_size fallback fixed 1 MiB → scaled.
#   Prior: fixed 1024*1024 → false raise at tight budgets (1 MiB >
#   250 KB state_limit at 500 KB) and false pass at large budgets
#   (1 MiB < 2% of 50 MB state_limit at 100 MB). New:
#   int(self._state_limit() * 0.1) — 10% of state_limit, proportional
#   at any scale. Warning message updated (was Cyrillic "неизвестная",
#   now clear English + guidance "override estimate_state_nbytes").
#   §11.5 documented limitation: an op whose init() OOMs despite
#   passing scaled fallback is op's responsibility — override
#   ReduceOp.estimate_state_nbytes for exact bound.
#
# TESTS UPDATED (§13.4/144, 6th occurrence this session — applied
# PROACTIVELY this time, not after fix-cycle):
#   test_reduce_state_bytes.py:
#     - test_legacy_op_falls_back_to_estimate_state_size (docstring)
#     - test_legacy_op_with_tight_budget_uses_1mib_fallback
#       → RENAMED test_legacy_op_fallback_scales_with_budget
#       (behavior change: 500 KB budget no longer raises)
#   test_memory_guard_v2.py:
#     - test_estimate_state_size_unknown_is_conservative
#       → RENAMED test_estimate_state_size_unknown_scales (×3 tests,
#       numeric values for 1/10/100 MB budgets)
#
# NEW TESTS (tests/test_v0925e.py):
#   - test_no_append_to_builders_ast — A.9 guard (AST)
#   - test_estimate_state_size_fallback_is_ast_scaled — A.10 AST guard
#   - test_estimate_state_size_returns_proportional — runtime scaling
#   - test_estimate_state_size_known_ops_unchanged — no regression
#   - test_no_per_row_as_py_in_sort — A.9 belt-and-suspenders (sort.py
#     post-cleanup has zero .as_py calls)
#
# LESSONS v0.9.25c/d CONSOLIDATED (this patch applies ALL):
#   §13.4/253 (per-container heuristic): AST removal by qualname,
#     not string anchors.
#   §13.4/144 (stale-test-on-refactor, 6th occurrence): grep for
#     "1mib_fallback|1 MiB|1024 * 1024" in tests/ BEFORE changing
#     A.10 — 2 tests updated PROACTIVELY in same patch.
#   §13.4/254 (harmless-over-guard): test_extra_factor_sites.SITES
#     untouched (no existing check_batch factor changed).
#
# Expected: 0-1 fix-cycle. Two independent sub-fixes, each with AST
# guard. If any test breaks it will be a specific old-test-assertion
# (fixed by apply_or_fail in fix1) — not a cascade.
#
# Run: uv run python scripts/patch_v0925e_phase_a_close.py

from __future__ import annotations
import ast
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
    print("\n=== verify: HEAD=v0.9.25d-b2, A.9/A.10 items present ===")
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "v0.9.25d-b2" not in log:
        raise SystemExit(f"expected HEAD=v0.9.25d-b2; got {log[:200]!r}")

    sort_src = Path("cadw/ops/sort.py").read_text()
    if "def _append_to_builders(" not in sort_src:
        raise SystemExit(
            "_append_to_builders not found — A.9 may be already applied"
        )

    mg_src = Path("cadw/core/memory_guard.py").read_text()
    if "return 1024 * 1024" not in mg_src:
        raise SystemExit(
            "1024 * 1024 fallback not found — A.10 may be already applied"
        )

    # Also verify diagnostic still clean (Phase A.6 baseline)
    import importlib.util as _iu
    spec = _iu.spec_from_file_location(
        "diag", "scripts/service/diagnose_unguarded_to_pylist.py",
    )
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.scan()
    if r.get("unguarded"):
        raise SystemExit(
            f"unguarded != 0; b2 not fully applied. "
            f"Sites: {[(s['file'], s['line']) for s in r['unguarded'][:3]]}"
        )
    print("  HEAD OK; A.9/A.10 items present; diagnostic unguarded=0")


# ============ 1. A.9: remove _append_to_builders ============

def remove_append_to_builders():
    print("\n=== A.9: remove _append_to_builders ===")
    path = Path("cadw/ops/sort.py")
    src = path.read_text()
    tree = ast.parse(src)
    target = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_append_to_builders"):
            target = node
            break
    if target is None:
        print("  already removed — skip")
        return

    lines = src.split("\n")
    before = lines[:target.lineno - 1]
    after = lines[target.end_lineno:]

    # Collapse to standard 2 blank lines (PEP8) between now-adjacent defs
    while before and before[-1].strip() == "":
        before.pop()
    while after and after[0].strip() == "":
        after.pop(0)
    new_src = "\n".join(before + ["", ""] + after)

    try:
        ast.parse(new_src)
    except SyntaxError as e:
        raise SystemExit(f"compile fail after removal: {e}")

    path.write_text(new_src)
    assert_compiles(path)
    check_no_invalid_escape(path)
    print(f"  removed lines {target.lineno}..{target.end_lineno}")


# ============ 2. A.10: change fallback ============

OLD_FALLBACK = '''        logger.warning("estimate_state_size: неизвестная операция '%s', "
                       "консервативно 1 MiB", op_name)
        return 1024 * 1024'''

NEW_FALLBACK = '''        # v0.9.25e (A.10): fallback scales with state_limit, not fixed
        # 1 MiB. Prior: fixed 1 MiB → false raise at tight budgets
        # (1 MiB > 250 KB state_limit at 500 KB budget) and false pass
        # at large budgets (1 MiB < 2% of state_limit at 100 MB).
        # Scaled fallback: unknown op "moderate" relative to available
        # memory. An op whose init() OOMs despite passing this fallback
        # is the op's responsibility (§11.5 documented limitation) —
        # override ReduceOp.estimate_state_nbytes() for exact bound.
        fallback = int(self._state_limit() * 0.1)
        logger.warning(
            "estimate_state_size: unknown op %r, fallback = "
            "10%% of state_limit (%d B). Override "
            "ReduceOp.estimate_state_nbytes() on the op for exact bound.",
            op_name, fallback,
        )
        return fallback'''


def change_fallback():
    print("\n=== A.10: change estimate_state_size fallback ===")
    applied = apply_or_fail(
        "cadw/core/memory_guard.py",
        OLD_FALLBACK, NEW_FALLBACK,
        label="scaled_fallback",
    )
    if applied:
        assert_compiles("cadw/core/memory_guard.py")
        check_no_invalid_escape("cadw/core/memory_guard.py")
    print("  applied" if applied else "  already applied — skip")


# ============ 3. Update existing tests (§13.4/144) ============

OLD_T1 = '''def test_legacy_op_falls_back_to_estimate_state_size(tmp_path):
    """Op without override uses estimate_state_size (1 MiB fallback)."""
    guard = MemoryGuard(max_memory_mb=100)  # 1 MiB > 100 MB? No, 1 < 100, OK
    op = LegacyOp()
    assert op.estimate_state_nbytes() is None
    # reduce() should use 1 MiB fallback (fits under 100 MB budget)
    reduce(make_iter([1.0]*5), op, guard, log=False)'''

NEW_T1 = '''def test_legacy_op_falls_back_to_estimate_state_size(tmp_path):
    """Op without override uses estimate_state_size (v0.9.25e: 10% scale)."""
    # 100 MB budget → state_limit 50 MB, fallback 5 MB → passes
    guard = MemoryGuard(max_memory_mb=100)
    op = LegacyOp()
    assert op.estimate_state_nbytes() is None
    reduce(make_iter([1.0]*5), op, guard, log=False)'''


OLD_T2 = '''def test_legacy_op_with_tight_budget_uses_1mib_fallback():
    """Legacy op (no override) → 1 MiB fallback → fits under 2 MB but not 0.5 MB."""
    # 500 KB budget → 1 MiB fallback > 500 KB * 0.5 = 250 KB → raise
    guard = MemoryGuard(max_memory_bytes=500_000)  # 500 KB
    op = LegacyOp()
    with pytest.raises(MemoryLimitExceeded):
        reduce(make_iter([1.0]*5), op, guard, log=False)'''

NEW_T2 = '''def test_legacy_op_fallback_scales_with_budget():
    """Legacy op (no override) → v0.9.25e (A.10): fallback = 10% of
    state_limit.

    Prior (<=v0.9.25d-b2): fixed 1 MiB → false raise at 500 KB budget
    (1 MiB > 250 KB state_limit). New: 10% of state_limit — proportional
    at any scale. See test_memory_guard_v2.py::
    test_estimate_state_size_unknown_scales for unit-level numeric checks.
    """
    # 500 KB budget → state_limit 250 KB, fallback 25 KB → passes
    guard = MemoryGuard(max_memory_bytes=500_000)
    op = LegacyOp()
    reduce(make_iter([1.0]*5), op, guard, log=False)
    # 100 KB budget → state_limit 50 KB, fallback 5 KB → passes
    guard = MemoryGuard(max_memory_bytes=100_000)
    op = LegacyOp()
    reduce(make_iter([1.0]*5), op, guard, log=False)'''


OLD_T3 = '''def test_estimate_state_size_unknown_is_conservative():
    g = MemoryGuard(max_memory_mb=10)
    assert g.estimate_state_size("mystery_op") == 1024 * 1024'''

NEW_T3 = '''def test_estimate_state_size_unknown_scales():
    """v0.9.25e (A.10): fallback = 10% of state_limit, not fixed 1 MiB.

    10 MB budget → state_limit 5,242,880 B; fallback = int(5,242,880 ×
    0.1) = 524,288 B. Prior (<=v0.9.25d-b2): fixed 1,048,576 B —
    false raise at 500 KB budget, false pass at 100 MB budget.
    """
    g = MemoryGuard(max_memory_mb=10)
    assert g.estimate_state_size("mystery_op") == 524_288


def test_estimate_state_size_unknown_scales_small_budget():
    """1 MB budget → state_limit 524,288 B; fallback 52,428 B."""
    g = MemoryGuard(max_memory_mb=1)
    assert g.estimate_state_size("mystery_op") == 52_428


def test_estimate_state_size_unknown_scales_large_budget():
    """100 MB budget → state_limit 52,428,800 B; fallback 5,242,880 B."""
    g = MemoryGuard(max_memory_mb=100)
    assert g.estimate_state_size("mystery_op") == 5_242_880'''


def update_existing_tests():
    print("\n=== update existing tests (proactive §13.4/144) ===")
    for path, old, new, label in [
        ("tests/test_reduce_state_bytes.py", OLD_T1, NEW_T1, "t1_docstring"),
        ("tests/test_reduce_state_bytes.py", OLD_T2, NEW_T2, "t2_rename"),
        ("tests/test_memory_guard_v2.py", OLD_T3, NEW_T3, "t3_scales"),
    ]:
        applied = apply_or_fail(path, old, new, label=label)
        if applied:
            assert_compiles(path)
            check_no_invalid_escape(path)
        print(f"  {label}: {'applied' if applied else 'skip'}")


# ============ 4. New tests file ============

NEW_TESTS = '''"""v0.9.25e: Phase A close — A.9 (dead code) + A.10 (fallback factor)."""
import ast as _ast
import inspect as _inspect
import textwrap as _tw

import pytest


# ============ A.9: _append_to_builders removed ============


def test_no_append_to_builders_ast():
    """v0.9.25e (A.9): _append_to_builders removed from ops/sort.py.

    Was v0.5.2a P5 helper: per-row .as_py() then builder.append() to
    avoid Python-list buffer overhead (peak 3-5x). v0.9.23c replaced
    merge with Arrow-native batched (pc.sort_indices), leaving
    _append_to_builders with zero callers. Per-row .as_py() is
    anti-pattern (§13.4/88) and INVISIBLE to diagnose_unguarded_
    to_pylist (scans .to_pylist, not .as_py). Audit caught it manually.
    """
    from cadw.ops import sort as sort_mod
    src = _inspect.getsource(sort_mod)
    tree = _ast.parse(src)
    defs = [
        n.name for n in _ast.walk(tree)
        if isinstance(n, _ast.FunctionDef)
        and n.name == "_append_to_builders"
    ]
    assert not defs, (
        "_append_to_builders re-introduced in ops/sort.py. Used "
        "per-row .as_py() (P5 anti-pattern). If you legitimately need "
        "per-row builder support, use a different helper."
    )


def test_no_per_row_as_py_in_sort():
    """v0.9.25e: zero .as_py() calls in ops/sort.py after A.9.

    _append_to_builders was the only per-row .as_py() in sort.py.
    Re-appearing without rationale = regression (P5 anti-pattern).
    """
    from cadw.ops import sort as sort_mod
    src = _inspect.getsource(sort_mod)
    tree = _ast.parse(src)
    as_py_calls = [
        n for n in _ast.walk(tree)
        if isinstance(n, _ast.Call)
        and isinstance(n.func, _ast.Attribute)
        and n.func.attr == "as_py"
    ]
    assert not as_py_calls, (
        f"ops/sort.py has {len(as_py_calls)} .as_py() call(s) — "
        f"v0.9.25e removed the last one (_append_to_builders). "
        f"Add new one only with explicit rationale."
    )


# ============ A.10: fallback scales ============


def test_estimate_state_size_fallback_is_ast_scaled():
    """v0.9.25e (A.10): last Return of estimate_state_size is scaled,
    not literal 1024*1024."""
    from cadw.core.memory_guard import MemoryGuard
    src = _tw.dedent(_inspect.getsource(MemoryGuard.estimate_state_size))
    tree = _ast.parse(src)
    func = tree.body[0]
    returns = [
        n for n in _ast.walk(func) if isinstance(n, _ast.Return)
    ]
    assert returns, "estimate_state_size has no return statement"
    last = returns[-1]
    if isinstance(last.value, _ast.BinOp):
        is_1mib_literal = (
            isinstance(last.value.op, _ast.Mult)
            and isinstance(last.value.left, _ast.Constant)
            and last.value.left.value == 1024
            and isinstance(last.value.right, _ast.Constant)
            and last.value.right.value == 1024
        )
        assert not is_1mib_literal, (
            "estimate_state_size fallback literal 1024*1024 — "
            "should be scaled (10% of state_limit)"
        )
    unparsed = _ast.unparse(last.value)
    assert "fallback" in unparsed or "state_limit" in unparsed, (
        f"fallback should reference scaled value, got {unparsed!r}"
    )


def test_estimate_state_size_returns_proportional():
    """Runtime: 3 budgets (1 / 10 / 100 MB) → proportionally scaled."""
    from cadw.core.memory_guard import MemoryGuard
    for mb, expected in [(1, 52_428), (10, 524_288), (100, 5_242_880)]:
        g = MemoryGuard(max_memory_mb=mb)
        got = g.estimate_state_size("__test_unknown_op__")
        assert got == expected, (
            f"budget={mb}MB: expected {expected} B (10% of state_limit), "
            f"got {got} B"
        )


def test_estimate_state_size_known_ops_unchanged():
    """Known ops keep fixed returns (no regression)."""
    from cadw.core.memory_guard import MemoryGuard
    g = MemoryGuard(max_memory_mb=10)
    assert g.estimate_state_size("count") == 16
    assert g.estimate_state_size("sum") == 16
    assert g.estimate_state_size("mean") == 16
    assert g.estimate_state_size("var") == 24
    assert g.estimate_state_size("std") == 24
    assert g.estimate_state_size("cov", d=3) == 8 + 8 * 3 + 8 * 9
    assert g.estimate_state_size("histogram", bins=10) == 8 * 10 + 16
'''


def add_new_tests():
    print("\n=== add tests/test_v0925e.py ===")
    p = Path("tests/test_v0925e.py")
    if p.exists() and "test_no_append_to_builders_ast" in p.read_text():
        print("  already present — skip")
        return
    p.write_text(NEW_TESTS)
    assert_compiles(p)
    check_no_invalid_escape(p)
    print("  OK")


# ============ 5. dogfood ============

def dogfood():
    print("\n=== dogfood: A.9 removed, A.10 scaled ===")

    def _verify():
        import ast as _ast
        import inspect as _inspect
        import textwrap as _tw
        from cadw.core.memory_guard import MemoryGuard
        from cadw.ops import sort as sort_mod

        # A.9
        sort_src = _inspect.getsource(sort_mod)
        tree = _ast.parse(sort_src)
        assert not any(
            isinstance(n, _ast.FunctionDef)
            and n.name == "_append_to_builders"
            for n in _ast.walk(tree)
        ), "_append_to_builders still present"
        as_py_count = sum(
            1 for n in _ast.walk(tree)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "as_py"
        )
        assert as_py_count == 0, f"sort.py has {as_py_count} .as_py() calls"

        # A.10: numeric values
        for mb, expected in [(1, 52_428), (10, 524_288), (100, 5_242_880)]:
            g = MemoryGuard(max_memory_mb=mb)
            got = g.estimate_state_size("__test_unknown_op__")
            assert got == expected, (
                f"budget={mb}MB: expected {expected}, got {got}"
            )

        # A.10: known ops unchanged
        g = MemoryGuard(max_memory_mb=10)
        for name, expected in [
            ("count", 16), ("sum", 16), ("mean", 16),
            ("var", 24), ("std", 24),
            ("cov", 8 + 8*3 + 8*9),
            ("histogram", 8*10+16),
        ]:
            if name == "cov":
                got = g.estimate_state_size(name, d=3)
            elif name == "histogram":
                got = g.estimate_state_size(name, bins=10)
            else:
                got = g.estimate_state_size(name)
            assert got == expected, f"{name}: expected {expected}, got {got}"

        print(
            "  A.9 removed (0 .as_py in sort.py); "
            "A.10 scaled (1/10/100 MB → 52K/524K/5.2M); known unchanged"
        )

    run_verify_in_subprocess(_verify)


# ============ 6. pytest ============

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


# ============ 7. main ============

def main():
    print("=== v0.9.25e: Phase A close (A.9 + A.10) ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  branch: {br}")

    verify_state()
    remove_append_to_builders()
    change_fallback()
    update_existing_tests()
    add_new_tests()
    dogfood()
    run_pytest()

    commit_with_guards(
        "v0.9.25e: Phase A close — A.9 (dead code) + A.10 (fallback "
        "factor). "
        "A.9: removed `_append_to_builders` from cadw/ops/sort.py. "
        "Was v0.5.2a P5 helper (per-row .as_py() then builder.append() "
        "to avoid Python-list buffer overhead). v0.9.23c replaced merge "
        "with Arrow-native batched (pc.sort_indices), leaving helper "
        "with zero callers. Per-row .as_py() is anti-pattern and "
        "INVISIBLE to diagnostic (scans .to_pylist, not .as_py) — "
        "audit caught it manually. AST-based removal (lineno.."
        "end_lineno), PEP8 blank-line normalization. "
        "A.10: MemoryGuard.estimate_state_size fallback 1024*1024 → "
        "int(self._state_limit() * 0.1). Prior fixed 1 MiB caused "
        "false raise at tight budgets (1 MiB > 250 KB state_limit at "
        "500 KB) and false pass at large (1 MiB < 2% of 50 MB at "
        "100 MB). New: 10% of state_limit — proportional at any scale. "
        "Warning message updated to clear English + guidance "
        "(override estimate_state_nbytes). §11.5 documented limitation: "
        "op whose init() OOMs despite passing scaled fallback — op's "
        "responsibility. "
        "Tests updated PROACTIVELY (§13.4/144, 6th occurrence this "
        "session — grep-tests-before-change, not after fix-cycle): "
        "test_reduce_state_bytes.py::test_legacy_op_falls_back_to_"
        "estimate_state_size (docstring 1MiB→10% scale) + "
        "test_legacy_op_with_tight_budget_uses_1mib_fallback → "
        "renamed test_legacy_op_fallback_scales_with_budget (500 KB "
        "no longer raises); test_memory_guard_v2.py::"
        "test_estimate_state_size_unknown_is_conservative → renamed "
        "test_estimate_state_size_unknown_scales ×3 (1/10/100 MB "
        "numeric values). "
        "New tests/test_v0925e.py: 5 tests — A.9 AST guard "
        "(no _append_to_builders, no .as_py); A.10 AST guard "
        "(fallback not literal 1024*1024) + runtime scaling (3 budgets) "
        "+ known-ops unchanged. "
        "Result: sort.py has 0 .as_py() calls (was 1); "
        "estimate_state_size('unknown', 10 MB) = 524,288 (was 1,048,576); "
        "500 KB budget no longer false-raises for tiny op. "
        "§13.4/253 (per-container heuristic): AST removal by qualname, "
        "not string anchors. §13.4/254 (harmless-over-guard): no "
        "existing check_batch factor changed — SITES untouched. "
        "Fast suite: ~2299 → ~2305 (+5 new tests, +1 net from t2/t3 "
        "renames replacing 1 test with 3). "
        "Phase A CLOSED (A.4–A.10 all done). "
        "Next: Phase B — dataframe.py split (~4h, 7 modules).",
        self_path="scripts/patch_v0925e_phase_a_close.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.25e applied. Phase A COMPLETE (A.4–A.10). "
          "Next: Phase B — dataframe.py split.")


if __name__ == "__main__":
    main()