"""v0.9.25e: Phase A close — A.9 (dead code) + A.10 (fallback factor)."""
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
    # §13.4/255: ast.walk is BFS, not source-order. Direct-child
    # Return(fallback) is yielded in Level 1; nested branch Returns
    # (Return(16), Return(24), ..., Return(8*bins+16)) are yielded
    # in Level 2 — AFTER. So `returns[-1]` picks the last Level-2
    # (histogram branch), not the fallback. Correct: order by source
    # line — `max(returns, key=lambda n: n.lineno)`.
    last = max(returns, key=lambda n: n.lineno)
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
