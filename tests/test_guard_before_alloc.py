"""G2: guard before alloc — check_state/check_batch before n-sized alloc.

For any function with a full-n allocation (same pattern as G1),
require that a guard.check_state or guard.check_batch call appears
at a lower line number in the same function.

This is stricter than G1: even whitelisted functions must have a
guard BEFORE the alloc (correct order). Whitelisted G1 functions
can pass G2 (e.g. gbdt multiclass has check_state before
np.zeros((n_total, K)) - correct).

Violation = a function that allocates n-sized state with NO prior
guard, meaning the guard raises too late (OOM possible on tight
budgets). All fix1a/b/c changes targeted this pattern.

See METHODOLOGY §6, §8 (gate G2).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

CADW = Path(__file__).parent.parent / "cadw"
ROOT = Path(__file__).parent.parent

FULL_N_NAMES = {"n_total", "N", "n_obs", "n_full", "n_samples"}
ALLOC_ATTRS = {"zeros", "empty", "full", "ones"}
GUARD_METHODS = {"check_state", "check_batch"}


def _walk_ml_ts():
    for sub in ("ml", "ts"):
        for p in sorted((CADW / sub).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            yield p


def _iter_functions(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            yield node


def _full_n_allocs(fn: ast.FunctionDef) -> list[int]:
    """Return line numbers of full-n allocs inside fn."""
    out = []
    for call in ast.walk(fn):
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr not in ALLOC_ATTRS:
            continue
        recv = call.func.value
        if not (isinstance(recv, ast.Name) and recv.id == "np"):
            continue
        if not call.args:
            continue
        shape = call.args[0]
        if not isinstance(shape, (ast.Tuple, ast.List)):
            continue
        if not shape.elts:
            continue
        first = shape.elts[0]
        if isinstance(first, ast.Name) and first.id in FULL_N_NAMES:
            out.append(call.lineno)
    return out


def _has_guard_before(fn: ast.FunctionDef, alloc_line: int) -> bool:
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in GUARD_METHODS:
            continue
        if node.lineno < alloc_line:
            return True
    return False


@pytest.mark.parametrize(
    "path", list(_walk_ml_ts()),
    ids=lambda p: str(p.relative_to(CADW)),
)
def test_g2_guard_before_alloc(path):
    """G2: for each n-sized alloc, guard must precede it in same fn."""
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        pytest.fail(f"{path.relative_to(ROOT)}: syntax error {e}")

    rel = str(path.relative_to(ROOT))
    violations: list[tuple[str, int]] = []
    for fn in _iter_functions(tree):
        for alloc_line in _full_n_allocs(fn):
            if not _has_guard_before(fn, alloc_line):
                violations.append((fn.name, alloc_line))

    assert not violations, (
        f"{rel}: n-sized alloc(s) without prior guard.check_state/"
        f"check_batch. Violations: {violations}. Move guard before "
        f"allocation (see fix1a-c pattern), or refactor to Reduce/"
        f"SpillDir (METHODOLOGY §5)."
    )
