"""G1: primitive reuse — no n-sized state without SpillDir/Reduce.

Flags np.zeros/empty/full/ones with first shape dim in a full-n
convention set (n_total, N, n_obs, n_full, n_samples) unless the
function is whitelisted in internal refactor plan (retired v0.9.223).

Whitelisted today (refactor pending):
  cadw/ml/gbdt.py::fit_gbdt_multiclass    (P1 via SpillDir)
  cadw/ml/lambdamart.py::fit_lambdamart   (P2 via SpillDir; not
                                            caught by current pattern
                                            because uses 'n' - audit
                                            flagged vstack instead)

Gate purpose: prevent NEW n-sized allocations from being added.
Existing violations tracked in plan whitelist; refactor removes.

See METHODOLOGY_ALGORITHM_SELECTION.md §6, §8 (gate G1).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests._gate_whitelist import is_whitelisted, load_whitelist

CADW = Path(__file__).parent.parent / "cadw"
ROOT = Path(__file__).parent.parent

FULL_N_NAMES = {"n_total", "N", "n_obs", "n_full", "n_samples"}
ALLOC_ATTRS = {"zeros", "empty", "full", "ones"}


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


def _is_full_n_alloc(call: ast.Call) -> bool:
    """np.{zeros,empty,full,ones}((n_total, ...)) - full-n pattern."""
    if not isinstance(call.func, ast.Attribute):
        return False
    if call.func.attr not in ALLOC_ATTRS:
        return False
    recv = call.func.value
    if not (isinstance(recv, ast.Name) and recv.id == "np"):
        return False
    if not call.args:
        return False
    shape = call.args[0]
    if not isinstance(shape, (ast.Tuple, ast.List)):
        return False
    if not shape.elts:
        return False
    first = shape.elts[0]
    if isinstance(first, ast.Name) and first.id in FULL_N_NAMES:
        return True
    return False


@pytest.mark.parametrize(
    "path", list(_walk_ml_ts()),
    ids=lambda p: str(p.relative_to(CADW)),
)
def test_g1_primitive_reuse(path):
    """G1: no full-n np.* alloc in cadw/ml/ or cadw/ts/ unless whitelisted."""
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        pytest.fail(f"{path.relative_to(ROOT)}: syntax error {e}")

    rel = str(path.relative_to(ROOT))
    wl = load_whitelist()["G1"]

    violations: list[tuple[str, int]] = []
    for fn in _iter_functions(tree):
        key = f"{rel}::{fn.name}"
        if is_whitelisted(wl, key):
            continue
        for call in ast.walk(fn):
            if isinstance(call, ast.Call) and _is_full_n_alloc(call):
                violations.append((fn.name, call.lineno))

    assert not violations, (
        f"{rel}: n-sized allocation without SpillDir/Reduce. "
        f"Violations: {violations}. Either refactor to streaming "
        f"(Reduce, or SpillDir + ArrowIPC per METHODOLOGY §5.3), or "
        f"add to internal refactor plan (retired) gate-whitelist with "
        f"a backlog item (e.g. 'P1', 'P2')."
    )
