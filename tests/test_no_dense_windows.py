"""G3: no dense windows — sliding_window_view(...).copy() is forbidden.

Flags any `.copy()` applied to a `sliding_window_view(...)` call in
cadw/. The result is a full O(n*w) dense array. Instead: keep the
strided view (0 bytes) and consume in chunks (O(chunk*w) RAM).

Whitelisted today (refactor pending — see plan gate-whitelist):
  cadw/ml/sub_pca.py::_prepare_windows     (P4 strided)
  cadw/ml/s2g.py::_score_queries           (P4 strided)
  cadw/ml/kshape.py::_sliding_windows      (P3 2-pass)
  cadw/ml/dwt_mlead.py::_sliding_windows   (P4 strided)
  cadw/ml/patch_memory_ad.py::fit_patch_memory_ad  (P4 inline)

Audit prototype (FINNEGAN; see internal audit):
  45.7 MB dense -> 4.9 MB strided, 5.4x faster, identical output.

See METHODOLOGY §5 (windowed state_type), §8 (gate G3).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests._gate_whitelist import is_whitelisted, load_whitelist

CADW = Path(__file__).parent.parent / "cadw"
ROOT = Path(__file__).parent.parent


def _walk_cadw():
    for p in sorted(CADW.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        yield p


def _is_sliding_copy(call: ast.Call) -> bool:
    """Detect .copy() on a Call to sliding_window_view(...)."""
    if not isinstance(call.func, ast.Attribute):
        return False
    if call.func.attr != "copy":
        return False
    recv = call.func.value
    if not isinstance(recv, ast.Call):
        return False
    if not isinstance(recv.func, ast.Attribute):
        return False
    if recv.func.attr != "sliding_window_view":
        return False
    return True


def _enclosing_function(tree, line: int):
    best = None
    best_end = 10**18
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        if node.lineno <= line <= end and end < best_end:
            best = node
            best_end = end
    return best


@pytest.mark.parametrize(
    "path", list(_walk_cadw()),
    ids=lambda p: str(p.relative_to(CADW)),
)
def test_g3_no_dense_windows(path):
    """G3: no sliding_window_view(...).copy() unless whitelisted."""
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        pytest.fail(f"{path.relative_to(ROOT)}: syntax error {e}")

    rel = str(path.relative_to(ROOT))
    wl = load_whitelist()["G3"]

    violations: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_sliding_copy(node):
            continue
        enclosing = _enclosing_function(tree, node.lineno)
        fname = enclosing.name if enclosing else "<module>"
        key = f"{rel}::{fname}"
        if is_whitelisted(wl, key):
            continue
        violations.append((fname, node.lineno))

    assert not violations, (
        f"{rel}: sliding_window_view(...).copy() without whitelist. "
        f"Violations: {violations}. Use strided view + chunked "
        f"consumption (METHODOLOGY §5 windowed state_type; audit "
        f"prototype: 45.7 MB -> 4.9 MB, 5.4x faster), or add to "
        f"internal refactor plan (retired) gate-whitelist with a "
        f"backlog item (P3/P4)."
    )
