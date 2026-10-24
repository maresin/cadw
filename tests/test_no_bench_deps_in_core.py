"""AST guard: cadw/ must not import benchmark-only reference libraries.

Rationale: requirements-bench.txt contains libraries used only for
benchmark reproduction in scripts/ and tests/. Core cadw/ must remain
dependency-light (v0.9.43, Phase G).

Mirror of tests/test_no_pandas_in_ops.py — same pattern, different set.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

CADW = Path(__file__).parent.parent / "cadw"

# Must match requirements-bench.txt.
BENCH_ONLY = {
    "pyod", "ruptures", "lifelines", "implicit",
    "statsmodels", "sklearn",
}


def _walk_py():
    for p in sorted(CADW.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        yield p


def _imported_modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module.split(".")[0]


@pytest.mark.parametrize(
    "path", list(_walk_py()),
    ids=lambda p: str(p.relative_to(CADW)),
)
def test_no_bench_dep_imports(path):
    """cadw/<path> does not import benchmark-only reference libraries."""
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        pytest.fail(f"{path.relative_to(CADW)}: syntax error {e}")
    bad = set(_imported_modules(tree)) & BENCH_ONLY
    assert not bad, (
        f"{path.relative_to(CADW)}: imports benchmark-only libraries "
        f"{sorted(bad)}. These live in requirements-bench.txt and must "
        f"be used only in scripts/ or tests/ (v0.9.43 AST guard)."
    )
