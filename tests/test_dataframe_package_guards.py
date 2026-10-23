"""v0.9.26h2-fix1: package-wide AST guards for cadw/dataframe/.

Complements per-file guards (test_facade_has_no_as_py in
test_dataframe.py, which only checks _core.py). These tests
iterate ALL modules in cadw/dataframe/, including extracted
modules like _exports.py and future B.1 extractions.
"""
import ast
from pathlib import Path

import pytest


PKG = Path("cadw/dataframe")


def _iter_package_sources():
    for py in sorted(PKG.rglob("*.py")):
        yield py, py.read_text()


def test_no_as_py_in_dataframe_package():
    """v0.9.26h2 (§13.4/292): no .as_py() anywhere in cadw/dataframe/."""
    violations = []
    for py, src in _iter_package_sources():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "as_py"):
                violations.append(f"{py}:{node.lineno}")
    assert not violations, (
        ".as_py() found in cadw/dataframe/ modules: "
        + str(violations)
        + ". Facade must be Arrow-native; use pc.* ops, not .as_py()."
    )


def test_no_row_loops_in_dataframe_package():
    """v0.9.26h2: package-wide row-loop guard."""
    violations = []
    forbidden = (".iterrows(", ".itertuples(")
    for py, src in _iter_package_sources():
        for i, ln in enumerate(src.splitlines(), 1):
            for pat in forbidden:
                if pat in ln:
                    violations.append(f"{py}:{i} {pat}")
    assert not violations, (
        "Row loops in cadw/dataframe/ modules: "
        + str(violations)
        + ". Facade must be batch-native."
    )


def test_no_pandas_import_in_dataframe_package():
    """v0.9.26h2: package-wide pandas-import guard."""
    violations = []
    for py, src in _iter_package_sources():
        for i, ln in enumerate(src.splitlines(), 1):
            stripped = ln.strip()
            if (stripped.startswith("import pandas")
                    or stripped.startswith("from pandas")):
                violations.append(f"{py}:{i} {stripped[:60]}")
    assert not violations, (
        "pandas import in cadw/dataframe/ modules: "
        + str(violations)
        + ". Package must not import pandas (opt-in only)."
    )


def test_package_modules_listed():
    """Sanity: expected modules present in package."""
    found = {py.name for py, _ in _iter_package_sources()}
    assert "__init__.py" in found, "missing __init__.py"
    assert "_core.py" in found, "missing _core.py"
    assert "_exports.py" in found, (
        "_exports.py not found. Actual: " + str(sorted(found))
    )
