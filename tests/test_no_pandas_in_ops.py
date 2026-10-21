"""v0.8.11a: AST guard — pandas removed from ops hot path.

After v0.8.11a, `cadw/ops/` has zero `import pandas` except in
`shuffle.py::_pandas_hash_column` — a documented perf exception
(pandas C-hash is 19–40× faster than Python fallback for
int/float/timestamp; §9.2.20).

Principle §3.11 (Arrow call memory discipline) is enforced separately in
`test_arrow_call_memory_guard.py`. This file guards the narrower claim:
no NEW pandas import sneaks into ops without an explicit decision.

If you need pandas in an op, whitelist it here with a rationale
(same discipline as `WHITELISTED_FUNCS` in test_arrow_call_memory_guard).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


CADW_OPS = Path(__file__).resolve().parent.parent / "cadw" / "ops"

# Modules that may import pandas, with rationale.
ALLOWED_PANDAS_MODULES: dict[str, str] = {
    # v0.9.6a: shuffle.py no longer imports pandas (own XXH64 hash).
    # Every other module in cadw/ops/ was already pandas-free
    # (v0.8.11a P-D1 Arrow-native dedup). Empty by design —
    # any new entry requires explicit rationale + review.
}


def _uses_pandas_import(tree: ast.Module) -> bool:
    """True if module has `import pandas` or `from pandas ...`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pandas" or alias.name.startswith("pandas."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == "pandas"
                or node.module.startswith("pandas.")
            ):
                return True
    return False


def _uses_pandas_import_at_module_level(tree: ast.Module) -> bool:
    """True if module-level (top-level) import of pandas.

    v0.9.1e-fix2 (§13.4/155): lazy `import pandas as pd` inside a
    function is OK — keeps `import cadw` light, pandas only loaded when
    needed (e.g., `_per_group_apply` constructs pd.Series/DataFrame from
    user fn output). Module-level import would eagerly load pandas.
    """
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pandas" or alias.name.startswith("pandas."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == "pandas"
                or node.module.startswith("pandas.")
            ):
                return True
    return False


def test_no_pandas_in_ops_except_shuffle():
    """`cadw/ops/*.py` — no `import pandas` outside whitelist."""
    violations: list[str] = []
    for py in sorted(CADW_OPS.glob("*.py")):
        if py.name in ALLOWED_PANDAS_MODULES:
            continue
        src = py.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        if _uses_pandas_import(tree):
            violations.append(py.name)

    assert not violations, (
        "Unexpected `import pandas` in cadw/ops/: "
        f"{violations}. "
        "Pandas was removed from hot path in v0.8.11a (P-D1). "
        f"Whitelist rationale: {ALLOWED_PANDAS_MODULES}. "
        "If needed, add to ALLOWED_PANDAS_MODULES with a rationale."
    )


def test_pandas_whitelist_has_rationale():
    """Every whitelisted module has a non-empty rationale."""
    for name, reason in ALLOWED_PANDAS_MODULES.items():
        assert isinstance(name, str) and name.endswith(".py")
        assert isinstance(reason, str) and reason, (
            f"Whitelisted {name!r} must have a rationale string"
        )


def test_no_pandas_in_dataframe_facade():
    """`cadw/dataframe.py` — no MODULE-LEVEL pandas import.

    v0.9.1e-fix2 (§13.4/155): lazy `import pandas as pd` inside
    `_emit_group` (used by `_per_group_apply` to construct pd.Series/
    pd.DataFrame from user fn output) is legitimate — keeps `import
    cadw` light. Only top-level import is forbidden.
    """
    py = CADW_OPS.parent / "dataframe.py"
    src = py.read_text()
    tree = ast.parse(src)
    assert not _uses_pandas_import_at_module_level(tree), (
        "cadw/dataframe.py must not import pandas at module level "
        "(lazy import inside functions OK — see §13.4/155)"
    )


def test_dedup_uses_arrow_native():
    """AST: `_local_dedup` uses dictionary_encode + is_in (Path C)."""
    import inspect
    from cadw.ops import dedup as dedup_mod
    src = inspect.getsource(dedup_mod._local_dedup)
    assert "dictionary_encode" in src, (
        "_local_dedup must use pc.dictionary_encode (Path C, v0.8.11a)"
    )
    assert "is_in" in src, (
        "_local_dedup must use pc.is_in (Path C, v0.8.11a)"
    )
    assert "to_pandas" not in src, (
        "_local_dedup must NOT call to_pandas (P-D1 regression)"
    )
    assert "sort_values" not in src, (
        "_local_dedup must NOT call pandas sort_values (P-D1 regression)"
    )


def test_dedup_no_pandas_import():
    """AST: `cadw/ops/dedup.py` has no `import pandas`."""
    py = CADW_OPS / "dedup.py"
    tree = ast.parse(py.read_text())
    assert not _uses_pandas_import(tree), (
        "cadw/ops/dedup.py must not import pandas (P-D1 closed v0.8.11a)"
    )


def test_join_no_pandas_import():
    """AST: `cadw/ops/join.py` has no `import pandas`."""
    py = CADW_OPS / "join.py"
    tree = ast.parse(py.read_text())
    assert not _uses_pandas_import(tree), (
        "cadw/ops/join.py must not import pandas (v0.8.11a cleanup)"
    )
