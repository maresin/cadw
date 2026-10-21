"""v0.8.11a: AST guard — pandas removed from ops hot path.

After v0.8.11a, `cadw/ops/` has zero `import pandas` except in
v0.9.6a replaced `_pandas_hash_column` with own XXH64-based
hash (`_vectorized_hash_numeric` + `_string_hash_column`).
`cadw/ops/` is pandas-free.

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
    """`cadw/dataframe.py` — no pandas import anywhere (v0.9.6b2).

    v0.9.6b2: _per_group_apply is CADW-native (fn returns
    scalar | dict | CadwDataFrame). No `import pandas` at
    module level or inside functions. Remaining pandas touch
    in dataframe.py is `Table.to_pandas()` in pivot_table —
    that is a METHOD, not an import; addressed separately
    (v0.9.6c PivotOp).
    """
    py = CADW_OPS.parent / "dataframe.py"
    src = py.read_text()
    tree = ast.parse(src)
    assert not _uses_pandas_import(tree), (
        "cadw/dataframe.py must not import pandas anywhere "
        "(v0.9.6b2: _per_group_apply is CADW-native). "
        "If a new lazy import is genuinely required, add a "
        "whitelist rationale — do not weaken this test."
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


def test_no_pandas_in_per_group_apply():
    """_per_group_apply + helpers must not import pandas
    (v0.9.6b2)."""
    import ast as _ast
    import inspect as _inspect
    import textwrap as _textwrap
    from cadw import dataframe as df_mod

    targets = [
        "_per_group_apply", "_detect_mode", "_normalize_scalar",
        "_build_out_schema", "_infer_arrow_type_from_value",
    ]
    for name in targets:
        fn = getattr(df_mod, name, None)
        if fn is None:
            continue
        src = _inspect.getsource(fn)
        tree = _ast.parse(_textwrap.dedent(src))
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith('pandas'), (
                        f'{name} imports pandas; must be '
                        'CADW-native'
                    )
            elif isinstance(node, _ast.ImportFrom):
                if node.module and node.module.startswith('pandas'):
                    raise AssertionError(
                        f'{name} imports from pandas; must be '
                        'CADW-native'
                    )
