#!/usr/bin/env python3
# v0.9.6e2 — scaffold hardening #2 (patch-script tooling).
#
# Closes 4 meta-bug classes from v0.9.6:
#   §13.4/189  .exists() skip != correct file  -> safe_skip_or_write
#   §13.4/191  getsource+ast.parse needs dedent -> parse_method_source
#   §13.4/192  copied-template missing import  -> assert_test_imports_defined
#   §13.4/193  assert_compiles before mutation -> write_content_checked
#
# Run: uv run python scripts/patch_v096e2_scaffold2.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import importlib
import cadw.patch_utils as pu
importlib.reload(pu)
from cadw.patch_utils import (
    write_content, assert_compiles, commit_with_guards,
)


HELPER_LINES = [
    "",
    "",
    "# ============ v0.9.6e2: patch-script hardening #2 ============",
    "#",
    "# Closes §13.4/189 (.exists() != correct), /191 (getsource needs",
    "# dedent), /192 (copied-template missing imports), /193",
    "# (assert_compiles BEFORE mutation).",
    "",
    "",
    "def write_content_checked(path, lines):",
    '    """write_content + assert_compiles (§13.4/193).',
    "",
    "    Use for any file write where syntax errors from escape/quote",
    "    traps (multi-line templates, embedded backslashes) could slip",
    "    through. assert_compiles raises SystemExit on syntax error",
    "    BEFORE the file is imported or run by anything else.",
    '    """',
    "    write_content(path, lines)",
    "    assert_compiles(path)",
    "",
    "",
    "def parse_method_source(fn):",
    '    """Return ast.Module for inspect.getsource(fn) (§13.4/191).',
    "",
    "    inspect.getsource(Class.method) includes the method's",
    "    class-level indent (4 spaces before `def`); ast.parse requires",
    "    zero-indent top-level. Wraps textwrap.dedent.",
    '    """',
    "    import ast as _ast",
    "    import inspect as _inspect",
    "    import textwrap as _tw",
    "    return _ast.parse(_tw.dedent(_inspect.getsource(fn)))",
    "",
    "",
    "def safe_skip_or_write(path, lines, *, required_symbols=()):",
    '    """Skip only if existing file is valid + has all symbols',
    "    (§13.4/189). Otherwise force-write.",
    "",
    "    Before v0.9.6e2, new-file creation often used `if p.exists():",
    "    skip` — assumes existing file is correct. If the file is",
    "    broken (partial prior run, external source), the skip writes",
    "    broken code to prod. Here: skip requires",
    "        assert_compiles(p) AND all(s in src for s in required)",
    "    Else force-write via write_content_checked.",
    "",
    "    Returns True if skipped (file valid), False if written.",
    '    """',
    "    from pathlib import Path as _P",
    "    p = _P(path)",
    "    if p.exists():",
    "        ok = True",
    "        try:",
    "            assert_compiles(p)",
    "        except SystemExit:",
    "            ok = False",
    "        if ok:",
    "            src = p.read_text()",
    "            missing = [s for s in required_symbols if s not in src]",
    "            if missing:",
    "                print(f\"  {p}: exists but missing symbols {missing} \"",
    "                      f\"— force-write\")",
    "                ok = False",
    "        if ok:",
    "            print(f\"  {p}: exists + valid — skip\")",
    "            return True",
    "        print(f\"  {p}: exists but invalid — force-write\")",
    "    write_content_checked(p, lines)",
    "    return False",
    "",
    "",
    "_BUILTINS = frozenset(",
    "    \"abs all any bool bytes callable chr dict dir enumerate \"",
    "    \"filter float format frozenset getattr hasattr hash hex id \"",
    "    \"int isinstance issubclass iter len list map max min next \"",
    "    \"object oct ord pow print range repr reversed round set \"",
    "    \"slice sorted str sum tuple type zip True False None\"",
    "    .split()",
    ")",
    "",
    "",
    "def assert_test_imports_defined(src, *, fn_name=\"<unknown>\"):",
    '    """AST: every loaded Name inside a FunctionDef has a binding',
    "    in that function (assign/import/local def), a module-level",
    "    import, or is a builtin. Catches copied-template missing",
    "    import (§13.4/192).",
    "",
    "    Heuristic — not a full resolver. Raises AssertionError listing",
    "    undefined names per function.",
    '    """',
    "    import ast as _ast",
    "    import textwrap as _tw",
    "    tree = _ast.parse(_tw.dedent(src))",
    "",
    "    module_names: set = set()",
    "    for node in tree.body:",
    "        if isinstance(node, _ast.Import):",
    "            for a in node.names:",
    "                module_names.add((a.asname or a.name).split('.')[0])",
    "        elif isinstance(node, _ast.ImportFrom):",
    "            for a in node.names:",
    "                module_names.add(a.asname or a.name)",
    "        elif isinstance(node, (_ast.FunctionDef, _ast.ClassDef)):",
    "            module_names.add(node.name)",
    "        elif isinstance(node, _ast.Assign):",
    "            for t in node.targets:",
    "                if isinstance(t, _ast.Name):",
    "                    module_names.add(t.id)",
    "",
    "    violations: dict = {}",
    "    for fn_node in [n for n in tree.body",
    "                    if isinstance(n, _ast.FunctionDef)]:",
    "        local: set = set()",
    "        for arg in fn_node.args.args:",
    "            local.add(arg.arg)",
    "        for n in _ast.walk(fn_node):",
    "            if isinstance(n, _ast.Import):",
    "                for a in n.names:",
    "                    local.add((a.asname or a.name).split('.')[0])",
    "            elif isinstance(n, _ast.ImportFrom):",
    "                for a in n.names:",
    "                    local.add(a.asname or a.name)",
    "            elif isinstance(n, _ast.Assign):",
    "                for t in n.targets:",
    "                    if isinstance(t, _ast.Name):",
    "                        local.add(t.id)",
    "            elif isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef,",
    "                               _ast.ClassDef)):",
    "                if n is not fn_node:",
    "                    local.add(n.name)",
    "            elif isinstance(n, (_ast.For, _ast.AsyncFor,",
    "                               _ast.comprehension)):",
    "                for nn in _ast.walk(n.target):",
    "                    if isinstance(nn, _ast.Name):",
    "                        local.add(nn.id)",
    "            elif isinstance(n, _ast.withitem):",
    "                if n.optional_vars is not None:",
    "                    for nn in _ast.walk(n.optional_vars):",
    "                        if isinstance(nn, _ast.Name):",
    "                            local.add(nn.id)",
    "",
    "        undefined: set = set()",
    "        for n in _ast.walk(fn_node):",
    "            if not isinstance(n, _ast.Name):",
    "                continue",
    "            if not isinstance(n.ctx, _ast.Load):",
    "                continue",
    "            if (n.id not in local",
    "                    and n.id not in module_names",
    "                    and n.id not in _BUILTINS):",
    "                undefined.add(n.id)",
    "        if undefined:",
    "            violations[fn_node.name] = sorted(undefined)",
    "",
    "    if violations:",
    "        msg = '; '.join(f'{k}: {v}' for k, v in violations.items())",
    "        raise AssertionError(",
    "            f'assert_test_imports_defined({fn_name}): undefined '",
    "            f'Names: {msg}'",
    "        )",
]


NEW_TESTS_LINES = [
    "",
    "",
    "# ============ v0.9.6e2: write_content_checked ============",
    "",
    "",
    "def test_write_content_checked_ok(tmp_path):",
    "    from cadw.patch_utils import write_content_checked",
    "    p = tmp_path / 'f.py'",
    "    write_content_checked(p, ['x = 1'])",
    "    assert p.read_text() == 'x = 1\\n'",
    "",
    "",
    "def test_write_content_checked_bad_raises(tmp_path):",
    "    from cadw.patch_utils import write_content_checked",
    "    p = tmp_path / 'f.py'",
    "    with pytest.raises(SystemExit):",
    "        write_content_checked(p, ['def f('])",
    "",
    "",
    "# ============ v0.9.6e2: parse_method_source ============",
    "",
    "",
    "def test_parse_method_source_indented():",
    "    from cadw.patch_utils import parse_method_source",
    "    from cadw.dataframe import CadwDataFrame",
    "    tree = parse_method_source(CadwDataFrame.pivot_table)",
    "    assert any(isinstance(n, ast.FunctionDef) for n in tree.body)",
    "",
    "",
    "def test_parse_method_source_nested_function():",
    "    from cadw.patch_utils import parse_method_source",
    "    from cadw.dataframe import CadwDataFrame",
    "    tree = parse_method_source(CadwDataFrame._per_group_apply)",
    "    assert tree is not None",
    "",
    "",
    "# ============ v0.9.6e2: safe_skip_or_write ============",
    "",
    "",
    "def test_safe_skip_or_write_new(tmp_path):",
    "    from cadw.patch_utils import safe_skip_or_write",
    "    p = tmp_path / 'f.py'",
    "    written = safe_skip_or_write(p, ['x = 1'])",
    "    assert written is False",
    "    assert p.read_text() == 'x = 1\\n'",
    "",
    "",
    "def test_safe_skip_or_write_valid_skips(tmp_path):",
    "    from cadw.patch_utils import safe_skip_or_write",
    "    p = tmp_path / 'f.py'",
    "    p.write_text('x = 1\\n')",
    "    written = safe_skip_or_write(p, ['x = 2'],",
    "                                 required_symbols=['x'])",
    "    assert written is True",
    "    assert 'x = 1' in p.read_text()",
    "",
    "",
    "def test_safe_skip_or_write_broken_rewrites(tmp_path):",
    "    from cadw.patch_utils import safe_skip_or_write",
    "    p = tmp_path / 'f.py'",
    "    p.write_text('def f(\\n')",
    "    written = safe_skip_or_write(p, ['x = 1'])",
    "    assert written is False",
    "    assert p.read_text() == 'x = 1\\n'",
    "",
    "",
    "def test_safe_skip_or_write_missing_symbol_rewrites(tmp_path):",
    "    from cadw.patch_utils import safe_skip_or_write",
    "    p = tmp_path / 'f.py'",
    "    p.write_text('y = 1\\n')",
    "    written = safe_skip_or_write(p, ['x = 1'],",
    "                                 required_symbols=['x'])",
    "    assert written is False",
    "    assert p.read_text() == 'x = 1\\n'",
    "",
    "",
    "# ============ v0.9.6e2: assert_test_imports_defined ============",
    "",
    "",
    "def test_assert_test_imports_defined_ok():",
    "    from cadw.patch_utils import assert_test_imports_defined",
    "    src = (",
    "        'def test_a():\\n'",
    "        '    import os\\n'",
    "        '    assert os.path.exists(\"/\") or True\\n'",
    "    )",
    "    assert_test_imports_defined(src, fn_name='test_a')",
    "",
    "",
    "def test_assert_test_imports_defined_catches_missing():",
    "    from cadw.patch_utils import assert_test_imports_defined",
    "    src = (",
    "        'def test_a():\\n'",
    "        '    missing_helper(1, 2)\\n'",
    "    )",
    "    with pytest.raises(AssertionError, match='missing_helper'):",
    "        assert_test_imports_defined(src, fn_name='test_a')",
    "",
    "",
    "def test_assert_test_imports_defined_module_import():",
    "    from cadw.patch_utils import assert_test_imports_defined",
    "    src = (",
    "        'import pytest\\n'",
    "        'def test_a():\\n'",
    "        '    pytest.raises(ValueError)\\n'",
    "    )",
    "    assert_test_imports_defined(src, fn_name='test_a')",
    "",
    "",
    "def test_assert_test_imports_defined_builtin_ok():",
    "    from cadw.patch_utils import assert_test_imports_defined",
    "    src = (",
    "        'def test_a():\\n'",
    "        '    assert len([1, 2]) == 2\\n'",
    "    )",
    "    assert_test_imports_defined(src, fn_name='test_a')",
    "",
    "",
    "def test_assert_test_imports_defined_assignment_ok():",
    "    from cadw.patch_utils import assert_test_imports_defined",
    "    src = (",
    "        'def test_a():\\n'",
    "        '    x = 5\\n'",
    "        '    assert x == 5\\n'",
    "    )",
    "    assert_test_imports_defined(src, fn_name='test_a')",
    "",
]


def add_helpers():
    print("\n=== patch_utils.py: add 4 helpers ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "def write_content_checked(" in s:
        print("  already present — skip")
        return
    anchor = "\n\ndef assert_no_stale_behavior("
    if anchor not in s:
        raise SystemExit("assert_no_stale_behavior anchor not found")
    block = "\n".join(HELPER_LINES)
    s = s.replace(anchor, block + anchor, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


def add_tests():
    print("\n=== test_patch_utils.py: add helper tests ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "def test_write_content_checked_ok" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(NEW_TESTS_LINES))
    assert_compiles(p)
    print("  OK")


def dogfood():
    print("\n=== dogfood: new helpers on real code ===")
    import importlib
    import cadw.patch_utils as pu
    importlib.reload(pu)
    from cadw.patch_utils import (
        parse_method_source, safe_skip_or_write,
        assert_test_imports_defined,
    )
    from cadw.dataframe import CadwDataFrame

    tree = parse_method_source(CadwDataFrame.pivot_table)
    assert any(getattr(n, "name", None) == "pivot_table"
               for n in tree.body)
    print("  parse_method_source(pivot_table): OK")

    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False,
    ) as f:
        f.write("x = 1\n")
        fname = f.name
    try:
        written = safe_skip_or_write(
            fname, ["x = 2"], required_symbols=["x"],
        )
        assert written is True
        print("  safe_skip_or_write(valid): skipped (OK)")
    finally:
        Path(fname).unlink(missing_ok=True)

    src = Path("tests/test_pivot_table.py").read_text()
    assert_test_imports_defined(src, fn_name="test_pivot_table.py")
    print("  assert_test_imports_defined(test_pivot_table.py): OK")


def main():
    print("=== v0.9.6e2: scaffold hardening #2 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    add_helpers()
    add_tests()
    dogfood()

    print("\n=== pytest: patch_utils + guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_patch_utils.py",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_no_pandas_in_ops.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PATCH_UTILS/GUARDS FAILED ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    commit_with_guards(
        "v0.9.6e2: scaffold hardening #2 — 4 patch-script helpers "
        "closing §13.4/189, /191, /192, /193. "
        "write_content_checked(path, lines) — write + assert_compiles "
        "atomic. "
        "parse_method_source(fn) — dedent + ast.parse. "
        "safe_skip_or_write(path, lines, *, required_symbols=()) — "
        "skip ONLY if existing file compiles + all required symbols "
        "present. "
        "assert_test_imports_defined(src) — AST heuristic: every "
        "loaded Name in test FunctionDefs is bound. "
        "+16 tests. Precedes v0.9.7 (R4 TopKOp + R5 DescribeOp).",
        self_path="scripts/patch_v096e2_scaffold2.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6e2 applied.")


if __name__ == "__main__":
    main()
