#!/usr/bin/env python3
# v0.9.6e2-fix2 — assert_test_imports_defined: narrower (no false positives).
#
# Third iteration on the resolver — pattern is clear: "all Names bound"
# heuristic is too broad without full scope analysis. Narrowed to the
# EXACT purpose (§13.4/192): catch bare `Call(func=Name(id))` where id
# is not local/module/builtin. Attribute calls, comprehensions, nested
# defs, AnnAssign (now handled) — out of scope; runtime catches.
#
# After fix1, dogfood still failed on 4 tests in test_apply.py:
#   test_apply_cadwdf_expand: ['g']  -- g is arg of nested `def fn(g)`
#     inside test. Walker descended into nested fn body, saw `g` as
#     Name Load. But nested `g` is BOUND in fn's scope; not a NameError.
#   test_apply_ast_uses_composite_sort: ['call_linenos']  -- AnnAssign
#     (`call_linenos: dict = {}`) not handled by Assign-only walker.
#
# Narrowed design:
#   - Walk own scope only: `_walk_own(node)` yields children, but does
#     not descend into nested FunctionDef/AsyncFunctionDef/Lambda/
#     ClassDef bodies.
#   - Bindings: args (all kinds) + Import/ImportFrom/Assign/AnnAssign/
#     AugAssign/NamedExpr/For/comprehension/with-as/nested-def-name.
#   - Violations: only `Call(func=Name(id))` where id not bound.
#   - Attribute calls skipped (obj.method() — obj undefined at runtime
#     → AttributeError/NameError caught by pytest collection/run;
#     not copy-paste-missing-import class).
#
# Original v0.9.6e2 tests preserved; 2 regression tests from fix1
# preserved. +1 dogfood case: real test_apply.py must pass.
#
# Run: uv run python scripts/patch_v096e2_fix2.py

from __future__ import annotations
import ast
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


NEW_FN_LINES = [
    'def assert_test_imports_defined(src, *, fn_name="<unknown>"):',
    '    """AST: bare-Call target in test OWN scope is bound.',
    '',
    '    Narrow scope (§13.4/195, v0.9.6e2-fix2): flags',
    '    `Call(func=Name(id))` inside the function\'s own scope (not',
    '    nested defs/lambdas) where `id` is not bound (args, imports,',
    '    assignments, for/comprehension/with targets, nested-def',
    '    names) nor module-level import/def/assign, nor builtin.',
    '',
    '    Does NOT flag:',
    '      - Attribute calls (`obj.method(...)`) — NameError on `obj`',
    '        or AttributeError caught at runtime by pytest.',
    '      - Names inside nested defs/lambdas — their own scopes.',
    '      - Names inside comprehensions — their own scopes.',
    '',
    '    Purpose (§13.4/192): catch "copied-template missing import"',
    '    in test bodies (e.g. test uses `assert_call_order(...)` but',
    '    forgot `from cadw.patch_utils import assert_call_order`).',
    '    """',
    '    import ast as _ast',
    '    import builtins as _bi',
    '    import textwrap as _tw',
    '',
    '    tree = _ast.parse(_tw.dedent(src))',
    '    _BUILTINS_LOCAL = frozenset(dir(_bi))',
    '',
    '    module_names: set = set()',
    '    for node in tree.body:',
    '        if isinstance(node, _ast.Import):',
    '            for a in node.names:',
    '                module_names.add((a.asname or a.name).split(".")[0])',
    '        elif isinstance(node, _ast.ImportFrom):',
    '            for a in node.names:',
    '                module_names.add(a.asname or a.name)',
    '        elif isinstance(node, (_ast.FunctionDef,',
    '                               _ast.AsyncFunctionDef,',
    '                               _ast.ClassDef)):',
    '            module_names.add(node.name)',
    '        elif isinstance(node, _ast.Assign):',
    '            for t in node.targets:',
    '                if isinstance(t, _ast.Name):',
    '                    module_names.add(t.id)',
    '        elif isinstance(node, _ast.AnnAssign):',
    '            if isinstance(node.target, _ast.Name):',
    '                module_names.add(node.target.id)',
    '',
    '    def _walk_own(node):',
    '        """Yield descendants; skip nested def/lambda/class bodies."""',
    '        for child in _ast.iter_child_nodes(node):',
    '            yield child',
    '            if not isinstance(child, (',
    '                _ast.FunctionDef, _ast.AsyncFunctionDef,',
    '                _ast.Lambda, _ast.ClassDef,',
    '            )):',
    '                yield from _walk_own(child)',
    '',
    '    violations: dict = {}',
    '    for fn_node in [',
    '        n for n in tree.body',
    '        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))',
    '    ]:',
    '        local: set = set()',
    '        for a in fn_node.args.args:',
    '            local.add(a.arg)',
    '        for a in getattr(fn_node.args, "posonlyargs", ()):',
    '            local.add(a.arg)',
    '        for a in getattr(fn_node.args, "kwonlyargs", ()):',
    '            local.add(a.arg)',
    '        if fn_node.args.vararg:',
    '            local.add(fn_node.args.vararg.arg)',
    '        if fn_node.args.kwarg:',
    '            local.add(fn_node.args.kwarg.arg)',
    '',
    '        for n in _walk_own(fn_node):',
    '            if isinstance(n, _ast.Import):',
    '                for a in n.names:',
    '                    local.add((a.asname or a.name).split(".")[0])',
    '            elif isinstance(n, _ast.ImportFrom):',
    '                for a in n.names:',
    '                    local.add(a.asname or a.name)',
    '            elif isinstance(n, _ast.Assign):',
    '                for t in n.targets:',
    '                    for nn in _ast.walk(t):',
    '                        if isinstance(nn, _ast.Name):',
    '                            local.add(nn.id)',
    '            elif isinstance(n, _ast.AnnAssign):',
    '                if isinstance(n.target, _ast.Name):',
    '                    local.add(n.target.id)',
    '            elif isinstance(n, _ast.AugAssign):',
    '                if isinstance(n.target, _ast.Name):',
    '                    local.add(n.target.id)',
    '            elif isinstance(n, _ast.NamedExpr):',
    '                if isinstance(n.target, _ast.Name):',
    '                    local.add(n.target.id)',
    '            elif isinstance(n, (_ast.FunctionDef,',
    '                               _ast.AsyncFunctionDef,',
    '                               _ast.ClassDef)):',
    '                local.add(n.name)',
    '            elif isinstance(n, (_ast.For, _ast.AsyncFor)):',
    '                for nn in _ast.walk(n.target):',
    '                    if isinstance(nn, _ast.Name):',
    '                        local.add(nn.id)',
    '            elif isinstance(n, _ast.comprehension):',
    '                for nn in _ast.walk(n.target):',
    '                    if isinstance(nn, _ast.Name):',
    '                        local.add(nn.id)',
    '            elif isinstance(n, _ast.withitem):',
    '                if n.optional_vars is not None:',
    '                    for nn in _ast.walk(n.optional_vars):',
    '                        if isinstance(nn, _ast.Name):',
    '                            local.add(nn.id)',
    '',
    '        missing: set = set()',
    '        for n in _walk_own(fn_node):',
    '            if not isinstance(n, _ast.Call):',
    '                continue',
    '            f = n.func',
    '            if not isinstance(f, _ast.Name):',
    '                continue  # attribute/subscript call: not a NameError',
    '            name = f.id',
    '            if (name in local',
    '                    or name in module_names',
    '                    or name in _BUILTINS_LOCAL):',
    '                continue',
    '            missing.add(name)',
    '        if missing:',
    '            violations[fn_node.name] = sorted(missing)',
    '',
    '    if violations:',
    '        msg = "; ".join(f"{k}: {v}" for k, v in violations.items())',
    '        raise AssertionError(',
    '            f"assert_test_imports_defined({fn_name}): undefined "',
    '            f"bare Call targets: {msg}"',
    '        )',
]


def replace_function():
    print("\n=== patch_utils.py: replace assert_test_imports_defined ===")
    p = Path("cadw/patch_utils.py")
    src = p.read_text()

    if "v0.9.6e2-fix2" in src and "bare-Call target in test OWN scope" in src:
        print("  already fix2 — skip")
        return

    # Locate old function via AST.
    tree = ast.parse(src)
    start = end = None
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name == "assert_test_imports_defined"):
            start, end = node.lineno, node.end_lineno
            break
    if start is None:
        raise SystemExit("assert_test_imports_defined not found")

    print(f"  replacing lines {start}..{end} ({end - start + 1} lines)")
    lines = src.split("\n")
    new_lines = lines[:start - 1] + NEW_FN_LINES + lines[end:]
    write_content(p, new_lines)
    assert_compiles(p)
    print("  OK")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: real test files (previous false positives) ===")
    import importlib
    import cadw.patch_utils as pu
    importlib.reload(pu)
    from cadw.patch_utils import assert_test_imports_defined

    for fname in (
        "tests/test_pivot_table.py",
        "tests/test_pivot_arrow_native.py",
        "tests/test_apply.py",       # was failing under fix1
        "tests/test_from_table.py",
        "tests/test_k_est_arrow_native.py",
    ):
        src = Path(fname).read_text()
        assert_test_imports_defined(src, fn_name=fname)
        print(f"  {fname}: OK")

    # Negative controls.
    assert_test_imports_defined(
        "def test_x():\n    totally_undefined_fn(1)\n",
        fn_name="negative-bare-call",
    ) if False else None
    try:
        assert_test_imports_defined(
            "def test_x():\n    totally_undefined_fn(1)\n",
            fn_name="negative-bare-call",
        )
    except AssertionError:
        print("  negative control (bare Call): correctly rejected")
    else:
        raise SystemExit("negative control failed")

    # Positive control: attribute call with unbound object must NOT trip
    # (that class is out of scope; pytest catches at runtime).
    assert_test_imports_defined(
        "def test_x():\n    some_obj.some_method()\n",
        fn_name="positive-attr-call",
    )
    print("  positive control (attribute call): allowed")


# ============ main ============

def main():
    print("=== v0.9.6e2-fix2 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    replace_function()
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
        "v0.9.6e2-fix2: assert_test_imports_defined narrowed. "
        "Third iteration — 'all Names bound' heuristic found new false "
        "positives (nested def arg `g` in test_apply.py, AnnAssign "
        "`call_linenos`) after fix1. New design: only flag "
        "`Call(func=Name(id))` in OWN scope (nested defs/lambdas/"
        "comprehensions excluded via _walk_own, which stops at nested "
        "FunctionDef/AsyncFunctionDef/Lambda/ClassDef). Bindings "
        "collected from own scope: args (all kinds), Import/"
        "ImportFrom/Assign/AnnAssign/AugAssign/NamedExpr/For/"
        "comprehension/with-as/nested-def-name. Attribute calls "
        "skipped — obj undefined at runtime → AttributeError/NameError "
        "caught by pytest, out of scope for copy-paste-missing-import "
        "class (§13.4/192). All original v0.9.6e2 + fix1 regression "
        "tests preserved. Dogfood: 5 real test files (previous "
        "false-positive test_apply.py now OK), negative bare-Call "
        "control, positive attribute-call control. v0.9.6e2 complete.",
        self_path="scripts/patch_v096e2_fix2.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6e2 complete (fix1 + fix2).")


if __name__ == "__main__":
    main()
