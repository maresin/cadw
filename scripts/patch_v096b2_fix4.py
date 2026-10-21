#!/usr/bin/env python3
# v0.9.6b2-fix4 — test_apply_ast_uses_composite_sort: AST-based ordering.
#
# Bug: fix2 upgraded no-to_pandas to AST, but kept substring
# `src.index("check_state") < src.index("from_table")`. Docstring
# mentions "CadwDataFrame.from_table(pa_table)" at pos 927; real
# check_state call at pos 6375 → assertion "check_state must
# precede" fails on docstring text.
#
# Third §13.4/161 occurrence in v0.9.6b2 (fix2 fixed one substring,
# left another). Rule: AST for ALL code checks in this test —
# comments/docstrings invisible.
#
# Run: uv run python scripts/patch_v096b2_fix4.py

from __future__ import annotations
import ast
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


NEW_TEST = '''def test_apply_ast_uses_composite_sort():
    """AST: _per_group_apply — external_sort + check_state before
    materialize, no .to_pandas() Call nodes.

    v0.9.6b2-fix4: ALL checks AST-based. Prior fix2 kept
    `src.index("check_state") < src.index("from_table")` — substring;
    docstring "CadwDataFrame.from_table(pa_table)" matched earlier than
    real check_state call → false failure (§13.4/161, third occurrence
    in v0.9.6b2 session: /182, /183, /184, /185).
    """
    import ast as _ast
    import inspect
    import textwrap
    from cadw.dataframe import CadwDataFrame

    src = inspect.getsource(CadwDataFrame._per_group_apply)
    tree = _ast.parse(textwrap.dedent(src))

    # Collect real Call-node linenos by attr/id. Docstrings and
    # comments are invisible to ast.walk over Call nodes.
    call_linenos: dict[str, list[int]] = {}
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        f = node.func
        if isinstance(f, _ast.Name):
            # bare function calls: external_sort(...)
            if f.id in ("external_sort",):
                call_linenos.setdefault(f.id, []).append(node.lineno)
        elif isinstance(f, _ast.Attribute):
            # method calls: obj.check_state(...), CadwDataFrame.from_table(...)
            if f.attr in ("check_state", "from_table", "to_pandas"):
                call_linenos.setdefault(f.attr, []).append(node.lineno)

    # 1. external_sort is called (composite sort by group+on).
    assert "external_sort" in call_linenos, (
        "v0.9.6b2: _per_group_apply must call external_sort"
    )

    # 2. No .to_pandas() Call nodes.
    assert "to_pandas" not in call_linenos, (
        f"_per_group_apply calls to_pandas at lines "
        f"{call_linenos.get('to_pandas')}"
    )

    # 3. Group materialized via CadwDataFrame.from_table.
    assert "from_table" in call_linenos, (
        "v0.9.6b2: group must be materialized via CadwDataFrame.from_table"
    )

    # 4. check_state Call precedes from_table Call (§3.10).
    assert "check_state" in call_linenos, (
        "_per_group_apply must call check_state"
    )
    first_guard = min(call_linenos["check_state"])
    first_materialize = min(call_linenos["from_table"])
    assert first_guard < first_materialize, (
        f"check_state call (line {first_guard}) must precede "
        f"CadwDataFrame.from_table call (line {first_materialize}) — "
        f"§3.10 guard before materialize"
    )
'''


def fix_test():
    print("\n=== test_apply.py: AST-based ordering (replace function) ===")
    p = Path("tests/test_apply.py")
    s = p.read_text()

    if "v0.9.6b2-fix4" in s:
        print("  already fixed — skip")
        return

    tree = ast.parse(s)
    start = end = None
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name == "test_apply_ast_uses_composite_sort"):
            start, end = node.lineno, node.end_lineno
            break
    if start is None:
        raise SystemExit("test_apply_ast_uses_composite_sort not found")

    lines = s.split("\n")
    p.write_text("\n".join(
        lines[:start - 1] + [NEW_TEST] + lines[end:]
    ))
    assert_compiles(p)
    print(f"  replaced lines {start}..{end}")


def main():
    print("=== v0.9.6b2-fix4 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_test()

    print("\n=== pytest: apply tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_apply.py",
         "tests/test_apply_e2e.py", "-v"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** APPLY TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: no-pandas guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_no_pandas_in_ops.py",
         "tests/test_arrow_call_memory_guard.py", "-q"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** AST GUARDS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: from_table/proxy regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_from_table.py",
         "tests/test_column_proxy_aggregators.py", "-q"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** FROM_TABLE/PROXY REGRESSION FAILED ***")
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
        "v0.9.6b2-fix4: test_apply_ast_uses_composite_sort — full AST. "
        "fix2 upgraded no-to_pandas to AST but kept substring "
        "`src.index('check_state') < src.index('from_table')`. "
        "Docstring 'CadwDataFrame.from_table(pa_table)' matched at "
        "pos 927 before real check_state call at pos 6375 → false "
        "failure. Third §13.4/161 occurrence in v0.9.6b2 (/182, "
        "/183, /184, /185). Fix: collect real Call-node linenos "
        "(Name: external_sort; Attribute: check_state/from_table/"
        "to_pandas), compare min(lineno). Comments/docstrings "
        "invisible to ast.walk over Call nodes. v0.9.6b2 complete.",
        self_path="scripts/patch_v096b2_fix4.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6b2 complete (fix1+fix2+fix3+fix4).")


if __name__ == "__main__":
    main()