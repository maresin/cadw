#!/usr/bin/env python3
# v0.9.6b2-fix2 — fix2 for fix1's own guard false-positive.
#
# fix1 replaced _per_group_apply + helpers correctly (scoped
# inspect.getsource guard). But smoke test failed:
#   AssertionError: _per_group_apply calls to_pandas
# Cause: comment `# v0.9.6b2: CADW-native group. No `to_pandas()`.` —
# literal `to_pandas` in code comment. Substring check catches comments
# (§13.4/161 — AST vs substring).
#
# Same pattern in test_apply.py::test_apply_ast_uses_composite_sort
# (updated by v0.9.6b2 to substring check `"to_pandas" not in src`).
#
# Fixes:
#   1. Comment rewrite: remove literal `to_pandas` word.
#   2. test_apply.py: replace substring check with AST Call-node walk.
#   3. Smoke test in fix2: AST-based check (same pattern).
#
# Run: uv run python scripts/patch_v096b2_fix2.py

from __future__ import annotations
import ast
import inspect
import os, subprocess, sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


# ============ 1. Fix comment in _per_group_apply ============

def fix_comment():
    print("\n=== dataframe.py: rewrite to_pandas comment ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()
    old = "            # v0.9.6b2: CADW-native group. No `to_pandas()`.\n"
    new = "            # v0.9.6b2: CADW-native group materialize (in-memory).\n"
    if old not in s:
        if new in s:
            print("  already fixed — skip")
            return
        raise SystemExit("comment anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. Fix test_apply.py AST guard ============

NEW_AST_TEST = '''def test_apply_ast_uses_composite_sort():
    """AST: _per_group_apply external_sort by group+on; check before group materialize.

    v0.9.6b2-fix2: no-to_pandas check is AST-based (Call nodes only,
    not substring). Substring catches `to_pandas` in comments
    (§13.4/161, /182) — e.g. "No `to_pandas()`" comment. Pandas IMPORT
    is separately guarded by test_no_pandas_in_per_group_apply in
    test_no_pandas_in_ops.py.
    """
    import ast as _ast
    import inspect
    import textwrap
    from cadw.dataframe import CadwDataFrame
    src = inspect.getsource(CadwDataFrame._per_group_apply)
    assert "external_sort" in src

    # AST walk: no `.to_pandas(...)` Call nodes.
    tree = _ast.parse(textwrap.dedent(src))
    violations = [
        n.lineno
        for n in _ast.walk(tree)
        if isinstance(n, _ast.Call)
        and isinstance(n.func, _ast.Attribute)
        and n.func.attr == "to_pandas"
    ]
    assert not violations, (
        f"v0.9.6b2: _per_group_apply must not call to_pandas() — "
        f"found at lines {violations}. Comments/docstrings do NOT "
        f"count (AST Call nodes only, §13.4/161)."
    )

    assert "CadwDataFrame.from_table" in src, (
        "v0.9.6b2: group must be materialized as CadwDataFrame"
    )
    assert "check_state" in src
    assert src.index("check_state") < src.index("from_table"), (
        "check_state must precede group materialization"
    )
'''


def fix_test_apply():
    print("\n=== test_apply.py: AST-based no-to_pandas check ===")
    p = Path("tests/test_apply.py")
    s = p.read_text()
    if "AST walk: no `.to_pandas(...)` Call nodes" in s:
        print("  already AST-based — skip")
        return

    # Find the existing test function span and replace it.
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
    new_s = "\n".join(
        lines[:start - 1] + [NEW_AST_TEST] + lines[end:]
    )
    p.write_text(new_s)
    assert_compiles(p)
    print("  OK")


# ============ 3. AST-based smoke test ============

def smoke_test():
    print("\n=== smoke test: AST-based no-to_pandas + no-pandas-import ===")
    script = textwrap.dedent("""
        import ast, inspect, textwrap, sys
        import cadw.dataframe as df_mod

        # 1. No `import pandas` anywhere in dataframe.py.
        src = inspect.getsource(df_mod)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith("pandas"), (
                        f"dataframe.py imports {a.name} at line {node.lineno}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("pandas"):
                    assert False, (
                        f"dataframe.py imports from {node.module} "
                        f"at line {node.lineno}"
                    )

        # 2. _per_group_apply: no .to_pandas() Call nodes.
        m_src = inspect.getsource(df_mod.CadwDataFrame._per_group_apply)
        m_tree = ast.parse(textwrap.dedent(m_src))
        viol = [
            n.lineno for n in ast.walk(m_tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "to_pandas"
        ]
        assert not viol, f"_per_group_apply calls to_pandas at lines {viol}"

        # 3. Positive markers.
        assert "CadwDataFrame.from_table" in m_src
        assert hasattr(df_mod.CadwDataFrame, "__len__")

        # 4. _detect_mode recognises dict.
        dm_src = inspect.getsource(df_mod._detect_mode)
        assert "dict" in dm_src
        dm_tree = ast.parse(textwrap.dedent(dm_src))
        for node in ast.walk(dm_tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith("pandas"), (
                        f"_detect_mode imports {a.name}"
                    )

        print("AST smoke: OK")
    """)
    # write to a temp file in scripts/ so cwd handling is identical
    tmp = Path("scripts/_smoke_v096b2_fix2.py")
    tmp.write_text(script)
    try:
        r = subprocess.run(
            ["uv", "run", "python", str(tmp)],
            capture_output=True, text=True, timeout=60,
        )
    finally:
        tmp.unlink(missing_ok=True)
    if r.returncode != 0:
        print("  STDOUT: " + r.stdout.strip()[-500:])
        print("  STDERR: " + r.stderr.strip()[-1200:])
        raise SystemExit("smoke test failed")
    print("  " + r.stdout.strip())
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.6b2-fix2 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    fix_comment()
    fix_test_apply()
    smoke_test()

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
         "tests/test_arrow_call_memory_guard.py", "-v"],
        timeout=180,
    )
    if r.returncode != 0:
        print("\n*** AST GUARDS FAILED ***")
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
        "v0.9.6b2-fix2: to_pandas-in-comment vs substring guard. "
        "fix1 successfully replaced _per_group_apply + helpers "
        "(scoped guard), but smoke test's `'to_pandas' not in src` "
        "caught the comment `# No to_pandas()` — substring check, "
        "not AST (§13.4/161, /182). Fix: (a) rewrite comment "
        "without the word; (b) upgrade test_apply_ast_uses_composite_"
        "sort to AST Call-node walk (comments/docstrings invisible "
        "to AST); (c) smoke test similarly AST-based. Both substring "
        "and AST guards now agree on the actual invariant: no "
        "`.to_pandas()` method call in execution path.",
        self_path="scripts/patch_v096b2_fix2.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6b2 complete (fix1 + fix2).")


if __name__ == "__main__":
    main()