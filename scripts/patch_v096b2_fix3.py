#!/usr/bin/env python3
# v0.9.6b2-fix3 — fix2 smoke test subprocess invocation.
#
# Bug: fix2 wrote scripts/_smoke_*.py and ran `uv run python
# scripts/_smoke_*.py`. Python adds the SCRIPT's directory
# (scripts/) to sys.path[0], NOT cwd. `import cadw` fails with
# ModuleNotFoundError (cadw lives in project root, not scripts/).
#
# fix1 used `uv run python -c "..."` — cwd on sys.path, import
# worked (failed on assertion, not import). Revert to inline -c.
#
# Run: uv run python scripts/patch_v096b2_fix3.py

from __future__ import annotations
import os, subprocess, sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import commit_with_guards


SMOKE_SCRIPT = textwrap.dedent("""
    import ast, inspect, textwrap
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
                raise AssertionError(
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

    # 4. _detect_mode recognises dict; no pandas import.
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
""").strip()


def smoke_test():
    print("\n=== smoke test: inline -c (cwd on sys.path) ===")
    r = subprocess.run(
        ["uv", "run", "python", "-c", SMOKE_SCRIPT],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        print("  STDOUT: " + r.stdout.strip()[-500:])
        print("  STDERR: " + r.stderr.strip()[-1200:])
        raise SystemExit("smoke test failed")
    print("  " + r.stdout.strip())
    print("  OK")


def main():
    print("=== v0.9.6b2-fix3 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    # fix1 (code) + fix2 (comment + AST test) already applied to disk.
    # fix3 only validates and commits.
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
        "v0.9.6b2-fix3: fix2 smoke test subprocess invocation. "
        "`uv run python scripts/_smoke.py` puts scripts/ on sys.path[0] "
        "(not cwd) → ModuleNotFoundError: cadw. Revert to "
        "`uv run python -c` (cwd on sys.path, same as fix1). No code "
        "changes — fix1 (CADW-native _per_group_apply + helpers) and "
        "fix2 (comment rewrite + AST guard in test_apply.py) already "
        "on disk, verified by AST smoke here. Full v0.9.6b2 commit.",
        self_path="scripts/patch_v096b2_fix3.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6b2 complete (fix1+fix2+fix3).")


if __name__ == "__main__":
    main()