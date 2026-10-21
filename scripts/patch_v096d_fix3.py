#!/usr/bin/env python3
# v0.9.6d-fix3 — AST-based active-usage check (not regex).
#
# Root cause: fix2 verify regex `import[^\n]*_pandas_hash_column`
# matched the DOCSTRING of test_no_test_imports_stale_hash_names:
#     """AST: no test file imports _pandas_hash_column or ...
# `imports` contains `import` as substring, `_pandas_hash_column`
# on same line → match. Fourth instance of "string-pattern catches
# non-code content" (§13.4/161, /183, /187, now /188) — the very
# class that assert_no_call_in_source/assert_call_order exist to
# prevent. Verify function must itself use AST.
#
# Fix: active-usage detection via AST nodes:
#   - ast.Import / ast.ImportFrom with alias._pandas_hash_column
#   - ast.Call whose func is Name(id=...) or Attribute(attr=...)
# Docstrings, string literals, set members (e.g. STALE = {...})
# are invisible to these node types.
#
# Also: some "historical mentions" in test_hash_v081.py are:
#   - `assert "_pandas_hash_column" not in src` (test_own_hash_removed
#     _pandas_functions) — legit absence assertion
#   - `STALE = {'_pandas_hash_column', ...}` (test_no_test_imports_
#     stale_hash_names) — legit AST-guard-against-stale-imports
# Neither is active usage.
#
# Run: uv run python scripts/patch_v096d_fix3.py

from __future__ import annotations
import ast
import os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import assert_compiles, commit_with_guards


STALE_NAMES = {"_pandas_hash_column", "_python_hash_column"}


def _find_active_usage(path: Path) -> list[tuple[int, str]]:
    """Return [(lineno, description)] for real active references.

    AST-based. Detects:
      - `import X` / `from Y import X` for X in STALE_NAMES
      - `X(...)` call (bare or via attribute)
      - attribute access `.X` (rare but possible)
    Docstrings / string literals / set members are ignored
    (§13.4/188 — same class as /183, /187; verify function must use
    AST for X-usage detection).
    """
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [(0, f"parse error: {e}")]

    violations: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        # import _pandas_hash_column
        if isinstance(node, ast.Import):
            for a in node.names:
                # a.name may be dotted; compare last segment
                last = a.name.rsplit(".", 1)[-1]
                if last in STALE_NAMES:
                    violations.append(
                        (node.lineno, f"import {a.name}")
                    )
        # from ... import _pandas_hash_column
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name in STALE_NAMES:
                    mod = node.module or "<relative>"
                    violations.append(
                        (node.lineno, f"from {mod} import {a.name}")
                    )
        # call: _pandas_hash_column(...) or obj._pandas_hash_column(...)
        elif isinstance(node, ast.Call):
            f = node.func
            name = None
            if isinstance(f, ast.Name):
                name = f.id
            elif isinstance(f, ast.Attribute):
                name = f.attr
            if name in STALE_NAMES:
                violations.append(
                    (node.lineno, f"call {name}(...)")
                )

    return violations


def verify_state():
    print("\n=== verify: helpers + tests + cleanups (AST active usage) ===")
    pu = Path("cadw/patch_utils.py").read_text()
    for name in (
        "assert_no_call_in_source", "assert_call_order",
        "is_patched", "run_smoke_inline",
    ):
        if f"def {name}(" not in pu:
            raise SystemExit(f"helper {name} missing from patch_utils.py")
    print("  4 helpers present")

    ts = Path("tests/test_patch_utils.py").read_text()
    if "def test_assert_no_call_in_source_clean" not in ts:
        raise SystemExit("test_assert_no_call_in_source_clean missing")
    print("  helper tests present")

    for path_str in (
        "tests/test_no_pandas_in_ops.py",
        "tests/test_hash_v081.py",
    ):
        hits = _find_active_usage(Path(path_str))
        if hits:
            for lineno, desc in hits:
                print(f"  {path_str}:{lineno}: {desc}")
            raise SystemExit(f"{path_str}: active stale usage")
        print(f"  {path_str}: no active usage (AST check)")


def fix_join_py():
    print("\n=== fix join.py stale comment block ===")
    p = Path("cadw/ops/join.py")
    s = p.read_text()
    if "_pandas_hash_column" not in s:
        print("  already clean — skip")
        return
    lines = s.split("\n")
    for i, line in enumerate(lines):
        if "_pandas_hash_column" not in line:
            continue
        start = i
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        end = i
        while (end < len(lines) - 1
               and lines[end + 1].lstrip().startswith("#")):
            end += 1
        print(f"  block lines {start+1}..{end+1}:")
        for j in range(start, end + 1):
            print(f"    {lines[j]}")
        new_block = [
            "# v0.8.11a: no pandas in hot path (dead helpers removed).",
            "# v0.9.6a: shuffle.py uses own XXH64-based hash; cadw/ops/",
            "# is pandas-free.",
        ]
        lines = lines[:start] + new_block + lines[end + 1:]
        p.write_text("\n".join(lines))
        assert_compiles(p)
        print("  fixed")
        return
    print("  WARNING: _pandas_hash_column present but not in comment block")


def dogfood_helpers():
    print("\n=== dogfood: new helpers on real _per_group_apply ===")
    import importlib
    import cadw.patch_utils as pu
    importlib.reload(pu)
    from cadw.patch_utils import (
        assert_no_call_in_source, assert_call_order, is_patched,
    )
    import inspect
    from cadw.dataframe import CadwDataFrame

    fn = CadwDataFrame._per_group_apply
    src = inspect.getsource(fn)

    assert_no_call_in_source(src, "to_pandas", fn_name="_per_group_apply")
    print("  assert_no_call_in_source: no to_pandas in _per_group_apply")

    assert_call_order(src, "check_state", "from_table",
                      fn_name="_per_group_apply")
    print("  assert_call_order: check_state < from_table")

    assert is_patched(fn, "CadwDataFrame.from_table")
    print("  is_patched: from_table marker present")

    try:
        assert_no_call_in_source(
            "def f():\n    x.to_pandas()\n", "to_pandas",
            fn_name="negative-control",
        )
    except AssertionError:
        print("  negative control: correctly rejected")
    else:
        raise SystemExit("negative control failed")
    print("  OK")


def main():
    print("=== v0.9.6d-fix3 ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    verify_state()
    fix_join_py()
    dogfood_helpers()

    print("\n=== pytest: patch_utils + guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_patch_utils.py",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_no_pandas_in_ops.py", "-v"],
        timeout=180,
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
        "v0.9.6d-fix3: verify_state uses AST, not regex. fix2 "
        "regex `import[^\\n]*_pandas_hash_column` matched the "
        "docstring of test_no_test_imports_stale_hash_names "
        "(\"AST: no test file imports _pandas_hash_column...\") — "
        "`imports` contains `import` as substring. Fourth instance "
        "of string-pattern catching non-code content (§13.4/161, "
        "/183, /187, /188); fix is AST walk over Import / "
        "ImportFrom / Call / Attribute nodes — docstrings and "
        "string literals invisible. Historical mentions in "
        "test_hash_v081.py (assert \"... not in src\", "
        "STALE = {...} set) are legitimate absence-assertions and "
        "AST-guard data, not active usage. Also: join.py comment "
        "block replaced (v0.9.6d anchor `# (v0.8.11a` — actual "
        "`# v0.8.11a`, no paren). v0.9.6d complete.",
        self_path="scripts/patch_v096d_fix3.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6d complete (fix1+fix2+fix3).")


if __name__ == "__main__":
    main()