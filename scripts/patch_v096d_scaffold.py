#!/usr/bin/env python3
# v0.9.6d — scaffold hardening: 4 patch-script helpers + cleanup.
#
# Six consecutive patch-script meta-bugs in v0.9.4b-2..v0.9.6b2:
#   §13.4/178: stale module cache (importlib.reload)
#   §13.4/179: parameter shadowing (verify_whitelist bool vs func)
#   §13.4/182: idempotency guard matched foreign tokens (scoped guard)
#   §13.4/183: substring guard caught comment "No `to_pandas()`"
#   §13.4/184: subprocess scripts/foo.py has scripts/ on sys.path
#   §13.4/185: *_ast_* test with substring ordering inside
#
# Helpers (in cadw/patch_utils.py):
#   - assert_no_call_in_source(src, *names, fn_name=)   [AST]
#   - assert_call_order(src, first, second, *, fn_name=) [AST]
#   - is_patched(fn, *markers) — inspect.getsource(fn) [scoped idempotency]
#   - run_smoke_inline(script, *, timeout=60) [uv run python -c]
#
# Cleanup: 3 stale references to v0.9.6a-removed `_pandas_hash_column`
# (test_no_pandas_in_ops.py, test_hash_v081.py, cadw/ops/join.py).
#
# Dogfood: new helpers used on real CadwDataFrame._per_group_apply
# source (positive: no to_pandas; ordering: check_state < from_table).
#
# Run: uv run python scripts/patch_v096d_scaffold.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    write_content, assert_compiles, commit_with_guards,
)


# ============ 1. Add helpers to patch_utils.py ============

HELPERS_BLOCK_LINES = [
    "",
    "",
    "# ============ v0.9.6d: AST + subprocess + scoped-idempotency ============",
    "#",
    "# Six consecutive patch-script meta-bugs (§13.4/178, /179, /182,",
    "# /183, /184, /185) — these helpers make the patterns structurally",
    "# impossible:",
    "#   - substring `assert 'X' not in src` catches comments",
    "#     → assert_no_call_in_source() / assert_call_order() [AST]",
    "#   - `python scripts/foo.py` puts scripts/ on sys.path[0]",
    "#     → run_smoke_inline() [uv run python -c]",
    "#   - whole-file idempotency guard matches tokens from other patches",
    "#     → is_patched(fn, *markers) [inspect.getsource(fn)]",
    "",
    "",
    "def _collect_call_linenos(tree):",
    "    \"\"\"Return {attr_or_name: [lineno, ...]} for all Call nodes.",
    "",
    "    Handles both `obj.attr(...)` (ast.Attribute, use .attr) and",
    "    `bare(...)` (ast.Name, use .id). Docstrings/comments are",
    "    invisible to ast.walk over Call nodes (§13.4/161, /183, /185).",
    "    \"\"\"",
    "    import ast as _ast",
    "    out: dict = {}",
    "    for node in _ast.walk(tree):",
    "        if not isinstance(node, _ast.Call):",
    "            continue",
    "        f = node.func",
    "        if isinstance(f, _ast.Name):",
    "            out.setdefault(f.id, []).append(node.lineno)",
    "        elif isinstance(f, _ast.Attribute):",
    "            out.setdefault(f.attr, []).append(node.lineno)",
    "    return out",
    "",
    "",
    "def assert_no_call_in_source(src, *names, fn_name=\"<unknown>\"):",
    "    \"\"\"AST: raise AssertionError if any Call node matches `name`.",
    "",
    "    `name` may be a bare function call (`foo()`) or an attribute",
    "    call (`obj.foo()`). Comments and docstrings are invisible",
    "    (§13.4/161, /183) — only ast.Call nodes are checked. Replaces",
    "    `assert 'foo' not in src`, which catches `# No foo() here`",
    "    and `Use CadwDataFrame.foo(...)` docstrings.",
    "    \"\"\"",
    "    import ast as _ast",
    "    import textwrap as _tw",
    "    try:",
    "        tree = _ast.parse(_tw.dedent(src))",
    "    except SyntaxError as e:",
    "        raise AssertionError(",
    "            f\"assert_no_call_in_source({fn_name}): parse error {e}\"",
    "        ) from None",
    "    calls = _collect_call_linenos(tree)",
    "    violations = {n: calls[n] for n in names if n in calls}",
    "    if violations:",
    "        detail = \"; \".join(",
    "            f\"{n!r} at line(s) {ls}\" for n, ls in violations.items()",
    "        )",
    "        raise AssertionError(",
    "            f\"assert_no_call_in_source({fn_name}): forbidden Call \"",
    "            f\"node(s) found: {detail}\"",
    "        )",
    "",
    "",
    "def assert_call_order(src, first_name, second_name, *, fn_name=\"<unknown>\"):",
    "    \"\"\"AST: assert min(lineno of first Call) < min(lineno of second).",
    "",
    "    Raises AssertionError if either Call is absent, or if first does",
    "    not precede second. Docstrings mentioning `first_name`/`second_name`",
    "    are invisible (§13.4/161, /185). Use instead of",
    "    `src.index('first') < src.index('second')`.",
    "    \"\"\"",
    "    import ast as _ast",
    "    import textwrap as _tw",
    "    try:",
    "        tree = _ast.parse(_tw.dedent(src))",
    "    except SyntaxError as e:",
    "        raise AssertionError(",
    "            f\"assert_call_order({fn_name}): parse error {e}\"",
    "        ) from None",
    "    calls = _collect_call_linenos(tree)",
    "    if first_name not in calls:",
    "        raise AssertionError(",
    "            f\"assert_call_order({fn_name}): first Call \"",
    "            f\"{first_name!r} not found\"",
    "        )",
    "    if second_name not in calls:",
    "        raise AssertionError(",
    "            f\"assert_call_order({fn_name}): second Call \"",
    "            f\"{second_name!r} not found\"",
    "        )",
    "    first_min = min(calls[first_name])",
    "    second_min = min(calls[second_name])",
    "    if first_min >= second_min:",
    "        raise AssertionError(",
    "            f\"assert_call_order({fn_name}): {first_name!r} \"",
    "            f\"(first line {first_min}) must precede {second_name!r} \"",
    "            f\"(first line {second_min})\"",
    "        )",
    "",
    "",
    "def is_patched(fn, *markers):",
    "    \"\"\"Scoped idempotency guard: fn source contains ALL markers.",
    "",
    "    Use instead of `if marker in whole_file: skip` (§13.4/182).",
    "    Whole-file substring matches tokens from OTHER patches in the",
    "    same file — e.g. v0.9.6b2's `if 'v0.9.6b2' in s and 'CADW-native'",
    "    in s` matched because add_len() wrote 'v0.9.6b2' to __len__",
    "    docstring and _ColumnProxy.mean had 'CADW-native' from v0.9.6b1.",
    "",
    "    Returns True iff `inspect.getsource(fn)` contains every marker.",
    "    \"\"\"",
    "    import inspect as _inspect",
    "    src = _inspect.getsource(fn)",
    "    return all(m in src for m in markers)",
    "",
    "",
    "def run_smoke_inline(script, *, timeout=60):",
    "    \"\"\"Run `uv run python -c <script>` (cwd on sys.path[0]).",
    "",
    "    Use instead of writing a temp file under scripts/ and running",
    "    `python scripts/foo.py`: that puts `scripts/` (not cwd) on",
    "    sys.path[0] → `import cadw` fails with ModuleNotFoundError",
    "    (§13.4/184). Stdout from the script is echoed (2-space indent);",
    "    on failure prints last 1200 chars of stderr, then sys.exit(1).",
    "    \"\"\"",
    "    import subprocess as _sp",
    "    import sys as _sys",
    "    r = _sp.run(",
    "        [\"uv\", \"run\", \"python\", \"-c\", script],",
    "        capture_output=True, text=True, timeout=timeout,",
    "    )",
    "    if r.stdout.strip():",
    "        for line in r.stdout.strip().splitlines():",
    "            print(\"  \" + line)",
    "    if r.returncode != 0:",
    "        print(\"  STDERR: \" + r.stderr.strip()[-1200:])",
    "        _sys.exit(1)",
    "",
]


def add_helpers():
    print("\n=== patch_utils.py: add AST + subprocess + scoped helpers ===")
    p = Path("cadw/patch_utils.py")
    s = p.read_text()
    if "def assert_call_order(" in s:
        print("  already present — skip")
        return
    anchor = "\n\ndef assert_no_stale_behavior("
    if anchor not in s:
        raise SystemExit("assert_no_stale_behavior anchor not found")
    new_block = "\n".join(HELPERS_BLOCK_LINES)
    s = s.replace(anchor, new_block + anchor, 1)
    p.write_text(s)
    assert_compiles(p)
    print("  OK")


# ============ 2. Add tests for helpers ============

NEW_TESTS_LINES = [
    "",
    "",
    "# ============ v0.9.6d: AST + subprocess + scoped helpers ============",
    "",
    "",
    "def test_assert_no_call_in_source_clean():",
    "    from cadw.patch_utils import assert_no_call_in_source",
    "    src = 'def f():\\n    return 1\\n'",
    "    assert_no_call_in_source(src, 'to_pandas')  # no raise",
    "",
    "",
    "def test_assert_no_call_in_source_finds_bare():",
    "    from cadw.patch_utils import assert_no_call_in_source",
    "    src = 'def f():\\n    to_pandas()\\n'",
    "    with pytest.raises(AssertionError, match=\"'to_pandas'\"):",
    "        assert_no_call_in_source(src, 'to_pandas')",
    "",
    "",
    "def test_assert_no_call_in_source_finds_attr():",
    "    from cadw.patch_utils import assert_no_call_in_source",
    "    src = 'def f(t):\\n    t.to_pandas()\\n'",
    "    with pytest.raises(AssertionError, match=\"'to_pandas'\"):",
    "        assert_no_call_in_source(src, 'to_pandas')",
    "",
    "",
    "def test_assert_no_call_in_source_ignores_comment():",
    "    # §13.4/183: substring would catch `# No to_pandas()`. AST does not.",
    "    from cadw.patch_utils import assert_no_call_in_source",
    "    src = 'def f():\\n    # No to_pandas() here\\n    return 1\\n'",
    "    assert_no_call_in_source(src, 'to_pandas')  # no raise",
    "",
    "",
    "def test_assert_no_call_in_source_ignores_docstring():",
    "    from cadw.patch_utils import assert_no_call_in_source",
    "    src = (",
    "        'def f():\\n'",
    "        '    \"\"\"Use CadwDataFrame.from_table(pa_table).\"\"\"\\n'",
    "        '    return 1\\n'",
    "    )",
    "    assert_no_call_in_source(src, 'from_table')  # no raise",
    "",
    "",
    "def test_assert_call_order_ok():",
    "    from cadw.patch_utils import assert_call_order",
    "    src = (",
    "        'def f(t):\\n'",
    "        '    t.check_state(1)\\n'",
    "        '    CadwDataFrame.from_table(t)\\n'",
    "    )",
    "    assert_call_order(src, 'check_state', 'from_table')",
    "",
    "",
    "def test_assert_call_order_wrong():",
    "    from cadw.patch_utils import assert_call_order",
    "    src = (",
    "        'def f(t):\\n'",
    "        '    CadwDataFrame.from_table(t)\\n'",
    "        '    t.check_state(1)\\n'",
    "    )",
    "    with pytest.raises(AssertionError, match='must precede'):",
    "        assert_call_order(src, 'check_state', 'from_table')",
    "",
    "",
    "def test_assert_call_order_missing_first():",
    "    from cadw.patch_utils import assert_call_order",
    "    src = 'def f():\\n    pass\\n'",
    "    with pytest.raises(AssertionError, match='not found'):",
    "        assert_call_order(src, 'check_state', 'from_table')",
    "",
    "",
    "def test_assert_call_order_ignores_docstring():",
    "    # §13.4/185: docstring mention of `from_table` must not affect ordering.",
    "    from cadw.patch_utils import assert_call_order",
    "    src = (",
    "        'def f(t):\\n'",
    "        '    \"\"\"Use CadwDataFrame.from_table(...).\"\"\"\\n'",
    "        '    t.check_state(1)\\n'",
    "        '    CadwDataFrame.from_table(t)\\n'",
    "    )",
    "    assert_call_order(src, 'check_state', 'from_table')",
    "",
    "",
    "def test_is_patched_scoped():",
    "    # §13.4/182: scoped idempotency uses inspect.getsource(fn).",
    "    from cadw.patch_utils import is_patched",
    "    def fn():",
    "        return 'marker_a'",
    "    assert is_patched(fn, 'marker_a')",
    "    assert not is_patched(fn, 'marker_a', 'marker_b')",
    "    assert not is_patched(fn, 'nonexistent')",
    "",
    "",
    "def test_run_smoke_inline_ok(capsys):",
    "    from cadw.patch_utils import run_smoke_inline",
    "    run_smoke_inline(\"print('hello-smoke')\")",
    "    captured = capsys.readouterr()",
    "    assert 'hello-smoke' in captured.out",
    "",
    "",
    "def test_run_smoke_inline_fail():",
    "    from cadw.patch_utils import run_smoke_inline",
    "    with pytest.raises(SystemExit):",
    "        run_smoke_inline(\"raise ValueError('boom')\")",
    "",
    "",
    "def test_run_smoke_inline_has_cadw_on_path():",
    "    # §13.4/184: cwd on sys.path. `import cadw` works.",
    "    from cadw.patch_utils import run_smoke_inline",
    "    run_smoke_inline(\"import cadw; print('cadw-import-ok')\")",
    "",
]


def add_tests():
    print("\n=== test_patch_utils.py: add helper tests ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "def test_assert_no_call_in_source_clean" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(NEW_TESTS_LINES))
    assert_compiles(p)
    print("  OK")


# ============ 3. Cleanup stale refs (best-effort) ============

CLEANUPS = [
    (
        "tests/test_no_pandas_in_ops.py",
        "`shuffle.py::_pandas_hash_column` — a documented perf exception\n"
        "(pandas C-hash is 19–40× faster than Python fallback for\n"
        "int/float/timestamp; §9.2.20).",
        "v0.9.6a replaced `_pandas_hash_column` with own XXH64-based\n"
        "hash (`_vectorized_hash_numeric` + `_string_hash_column`).\n"
        "`cadw/ops/` is pandas-free.",
    ),
    (
        "tests/test_hash_v081.py",
        "    Without types_mapper in _pandas_hash_column:\n"
        "      - batch without nulls: to_pandas → int64\n"
        "      - batch with nulls:    to_pandas → float64 (NaN)\n"
        "      → hash(1) ≠ hash(1.0) → partition inconsistency → WRONG results.",
        "    v0.9.6a: hash is deterministic on value regardless of\n"
        "    nulls present in the batch (own XXH64 for numeric;\n"
        "    nulls sentinel).",
    ),
    (
        "cadw/ops/join.py",
        "# (v0.8.11a: no pandas in hot path (dead helpers removed; only\n"
        "# `_pandas_hash_column` in cadw/ops/shuffle.py remains, documented).",
        "# (v0.8.11a: no pandas in hot path (dead helpers removed).\n"
        "# v0.9.6a: shuffle.py uses own XXH64-based hash; cadw/ops/\n"
        "# is pandas-free.)",
    ),
]


def cleanup_stale_refs():
    print("\n=== cleanup stale docstring/comment refs ===")
    for path_str, old, new in CLEANUPS:
        p = Path(path_str)
        if not p.exists():
            print(f"  {path_str}: missing — skip")
            continue
        s = p.read_text()
        if old in s:
            s = s.replace(old, new, 1)
            p.write_text(s)
            if path_str.endswith(".py"):
                assert_compiles(p)
            print(f"  {path_str}: cleaned")
        else:
            print(f"  {path_str}: anchor not found — skip (non-fatal)")


# ============ 4. Dogfood new helpers on real source ============

def dogfood_helpers():
    print("\n=== dogfood: new helpers on real _per_group_apply ===")
    import inspect
    from cadw.patch_utils import (
        assert_no_call_in_source, assert_call_order, is_patched,
    )
    from cadw.dataframe import CadwDataFrame

    fn = CadwDataFrame._per_group_apply
    src = inspect.getsource(fn)

    # AST no-call check: no to_pandas Call node.
    assert_no_call_in_source(src, "to_pandas", fn_name="_per_group_apply")
    print("  assert_no_call_in_source: no to_pandas in _per_group_apply")

    # AST ordering: check_state precedes from_table.
    assert_call_order(src, "check_state", "from_table",
                      fn_name="_per_group_apply")
    print("  assert_call_order: check_state < from_table")

    # Scoped idempotency: from_table present, to_pandas Call absent.
    assert is_patched(fn, "CadwDataFrame.from_table")
    print("  is_patched: from_table marker present")

    # Negative control: assert_no_call_in_source should raise on
    # a synthetic src with forbidden call.
    try:
        assert_no_call_in_source(
            "def f():\n    x.to_pandas()\n", "to_pandas",
            fn_name="negative-control",
        )
    except AssertionError:
        print("  negative control: forbidden Call correctly rejected")
    else:
        raise SystemExit("negative control failed: no AssertionError")
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.6d: scaffold hardening ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    add_helpers()
    add_tests()
    cleanup_stale_refs()
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
        "v0.9.6d: scaffold hardening — 4 patch-script helpers to "
        "prevent 6 consecutive meta-bug classes (§13.4/178, /179, "
        "/182, /183, /184, /185). patch_utils: "
        "assert_no_call_in_source(src, *names) [AST, no .foo() Call "
        "nodes, comments/docstrings invisible]; assert_call_order"
        "(src, first, second) [AST ordering]; is_patched(fn, "
        "*markers) [inspect.getsource(fn), scoped idempotency]; "
        "run_smoke_inline(script) [uv run python -c, cwd on sys.path]. "
        "+13 tests (positive + negative for each). Cleanup 3 stale "
        "docstring/comment refs to v0.9.6a-removed _pandas_hash_column "
        "(test_no_pandas_in_ops.py, test_hash_v081.py, cadw/ops/join.py). "
        "Dogfood: new helpers verified on real _per_group_apply source.",
        self_path="scripts/patch_v096d_scaffold.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.6d applied.")


if __name__ == "__main__":
    main()