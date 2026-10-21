#!/usr/bin/env python3
# v0.9.14-fix2 — complete v0.9.14: T2.2 helper + F12-6/T2.2 tests.
# v0.9.14 COMPLETE.
#
# v0.9.14-fix1 applied U3 (validate=toml), apply_or_fail validate=
# kwarg, T2.9 tests — but v0.9.14-fix1 verify_state found 2 missing
# anchors:
#   cadw/patch_utils.py: 'v0.9.14 (T2.2, §13.4/217)'  (T2.2 helper)
#   tests/test_apply.py: 'test_apply_mixed_bool_float_widens'  (F12-6 tests)
#
# Root: v0.9.14 (initial) died at U3 compile step. Order:
#   F12-6a (OK) → F12-6b (OK) → F12-7 (OK) → U3 (DIED) →
#   T2.2 (never) → F12-6 tests (never) → T2.2 tests (never) →
#   commit (never)
# So v0.9.14 partial state: F12-6a/6b/7 + F12-7 §11.5 note
# applied; U3 + T2.2 + all tests + commit not applied.
# v0.9.14-fix1 applied: apply_or_fail validate= + T2.9 tests + U3.
# Still missing: T2.2 (_collect_call_linenos + assert_no_call_in_source
# receiver_type kwarg), F12-6 tests (3), T2.2 tests (2), commit.
#
# §13.4/222 (new): multi-step patch abort leaves partial state —
# verify_state must check ALL steps (idempotent-safe) so fix-scripts
# know which remain. Every anchor in the verify list must be
# independently re-appliable.
#
# State on disk (from v0.9.14 partial + fix1):
#   F12-6a bool→float widen          — applied (v0.9.14)
#   F12-6b clear RuntimeError        — applied (v0.9.14)
#   F12-7 §11.5 empty-input note     — applied (v0.9.14)
#   apply_or_fail validate= kwarg    — applied (fix1)
#   T2.9 tests (4)                   — applied (fix1)
#   U3 pyproject version 0.9.14      — applied (fix1)
#   T2.2 receiver_type kwarg         — NOT APPLIED
#   F12-6 tests (3)                  — NOT APPLIED
#   T2.2 tests (2)                   — NOT APPLIED
#   commit                           — never reached
#
# fix2: apply T2.2 (helper) + F12-6 tests + T2.2 tests + commit.
#
# Run: uv run python scripts/patch_v0914_fix2.py

from __future__ import annotations
import importlib
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cadw.patch_utils as pu


# ============ (1) T2.2: _collect_call_linenos receiver_type kwarg ============

def t2_2_collect():
    print("\n=== (1) _collect_call_linenos: receiver_type kwarg ===")
    pu.apply_or_fail(
        "cadw/patch_utils.py",
        (
            "def _collect_call_linenos(tree):\n"
            "    \"\"\"Return {attr_or_name: [lineno, ...]} for all Call nodes.\n"
            "\n"
            "    Handles both `obj.attr(...)` (ast.Attribute, use .attr) and\n"
            "    `bare(...)` (ast.Name, use .id). Docstrings/comments are\n"
            "    invisible to ast.walk over Call nodes (§13.4/161, /183, /185).\n"
            "    \"\"\"\n"
            "    import ast as _ast\n"
            "    out: dict = {}\n"
            "    for node in _ast.walk(tree):\n"
            "        if not isinstance(node, _ast.Call):\n"
            "            continue\n"
            "        f = node.func\n"
            "        if isinstance(f, _ast.Name):\n"
            "            out.setdefault(f.id, []).append(node.lineno)\n"
            "        elif isinstance(f, _ast.Attribute):\n"
            "            out.setdefault(f.attr, []).append(node.lineno)\n"
            "    return out\n"
        ),
        (
            "def _collect_call_linenos(tree, *, receiver_type=None):\n"
            "    \"\"\"Return {attr_or_name: [lineno, ...]} for all Call nodes.\n"
            "\n"
            "    Handles both `obj.attr(...)` (ast.Attribute, use .attr) and\n"
            "    `bare(...)` (ast.Name, use .id). Docstrings/comments are\n"
            "    invisible to ast.walk over Call nodes (§13.4/161, /183, /185).\n"
            "\n"
            "    v0.9.14 (T2.2, §13.4/217): optional `receiver_type` filter\n"
            "    (ast class). When set, only Calls where\n"
            "    `node.func.value` is `isinstance(..., receiver_type)` are\n"
            "    collected. Use ast.Subscript for per-row `X[i].method()`,\n"
            "    ast.Call for aggregate `fn(...).method()`.\n"
            "    \"\"\"\n"
            "    import ast as _ast\n"
            "    out: dict = {}\n"
            "    for node in _ast.walk(tree):\n"
            "        if not isinstance(node, _ast.Call):\n"
            "            continue\n"
            "        f = node.func\n"
            "        if receiver_type is not None:\n"
            "            if not (isinstance(f, _ast.Attribute)\n"
            "                    and isinstance(f.value, receiver_type)):\n"
            "                continue\n"
            "        if isinstance(f, _ast.Name):\n"
            "            out.setdefault(f.id, []).append(node.lineno)\n"
            "        elif isinstance(f, _ast.Attribute):\n"
            "            out.setdefault(f.attr, []).append(node.lineno)\n"
            "    return out\n"
        ),
        label="T2.2 _collect",
    )
    print("  OK")


# ============ (2) T2.2: assert_no_call_in_source receiver_type kwarg ============

def t2_2_assert():
    print("\n=== (2) assert_no_call_in_source: receiver_type kwarg ===")
    pu.apply_or_fail(
        "cadw/patch_utils.py",
        (
            "def assert_no_call_in_source(src, *names, fn_name=\"<unknown>\"):\n"
            "    \"\"\"AST: raise AssertionError if any Call node matches `name`.\n"
            "\n"
            "    `name` may be a bare function call (`foo()`) or an attribute\n"
            "    call (`obj.foo()`). Comments and docstrings are invisible\n"
            "    (§13.4/161, /183) — only ast.Call nodes are checked. Replaces\n"
            "    `assert 'foo' not in src`, which catches `# No foo() here`\n"
            "    and `Use CadwDataFrame.foo(...)` docstrings.\n"
            "    \"\"\"\n"
            "    import ast as _ast\n"
            "    import textwrap as _tw\n"
            "    try:\n"
            "        tree = _ast.parse(_tw.dedent(src))\n"
            "    except SyntaxError as e:\n"
            "        raise AssertionError(\n"
            "            f\"assert_no_call_in_source({fn_name}): parse error {e}\"\n"
            "        ) from None\n"
            "    calls = _collect_call_linenos(tree)\n"
        ),
        (
            "def assert_no_call_in_source(\n"
            "    src, *names, fn_name=\"<unknown>\", receiver_type=None,\n"
            "):\n"
            "    \"\"\"AST: raise AssertionError if any Call node matches `name`.\n"
            "\n"
            "    `name` may be a bare function call (`foo()`) or an attribute\n"
            "    call (`obj.foo()`). Comments and docstrings are invisible\n"
            "    (§13.4/161, /183) — only ast.Call nodes are checked. Replaces\n"
            "    `assert 'foo' not in src`, which catches `# No foo() here`\n"
            "    and `Use CadwDataFrame.foo(...)` docstrings.\n"
            "\n"
            "    v0.9.14 (T2.2, §13.4/217): optional `receiver_type` ast class.\n"
            "    When set, only Calls where `node.func.value` is instance\n"
            "    of `receiver_type` match. Common: ast.Subscript for\n"
            "    per-row `X[i].method()` (data-loss anti-pattern);\n"
            "    ast.Call for aggregate `fn(...).method()` (legitimate).\n"
            "    \"\"\"\n"
            "    import ast as _ast\n"
            "    import textwrap as _tw\n"
            "    try:\n"
            "        tree = _ast.parse(_tw.dedent(src))\n"
            "    except SyntaxError as e:\n"
            "        raise AssertionError(\n"
            "            f\"assert_no_call_in_source({fn_name}): parse error {e}\"\n"
            "        ) from None\n"
            "    calls = _collect_call_linenos(tree, receiver_type=receiver_type)\n"
        ),
        label="T2.2 assert",
    )
    print("  OK")


# ============ (3) F12-6 tests ============

F12_6_TESTS = [
    "",
    "",
    "# ============ v0.9.14 (F12-6): bool→float + clear error on mismatch ============",
    "",
    "",
    "def test_apply_mixed_bool_float_widens(tmp_path):",
    "    \"\"\"F12-6a: first group bool, later float → widen to float64.",
    "",
    "    Prior: schema inferred bool from first group; 0.5 at flush →",
    "    ArrowInvalid (cryptic). Fix: bool field + non-bool numeric →",
    "    widen to float64. 0.5 lossless; True becomes 1.0.",
    "    \"\"\"",
    "    import pyarrow.parquet as pq",
    "    df = make_df(tmp_path, {",
    "        'g': ['a', 'a', 'b', 'b'],",
    "        'v': [1, 2, 3, 4],",
    "    })",
    "    def fn(g):",
    "        if g['g'].to_list()[0] == 'a':",
    "            return True  # bool (first group, schema=bool_)",
    "        return 2.5     # float (later group, needs widen)",
    "    result = df.groupby('g').apply(fn)",
    "    out = tmp_path / 'out.parquet'",
    "    result.write_parquet(out)",
    "    rows = {r['g']: r['apply_result'] for r in",
    "            pq.read_table(out).to_pylist()}",
    "    assert rows['a'] == 1.0",
    "    assert rows['b'] == 2.5",
    "",
    "",
    "def test_apply_mixed_str_int_clear_error(tmp_path):",
    "    \"\"\"F12-6b: str then int → RuntimeError with hint, not cryptic",
    "    ArrowInvalid from from_pylist.\"\"\"",
    "    import pytest",
    "    df = make_df(tmp_path, {",
    "        'g': ['a', 'a', 'b', 'b'],",
    "        'v': [1, 2, 3, 4],",
    "    })",
    "    def fn(g):",
    "        if g['g'].to_list()[0] == 'a':",
    "            return 'hello'  # str",
    "        return 5            # int",
    "    with pytest.raises(RuntimeError, match='inconsistent types'):",
    "        df.groupby('g').apply(fn).write_parquet(tmp_path / 'o.parquet')",
    "",
    "",
    "def test_apply_all_bool_stays_bool(tmp_path):",
    "    \"\"\"F12-6a regression: all-bool → schema stays bool (no widen).\"\"\"",
    "    import pyarrow as pa",
    "    import pyarrow.parquet as pq",
    "    df = make_df(tmp_path, {",
    "        'g': ['a', 'a', 'b', 'b'],",
    "        'v': [1, 2, 3, 4],",
    "    })",
    "    def fn(g):",
    "        return True",
    "    result = df.groupby('g').apply(fn)",
    "    out = tmp_path / 'out.parquet'",
    "    result.write_parquet(out)",
    "    schema = pq.read_schema(out)",
    "    assert schema.field('apply_result').type == pa.bool_()",
]


def add_f12_6_tests():
    print("\n=== (3) F12-6 tests ===")
    p = Path("tests/test_apply.py")
    s = p.read_text()
    if "def test_apply_mixed_bool_float_widens" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(F12_6_TESTS))
    pu.assert_compiles(p)
    print("  OK")


# ============ (4) T2.2 tests ============

T2_2_TESTS = [
    "",
    "",
    "# ============ v0.9.14 (T2.2): assert_no_call_in_source receiver_type ============",
    "",
    "",
    "def test_assert_no_call_in_source_receiver_subscript():",
    "    \"\"\"T2.2: ast.Subscript receiver filter catches X[i].as_py() only.\"\"\"",
    "    import ast",
    "    from cadw.patch_utils import assert_no_call_in_source",
    "    # Per-row: X[i].as_py() — Subscript receiver → should match",
    "    src_per_row = 'def f(x, i):\\n    return x[i].as_py()\\n'",
    "    with pytest.raises(AssertionError, match='as_py'):",
    "        assert_no_call_in_source(",
    "            src_per_row, 'as_py', receiver_type=ast.Subscript,",
    "        )",
    "    # Aggregate: pc.sum(x).as_py() — Call receiver → should NOT match",
    "    src_aggregate = 'def f(x):\\n    return pc.sum(x).as_py()\\n'",
    "    assert_no_call_in_source(  # no raise",
    "        src_aggregate, 'as_py', receiver_type=ast.Subscript,",
    "    )",
    "",
    "",
    "def test_assert_no_call_in_source_receiver_none_unchanged():",
    "    \"\"\"Backward compat: receiver_type=None matches any receiver.\"\"\"",
    "    from cadw.patch_utils import assert_no_call_in_source",
    "    src = 'def f(x):\\n    return pc.sum(x).as_py()\\n'",
    "    with pytest.raises(AssertionError, match='as_py'):",
    "        assert_no_call_in_source(src, 'as_py')",
]


def add_t2_2_tests():
    print("\n=== (4) T2.2 tests ===")
    p = Path("tests/test_patch_utils.py")
    s = p.read_text()
    if "def test_assert_no_call_in_source_receiver_subscript" in s:
        print("  already present — skip")
        return
    p.write_text(s + "\n".join(T2_2_TESTS))
    pu.assert_compiles(p)
    print("  OK")


# ============ verify ============

def verify_state():
    print("\n=== verify: v0.9.14 COMPLETE (all steps) ===")
    pu.verify_anchors([
        # fix2 (T2.2 + tests)
        ("cadw/patch_utils.py", "v0.9.14 (T2.2, §13.4/217)"),
        ("cadw/patch_utils.py", "receiver_type=None"),
        ("tests/test_apply.py", "test_apply_mixed_bool_float_widens"),
        ("tests/test_apply.py", "test_apply_mixed_str_int_clear_error"),
        ("tests/test_apply.py", "test_apply_all_bool_stays_bool"),
        ("tests/test_patch_utils.py",
         "test_assert_no_call_in_source_receiver_subscript"),
        ("tests/test_patch_utils.py",
         "test_assert_no_call_in_source_receiver_none_unchanged"),
        # fix1
        ("cadw/patch_utils.py", "v0.9.14-fix1 (T2.9, §13.4/221)"),
        ("cadw/patch_utils.py", 'elif validate == "toml":'),
        ("tests/test_patch_utils.py", "test_apply_or_fail_toml_valid"),
        ("pyproject.toml", 'version = "0.9.14"'),
        # v0.9.14 partial
        ("cadw/dataframe.py", "v0.9.14 (F12-6a)"),
        ("cadw/dataframe.py", "v0.9.14 (F12-6b)"),
        ("cadw/dataframe.py", "§11.5, v0.9.14 F12-7"),
        # v0.9.13/12/11/10/9 regression
        ("cadw/ops/quantile.py", "v0.9.13 (F12-5, S3 perf)"),
        ("cadw/ops/groupby.py", "v0.9.12-fix2 (§13.4/218)"),
        ("cadw/dataframe.py", "def _maybe_upgrade_schema("),
        ("cadw/patch_utils.py", "def apply_or_fail("),
        ("cadw/patch_utils.py", "_tw.dedent(_inspect.getsource"),
    ])
    print("  all anchors present")


# ============ dogfood ============

def dogfood():
    print("\n=== dogfood: v0.9.14 COMPLETE via subprocess ===")

    def _verify_all():
        import ast as _ast
        import inspect
        import textwrap as _tw
        import tomllib
        from pathlib import Path as _P

        # T2.2: receiver_type filter works
        from cadw.patch_utils import assert_no_call_in_source
        src_per_row = "def f(x, i):\n    return x[i].as_py()\n"
        try:
            assert_no_call_in_source(
                src_per_row, "as_py", receiver_type=_ast.Subscript,
            )
            raise AssertionError("T2.2: per-row as_py not flagged")
        except AssertionError as e:
            if "T2.2: per-row" in str(e):
                raise
        # Aggregate NOT flagged by Subscript filter
        src_agg = "def f(x):\n    return pc.sum(x).as_py()\n"
        assert_no_call_in_source(
            src_agg, "as_py", receiver_type=_ast.Subscript,
        )

        # F12-6a: bool widen present
        from cadw.dataframe import CadwDataFrame
        src_pga = _tw.dedent(inspect.getsource(CadwDataFrame._per_group_apply))
        assert "v0.9.14 (F12-6a)" in src_pga
        assert "is_boolean(field.type)" in src_pga
        assert "v0.9.14 (F12-6b)" in src_pga
        assert "inconsistent types across groups" in src_pga
        assert "§11.5, v0.9.14 F12-7" in src_pga

        # F12-6 end-to-end: mixed bool/float widen
        import tempfile
        import pyarrow as pa
        import pyarrow.parquet as pq
        from cadw import CadwDataFrame as _CDF, MemoryGuard as _MG
        with tempfile.TemporaryDirectory() as td:
            import pathlib as _pl
            td = _pl.Path(td)
            t = pa.table({"g": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]})
            src = td / "src.parquet"
            pq.write_table(t, src)
            df = _CDF.from_parquet(
                src, td / "ds", chunk_size=2,
                memory_guard=_MG(max_memory_mb=50),
                spill_dir=td / "spill",
            )
            def fn(g):
                return True if g["g"].to_list()[0] == "a" else 2.5
            result = df.groupby("g").apply(fn)
            out = td / "out.parquet"
            result.write_parquet(out)
            rows = {r["g"]: r["apply_result"]
                    for r in pq.read_table(out).to_pylist()}
            assert rows["a"] == 1.0
            assert rows["b"] == 2.5

        # fix1: U3 pyproject version, apply_or_fail validate=
        pj = _P("pyproject.toml").read_text()
        assert 'version = "0.9.14"' in pj
        data = tomllib.loads(pj)
        assert data["project"]["version"] == "0.9.14"

        from cadw.patch_utils import apply_or_fail
        sig = inspect.signature(apply_or_fail)
        assert "validate" in sig.parameters
        assert sig.parameters["validate"].default is None

        print("  T2.2 receiver_type filter OK")
        print("  F12-6a bool→float widen (source + end-to-end) OK")
        print("  F12-6b clear RuntimeError OK")
        print("  F12-7 §11.5 note OK")
        print("  fix1: apply_or_fail validate= kwarg + pyproject 0.9.14 OK")

    pu.run_verify_in_subprocess(_verify_all)


# ============ pytest ============

def run_pytest():
    print("\n=== pytest: target ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_patch_utils.py",
         "tests/test_apply.py",
         "tests/test_apply_e2e.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** TARGET FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"], timeout=1800,
    )
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "slow"], timeout=900,
    )
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)


# ============ main ============

def main():
    print("=== v0.9.14-fix2 ===")
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    print("  branch: " + r.stdout.strip())

    t2_2_collect()
    t2_2_assert()
    add_f12_6_tests()
    add_t2_2_tests()
    importlib.reload(pu)
    verify_state()
    dogfood()
    run_pytest()

    pu.commit_with_guards(
        "v0.9.14-fix2: complete v0.9.14 (T2.2 helper + F12-6/T2.2 "
        "tests). v0.9.14-fix1 applied U3, apply_or_fail validate=, "
        "T2.9 tests — but verify_state found 2 missing anchors: T2.2 "
        "helper (never applied) and F12-6 tests (never added). Root: "
        "v0.9.14 (initial) died at U3 compile step. Order was "
        "F12-6a (OK) → F12-6b (OK) → F12-7 (OK) → U3 (DIED) → "
        "T2.2 (never) → tests (never) → commit (never). v0.9.14-"
        "fix1 completed U3 + apply_or_fail validate + T2.9 tests, "
        "but T2.2 + F12-6/T2.2 tests remained. §13.4/222 (new): "
        "multi-step patch abort leaves partial state — verify_state "
        "must check ALL steps idempotent-safe so fix-scripts know "
        "which remain. Every anchor in verify list must be "
        "independently re-appliable. fix2: (1) T2.2 _collect_call_"
        "linenos receiver_type kwarg; (2) T2.2 assert_no_call_in_"
        "source receiver_type kwarg; (3) F12-6 tests (bool/float "
        "widen, str/int clear error, all-bool stays bool); "
        "(4) T2.2 tests (Subscript receiver match, backward-compat "
        "default); (5) verify (18 anchors, all applied) + dogfood "
        "(T2.2 filter, F12-6 end-to-end) + pytest + commit. "
        "v0.9.14 COMPLETE: F12-6a/b (bool→float widen, clear "
        "RuntimeError on non-widenable mismatch), F12-7 (§11.5 "
        "empty-input note), T2.2 (precise AST check helper — "
        "receiver_type kwarg), T2.9 (apply_or_fail validate= "
        "toml/json/python/none, format-aware), U3 (pyproject "
        "version 0.1.0→0.9.14). 15 patches this session "
        "(v0.9.8a–v0.9.14-fix2). Deferred: A1 (dataframe.py split "
        "5404 lines), T2.5/T2.6/T2.7 (scaffold v4 tooling), N6 "
        "(Dataset.iter_batches guard, user responsibility), R3 "
        "(planner spill stub, v0.2 backlog). Next: Pingis formal "
        "(fresh snapshot + bench TASK2/TASK3-style) OR A1 split.",
        self_path="scripts/patch_v0914_fix2.py",
        check_whitelist=True,
    )
    print("\n==> v0.9.14-fix2 applied. v0.9.14 COMPLETE.")


if __name__ == "__main__":
    main()
